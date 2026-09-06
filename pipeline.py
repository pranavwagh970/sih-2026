"""
Core pipeline logic for the SIH 26043 jansetu platform.

Three stages, each independently testable:
  1. DomainClassifier   - text -> domain category (+ confidence)
  2. DuplicateDetector  - text -> similar existing complaints (+ similarity score)
  3. PartnerMatcher     - domain (+ optional district) -> matching universities & industry/CSR orgs

Design notes (from real experiments on the dataset, not assumptions):
  - Char n-grams (3-5), not word n-grams: robust to Hinglish spelling variation.
  - Domain classifier trained on a MERGED 8-class taxonomy, not the original 11:
      agriculture + rural_livelihoods            -> agri_rural
      environment + sanitation + urban_infrastructure -> env_infra_sanitation
    These pairs were the dominant confusion sources in the original 11-class
    model (real-world overlap, not a modeling failure) and merging them
    measured +10pts accuracy. For MATCHING against universities/industry
    (which still use the original 11-domain vocabulary), the merged
    prediction is expanded back to its member domains, so no accuracy is
    lost on the matching side.
  - Logistic Regression (class_weight='balanced') outperformed both
    MultinomialNB and ComplementNB on the merged taxonomy in testing.
  - Duplicate detection uses TF-IDF char-ngram cosine similarity, not a
    trained classifier: on the 70 labeled pairs, a similarity >= 0.3
    threshold gave 100% accuracy (duplicates cluster ~0.61 similarity,
    non-duplicates ~0.04) - clean enough that a learned classifier would
    add complexity without adding accuracy.
  - IMPORTANT caveat found during testing: whole-document cosine similarity
    breaks down when comparing a short complaint against a much longer,
    formally-worded one describing the same issue - the extra words in
    the long version dilute the vector and pull similarity below 0.3 even
    for genuine duplicates. The 70 labeled pairs never tested this case
    (they're all similar-length pairs, ~13-15 words each), so the original
    100% accuracy score didn't catch it. Fix: also compare the short text
    against each clause/chunk of the longer one and take the best score,
    not just the whole-document score. This still scores 100% on the
    original labeled pairs and correctly catches the length-mismatch case.
"""

import json
import pickle
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"

MERGE_MAP = {
    "rural_livelihoods": "agri_rural",
    "agriculture": "agri_rural",
    "environment": "env_infra_sanitation",
    "sanitation": "env_infra_sanitation",
    "urban_infrastructure": "env_infra_sanitation",
}

# Reverse map: merged category -> list of original domains it stands for.
# Domains that were never merged map to themselves.
REVERSE_MERGE_MAP = {}
for original, merged in MERGE_MAP.items():
    REVERSE_MERGE_MAP.setdefault(merged, []).append(original)
for original in [
    "water_management", "healthcare", "education", "energy",
    "accessibility", "public_administration",
]:
    REVERSE_MERGE_MAP.setdefault(original, []).append(original)

DUPLICATE_SIMILARITY_THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# Stage 1: Domain classification
# ---------------------------------------------------------------------------

class DomainClassifier:
    def __init__(self):
        self.pipe = None

    def train(self, descriptions, primary_domains):
        merged_labels = [MERGE_MAP.get(d, d) for d in primary_domains]
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.pipe = Pipeline([("tfidf", vectorizer), ("clf", clf)])
        self.pipe.fit(descriptions, merged_labels)
        return self

    def predict(self, text: str):
        """Returns (merged_domain, confidence, expanded_original_domains, all_scores)."""
        proba = self.pipe.predict_proba([text])[0]
        classes = self.pipe.classes_
        best_idx = proba.argmax()
        merged_domain = classes[best_idx]
        confidence = float(proba[best_idx])
        expanded = REVERSE_MERGE_MAP.get(merged_domain, [merged_domain])
        all_scores = sorted(
            zip(classes, proba), key=lambda x: x[1], reverse=True
        )
        return merged_domain, confidence, expanded, all_scores

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.pipe, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.pipe = pickle.load(f)
        return self


# ---------------------------------------------------------------------------
# Stage 2: Duplicate detection
# ---------------------------------------------------------------------------

class DuplicateDetector:
    def __init__(self, threshold: float = DUPLICATE_SIMILARITY_THRESHOLD):
        self.threshold = threshold
        self.vectorizer = None
        self.corpus_vectors = None
        self.corpus_df = None

    def fit(self, problems_df: pd.DataFrame):
        self.corpus_df = problems_df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        self.corpus_vectors = self.vectorizer.fit_transform(self.corpus_df["description"])
        return self

    @staticmethod
    def _split_clauses(doc: str) -> list:
        """Splits a long description into shorter clauses so a short query
        isn't unfairly diluted when compared against a much longer document."""
        chunks = re.split(r"[,;]| and | but | despite | which | that ", doc)
        chunks = [c.strip() for c in chunks if len(c.strip()) > 5]
        return chunks or [doc]

    def _best_similarity(self, query_vec, doc_index: int, doc_text: str) -> float:
        """Whole-document similarity, or the best-matching clause similarity
        if that's higher - handles short-query-vs-long-formal-doc cases."""
        whole_sim = float(cosine_similarity(query_vec, self.corpus_vectors[doc_index])[0][0])
        if len(doc_text.split()) <= 20:
            # short enough that whole-document comparison is already fair
            return whole_sim
        clause_sims = [
            float(cosine_similarity(query_vec, self.vectorizer.transform([c]))[0][0])
            for c in self._split_clauses(doc_text)
        ]
        return max(whole_sim, max(clause_sims, default=0.0))

    def find_similar(self, text: str, top_k: int = 5, domain_filter: list = None):
        """Returns a list of dicts: issue_id, description, district, similarity, is_duplicate.

        domain_filter: if given, only compares against corpus rows whose
        primary_domain is in this list. Strongly recommended - without it,
        two complaints in totally different domains (e.g. a power outage
        and a water outage) can still score a high "duplicate" similarity
        purely from shared generic phrasing template ("has been X for the
        past Y days"), which is a false positive, not a real duplicate.
        """
        candidate_df = self.corpus_df
        candidate_vectors = self.corpus_vectors
        if domain_filter:
            mask = self.corpus_df["primary_domain"].isin(domain_filter)
            candidate_df = self.corpus_df[mask]
            candidate_vectors = self.corpus_vectors[mask.values]
            if candidate_df.empty:
                return []

        qv = self.vectorizer.transform([text])
        whole_sims = cosine_similarity(qv, candidate_vectors)[0]

        # Cheap first pass: take a wider candidate pool by whole-document
        # similarity, then re-score candidates with the clause-aware method
        # (avoids recomputing clause similarity for every candidate row).
        pool_size = min(max(top_k * 4, 20), len(candidate_df))
        candidate_pool = whole_sims.argsort()[::-1][:pool_size]

        scored = []
        for pos in candidate_pool:
            row = candidate_df.iloc[pos]
            # locate this row's vector within candidate_vectors by position
            doc_vec_index = pos
            whole_sim = float(cosine_similarity(qv, candidate_vectors[doc_vec_index])[0][0])
            if len(row["description"].split()) <= 20:
                best_sim = whole_sim
            else:
                clause_sims = [
                    float(cosine_similarity(qv, self.vectorizer.transform([c]))[0][0])
                    for c in self._split_clauses(row["description"])
                ]
                best_sim = max(whole_sim, max(clause_sims, default=0.0))
            scored.append((best_sim, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, row in scored[:top_k]:
            results.append({
                "issue_id": row["issue_id"],
                "description": row["description"],
                "district": row.get("district", ""),
                "similarity": float(sim),
                "is_duplicate": bool(sim >= self.threshold),
            })
        return results


# ---------------------------------------------------------------------------
# Stage 3: University + Industry/CSR matching
# ---------------------------------------------------------------------------

class PartnerMatcher:
    def __init__(self, universities: list, industry_orgs: list):
        self.universities = universities
        self.industry_orgs = industry_orgs

    def match_universities(self, domains: list, district: str = None, top_k: int = 5):
        scored = []
        for u in self.universities:
            overlap = set(u.get("domains", [])) & set(domains)
            if not overlap:
                continue
            same_district = district is not None and u.get("district") == district
            capacity_left = u.get("max_capacity", 0) - u.get("active_projects", 0)
            score = (len(overlap), same_district, capacity_left)
            scored.append((score, u, overlap))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {**u, "matched_domains": list(overlap), "same_district": district is not None and u.get("district") == district}
            for _, u, overlap in scored[:top_k]
        ]

    def match_industry(self, domains: list, district: str = None, top_k: int = 5):
        scored = []
        for c in self.industry_orgs:
            overlap = set(c.get("domains", [])) & set(domains)
            if not overlap:
                continue
            same_district = district is not None and c.get("district") == district
            capacity_left = c.get("max_projects", 0) - c.get("active_projects", 0)
            score = (len(overlap), same_district, c.get("mentorship_available", False), capacity_left)
            scored.append((score, c, overlap))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {**c, "matched_domains": list(overlap), "same_district": district is not None and c.get("district") == district}
            for _, c, overlap in scored[:top_k]
        ]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_problems(path=None) -> pd.DataFrame:
    path = path or (DATA_DIR / "problems_dataset_final.csv")
    return pd.read_csv(path)


def load_universities(path=None) -> list:
    path = path or (DATA_DIR / "universities_dataset_final.json")
    with open(path) as f:
        return json.load(f)


def load_industry(path=None) -> list:
    path = path or (DATA_DIR / "industry_csr_dataset_final.json")
    with open(path) as f:
        return json.load(f)


def get_all_districts(universities, industry_orgs) -> list:
    districts = set()
    for u in universities:
        if u.get("district"):
            districts.add(u["district"])
    for c in industry_orgs:
        if c.get("district"):
            districts.add(c["district"])
    return sorted(districts)
