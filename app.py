"""
SIH 26043 — Quadruple Helix Innovation Platform (Demo)
Citizen -> Academia -> Industry -> Government

Streamlit demo: paste a citizen complaint, and the app will:
  1. Classify its domain (with confidence)
  2. Check it against existing complaints for duplicates
  3. Recommend matching universities and industry/CSR partners
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline import (
    DomainClassifier,
    DuplicateDetector,
    PartnerMatcher,
    load_problems,
    load_universities,
    load_industry,
    get_all_districts,
    MODEL_DIR,
)

st.set_page_config(
    page_title="Quadruple Helix Innovation Platform — SIH 26043",
    page_icon="🏛️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached resource loading — runs once per app session, not per interaction
# ---------------------------------------------------------------------------

@st.cache_resource
def get_domain_classifier():
    clf = DomainClassifier()
    model_path = MODEL_DIR / "domain_classifier.pkl"
    if model_path.exists():
        clf.load(model_path)
    else:
        # Train on the fly if no cached model is committed to the repo
        problems = load_problems()
        clf.train(problems["description"], problems["primary_domain"])
    return clf


@st.cache_resource
def get_duplicate_detector():
    problems = load_problems()
    detector = DuplicateDetector()
    detector.fit(problems)
    return detector


@st.cache_resource
def get_matcher():
    universities = load_universities()
    industry = load_industry()
    return PartnerMatcher(universities, industry), universities, industry


@st.cache_data
def get_districts(_universities, _industry):
    return get_all_districts(_universities, _industry)


domain_clf = get_domain_classifier()
dup_detector = get_duplicate_detector()
matcher, universities, industry_orgs = get_matcher()
districts = get_districts(universities, industry_orgs)


# ---------------------------------------------------------------------------
# Sidebar — project context
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🏛️ Quadruple Helix")
    st.caption("SIH 2026 — Problem Statement 26043")
    st.markdown(
        """
        **Department of Higher & Technical Education, Government of Jharkhand**

        A pipeline connecting:
        - 👥 **Citizens** who report local problems
        - 🎓 **Academia** (18 HEIs) who can research them
        - 🏭 **Industry/CSR** (25 orgs) who fund/mentor projects
        - 🏛️ **Government** who tracks impact

        Aligned with NEP 2020's push to connect higher education
        to real, local research problems.
        """
    )
    st.divider()
    st.markdown("**How this demo works**")
    st.markdown(
        """
        1. Type a citizen complaint below
        2. The model classifies its domain
        3. It's checked against existing complaints for duplicates
        4. Matching universities + industry partners are recommended
        """
    )
    st.divider()
    st.markdown("**Model notes**")
    st.caption(
        "Domain classifier: char n-gram TF-IDF + Logistic Regression on a "
        "merged 8-category taxonomy (72% CV accuracy vs 62% on the original "
        "11 categories, since the merged categories fixed genuine real-world "
        "label overlap). Duplicate detection: TF-IDF cosine similarity, "
        "domain-gated and clause-aware to handle short-vs-long-formal "
        "phrasing of the same issue; 100% accuracy on 70 labeled test pairs."
    )


# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["📝 Submit a Complaint", "📊 Dataset Overview", "ℹ️ About the Models"])

with tab1:
    st.header("Submit a citizen complaint")
    st.caption("Try English, Hindi-transliterated (Hinglish), or mixed text.")

    example_options = {
        "— pick an example or write your own —": "",
        "Road / infrastructure": "Sadak par bahut bada gaddha hai, roz accident hone ka dar hai",
        "Water supply": "The water supply in our village has been cut off for the past 5 days",
        "Education": "Local school has no proper toilets for girl students",
        "Agriculture": "Farmers are not getting fair price for their crops this season",
        "Environment": "Illegal garbage dumping near the river is causing bad smell and disease",
    }
    example_choice = st.selectbox("Quick examples", list(example_options.keys()))

    complaint_text = st.text_area(
        "Complaint text",
        value=example_options[example_choice],
        height=100,
        placeholder="e.g. Our block has no proper drainage and water logs every monsoon...",
    )

    col_a, col_b = st.columns([2, 1])
    with col_a:
        district_filter = st.selectbox(
            "District (optional — used to prioritize local partners)",
            ["Any district"] + districts,
        )
    with col_b:
        top_k = st.slider("Partners to show", min_value=1, max_value=5, value=3)

    submitted = st.button("🔍 Analyze Complaint", type="primary", use_container_width=True)

    if submitted:
        if not complaint_text.strip():
            st.warning("Please enter a complaint description first.")
        else:
            district_arg = None if district_filter == "Any district" else district_filter

            # --- Stage 1: Domain classification ---
            st.subheader("1️⃣ Domain Classification")
            merged_domain, confidence, expanded_domains, all_scores = domain_clf.predict(complaint_text)

            score_col, label_col = st.columns([1, 2])
            with score_col:
                st.metric("Predicted category", merged_domain.replace("_", " ").title())
                st.metric("Confidence", f"{confidence*100:.1f}%")
                if confidence < 0.35:
                    st.warning("Low confidence — consider routing this to a human officer for domain confirmation.")
            with label_col:
                scores_df = pd.DataFrame(all_scores, columns=["Category", "Score"])
                scores_df["Category"] = scores_df["Category"].str.replace("_", " ").str.title()
                scores_df["Score"] = (scores_df["Score"] * 100).round(1)
                st.bar_chart(scores_df.set_index("Category"))

            st.caption(f"Matches against these original problem domains: {', '.join(expanded_domains)}")

            # --- Stage 2: Duplicate detection ---
            # Gated by the classified domain(s) - comparing across domains
            # produces false positives from shared generic phrasing
            # ("has been X for the past Y days") that isn't a real match.
            st.subheader("2️⃣ Duplicate Check")
            similar = dup_detector.find_similar(complaint_text, top_k=5, domain_filter=expanded_domains)
            any_duplicate = any(s["is_duplicate"] for s in similar)

            if any_duplicate:
                st.error("⚠️ Possible duplicate(s) found in existing records.")
            else:
                st.success("✅ No duplicates found — this looks like a new issue.")

            sim_df = pd.DataFrame(similar)
            sim_df["similarity"] = (sim_df["similarity"] * 100).round(1).astype(str) + "%"
            sim_df["is_duplicate"] = sim_df["is_duplicate"].map({True: "🔴 Duplicate", False: "🟢 Distinct"})
            st.dataframe(
                sim_df.rename(columns={
                    "issue_id": "Issue ID", "description": "Existing Complaint",
                    "district": "District", "similarity": "Similarity", "is_duplicate": "Verdict",
                }),
                use_container_width=True, hide_index=True,
            )

            # --- Stage 3: Partner matching ---
            st.subheader("3️⃣ Recommended Partners")
            uni_col, ind_col = st.columns(2)

            with uni_col:
                st.markdown("**🎓 Matching Universities**")
                uni_matches = matcher.match_universities(expanded_domains, district=district_arg, top_k=top_k)
                if not uni_matches:
                    st.info("No matching universities found for this domain.")
                for u in uni_matches:
                    with st.container(border=True):
                        badge = " 📍 Same district" if u["same_district"] else ""
                        st.markdown(f"**{u['name']}**{badge}")
                        st.caption(f"{u['department']} — {u['district']}")
                        st.caption(f"Faculty lead: {u.get('faculty_lead', 'N/A')}")
                        st.caption(f"Matched on: {', '.join(u['matched_domains'])}")
                        st.caption(f"Capacity: {u.get('active_projects', 0)}/{u.get('max_capacity', 0)} projects active")

            with ind_col:
                st.markdown("**🏭 Matching Industry/CSR Partners**")
                ind_matches = matcher.match_industry(expanded_domains, district=district_arg, top_k=top_k)
                if not ind_matches:
                    st.info("No matching industry partners found for this domain.")
                for c in ind_matches:
                    with st.container(border=True):
                        badge = " 📍 Same district" if c["same_district"] else ""
                        st.markdown(f"**{c['company_name']}**{badge}")
                        st.caption(f"{c['district']}")
                        st.caption(f"Matched on: {', '.join(c['matched_domains'])}")
                        st.caption(f"Funding capacity: ₹{c.get('funding_capacity', 0):,}")
                        st.caption(f"Mentorship available: {'Yes' if c.get('mentorship_available') else 'No'}")


with tab2:
    st.header("Dataset Overview")
    problems_df = load_problems()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Citizen problems", len(problems_df))
    m2.metric("Universities", len(universities))
    m3.metric("Industry/CSR orgs", len(industry_orgs))
    m4.metric("Districts covered", len(districts))

    st.subheader("Problems by domain")
    st.bar_chart(problems_df["primary_domain"].value_counts())

    st.subheader("Problems by district")
    st.bar_chart(problems_df["district"].value_counts())

    with st.expander("View raw problem records"):
        st.dataframe(problems_df, use_container_width=True)

    with st.expander("View university catalog"):
        st.dataframe(pd.json_normalize(universities), use_container_width=True)

    with st.expander("View industry/CSR catalog"):
        st.dataframe(pd.json_normalize(industry_orgs), use_container_width=True)


with tab3:
    st.header("About the models")

    st.subheader("Domain Classifier")
    st.markdown(
        """
        - **Input:** raw complaint text (English, Hinglish, or mixed)
        - **Output:** one of 8 merged domain categories + a confidence score per category
        - **Method:** TF-IDF over character 3-5 grams (robust to spelling variation) +
          Logistic Regression with `class_weight='balanced'`
        - **Why merged categories:** the original 11-domain labels had two clusters of
          genuine real-world overlap — `agriculture`/`rural_livelihoods` and
          `environment`/`sanitation`/`urban_infrastructure`. Confusion-matrix analysis
          showed these were the dominant error sources, not a modeling weakness.
          Merging them raised 5-fold cross-validated accuracy from **62.4% → 72.1%**.
        - **Why not plain Naive Bayes:** `MultinomialNB` is biased toward whichever
          class has more/more-varied training text once classes are imbalanced — on
          the merged (imbalanced) taxonomy its accuracy collapsed to 37%, because it
          defaulted to predicting the largest class. Logistic Regression with balanced
          class weights avoided this entirely.
        """
    )

    st.subheader("Duplicate Detector")
    st.markdown(
        """
        - **Input:** a new complaint + the full corpus of existing complaints
        - **Output:** top-5 most similar existing complaints with a similarity score
        - **Method:** TF-IDF cosine similarity over the same character n-gram features
        - **Threshold:** similarity ≥ 0.3 flags a duplicate. On 70 labeled test pairs
          (35 genuine paraphrase-duplicates, 35 hard negatives), this threshold gave
          **100% accuracy** — duplicates cluster around 0.61 similarity, distinct
          complaints around 0.04, with a clean gap between them.
        """
    )

    st.subheader("Partner Matcher")
    st.markdown(
        """
        - **Input:** the classifier's predicted domain(s) + optional district
        - **Output:** ranked universities and industry/CSR orgs
        - **Method:** rule-based join, not ML — filters universities/industry entries
          whose declared `domains` overlap with the predicted domain(s), then ranks by
          (number of overlapping domains, same district, available capacity/mentorship)
        - **Why not ML here:** this is a well-defined relational join with only ~40
          total partner records — a trained model would add complexity without adding
          accuracy over straightforward rule-based ranking.
        """
    )
