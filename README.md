# Quadruple Helix Innovation Platform — Demo (SIH 26043)

A working prototype of the citizen → academia → industry → government
pipeline proposed for Jharkhand's Dept. of Higher & Technical Education
(NEP 2020 aligned), built for Smart India Hackathon 2026.

**Live pipeline in this app:**
```
citizen complaint (text)
   -> domain classification (char n-grams + Logistic Regression)
   -> duplicate check against existing complaints (TF-IDF cosine similarity)
   -> matched to universities + industry/CSR partners (domain + district join)
```

## Repo structure

```
.
├── app.py                          # Streamlit UI — run this
├── pipeline.py                     # Core classifier / duplicate detector / matcher logic
├── train.py                        # Builds and caches the domain classifier model
├── requirements.txt
├── models/
│   └── domain_classifier.pkl       # pre-trained model (commit this so the app
│                                     starts instantly — see note below)
└── data/
    ├── problems_dataset_final.csv
    ├── universities_dataset_final.json
    ├── industry_csr_dataset_final.json
    ├── projects_dataset_final.json
    └── duplicate_pairs_dataset.csv
```

## Run locally

```bash
pip install -r requirements.txt
python3 train.py        # builds models/domain_classifier.pkl (~1-2 sec, tiny dataset)
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

Note: if you skip `train.py`, `app.py` will train the model automatically
on first load anyway (it's a fast, tiny dataset) — but committing the
`.pkl` to the repo makes cloud deployment start instantly.

## Deploy on Streamlit Community Cloud (free, via GitHub)

1. **Create a GitHub repo** and push this entire folder to it:
   ```bash
   git init
   git add .
   git commit -m "Initial commit — Quadruple Helix demo"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```

2. **Go to** [share.streamlit.io](https://share.streamlit.io) and sign in
   with your GitHub account.

3. Click **"New app"**, then:
   - Repository: `<your-username>/<repo-name>`
   - Branch: `main`
   - Main file path: `app.py`

4. Click **Deploy**. Streamlit Cloud installs `requirements.txt`
   automatically and gives you a public URL
   (`https://<something>.streamlit.app`) — that's the link you'll show
   your teacher.

5. Any time you `git push` a change, the deployed app auto-updates in
   ~1 minute.

## What to point out during the demo

- **The confidence score matters as much as the label** — try a genuinely
  ambiguous complaint (e.g. the "pothole" example) and show the confidence
  drops below 35%, which the app flags for human review rather than
  auto-routing blindly.
- **The duplicate detector isn't guessing** — the 0.3 similarity threshold
  was chosen by testing against 70 hand-labeled pairs and got 100%
  accuracy on them; show the "About the Models" tab for the numbers.
- **The domain taxonomy was redesigned from evidence, not assumption** —
  the confusion matrix on the original 11 domains showed
  `agriculture`/`rural_livelihoods` and
  `environment`/`sanitation`/`urban_infrastructure` were the two biggest
  error clusters. Merging them (while still matching against all their
  original domains for university/industry pairing) raised accuracy from
  62% to 72%. This is a good story for judges: you diagnosed *why* the
  model was wrong, not just tuned parameters blindly.
- **The matching stage is deliberately rule-based, not ML** — with ~40
  total partner records, a trained model would be overkill; a
  transparent, explainable ranking is actually the right engineering
  choice here, and worth saying so explicitly.

## Dataset provenance

All datasets are **synthetic but structurally realistic** (real Jharkhand
districts, plausible HEI names, audited for referential integrity — see
the original `AUDIT_REPORT.md` if you kept it). For a hackathon prototype
this is standard and expected; be upfront about it if asked.
