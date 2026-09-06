"""
Builds the trained model artifacts used by app.py.
Run this once before deploying (and commit the .pkl files to the repo,
or let app.py train on the fly at startup - see app.py for the fallback).

Usage:
    python3 train.py
"""

from pathlib import Path

from pipeline import (
    DomainClassifier,
    DuplicateDetector,
    load_problems,
    MODEL_DIR,
)


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    problems = load_problems()
    print(f"Loaded {len(problems)} problem records")

    print("Training domain classifier (char n-grams + Logistic Regression, merged taxonomy)...")
    clf = DomainClassifier()
    clf.train(problems["description"], problems["primary_domain"])
    clf.save(MODEL_DIR / "domain_classifier.pkl")
    print(f"Saved -> {MODEL_DIR / 'domain_classifier.pkl'}")

    # DuplicateDetector is cheap to fit (just TF-IDF on the corpus), so
    # app.py fits it fresh at startup rather than pickling it. Nothing
    # else to train here.

    print("\nDone. Domain classifier is ready for app.py.")


if __name__ == "__main__":
    main()
