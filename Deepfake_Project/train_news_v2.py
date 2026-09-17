"""Clean V2 text-pattern classifier with source-aware evaluation.

This is not a fact-checking engine: it reports an uncertain result for claims
that are not strongly represented by the training data.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

DATA_PATH = Path("news_dataset/combined_news.csv")
MODEL_PATH = Path("models/fake_news_v2.joblib")
META_PATH = Path("models/fake_news_v2_meta.json")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run prepare_all_news_datasets.py first.")
    data = pd.read_csv(DATA_PATH, usecols=["text", "label", "source"]).dropna()
    data["text"] = data["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    data = data[data["text"].str.len() >= 20]
    data = data[data["label"].isin(["fake", "real"])]
    # Keep an entire dataset source out of evaluation. This is harder but more
    # honest than random rows from the same source in both train and test sets.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(data["text"], data["label"], data["source"]))
    train, test = data.iloc[train_idx], data.iloc[test_idx]
    print("Held-out source(s):", ", ".join(sorted(test["source"].unique())))
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2),
                                  min_df=3, max_df=0.95, sublinear_tf=True, max_features=100_000)),
        ("classifier", LogisticRegression(C=1.0, max_iter=2_000, class_weight="balanced")),
    ])
    model.fit(train["text"], train["label"])
    predicted = model.predict(test["text"])
    print(classification_report(test["label"], predicted, digits=3))
    print("Cross-source macro-F1:", round(f1_score(test["label"], predicted, average="macro"), 3))
    # Fit final model on all available training data after reporting an honest
    # cross-source score. The probability margin controls the Uncertain result.
    model.fit(data["text"], data["label"])
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    META_PATH.write_text(json.dumps({"version": "v2", "uncertain_low": 0.35,
                                     "uncertain_high": 0.65,
                                     "classes": list(model.classes_)}, indent=2), encoding="utf-8")
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
