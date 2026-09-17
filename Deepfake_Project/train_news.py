"""Train a fake-news text classifier from news_dataset/news.csv.

The CSV must contain a `text` column and a `label` column. Labels may be
`fake`/`real` (recommended) or 0/1, where 0 means fake and 1 means real.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
import numpy as np


COMBINED_DATA_PATH = Path("news_dataset/combined_news.csv")
DATA_PATH = COMBINED_DATA_PATH if COMBINED_DATA_PATH.exists() else Path("news_dataset/kaggle_fake_news.csv")
MODEL_PATH = Path("models/fake_news_model.joblib")
META_PATH = Path("models/fake_news_meta.json")


def normalise_labels(labels):
    labels = labels.astype(str).str.strip().str.lower()
    mapping = {"fake": "fake", "false": "fake", "0": "fake",
               "real": "real", "true": "real", "1": "real"}
    result = labels.map(mapping)
    if result.isna().any():
        values = ", ".join(sorted(labels[result.isna()].unique())[:5])
        raise ValueError(f"Unknown label(s): {values}. Use fake/real or 0/1.")
    return result


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Run prepare_all_news_datasets.py, or add news_dataset/kaggle_fake_news.csv with text,label columns")
    data = pd.read_csv(DATA_PATH)
    required = {"text", "label"}
    if not required.issubset(data.columns):
        raise ValueError("CSV must contain columns named: text,label")

    data = data.dropna(subset=["text", "label"]).copy()
    # Kaggle datasets often provide a headline as well as article text. Use both
    # when available, while keeping simple text,label CSV files compatible.
    if "title" in data.columns:
        data["text"] = data["title"].fillna("").astype(str) + ". " + data["text"].astype(str)
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"].str.len() >= 20]
    data["label"] = normalise_labels(data["label"])
    if data["label"].nunique() != 2:
        raise ValueError("The CSV needs examples of both fake and real news.")

    train_text, val_text, train_labels, val_labels = train_test_split(
        data["text"], data["label"], test_size=0.20, random_state=42,
        stratify=data["label"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english",
                                  ngram_range=(1, 2), min_df=2, max_df=0.95,
                                  sublinear_tf=True, max_features=150_000)),
        ("classifier", LogisticRegression(C=2.0, max_iter=2_000,
                                           class_weight="balanced")),
    ])
    model.fit(train_text, train_labels)

    predictions = model.predict(val_text)
    print(classification_report(val_labels, predictions, digits=3))

    # A report should need strong evidence before it is labelled fake. This
    # threshold keeps 98% of validation real-news samples classified as real.
    probabilities = model.predict_proba(val_text)
    fake_column = list(model.classes_).index("fake")
    real_fake_scores = probabilities[val_labels.to_numpy() == "real", fake_column]
    fake_threshold = float(np.clip(np.quantile(real_fake_scores, 0.98), 0.5, 0.95))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    META_PATH.write_text(json.dumps({"fake_threshold": fake_threshold,
                                     "classes": list(model.classes_)}, indent=2),
                         encoding="utf-8")
    print(f"Saved {MODEL_PATH}; fake threshold={fake_threshold:.3f}")


if __name__ == "__main__":
    main()
