"""Standardise WELFake, FakeNewsNet, and FEVER into one CSV.

Place downloaded files in the layout documented in news_dataset/DATASETS.md,
then run: python prepare_all_news_datasets.py
"""
import json
from pathlib import Path

import pandas as pd


RAW = Path("news_dataset/raw")
OUTPUT = Path("news_dataset/combined_news.csv")


def first_existing(*paths):
    return next((path for path in paths if path.exists()), None)


def standardise(frame, text, label, source):
    result = pd.DataFrame({"text": frame[text].fillna("").astype(str),
                           "label": frame[label].astype(str).str.lower(),
                           "source": source})
    return result[result["text"].str.len() >= 20]


def load_welfake():
    path = first_existing(RAW / "welfake" / "WELFake_Dataset.csv",
                          Path("news_dataset/kaggle_fake_news.csv"))
    if path is None:
        return pd.DataFrame(columns=["text", "label", "source"])
    data = pd.read_csv(path)
    if not {"text", "label"}.issubset(data.columns):
        print(f"Skipping {path}: it does not have text,label columns.")
        return pd.DataFrame(columns=["text", "label", "source"])
    text = data.get("title", "").fillna("").astype(str) + ". " + data["text"].fillna("").astype(str)
    labels = data["label"].map({0: "fake", 1: "real"})
    return pd.DataFrame({"text": text, "label": labels, "source": "welfake"}).dropna(subset=["label"])


def load_fakenewsnet():
    folder = first_existing(RAW / "fakenewsnet" / "dataset",
                            RAW / "FakeNewsNet-master" / "dataset")
    parts = []
    for filename in ("politifact_fake.csv", "gossipcop_fake.csv"):
        path = folder / filename if folder else None
        if path and path.exists():
            data = pd.read_csv(path)
            parts.append(pd.DataFrame({"text": data["title"], "label": "fake", "source": "fakenewsnet"}))
    for filename in ("politifact_real.csv", "gossipcop_real.csv"):
        path = folder / filename if folder else None
        if path and path.exists():
            data = pd.read_csv(path)
            parts.append(pd.DataFrame({"text": data["title"], "label": "real", "source": "fakenewsnet"}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["text", "label", "source"])


def load_fever():
    path = first_existing(RAW / "fever" / "train.jsonl", RAW / "train.jsonl")
    if path is None:
        return pd.DataFrame(columns=["text", "label", "source"])
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            label = {"REFUTES": "fake", "SUPPORTS": "real"}.get(item.get("label"))
            if label:
                rows.append({"text": item["claim"], "label": label, "source": "fever"})
    # NOT ENOUGH INFO is intentionally excluded; it cannot be called fake.
    return pd.DataFrame(rows)


def main():
    frames = [load_welfake(), load_fakenewsnet(), load_fever()]
    data = pd.concat(frames, ignore_index=True).dropna(subset=["text", "label"])
    data["text"] = data["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    data = data[data["text"].str.len() >= 20].drop_duplicates(subset=["text"])
    if data.empty:
        raise FileNotFoundError("No datasets found. Follow news_dataset/DATASETS.md first.")
    data.to_csv(OUTPUT, index=False)
    print(data.groupby(["source", "label"]).size())
    print(f"Saved {len(data):,} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
