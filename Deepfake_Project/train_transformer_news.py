"""Fine-tune DistilBERT for fake-news text classification.

Requires: pip install torch transformers scikit-learn pandas
Training uses news_dataset/combined_news.csv when present, otherwise the
single CSV used by train_news.py. A GPU is strongly recommended.
"""
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = (PROJECT_DIR / "news_dataset/combined_news.csv"
             if (PROJECT_DIR / "news_dataset/combined_news.csv").exists()
             else PROJECT_DIR / "news_dataset/kaggle_fake_news.csv")
MODEL_OPTIONS = {
    "distilbert": ("distilbert-base-uncased", PROJECT_DIR / "models/news_transformer"),
    "roberta-large": ("FacebookAI/roberta-large", PROJECT_DIR / "models/news_roberta_large"),
    "deberta-v3": ("microsoft/deberta-v3-base", PROJECT_DIR / "models/news_deberta_v3"),
}
MAX_LENGTH = 192
BATCH_SIZE = 8
EPOCHS = 2
SEED = 42


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts, self.labels, self.tokenizer = list(texts), list(labels), tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        encoded = self.tokenizer(self.texts[index], truncation=True, max_length=MAX_LENGTH,
                                 padding="max_length", return_tensors="pt")
        return {key: value.squeeze(0) for key, value in encoded.items()} | {
            "labels": torch.tensor(self.labels[index], dtype=torch.long)}


def evaluate(model, loader, device):
    model.eval()
    actual, predicted, fake_scores = [], [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            output = model(**{key: value.to(device) for key, value in batch.items()})
            probabilities = torch.softmax(output.logits, dim=1)
            actual.extend(labels.cpu().tolist())
            predicted.extend(probabilities.argmax(dim=1).cpu().tolist())
            fake_scores.extend(probabilities[:, 0].cpu().tolist())
    return np.array(actual), np.array(predicted), np.array(fake_scores)


def main():
    parser = argparse.ArgumentParser(description="Train a transformer fake-news classifier")
    parser.add_argument("--model", choices=MODEL_OPTIONS, default="distilbert")
    args = parser.parse_args()
    model_name, model_dir = MODEL_OPTIONS[args.model]
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run prepare_all_news_datasets.py first, then retry.")
    torch.manual_seed(SEED)
    data = pd.read_csv(DATA_PATH, usecols=["text", "label"]).dropna()
    data["text"] = data["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    data = data[data["text"].str.len() >= 20]
    data["label"] = data["label"].astype(str).str.lower().map({"fake": 0, "real": 1})
    data = data.dropna(subset=["label"])
    train_text, val_text, train_label, val_label = train_test_split(
        data["text"], data["label"].astype(int), test_size=0.15, random_state=SEED,
        stratify=data["label"]
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label={0: "fake", 1: "real"},
        label2id={"fake": 0, "real": 1}
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} with {len(train_text):,} rows; validation={len(val_text):,}")
    if device.type == "cpu":
        print("CPU training may take many hours. A CUDA GPU is strongly recommended.")
    model.to(device)
    train_loader = DataLoader(NewsDataset(train_text, train_label, tokenizer), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(NewsDataset(val_text, val_label, tokenizer), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(len(train_loader) * 0.1),
        num_training_steps=len(train_loader) * EPOCHS
    )

    best_f1 = -1.0
    model_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            optimizer.zero_grad()
            loss = model(**{key: value.to(device) for key, value in batch.items()}, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        actual, predicted, fake_scores = evaluate(model, val_loader, device)
        score = f1_score(actual, predicted, average="macro")
        print(f"Epoch {epoch + 1}/{EPOCHS} validation macro-F1: {score:.3f}")
        if score > best_f1:
            best_f1 = score
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            # Only call fake when there is strong evidence, preserving real-news recall.
            threshold = float(np.clip(np.quantile(fake_scores[actual == 1], 0.98), 0.5, 0.95))
            (model_dir / "metadata.json").write_text(json.dumps({
                "fake_threshold": threshold, "max_length": MAX_LENGTH,
                "decision_threshold": 0.5,
                "model_name": model_name,
                "classes": ["fake", "real"]
            }, indent=2), encoding="utf-8")
            print(classification_report(actual, predicted, target_names=["fake", "real"], digits=3))
    print(f"Saved best {args.model} model to {model_dir}")


if __name__ == "__main__":
    main()
