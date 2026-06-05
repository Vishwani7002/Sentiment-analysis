import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import DistilBertTokenizerFast
import pandas as pd
import numpy as np
from datasets import load_dataset
from collections import Counter


LABEL_MAP = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}  # stars → {0:neg, 1:neu, 2:pos}
LABEL_NAMES = ["negative", "neutral", "positive"]
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256


def load_amazon_reviews(n_samples: int = 50000, seed: int = 42) -> pd.DataFrame:
    print("Loading dataset from Hugging Face Hub...")
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_review_All_Beauty",
        split="full",
        trust_remote_code=True,
    )
    df = dataset.to_pandas()[["text", "rating"]].dropna()
    df["label"] = df["rating"].astype(int).map(LABEL_MAP)
    df = df.dropna(subset=["label"]).astype({"label": int})

    df = (
        df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), n_samples // 3), random_state=seed))
        .reset_index(drop=True)
    )
    print(f"Dataset loaded: {len(df)} samples")
    print(df["label"].value_counts().to_string())
    return df[["text", "label"]]


def clean_text(text: str) -> str:
    import re
    text = re.sub(r"<[^>]+>", " ", text)      
    text = re.sub(r"\s+", " ", text).strip()    
    return text[:2000]                           
class ReviewDataset(Dataset):
   
    def __init__(self, texts: list, labels: list, tokenizer, max_len: int = MAX_LEN):
        cleaned = [clean_text(t) for t in texts]
        self.encodings = tokenizer(
            cleaned,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def make_weighted_sampler(labels: list) -> WeightedRandomSampler:
    counts = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def get_dataloaders(
    df: pd.DataFrame,
    tokenizer,
    batch_size: int = 32,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42,
) -> tuple:

    from sklearn.model_selection import train_test_split

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        texts, labels, test_size=(val_split + test_split), random_state=seed, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp
    )

    train_ds = ReviewDataset(X_train, y_train, tokenizer)
    val_ds   = ReviewDataset(X_val,   y_val,   tokenizer)
    test_ds  = ReviewDataset(X_test,  y_test,  tokenizer)

    sampler = make_weighted_sampler(y_train)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    df = load_amazon_reviews(n_samples=1000)
    train_loader, val_loader, test_loader = get_dataloaders(df, tokenizer, batch_size=8)
    batch = next(iter(train_loader))
    print("Sample batch keys:", list(batch.keys()))
    print("Input IDs shape:", batch["input_ids"].shape)
