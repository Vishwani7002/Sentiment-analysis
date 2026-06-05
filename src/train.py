import os
import time
import torch
import numpy as np
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torch.optim import AdamW

from dataset import load_amazon_reviews, get_dataloaders, LABEL_NAMES, MODEL_NAME

EPOCHS       = 3
BATCH_SIZE   = 32
LR           = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
NUM_LABELS   = 3
SAVE_DIR     = "models"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}/{len(loader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, f1


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item()

        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1  = f1_score(all_labels, all_preds, average="macro")
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, f1, acc, all_preds, all_labels


def main():
    print(f"Device: {DEVICE}")

    # Load data + tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    df = load_amazon_reviews(n_samples=50000)
    train_loader, val_loader, test_loader = get_dataloaders(
        df, tokenizer, batch_size=BATCH_SIZE
    )

    # Model
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
    )

    best_val_f1 = 0.0
    history = []

    print("\n" + "=" * 60)
    print("Starting fine-tuning...")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        val_loss, val_f1, val_acc, _, _ = evaluate(model, val_loader, DEVICE)

        elapsed = time.time() - t0
        print(f"  Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}  | Val F1:   {val_f1:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  Epoch time: {elapsed:.1f}s")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_f1": val_f1})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            ckpt_path = os.path.join(SAVE_DIR, "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved best model (val F1={val_f1:.4f}) → {ckpt_path}")

    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pt")))
    _, test_f1, test_acc, test_preds, test_labels = evaluate(model, test_loader, DEVICE)
    print(f"Test F1 (macro): {test_f1:.4f}")
    print(f"Test Accuracy:   {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=LABEL_NAMES))

    tokenizer.save_pretrained(os.path.join(SAVE_DIR, "tokenizer"))
    print(f"\nDone. Best val F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()
