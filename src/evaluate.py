import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix
)
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from dataset import load_amazon_reviews, get_dataloaders, LABEL_NAMES

SAVE_DIR   = "models"
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def run_evaluation(model, loader, device) -> dict:
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"]

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds  = logits.argmax(dim=-1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    return {
        "preds":  all_preds,
        "labels": all_labels,
        "f1":     f1_score(all_labels, all_preds, average="macro"),
        "acc":    accuracy_score(all_labels, all_preds),
        "report": classification_report(all_labels, all_preds,
                                        target_names=LABEL_NAMES),
        "cm":     confusion_matrix(all_labels, all_preds),
    }


def plot_confusion_matrix(cm: np.ndarray, save_path: str = None):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Sentiment Classifier")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved confusion matrix → {save_path}")
    plt.show()


def main():
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        os.path.join(SAVE_DIR, "tokenizer")
    )
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )
    model.load_state_dict(
        torch.load(os.path.join(SAVE_DIR, "best_model.pt"), map_location=DEVICE)
    )
    model.to(DEVICE)

    df = load_amazon_reviews(n_samples=5000)
    _, _, test_loader = get_dataloaders(df, tokenizer, batch_size=64)

    print("Evaluating on test set...")
    results = run_evaluation(model, test_loader, DEVICE)

    print(f"\nMacro F1:  {results['f1']:.4f}")
    print(f"Accuracy:  {results['acc']:.4f}")
    print("\n" + results["report"])

    plot_confusion_matrix(results["cm"],
                          save_path=os.path.join(SAVE_DIR, "confusion_matrix.png"))


if __name__ == "__main__":
    main()
