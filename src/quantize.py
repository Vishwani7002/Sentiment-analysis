import os
import time
import torch
import numpy as np
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from sklearn.metrics import f1_score
from torch.quantization import quantize_dynamic

SAVE_DIR   = "models"
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 3


def load_model(checkpoint_path: str) -> DistilBertForSequenceClassification:
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def apply_quantization(model) -> torch.nn.Module:
    """Apply dynamic INT8 quantization to all Linear layers."""
    quantized = quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    quantized.eval()
    return quantized


def get_model_size_mb(model) -> float:
    tmp_path = os.path.join(SAVE_DIR, "_tmp_size_check.pt")
    torch.save(model.state_dict(), tmp_path)
    size_mb = os.path.getsize(tmp_path) / 1e6
    os.remove(tmp_path)
    return size_mb


def benchmark_latency(model, tokenizer, texts: list, n_runs: int = 100) -> dict:
    latencies = []
    for text in texts[:n_runs]:
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        padding="max_length", max_length=256)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(**enc)
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "mean_ms": round(float(np.mean(latencies)), 2),
    }


def evaluate_f1(model, tokenizer, texts: list, labels: list) -> float:
    preds = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        padding="max_length", max_length=256)
        with torch.no_grad():
            logits = model(**enc).logits
        preds.append(logits.argmax(dim=-1).item())
    return f1_score(labels, preds, average="macro")


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        os.path.join(SAVE_DIR, "tokenizer")
    )

    print("Loading FP32 model...")
    fp32_model = load_model(os.path.join(SAVE_DIR, "best_model.pt"))

    print("Applying dynamic INT8 quantization...")
    int8_model = apply_quantization(fp32_model)

    fp32_size = get_model_size_mb(fp32_model)
    int8_size = get_model_size_mb(int8_model)
    reduction = (1 - int8_size / fp32_size) * 100

    print("\nModel Size")
    print(f"  FP32 model: {fp32_size:.1f} MB")
    print(f"  INT8 model: {int8_size:.1f} MB")
    print(f"  Reduction:  {reduction:.1f}%")

    sample_texts = [
        "This product is absolutely amazing, best purchase ever!",
        "Complete waste of money, broke after one day.",
        "It was okay, nothing special really.",
        "Exceeded my expectations in every way.",
        "Would not recommend, very disappointing quality.",
    ] * 20  

    print("\n── Latency Benchmark (100 samples) ─────────────────")
    fp32_lat = benchmark_latency(fp32_model, tokenizer, sample_texts)
    int8_lat = benchmark_latency(int8_model, tokenizer, sample_texts)

    print(f"  FP32 → p50: {fp32_lat['p50_ms']} ms | p95: {fp32_lat['p95_ms']} ms")
    print(f"  INT8 → p50: {int8_lat['p50_ms']} ms | p95: {int8_lat['p95_ms']} ms")
    print(f"  Speedup: {fp32_lat['p50_ms'] / int8_lat['p50_ms']:.2f}x")

    int8_path = os.path.join(SAVE_DIR, "model_int8.pt")
    torch.save(int8_model.state_dict(), int8_path)
    print(f"\n✓ Saved INT8 model → {int8_path}")
    print(f"  Final size: {int8_size:.1f} MB (was {fp32_size:.1f} MB, -{reduction:.0f}%)")


if __name__ == "__main__":
    main()
