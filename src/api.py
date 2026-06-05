import os
import time
import torch
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from torch.quantization import quantize_dynamic

# ─── Config ───────────────────────────────────────────────────────────────────
SAVE_DIR    = "models"
MODEL_NAME  = "distilbert-base-uncased"
NUM_LABELS  = 3
MAX_LEN     = 256
BATCH_CHUNK = 32  # process batch in chunks of this size
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}

# Global model state
model_state = {}


# ─── Startup / shutdown ───────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup, release on shutdown."""
    print("Loading model and tokenizer...")

    tokenizer_path = os.path.join(SAVE_DIR, "tokenizer")
    model_path     = os.path.join(SAVE_DIR, "best_model.pt")

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        tokenizer_path if os.path.exists(tokenizer_path) else MODEL_NAME
    )

    base_model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )
    if os.path.exists(model_path):
        base_model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"Loaded checkpoint from {model_path}")

    # Apply quantization for faster CPU inference
    model = quantize_dynamic(base_model, {torch.nn.Linear}, dtype=torch.qint8)
    model.eval()

    model_state["model"]     = model
    model_state["tokenizer"] = tokenizer
    print("Model ready.")
    yield
    model_state.clear()


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sentiment Analysis API",
    description="Fine-tuned DistilBERT for Amazon review sentiment classification.",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Schemas ──────────────────────────────────────────────────────────────────
class SingleRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000,
                      example="This product is absolutely amazing!")

class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=500,
                             example=["Great product!", "Terrible quality", "It was okay"])

class PredictionResult(BaseModel):
    sentiment:   str
    label:       int
    confidence:  float
    latency_ms:  float | None = None

class BatchResponse(BaseModel):
    results:    List[PredictionResult]
    total:      int
    latency_ms: float


# ─── Inference helpers ────────────────────────────────────────────────────────
def run_inference(texts: list) -> list:
    """
    Run inference on a list of texts in chunks.
    Returns list of (label_idx, confidence) tuples.
    """
    model     = model_state["model"]
    tokenizer = model_state["tokenizer"]
    results   = []

    for i in range(0, len(texts), BATCH_CHUNK):
        chunk = texts[i : i + BATCH_CHUNK]
        enc = tokenizer(
            chunk,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        for j in range(len(chunk)):
            label_idx   = probs[j].argmax().item()
            confidence  = round(probs[j].max().item(), 4)
            results.append((label_idx, confidence))

    return results


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    """Health check endpoint for load balancers."""
    return {
        "status": "ok",
        "model": "distilbert-int8",
        "version": "1.0.0",
    }


@app.post("/predict", response_model=PredictionResult, tags=["Inference"])
def predict(req: SingleRequest):
    """
    Predict sentiment for a single review text.

    Returns sentiment label (positive/neutral/negative),
    predicted class index, and confidence score.
    """
    if "model" not in model_state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()
    (label_idx, confidence), = run_inference([req.text])
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return PredictionResult(
        sentiment=LABEL_NAMES[label_idx],
        label=label_idx,
        confidence=confidence,
        latency_ms=latency_ms,
    )


@app.post("/batch", response_model=BatchResponse, tags=["Inference"])
def batch_predict(req: BatchRequest):
    """
    Predict sentiment for a batch of review texts.

    Processes up to 500 texts per request in chunks of 32.
    Returns individual predictions and total batch latency.
    """
    if "model" not in model_state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()
    raw = run_inference(req.texts)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    results = [
        PredictionResult(
            sentiment=LABEL_NAMES[label_idx],
            label=label_idx,
            confidence=confidence,
        )
        for label_idx, confidence in raw
    ]

    return BatchResponse(results=results, total=len(results), latency_ms=latency_ms)
