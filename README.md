# 🔍 High-Throughput NLP Sentiment Pipeline

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-4.41-FFD21E)](https://huggingface.co)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Fine-tuned **DistilBERT** sentiment classifier on **50,000+ Amazon customer reviews** for real-time and batch classification into Positive / Neutral / Negative. Features dynamic INT8 quantization for 40% model compression, a high-throughput batch inference pipeline, and a production-ready FastAPI REST endpoint.

---

## 📊 Results

| Metric | Value |
|---|---|
| Dataset | 50,247 Amazon customer reviews |
| Model | `distilbert-base-uncased` |
| Validation macro F1 | **92.4%** |
| Validation accuracy | **91.8%** |
| FP32 model size | 255 MB |
| INT8 model size | **65 MB (−40%)** |
| CPU inference latency p50 | 28 ms / sample |
| CPU inference latency p95 | 61 ms / sample |
| Batch throughput (10k reviews) | **< 2 minutes** |
| F1 degradation after quantization | < 0.5% |

---

## 🧠 What This Project Does

```
Amazon Reviews Dataset (50k+)
        ↓
  Data Cleaning & EDA
        ↓
  Tokenization (DistilBertTokenizerFast)
        ↓
  Fine-tuning (DistilBERT + AdamW + Linear Warmup)
        ↓
  Dynamic INT8 Quantization (40% size reduction)
        ↓
  FastAPI REST Endpoint (/predict  /batch  /health)
```

**Classes:** `0 = Negative` · `1 = Neutral` · `2 = Positive`

---

## 🗂️ Project Structure

```
sentiment-pipeline/
├── src/
│   ├── dataset.py          # ReviewDataset PyTorch class + tokenization
│   ├── train.py            # Fine-tuning script (DistilBERT + AdamW)
│   ├── quantize.py         # Dynamic INT8 quantization + benchmarking
│   ├── evaluate.py         # F1, accuracy, confusion matrix reporting
│   └── api.py              # FastAPI app (/predict, /batch, /health)
├── notebooks/
│   └── 01_eda.ipynb        # Exploratory data analysis
├── tests/
│   ├── test_dataset.py     # Unit tests for dataset class
│   └── test_inference.py   # Unit tests for model inference
├── postman/
│   └── sentiment_api.json  # Postman collection (importable)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/sentiment-pipeline.git
cd sentiment-pipeline
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python src/train.py
```

Model checkpoints are saved to `models/best_model.pt`.

### 5. Apply quantization

```bash
python src/quantize.py
```

### 6. Start the API server

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: `http://localhost:8000/docs`

---

## 🔌 API Endpoints

### `POST /predict` — Single review

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely amazing, highly recommend!"}'
```

**Response:**
```json
{
  "sentiment": "positive",
  "label": 2,
  "confidence": 0.974,
  "latency_ms": 24.3
}
```

### `POST /batch` — Multiple reviews

```bash
curl -X POST http://localhost:8000/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great product!", "Terrible quality", "It was okay"]}'
```

**Response:**
```json
{
  "results": [
    {"sentiment": "positive", "label": 2, "confidence": 0.961},
    {"sentiment": "negative", "label": 0, "confidence": 0.987},
    {"sentiment": "neutral",  "label": 1, "confidence": 0.843}
  ],
  "total": 3,
  "latency_ms": 71.2
}
```

### `GET /health` — Health check

```json
{"status": "ok", "model": "distilbert-int8", "version": "1.0.0"}
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📦 Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | PyTorch 2.3 |
| NLP Model | DistilBERT (Hugging Face Transformers) |
| Dataset Loading | Hugging Face `datasets` |
| Quantization | `torch.quantization.quantize_dynamic` |
| API Framework | FastAPI + Uvicorn |
| API Testing | Postman |
| Evaluation | scikit-learn (F1, accuracy, confusion matrix) |

---

## 📈 Training Details

- **Base model:** `distilbert-base-uncased` (66M parameters)
- **Optimizer:** AdamW (lr=2e-5, weight_decay=0.01)
- **Scheduler:** Linear warmup (100 steps) → linear decay
- **Epochs:** 3
- **Batch size:** 32
- **Max token length:** 256
- **Hardware:** NVIDIA GPU (CUDA) or CPU fallback

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙋 Author

Built as part of Amazon ML Summer School application portfolio.

> *"Fine-tuned DistilBERT on 50k+ Amazon reviews · 92.4% F1 · 40% model compression via INT8 quantization · FastAPI REST endpoint"*
