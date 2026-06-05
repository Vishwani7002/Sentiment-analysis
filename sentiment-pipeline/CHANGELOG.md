# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2025-06

### Added
- Fine-tuned DistilBERT on 50,247 Amazon reviews for 3-class sentiment classification
- Dynamic INT8 quantization with 40% model size reduction and <0.5% F1 degradation
- High-throughput batch inference pipeline (10k reviews in <2 minutes)
- FastAPI REST API with `/predict`, `/batch`, and `/health` endpoints
- Postman collection for API testing and documentation
- WeightedRandomSampler for class imbalance handling
- Confusion matrix visualization and per-class F1 reporting
- Unit tests for dataset class and API endpoints
