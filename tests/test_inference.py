import pytest
from fastapi.testclient import TestClient
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))


from unittest.mock import patch, MagicMock
import torch


@pytest.fixture
def mock_model_state():
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids":      torch.zeros(1, 256, dtype=torch.long),
        "attention_mask": torch.ones(1, 256, dtype=torch.long),
    }

    mock_logits = torch.tensor([[0.1, 0.2, 0.9]])  # → positive
    mock_output = MagicMock()
    mock_output.logits = mock_logits

    mock_model = MagicMock()
    mock_model.return_value = mock_output

    return {"model": mock_model, "tokenizer": mock_tokenizer}


def test_predict_returns_valid_sentiment(mock_model_state):
    with patch("api.model_state", mock_model_state):
        from src.api import app
        client = TestClient(app)
        response = client.post("/predict", json={"text": "This is a great product!"})

    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ["negative", "neutral", "positive"]
    assert 0 <= data["confidence"] <= 1
    assert "latency_ms" in data


def test_predict_returns_positive_for_positive_text(mock_model_state):
    with patch("api.model_state", mock_model_state):
        from src.api import app
        client = TestClient(app)
        response = client.post("/predict", json={"text": "Amazing product, love it!"})

    assert response.json()["sentiment"] == "positive"
    assert response.json()["label"] == 2


def test_batch_predict_returns_list(mock_model_state):
    texts = ["Great!", "Terrible.", "It was okay."]
    with patch("api.model_state", mock_model_state):
        from src.api import app
        client = TestClient(app)
        response = client.post("/batch", json={"texts": texts})

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == len(texts)
    assert len(data["results"]) == len(texts)


def test_health_endpoint():
    from src.api import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_rejects_empty_text(mock_model_state):
    with patch("api.model_state", mock_model_state):
        from src.api import app
        client = TestClient(app)
        response = client.post("/predict", json={"text": ""})

    assert response.status_code == 422


def test_predict_rejects_missing_text(mock_model_state):
    with patch("api.model_state", mock_model_state):
        from src.api import app
        client = TestClient(app)
        response = client.post("/predict", json={})

    assert response.status_code == 422
