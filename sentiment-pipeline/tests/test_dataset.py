"""
test_dataset.py
---------------
Unit tests for the ReviewDataset class and helper functions.
Run with: pytest tests/ -v
"""

import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from unittest.mock import patch, MagicMock
from dataset import ReviewDataset, clean_text, make_weighted_sampler, LABEL_MAP


# ─── clean_text ───────────────────────────────────────────────────────────────

def test_clean_text_removes_html():
    result = clean_text("<b>Great product</b>!")
    assert "<b>" not in result
    assert "Great product" in result


def test_clean_text_collapses_whitespace():
    result = clean_text("too   many    spaces")
    assert "  " not in result


def test_clean_text_truncates_long_strings():
    long_text = "a" * 5000
    result = clean_text(long_text)
    assert len(result) <= 2000


# ─── LABEL_MAP ────────────────────────────────────────────────────────────────

def test_label_map_covers_all_ratings():
    for star in [1, 2, 3, 4, 5]:
        assert star in LABEL_MAP


def test_label_map_negative():
    assert LABEL_MAP[1] == 0
    assert LABEL_MAP[2] == 0


def test_label_map_neutral():
    assert LABEL_MAP[3] == 1


def test_label_map_positive():
    assert LABEL_MAP[4] == 2
    assert LABEL_MAP[5] == 2


# ─── ReviewDataset ────────────────────────────────────────────────────────────

@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.return_value = {
        "input_ids":      torch.zeros(3, 256, dtype=torch.long),
        "attention_mask": torch.ones(3, 256, dtype=torch.long),
    }
    return tok


def test_dataset_length(mock_tokenizer):
    texts  = ["Good", "Bad", "Okay"]
    labels = [2, 0, 1]
    ds = ReviewDataset(texts, labels, mock_tokenizer)
    assert len(ds) == 3


def test_dataset_returns_correct_label(mock_tokenizer):
    texts  = ["Good", "Bad", "Okay"]
    labels = [2, 0, 1]
    ds = ReviewDataset(texts, labels, mock_tokenizer)
    item = ds[1]
    assert item["labels"].item() == 0  # "Bad" → negative


def test_dataset_item_has_required_keys(mock_tokenizer):
    ds = ReviewDataset(["hello"], [1], mock_tokenizer)
    item = ds[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item


# ─── make_weighted_sampler ────────────────────────────────────────────────────

def test_weighted_sampler_length():
    labels = [0, 0, 0, 1, 2, 2]
    sampler = make_weighted_sampler(labels)
    assert sampler.num_samples == len(labels)
