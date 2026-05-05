"""
Legacy model compatibility wrapper.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from guardify.models import TransformerClassifier, create_transformer_model


BullyingDetectionModel = TransformerClassifier


def create_model(
    model_name: str = "google/muril-base-cased",
    num_labels: int = 2,
    device: str | None = None,
) -> BullyingDetectionModel:
    assets = create_transformer_model(model_name=model_name, num_labels=num_labels)
    model = assets.model
    model.tokenizer = assets.tokenizer  # type: ignore[attr-defined]
    model.get_tokenizer = lambda: assets.tokenizer  # type: ignore[attr-defined]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
    return model.to(device)


if __name__ == "__main__":
    create_model()
