"""
Transformer classifier implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


@dataclass
class TransformerAssets:
    model: "TransformerClassifier"
    tokenizer: object


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "google/muril-base-cased",
        num_labels: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = getattr(outputs, "pooler_output", None)
        if pooled_output is None:
            pooled_output = outputs.last_hidden_state[:, 0]
        return self.classifier(self.dropout(pooled_output))


def create_transformer_model(
    *,
    model_name: str,
    num_labels: int = 2,
    dropout: float = 0.3,
) -> TransformerAssets:
    model = TransformerClassifier(model_name=model_name, num_labels=num_labels, dropout=dropout)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return TransformerAssets(model=model, tokenizer=tokenizer)

