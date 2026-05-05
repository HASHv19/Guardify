"""
Unified inference service for Guardify bundles.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import torch
from transformers import AutoTokenizer

from guardify.models import TransformerClassifier
from guardify.preprocessing import PreprocessingPipeline


ID_TO_LABEL = {0: "Non-Bullying", 1: "Bullying"}


class InferenceService:
    def __init__(self, bundle_path: str | Path) -> None:
        self.bundle_path = Path(bundle_path).resolve()
        self.metadata = json.loads((self.bundle_path / "metadata.json").read_text(encoding="utf-8"))
        self.pipeline = PreprocessingPipeline()
        self.model_type = self.metadata["model_type"]
        self.model_name = self.metadata["model_name"]
        self.device = self._device()
        self.estimator = None
        self.transformer = None
        self.tokenizer = None
        self._load()

    def _device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load(self) -> None:
        if self.model_type == "baseline":
            self.estimator = joblib.load(self.bundle_path / "estimator.joblib")
            return

        if self.model_type == "transformer":
            checkpoint = torch.load(self.bundle_path / "model.pt", map_location=self.device)
            model_name = checkpoint["model_name"]
            dropout = float(checkpoint.get("dropout", 0.3))
            self.transformer = TransformerClassifier(model_name=model_name, num_labels=2, dropout=dropout)
            self.transformer.load_state_dict(checkpoint["model_state_dict"])
            self.transformer = self.transformer.to(self.device)
            self.transformer.eval()
            tokenizer_dir = self.bundle_path / "tokenizer"
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            return

        raise ValueError(f"Unsupported model bundle type: {self.model_type}")

    def model_info(self) -> dict[str, Any]:
        return {
            "bundle_path": str(self.bundle_path),
            "model_type": self.model_type,
            "model_name": self.model_name,
            "metrics": self.metadata.get("metrics", {}),
        }

    def _get_sub_category(self, label: str, flagged_tokens: list[str]) -> str | None:
        if label != "Bullying":
            return None
        if not flagged_tokens:
            return "Contextual Abuse"
        from guardify.preprocessing.lexicon import TOKEN_CATEGORIES
        severity_map = {"Threat / Violence": 4, "Hate Speech": 3, "Profanity": 2, "Harassment / Insult": 1}
        highest_severity = 0
        sub_category = "General Abuse"
        for token in flagged_tokens:
            cat = TOKEN_CATEGORIES.get(token.lower())
            if cat and severity_map.get(cat, 0) > highest_severity:
                highest_severity = severity_map.get(cat, 0)
                sub_category = cat
        return sub_category

    def predict_text(self, text: str) -> dict[str, Any]:
        if not text.strip():
            raise ValueError("Text must not be empty or whitespace.")

        result = self.pipeline.normalize(text)
        if not result.normalized_text:
            raise ValueError("Text contains no processable characters after normalization.")

        if self.model_type == "baseline":
            assert self.estimator is not None
            vectorizer = self.estimator["vectorizer"]
            model = self.estimator["model"]
            features = vectorizer.transform([result.normalized_text])
            if hasattr(model, "decision_function"):
                raw_scores = model.decision_function(features)
                prob_pos = float(1 / (1 + math.exp(-raw_scores[0])))
                prob_neg = 1 - prob_pos
                prediction = model.predict(features)[0]
                probabilities = {str(model.classes_[0]): prob_neg, str(model.classes_[1]): prob_pos}
                confidence = float(max(probabilities.values()))
            else:
                proba = model.predict_proba(features)[0]
                prediction = model.predict(features)[0]
                probabilities = {str(model.classes_[0]): float(proba[0]), str(model.classes_[1]): float(proba[1])}
                confidence = float(max(probabilities.values()))

            return {
                "label": prediction,
                "confidence": confidence,
                "probabilities": probabilities,
                "flagged_tokens": result.flagged_tokens,
                "normalized_text": result.normalized_text,
                "model_version": self.model_name,
                "sub_category": self._get_sub_category(prediction, result.flagged_tokens),
                "needs_review": confidence < 0.60,
            }

        assert self.transformer is not None and self.tokenizer is not None
        encoding = self.tokenizer(
            result.normalized_text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.transformer(
                input_ids=encoding["input_ids"].to(self.device),
                attention_mask=encoding["attention_mask"].to(self.device),
            )
            probabilities_tensor = torch.softmax(logits, dim=1)[0].tolist()
        probabilities = {
            "Non-Bullying": float(probabilities_tensor[0]),
            "Bullying": float(probabilities_tensor[1]),
        }
        predicted_id = 1 if probabilities["Bullying"] >= probabilities["Non-Bullying"] else 0
        return {
            "label": ID_TO_LABEL[predicted_id],
            "confidence": float(max(probabilities.values())),
            "probabilities": probabilities,
            "flagged_tokens": result.flagged_tokens,
            "normalized_text": result.normalized_text,
            "model_version": self.model_name,
            "sub_category": self._get_sub_category(ID_TO_LABEL[predicted_id], result.flagged_tokens),
            "needs_review": float(max(probabilities.values())) < 0.60,
        }
