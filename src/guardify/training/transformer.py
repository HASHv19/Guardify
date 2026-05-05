"""
Transformer training workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from guardify.evaluation.metrics import evaluate_predictions, export_misclassifications, write_metrics
from guardify.models import create_transformer_model
from guardify.training.artifacts import create_bundle_dir, write_metadata, write_json


LABEL_TO_ID = {"Non-Bullying": 0, "Bullying": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[str], tokenizer: Any, max_length: int) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(LABEL_TO_ID[self.labels[idx]], dtype=torch.long),
        }


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _split_dataframe(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    training_cfg = config.get("training", {})
    test_size = float(training_cfg.get("test_size", 0.2))
    val_size = float(training_cfg.get("val_size", 0.2))
    random_state = int(training_cfg.get("random_state", 42))

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["binary_label"],
    )
    val_ratio = val_size / max(1e-6, (1 - test_size))
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        random_state=random_state,
        stratify=train_val_df["binary_label"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _evaluate_loader(model: torch.nn.Module, loader: DataLoader, device: str) -> tuple[float, list[str], list[str]]:
    loss_fn = torch.nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    predictions: list[str] = []
    truth: list[str] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            total_loss += loss_fn(logits, labels).item()
            predicted_ids = torch.argmax(logits, dim=1).tolist()
            predictions.extend(ID_TO_LABEL[item] for item in predicted_ids)
            truth.extend(ID_TO_LABEL[item] for item in labels.tolist())
    average_loss = total_loss / max(len(loader), 1)
    return average_loss, predictions, truth


def train_transformer_model(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    transformer_cfg = config.get("transformer", {})
    train_df, val_df, test_df = _split_dataframe(df, config)

    assets = create_transformer_model(
        model_name=transformer_cfg.get("model_name", "google/muril-base-cased"),
        num_labels=2,
        dropout=float(transformer_cfg.get("dropout", 0.3)),
    )
    tokenizer = assets.tokenizer
    model = assets.model.to(_device())
    max_length = int(transformer_cfg.get("max_length", 128))
    batch_size = int(transformer_cfg.get("batch_size", 4))
    epochs = int(transformer_cfg.get("epochs", 2))
    learning_rate = float(transformer_cfg.get("learning_rate", 2e-5))

    train_loader = DataLoader(
        TextDataset(train_df["normalized_text"].tolist(), train_df["binary_label"].tolist(), tokenizer, max_length),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TextDataset(val_df["normalized_text"].tolist(), val_df["binary_label"].tolist(), tokenizer, max_length),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TextDataset(test_df["normalized_text"].tolist(), test_df["binary_label"].tolist(), tokenizer, max_length),
        batch_size=batch_size,
        shuffle=False,
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = max(len(train_loader) * epochs, 1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_state = None
    best_metrics = None
    best_epoch = 0
    device = _device()

    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

        val_loss, val_predictions, val_truth = _evaluate_loader(model, val_loader, device)
        metrics = evaluate_predictions(val_truth, val_predictions)
        metrics["loss"] = val_loss
        if best_metrics is None or metrics["f1_macro"] > best_metrics["f1_macro"]:
            best_metrics = metrics
            best_state = model.state_dict()
            best_epoch = epoch + 1

    assert best_state is not None and best_metrics is not None
    model.load_state_dict(best_state)

    test_loss, test_predictions, test_truth = _evaluate_loader(model, test_loader, device)
    test_frame = test_df.copy()
    test_frame["prediction"] = test_predictions
    final_metrics = evaluate_predictions(test_truth, test_predictions, dataframe=test_frame)
    final_metrics["test_loss"] = test_loss
    final_metrics["best_epoch"] = best_epoch

    bundle_dir = create_bundle_dir(config, "transformer")
    tokenizer.save_pretrained(bundle_dir / "tokenizer")
    torch.save(
        {
            "model_state_dict": best_state,
            "model_name": transformer_cfg.get("model_name", "google/muril-base-cased"),
            "dropout": transformer_cfg.get("dropout", 0.3),
            "label_map": LABEL_TO_ID,
        },
        bundle_dir / "model.pt",
    )
    write_json(bundle_dir / "label_map.json", {"0": "Non-Bullying", "1": "Bullying"})
    write_metrics(final_metrics, bundle_dir / "metrics.json")
    export_misclassifications(test_frame, bundle_dir / "misclassified.csv")
    write_metadata(
        bundle_dir,
        model_type="transformer",
        model_name=transformer_cfg.get("model_name", "google/muril-base-cased"),
        metrics=final_metrics,
        config=config,
    )

    return {
        "bundle_dir": str(bundle_dir),
        "best_model": transformer_cfg.get("model_name", "google/muril-base-cased"),
        "metrics": final_metrics,
        "splits": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    }

