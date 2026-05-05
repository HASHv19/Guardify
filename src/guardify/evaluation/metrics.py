"""
Evaluation metrics utilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


LABEL_ORDER = ["Non-Bullying", "Bullying"]


def evaluate_predictions(
    y_true: list[str],
    y_pred: list[str],
    *,
    dataframe: pd.DataFrame | None = None,
) -> dict[str, Any]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABEL_ORDER).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=LABEL_ORDER,
            zero_division=0,
            output_dict=True,
        ),
    }

    if dataframe is not None and "source" in dataframe.columns:
        source_breakdown: dict[str, Any] = {}
        for source, group in dataframe.groupby("source"):
            source_breakdown[source] = {
                "rows": int(len(group)),
                "f1_macro": f1_score(group["binary_label"], group["prediction"], average="macro", zero_division=0),
                "accuracy": accuracy_score(group["binary_label"], group["prediction"]),
            }
        metrics["per_source"] = source_breakdown

    return metrics


def export_misclassifications(df: pd.DataFrame, output_path: Path) -> None:
    mistakes = df[df["binary_label"] != df["prediction"]].copy()
    mistakes.to_csv(output_path, index=False)


def write_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

