"""
Dataset validation helpers.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .schema import CANONICAL_COLUMNS, describe_class_balance


def validate_canonical_dataset(df: pd.DataFrame) -> dict[str, Any]:
    missing = [column for column in CANONICAL_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing canonical columns: {missing}")

    empty_text = int(df["text"].fillna("").astype(str).str.strip().eq("").sum())
    duplicates = int(df.duplicated(subset=["text", "binary_label", "source"]).sum())
    unknown_labels = sorted(set(df["binary_label"]) - {"Bullying", "Non-Bullying"})
    if unknown_labels:
        raise ValueError(f"Unexpected labels found: {unknown_labels}")

    report = {
        "rows": int(len(df)),
        "empty_text_rows": empty_text,
        "duplicate_rows": duplicates,
        "class_balance": describe_class_balance(df),
    }
    return report

