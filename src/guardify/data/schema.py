"""
Canonical dataset schema helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


CANONICAL_COLUMNS = [
    "id",
    "text",
    "original_label",
    "binary_label",
    "source",
    "language_hint",
    "split",
]


DEFAULT_BINARY_LABEL_MAP = {
    "bullying": "Bullying",
    "aggressive": "Bullying",
    "offensive": "Bullying",
    "hate": "Bullying",
    "abusive": "Bullying",
    "hof": "Bullying",
    "1": "Bullying",
    "true": "Bullying",
    "non-bullying": "Non-Bullying",
    "none": "Non-Bullying",
    "not bullying": "Non-Bullying",
    "non aggressive": "Non-Bullying",
    "non-aggressive": "Non-Bullying",
    "neutral": "Non-Bullying",
    "normal": "Non-Bullying",
    "not": "Non-Bullying",
    "0": "Non-Bullying",
    "false": "Non-Bullying",
}


def infer_language_hint(text: str) -> str:
    has_devanagari = any("\u0900" <= char <= "\u097f" for char in text)
    has_urdu = any("\u0600" <= char <= "\u06ff" for char in text)
    has_latin = any("a" <= char.lower() <= "z" for char in text)

    if has_latin and (has_devanagari or has_urdu):
        return "mixed"
    if has_devanagari:
        return "hindi"
    if has_urdu:
        return "urdu"
    if has_latin:
        return "latin"
    return "unknown"


def map_label(value: Any, label_map: dict[str, str] | None = None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raise ValueError("Missing label value.")

    candidate = str(value).strip()
    if candidate in {"Bullying", "Non-Bullying"}:
        return candidate

    normalized = candidate.lower().replace("_", " ")
    normalized = " ".join(normalized.split())
    merged_map = {**DEFAULT_BINARY_LABEL_MAP}
    if label_map:
        merged_map.update({str(key).strip().lower(): value for key, value in label_map.items()})

    if normalized not in merged_map:
        raise ValueError(f"Unknown label '{candidate}'.")
    mapped = merged_map[normalized]
    if mapped not in {"Bullying", "Non-Bullying"}:
        raise ValueError(f"Mapped label must be Bullying or Non-Bullying, got '{mapped}'.")
    return mapped


def describe_class_balance(df: pd.DataFrame) -> dict[str, int]:
    counts = df["binary_label"].value_counts(dropna=False).to_dict()
    return {
        "Bullying": int(counts.get("Bullying", 0)),
        "Non-Bullying": int(counts.get("Non-Bullying", 0)),
        "total": int(len(df)),
    }


def canonicalize_dataset(
    df: pd.DataFrame,
    *,
    source_name: str,
    text_column: str,
    label_column: str,
    label_map: dict[str, str] | None = None,
    id_column: str | None = None,
    language_column: str | None = None,
    split_column: str | None = None,
) -> pd.DataFrame:
    if text_column not in df.columns:
        raise ValueError(f"Missing text column '{text_column}'.")
    if label_column not in df.columns:
        raise ValueError(f"Missing label column '{label_column}'.")

    canonical = pd.DataFrame()
    canonical["text"] = df[text_column].fillna("").astype(str)
    canonical["original_label"] = df[label_column].astype(str)
    canonical["binary_label"] = df[label_column].apply(lambda value: map_label(value, label_map))
    canonical["source"] = source_name

    if id_column and id_column in df.columns:
        canonical["id"] = df[id_column].astype(str)
    else:
        canonical["id"] = [f"{source_name}-{idx}" for idx in range(len(df))]

    if language_column and language_column in df.columns:
        canonical["language_hint"] = df[language_column].fillna("").astype(str)
    else:
        canonical["language_hint"] = canonical["text"].apply(infer_language_hint)

    if split_column and split_column in df.columns:
        canonical["split"] = df[split_column].fillna("").astype(str)
    else:
        canonical["split"] = "unspecified"

    return canonical[CANONICAL_COLUMNS]

