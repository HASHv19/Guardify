"""
Dataset loader registry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import canonicalize_dataset
from .validation import validate_canonical_dataset


def _read_source(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if "records" in data and isinstance(data["records"], list):
                return pd.DataFrame(data["records"])
            raise ValueError("JSON object inputs must contain a 'records' list.")
        if isinstance(data, list):
            return pd.DataFrame(data)
        raise ValueError("Unsupported JSON dataset structure.")
    raise ValueError(f"Unsupported dataset extension: {suffix}")


def _dataset_type_defaults(dataset_type: str) -> dict[str, Any]:
    defaults: dict[str, dict[str, Any]] = {
        "sample_csv": {"text_column": "text", "label_column": "label"},
        "hasoc_csv": {"text_column": "text", "label_column": "task_1"},
        "trac_csv": {"text_column": "text", "label_column": "label"},
        "custom_csv": {"text_column": "text", "label_column": "label"},
        "custom_json": {"text_column": "text", "label_column": "label"},
    }
    return defaults.get(dataset_type, {"text_column": "text", "label_column": "label"})


def load_dataset(dataset_config: dict[str, Any], project_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset_type = dataset_config.get("type", "custom_csv")
    defaults = _dataset_type_defaults(dataset_type)
    source_name = dataset_config.get("name", dataset_type)
    source_path = Path(dataset_config["path"])
    if not source_path.is_absolute():
        source_path = (project_root / source_path).resolve()

    raw_df = _read_source(source_path)
    canonical_df = canonicalize_dataset(
        raw_df,
        source_name=source_name,
        text_column=dataset_config.get("text_column", defaults["text_column"]),
        label_column=dataset_config.get("label_column", defaults["label_column"]),
        label_map=dataset_config.get("label_map"),
        id_column=dataset_config.get("id_column"),
        language_column=dataset_config.get("language_column"),
        split_column=dataset_config.get("split_column"),
    )
    report = validate_canonical_dataset(canonical_df)
    report["path"] = str(source_path)
    report["source"] = source_name
    return canonical_df, report


def load_datasets_from_config(config: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    datasets = config.get("datasets", [])
    if not datasets:
        raise ValueError("Config must declare at least one dataset.")

    project_root = Path(config["_project_root"])
    frames: list[pd.DataFrame] = []
    reports: list[dict[str, Any]] = []

    for dataset_config in datasets:
        frame, report = load_dataset(dataset_config, project_root)
        frames.append(frame)
        reports.append(report)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["text", "binary_label", "source"]).reset_index(drop=True)
    return combined, reports

