"""
Configuration helpers for Guardify.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {"name": "Guardify"},
    "artifacts": {"root": "artifacts"},
    "preprocessing": {
        "lowercase": True,
        "transliterate": False,
        "keep_hashtag_text": True,
    },
    "datasets": [],
    "training": {
        "task": "baseline",
        "test_size": 0.2,
        "val_size": 0.2,
        "random_state": 42,
        "stratify": True,
        "output_name": "latest",
        "kfold": 0,
    },
    "baseline": {
        "models": ["logistic_regression", "linear_svm"],
        "ngram_range": [1, 2],
        "min_df": 1,
        "max_features": 5000,
    },
    "transformer": {
        "model_name": "google/muril-base-cased",
        "epochs": 2,
        "batch_size": 4,
        "learning_rate": 2e-5,
        "max_length": 128,
        "dropout": 0.3,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_config_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    raw = path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML config files.")
        data = yaml.safe_load(raw)
    elif suffix == ".json":
        data = json.loads(raw)
    elif suffix == ".toml":
        if tomllib is None:
            raise RuntimeError("tomllib is unavailable in this Python version.")
        data = tomllib.loads(raw)
    else:
        raise ValueError(f"Unsupported config format: {suffix}")
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping/object.")
    return data


def resolve_path(base_dir: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_config(config_path: str | os.PathLike[str]) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config = _deep_merge(DEFAULT_CONFIG, _read_config_file(config_file))
    config["_config_path"] = str(config_file)
    config["_project_root"] = str(config_file.parent.parent.resolve())
    return config

