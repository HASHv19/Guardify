"""
Artifact helpers for model bundles.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def create_bundle_dir(config: dict[str, Any], model_type: str) -> Path:
    project_root = Path(config["_project_root"])
    root = project_root / config.get("artifacts", {}).get("root", "artifacts")
    bundle_name = config.get("training", {}).get("output_name", "latest")
    bundle_dir = root / model_type / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)
    return bundle_dir


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_metadata(
    bundle_dir: Path,
    *,
    model_type: str,
    model_name: str,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    metadata = {
        "model_type": model_type,
        "model_name": model_name,
        "created_at": datetime.now(UTC).isoformat(),
        "metrics": metrics,
        "config_path": config.get("_config_path"),
    }
    write_json(bundle_dir / "metadata.json", metadata)
    write_json(bundle_dir / "config_snapshot.json", config)

