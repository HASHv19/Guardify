"""
Legacy dataset helpers that proxy to the new Guardify data layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from guardify.data.schema import canonicalize_dataset, describe_class_balance


def load_data(data_path: str | None = None) -> pd.DataFrame:
    project_root = Path(__file__).resolve().parent.parent
    source_path = Path(data_path) if data_path else project_root / "data" / "raw" / "data.csv"
    frame = pd.read_csv(source_path)
    print(f"✅ Loaded {len(frame)} rows from {source_path}")
    return frame


def get_data_info(df: pd.DataFrame) -> dict:
    canonical = canonicalize_dataset(
        df,
        source_name="legacy",
        text_column="text",
        label_column="label",
    )
    balance = describe_class_balance(canonical)
    return {
        "total_samples": balance["total"],
        "bullying_count": balance["Bullying"],
        "non_bullying_count": balance["Non-Bullying"],
        "columns": list(df.columns),
    }


if __name__ == "__main__":
    dataset = load_data()
    print(get_data_info(dataset))
