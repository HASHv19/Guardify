"""
Legacy preprocessing helpers that proxy to the new pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from guardify.preprocessing import PreprocessingPipeline, preprocess_dataframe as _preprocess_dataframe


def clean_text(text: str, lowercase: bool = True) -> str:
    pipeline = PreprocessingPipeline(lowercase=lowercase)
    return pipeline.normalize(text).normalized_text


def preprocess_dataframe(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    processed = _preprocess_dataframe(df, text_column=text_column)
    processed[f"{text_column}_cleaned"] = processed["normalized_text"]
    return processed


if __name__ == "__main__":
    from dataset import load_data

    dataset = load_data()
    print(preprocess_dataframe(dataset).head())
