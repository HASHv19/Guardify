"""
Simple Guardify workflow walkthrough.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.dataset import get_data_info, load_data
from src.preprocess import preprocess_dataframe

print("=" * 70)
print("GUARDIFY V1 WORKFLOW")
print("=" * 70)

dataset = load_data()
info = get_data_info(dataset)
processed = preprocess_dataframe(dataset)

print("\n1. DATASET SNAPSHOT")
print(info)
print("\n2. NORMALIZATION PREVIEW")
print(processed[["text", "normalized_text", "flagged_tokens"]].head())

print("\n3. TRAINING COMMANDS")
print("Baseline:")
print("  PYTHONPATH=src venv/bin/python3 -m guardify.train --config configs/baseline.yaml")
print("Transformer:")
print("  PYTHONPATH=src venv/bin/python3 -m guardify.train --config configs/muril.yaml")

print("\n4. API COMMAND")
print(
    "  GUARDIFY_MODEL_BUNDLE=artifacts/baseline/sample-baseline "
    "PYTHONPATH=src venv/bin/python3 -m uvicorn apps.api.main:app --reload"
)

print("\n5. FRONTEND COMMAND")
print("  cd apps/web && npm install && npm run dev")

print("\n6. PREDICTION COMMAND")
print(
    "  PYTHONPATH=src venv/bin/python3 -m guardify.predict "
    "--bundle artifacts/baseline/sample-baseline --text \"Tum bohot cheap ho\""
)

print("\nArtifacts directory exists:", Path("artifacts").exists())
