from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from guardify.preprocessing import PreprocessingPipeline


project_root = Path(__file__).resolve().parent
fixture = project_root / "data" / "raw" / "data.csv"

df = pd.read_csv(fixture)
pipeline = PreprocessingPipeline()
preview = pipeline.normalize(df["text"].iloc[0])

print("✅ Data Loaded Successfully:")
print(df.head())
print(f"\n✅ Computing Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print("✅ Preprocessing Preview:")
print(f"   normalized_text={preview.normalized_text}")
print(f"   flagged_tokens={preview.flagged_tokens}")
print("✅ Ready for Guardify Development!")
