from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from guardify.train import run_training


def _write_fixture_dataset(path: Path) -> None:
    rows = [
        {"text": "You are a loser", "label": "Bullying"},
        {"text": "Tum bohot cheap ho", "label": "Bullying"},
        {"text": "What a clown move", "label": "Bullying"},
        {"text": "Idiot behavior", "label": "Bullying"},
        {"text": "Have a good day", "label": "Non-Bullying"},
        {"text": "Kal milte hain", "label": "Non-Bullying"},
        {"text": "This song is amazing", "label": "Non-Bullying"},
        {"text": "Hope your exam goes well", "label": "Non-Bullying"},
        {"text": "You are useless", "label": "Bullying"},
        {"text": "Let's grab coffee", "label": "Non-Bullying"},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_baseline_training_writes_bundle(tmp_path):
    dataset_path = tmp_path / "fixture.csv"
    _write_fixture_dataset(dataset_path)
    config_path = tmp_path / "baseline.json"
    config_path.write_text(
        json.dumps(
            {
                "datasets": [{"name": "fixture", "type": "sample_csv", "path": str(dataset_path)}],
                "training": {
                    "task": "baseline",
                    "test_size": 0.2,
                    "val_size": 0.2,
                    "random_state": 7,
                    "output_name": "smoke",
                    "kfold": 2,
                },
                "artifacts": {"root": str(tmp_path / "artifacts")},
            }
        ),
        encoding="utf-8",
    )

    result = run_training(str(config_path))
    bundle_dir = Path(result["bundle_dir"])

    assert result["best_model"] in {"logistic_regression", "linear_svm"}
    assert (bundle_dir / "estimator.joblib").exists()
    assert (bundle_dir / "metrics.json").exists()
    assert (bundle_dir / "metadata.json").exists()

