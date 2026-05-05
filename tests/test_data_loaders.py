from __future__ import annotations

import json

import pandas as pd

from guardify.data import load_datasets_from_config


def test_loads_csv_and_json_sources(tmp_path):
    csv_path = tmp_path / "sample.csv"
    json_path = tmp_path / "custom.json"
    pd.DataFrame(
        [
            {"text": "You are a loser", "label": "Bullying"},
            {"text": "Hope you have a great day", "label": "Non-Bullying"},
        ]
    ).to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            [
                {"text": "Tum bohot cheap ho", "label": "Bullying"},
                {"text": "Kal milte hain", "label": "Non-Bullying"},
            ]
        ),
        encoding="utf-8",
    )

    config = {
        "_project_root": str(tmp_path),
        "datasets": [
            {"name": "csv_fixture", "type": "sample_csv", "path": str(csv_path)},
            {"name": "json_fixture", "type": "custom_json", "path": str(json_path)},
        ],
    }

    dataset, reports = load_datasets_from_config(config)

    assert len(dataset) == 4
    assert set(dataset.columns) >= {"id", "text", "binary_label", "source"}
    assert {report["source"] for report in reports} == {"csv_fixture", "json_fixture"}


def test_unknown_label_raises(tmp_path):
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame([{"text": "hello", "label": "maybe"}]).to_csv(csv_path, index=False)
    config = {
        "_project_root": str(tmp_path),
        "datasets": [{"name": "bad_fixture", "type": "sample_csv", "path": str(csv_path)}],
    }

    try:
        load_datasets_from_config(config)
    except ValueError as exc:
        assert "Unknown label" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for an unknown label.")
