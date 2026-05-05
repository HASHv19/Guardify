from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from apps.api.main import create_app
from guardify.train import run_training


@pytest.fixture(scope="session")
def bundle_path(tmp_path_factory) -> str:
    tmp_path = tmp_path_factory.mktemp("guardify_api_tests")
    dataset_path = tmp_path / "fixture.csv"
    pd.DataFrame(
        [
            {"text": "You are a loser", "label": "Bullying"},
            {"text": "Tum bohot cheap ho", "label": "Bullying"},
            {"text": "What a clown move", "label": "Bullying"},
            {"text": "Idiot behavior", "label": "Bullying"},
            {"text": "Have a good day", "label": "Non-Bullying"},
            {"text": "Kal milte hain", "label": "Non-Bullying"},
            {"text": "This song is amazing", "label": "Non-Bullying"},
            {"text": "Hope your exam goes well", "label": "Non-Bullying"},
        ]
    ).to_csv(dataset_path, index=False)
    
    config_path = tmp_path / "baseline.json"
    config_path.write_text(
        json.dumps(
            {
                "datasets": [{"name": "fixture", "type": "sample_csv", "path": str(dataset_path)}],
                "training": {
                    "task": "baseline",
                    "test_size": 0.25,
                    "val_size": 0.25,
                    "random_state": 7,
                    "output_name": "api-smoke",
                    "kfold": 0,
                },
                "artifacts": {"root": str(tmp_path / "artifacts")},
            }
        ),
        encoding="utf-8",
    )
    result = run_training(str(config_path))
    return result["bundle_dir"]


def test_predict_endpoint(bundle_path):
    os.environ["GUARDIFY_MODEL_BUNDLE"] = bundle_path
    with TestClient(create_app()) as client:
        response = client.post("/predict", json={"text": "You are a loser"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["label"] in {"Bullying", "Non-Bullying"}
        assert "normalized_text" in payload
        assert "needs_review" in payload


def test_predict_requires_non_empty_text(bundle_path):
    os.environ["GUARDIFY_MODEL_BUNDLE"] = bundle_path
    with TestClient(create_app()) as client:
        # Schema validation (min_length=1)
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422
        
        # Whitespace validation (raises ValueError inside predict_text -> 422)
        response2 = client.post("/predict", json={"text": "   "})
        assert response2.status_code == 422


def test_predict_long_input_rejection(bundle_path):
    os.environ["GUARDIFY_MODEL_BUNDLE"] = bundle_path
    with TestClient(create_app()) as client:
        long_text = "a" * 2001
        response = client.post("/predict", json={"text": long_text})
        assert response.status_code == 422


def test_health_and_info_endpoints(bundle_path):
    os.environ["GUARDIFY_MODEL_BUNDLE"] = bundle_path
    with TestClient(create_app()) as client:
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "ok"
        assert health_resp.json()["bundle_loaded"] is True
        
        info_resp = client.get("/model-info")
        assert info_resp.status_code == 200
        assert "model_name" in info_resp.json()


def test_missing_bundle_raises_runtime_error(monkeypatch):
    monkeypatch.delenv("GUARDIFY_MODEL_BUNDLE", raising=False)
    with pytest.raises(RuntimeError, match="GUARDIFY_MODEL_BUNDLE is not configured"):
        with TestClient(create_app()):
            pass

