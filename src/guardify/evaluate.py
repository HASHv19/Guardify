"""
Standalone evaluation CLI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from guardify.config import load_config
from guardify.data import load_datasets_from_config
from guardify.evaluation.metrics import evaluate_predictions, export_misclassifications, write_metrics
from guardify.inference import InferenceService
from guardify.preprocessing import PreprocessingPipeline, preprocess_dataframe


def run_evaluation(config_path: str, bundle_path: str) -> dict:
    config = load_config(config_path)
    dataset, reports = load_datasets_from_config(config)
    dataset = preprocess_dataframe(dataset, pipeline=PreprocessingPipeline(**config.get("preprocessing", {})))
    service = InferenceService(bundle_path)

    predictions = [service.predict_text(text)["label"] for text in dataset["text"].tolist()]
    frame = dataset.copy()
    frame["prediction"] = predictions
    metrics = evaluate_predictions(frame["binary_label"].tolist(), predictions, dataframe=frame)
    output_dir = Path(bundle_path)
    write_metrics(metrics, output_dir / "evaluation_metrics.json")
    export_misclassifications(frame, output_dir / "evaluation_misclassified.csv")
    return {"metrics": metrics, "dataset_reports": reports}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Guardify bundle.")
    parser.add_argument("--config", required=True, help="Path to config file.")
    parser.add_argument("--bundle", required=True, help="Path to a model bundle.")
    args = parser.parse_args()
    result = run_evaluation(args.config, args.bundle)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

