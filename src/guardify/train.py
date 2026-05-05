"""
Training CLI for Guardify.
"""

from __future__ import annotations

import argparse
import json

from guardify.config import load_config
from guardify.data import load_datasets_from_config
from guardify.preprocessing import PreprocessingPipeline, preprocess_dataframe
from guardify.training import train_baseline_models, train_transformer_model


def run_training(config_path: str) -> dict:
    config = load_config(config_path)
    dataset, reports = load_datasets_from_config(config)
    pipeline = PreprocessingPipeline(**config.get("preprocessing", {}))
    dataset = preprocess_dataframe(dataset, pipeline=pipeline)

    task = config.get("training", {}).get("task", "baseline")
    if task == "baseline":
        result = train_baseline_models(dataset, config)
    elif task == "transformer":
        result = train_transformer_model(dataset, config)
    else:
        raise ValueError(f"Unsupported training task '{task}'.")

    result["dataset_reports"] = reports
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Guardify models from config.")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON/TOML config file.")
    args = parser.parse_args()
    result = run_training(args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

