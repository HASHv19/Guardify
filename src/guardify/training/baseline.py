"""
Baseline training workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import StratifiedKFold, train_test_split

from guardify.evaluation.metrics import evaluate_predictions, export_misclassifications, write_metrics
from guardify.features import build_tfidf_vectorizer
from guardify.models import build_baseline_estimators
from guardify.training.artifacts import create_bundle_dir, write_metadata, write_json


def _split_dataframe(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    training_cfg = config.get("training", {})
    test_size = float(training_cfg.get("test_size", 0.2))
    val_size = float(training_cfg.get("val_size", 0.2))
    random_state = int(training_cfg.get("random_state", 42))
    stratify = df["binary_label"] if training_cfg.get("stratify", True) else None

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    val_ratio = val_size / max(1e-6, (1 - test_size))
    stratify_train_val = train_val_df["binary_label"] if training_cfg.get("stratify", True) else None
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        random_state=random_state,
        stratify=stratify_train_val,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _kfold_summary(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, list[float]]:
    folds = int(config.get("training", {}).get("kfold", 0))
    if folds < 2:
        return {}

    feature_cfg = config.get("baseline", {})
    vectorizer = build_tfidf_vectorizer(
        ngram_range=tuple(feature_cfg.get("ngram_range", [1, 2])),
        min_df=int(feature_cfg.get("min_df", 1)),
        max_features=int(feature_cfg.get("max_features", 5000)),
    )
    estimators = build_baseline_estimators(feature_cfg.get("models"))
    scores = {name: [] for name in estimators}
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(config["training"]["random_state"]))

    for train_idx, test_idx in splitter.split(df["normalized_text"], df["binary_label"]):
        train_texts = df.iloc[train_idx]["normalized_text"]
        test_texts = df.iloc[test_idx]["normalized_text"]
        y_train = df.iloc[train_idx]["binary_label"]
        y_test = df.iloc[test_idx]["binary_label"]

        features_train = vectorizer.fit_transform(train_texts)
        features_test = vectorizer.transform(test_texts)
        for name, estimator in estimators.items():
            estimator.fit(features_train, y_train)
            predictions = estimator.predict(features_test)
            score = evaluate_predictions(y_test.tolist(), predictions.tolist())
            scores[name].append(score["f1_macro"])
    return scores


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def train_baseline_models(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("training", {}).get("random_state", 42))
    _set_seeds(seed)
    train_df, val_df, test_df = _split_dataframe(df, config)
    feature_cfg = config.get("baseline", {})
    vectorizer = build_tfidf_vectorizer(
        ngram_range=tuple(feature_cfg.get("ngram_range", [1, 2])),
        min_df=int(feature_cfg.get("min_df", 1)),
        max_features=int(feature_cfg.get("max_features", 5000)),
    )
    estimators = build_baseline_estimators(feature_cfg.get("models"))

    x_train = vectorizer.fit_transform(train_df["normalized_text"])
    x_val = vectorizer.transform(val_df["normalized_text"])
    x_test = vectorizer.transform(test_df["normalized_text"])
    y_train = train_df["binary_label"]
    y_val = val_df["binary_label"]
    y_test = test_df["binary_label"]

    best_name = None
    best_model = None
    best_metrics = None
    model_reports: dict[str, Any] = {}

    for name, estimator in estimators.items():
        estimator.fit(x_train, y_train)
        val_predictions = estimator.predict(x_val)
        metrics = evaluate_predictions(y_val.tolist(), val_predictions.tolist())
        model_reports[name] = metrics
        if best_metrics is None or metrics["f1_macro"] > best_metrics["f1_macro"]:
            best_name = name
            best_model = estimator
            best_metrics = metrics

    assert best_name is not None and best_model is not None and best_metrics is not None

    test_predictions = best_model.predict(x_test)
    test_frame = test_df.copy()
    test_frame["prediction"] = test_predictions
    final_metrics = evaluate_predictions(y_test.tolist(), test_predictions.tolist(), dataframe=test_frame)
    final_metrics["validation_comparison"] = model_reports
    final_metrics["kfold"] = _kfold_summary(df, config)

    bundle_dir = create_bundle_dir(config, "baseline")
    joblib.dump({"vectorizer": vectorizer, "model": best_model}, bundle_dir / "estimator.joblib")
    write_json(bundle_dir / "label_map.json", {"0": "Non-Bullying", "1": "Bullying"})
    write_metrics(final_metrics, bundle_dir / "metrics.json")
    export_misclassifications(test_frame, bundle_dir / "misclassified.csv")
    write_metadata(
        bundle_dir,
        model_type="baseline",
        model_name=best_name,
        metrics=final_metrics,
        config=config,
    )

    print("\n" + "="*40)
    print("Training Completed!")
    print(f"Best Model: {best_name}")
    print(f"F1 Macro: {final_metrics['f1_macro']:.4f}")
    print("\nConfusion Matrix (Non-Bullying, Bullying):")
    for row in final_metrics["confusion_matrix"]:
        print(f"  {row}")
    print("\nClassification Report:")
    for label, stats in final_metrics["classification_report"].items():
        if isinstance(stats, dict):
            print(f"  {label:<15}: Precision: {stats.get('precision', 0):.2f}, Recall: {stats.get('recall', 0):.2f}, F1: {stats.get('f1-score', 0):.2f}")
    print("="*40 + "\n")

    return {
        "bundle_dir": str(bundle_dir),
        "best_model": best_name,
        "metrics": final_metrics,
        "splits": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    }
