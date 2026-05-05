"""
Baseline estimator builders.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def build_baseline_estimators(selected: list[str] | None = None) -> dict[str, object]:
    estimators = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "linear_svm": LinearSVC(class_weight="balanced"),
    }
    if not selected:
        return estimators
    return {name: estimator for name, estimator in estimators.items() if name in selected}
