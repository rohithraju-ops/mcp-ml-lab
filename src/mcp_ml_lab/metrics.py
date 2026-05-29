"""Metric computation for classification.

Binary: accuracy, F1, AUC, log_loss.
Multi-class: accuracy, F1_macro, F1_weighted, AUC_ovr, log_loss.
AUC and log_loss require predict_proba; silently skipped when it returns None.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    n_classes: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    if n_classes == 2:
        metrics["f1"] = float(f1_score(y_true, y_pred))
        if y_proba is not None:
            # For binary AUC, sklearn expects probability of the POSITIVE class only.
            metrics["auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            metrics["log_loss"] = float(log_loss(y_true, y_proba))
    else:
        # Macro: each class weighted equally — best for "treat all classes the same"
        # Weighted: each sample weighted equally — best for "production class mix"
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
        metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted"))
        if y_proba is not None:
            try:
                # One-vs-Rest AUC: AUC per class vs all-others, then averaged.
                metrics["auc_ovr"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                )
            except ValueError:
                # AUC undefined if a class is missing from y_true in this fold
                pass
            metrics["log_loss"] = float(log_loss(y_true, y_proba))

    return metrics


def primary_metric_name(n_classes: int, has_proba: bool) -> str:
    """Conventional 'best score' metric used to compare trials and runs."""
    if n_classes == 2:
        return "auc" if has_proba else "f1"
    return "auc_ovr" if has_proba else "f1_macro"
