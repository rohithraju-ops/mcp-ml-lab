"""Cross-validation runner. Optuna-based tuning lands on Day 4."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold

from mcp_ml_lab.metrics import compute_classification_metrics
from mcp_ml_lab.trainers.base import BaseTrainer


def cross_validate_classification(
    trainer: BaseTrainer,
    X: pd.DataFrame,
    y: np.ndarray,
    preprocessor: ColumnTransformer,
    params: dict,
    n_splits: int = 5,
    seed: int = 42,
    n_classes: int = 2,) -> dict[str, Any]:
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics: list[dict[str, float]] = []
    fold_durations: list[float] = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fresh, unfit copy of the preprocessor for THIS fold.
        # clone() copies the recipe (transformer types, column lists) but NOT
        # any learned state. This is the entire reason yesterday's
        # build_preprocessor returned an unfit object.
        pre = clone(preprocessor)
        X_train_t = pre.fit_transform(X_train)
        X_test_t = pre.transform(X_test)

        t0 = time.perf_counter()
        model = trainer.fit(X_train_t, y_train, params)
        duration = time.perf_counter() - t0

        y_pred = trainer.predict(model, X_test_t)
        y_proba = trainer.predict_proba(model, X_test_t)

        fold_metrics.append(
            compute_classification_metrics(y_test, y_pred, y_proba, n_classes)
        )
        fold_durations.append(duration)

    # Aggregate: mean and std across folds for every metric that appeared
    all_keys = set().union(*[m.keys() for m in fold_metrics])
    aggregated: dict[str, dict[str, float]] = {}
    for k in all_keys:
        vals = [m[k] for m in fold_metrics if k in m]
        aggregated[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

    return {
        "fold_metrics": fold_metrics,
        "aggregated": aggregated,
        "mean_fit_seconds": float(np.mean(fold_durations)),
        "total_fit_seconds": float(np.sum(fold_durations)),
        "n_splits": n_splits,
    }