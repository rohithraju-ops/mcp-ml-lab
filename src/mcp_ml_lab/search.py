"""Cross-validation runner and Optuna-based hyperparameter search."""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold

from mcp_ml_lab.metrics import compute_classification_metrics
from mcp_ml_lab.trainers.base import BaseTrainer, ParamsSpace

optuna.logging.set_verbosity(optuna.logging.WARNING)

def cross_validate_classification(
    trainer: BaseTrainer,
    X: pd.DataFrame,
    y: np.ndarray,
    preprocessor: ColumnTransformer,
    params: dict,
    n_splits: int = 5,
    seed: int = 42,
    n_classes: int = 2,
) -> dict[str, Any]:

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics: list[dict[str, float]] = []
    fold_durations: list[float] = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # clone() copies transformer types + column lists, not learned state.
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

def sample_params(trial: optuna.Trial, space: ParamsSpace) -> dict:
    """Map a declarative ParamsSpace to Optuna suggest_* calls for one trial."""
    out: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "int":
            out[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "uniform":
            out[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "loguniform":
            out[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == "categorical":
            out[name] = trial.suggest_categorical(name, spec[1])
        else:
            raise ValueError(f"Unknown param kind {kind!r} for {name!r}")
    return out


def tune(
    trainer: BaseTrainer,
    X: pd.DataFrame,
    y: np.ndarray,
    preprocessor: ColumnTransformer,
    n_classes: int,
    primary_metric: str,
    time_budget_seconds: int = 60,
    n_trials_max: int = 100,
    n_splits: int = 5,
    seed: int = 42,
    on_trial_complete: Callable[[dict], None] | None = None,
) -> dict:
    """Run an Optuna TPE study to tune one trainer's hyperparameters.

    Each trial runs k-fold CV and returns mean(primary_metric) for Optuna to
    maximize. Stops at whichever limit arrives first: time_budget_seconds or
    n_trials_max. on_trial_complete is called after each trial for persistence;
    the search itself has no storage dependency.
    """
    space = trainer.params_space()

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial, space)
        merged = {**trainer.default_params(), **params}

        cv_result = cross_validate_classification(
            trainer=trainer,
            X=X,
            y=y,
            preprocessor=preprocessor,
            params=merged,
            n_splits=n_splits,
            seed=seed,
            n_classes=n_classes,
        )

        score = cv_result["aggregated"][primary_metric]["mean"]

        if on_trial_complete is not None:
            on_trial_complete(
                {
                    "params": merged,
                    "metrics": cv_result["aggregated"],
                    "duration_s": cv_result["total_fit_seconds"],
                }
            )

        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        pruner=optuna.pruners.NopPruner(),
    )

    t0 = time.perf_counter()
    study.optimize(
        objective,
        timeout=time_budget_seconds,
        n_trials=n_trials_max,
        gc_after_trial=True,
        show_progress_bar=False,
    )
    elapsed = time.perf_counter() - t0

    return {
        "best_params": dict(study.best_trial.params),
        "best_score": float(study.best_value),
        "n_trials": len(study.trials),
        "elapsed_seconds": elapsed,
    }
