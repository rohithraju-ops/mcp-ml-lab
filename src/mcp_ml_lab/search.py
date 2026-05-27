"""Cross-validation runner. Optuna-based tuning lands on Day 4."""
from __future__ import annotations

import time
from typing import Any, Callable

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

# src/mcp_ml_lab/search.py  (append after cross_validate_classification)

def sample_params(trial: optuna.Trial, space: ParamsSpace) -> dict:
    """Materialize concrete hyperparameters from a declarative space dict.

    This is the bridge between trainer.params_space() (declarative, Optuna-free)
    and an actual trial's suggest_* calls.
    """
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
    on_trial_complete: Callable[[dict], None] | None = None,) -> dict:
    """Run an Optuna study to tune one trainer's hyperparameters.

    The objective function evaluates each suggested config with k-fold CV
    and returns the mean of `primary_metric` across folds (the value Optuna
    maximizes). Every completed trial is forwarded to `on_trial_complete` for
    persistence — the search itself doesn't touch storage.

    Args:
        trainer: An instance from trainers.get_trainer().
        X: Feature DataFrame.
        y: Target array.
        preprocessor: UNFIT ColumnTransformer. Cloned per fold inside CV.
        n_classes: 2 for binary, >2 for multi-class.
        primary_metric: Key into the aggregated metrics dict to optimize.
        time_budget_seconds: Wall-clock budget. Optuna stops at first arrival
            of (timeout, n_trials_max).
        n_trials_max: Hard cap on trial count. Acts as a circuit breaker.
        n_splits: CV folds per trial.
        seed: Used by both the TPE sampler AND the per-trial CV. Same seed +
            same data → identical trial sequence (useful for debugging).
        on_trial_complete: Optional callback invoked once per trial with
            {"params", "metrics", "duration_s"}. Used by run_experiment to
            persist trials to SQLite.

    Returns:
        dict with `best_params`, `best_score`, `n_trials`, and `study_summary`.
    """
    space = trainer.params_space()

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial, space)
        # Merge with trainer defaults so non-searched params (random_state,
        # n_jobs, etc.) still get set.
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