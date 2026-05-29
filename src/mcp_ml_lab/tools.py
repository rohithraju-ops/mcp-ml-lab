from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mcp_ml_lab import data, reporting, search, storage, trainers
from mcp_ml_lab.metrics import primary_metric_name


def _to_native(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        f = float(value)
        if not np.isfinite(f):
            return None
        return f
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_to_native(v) for v in value.tolist()]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def inspect_data_impl(csv_path: str) -> dict:
    """Profile a CSV file: shape, dtypes, null counts, summary stats.

    Returns a flat dict of JSON-serializable values suitable for LLM consumption.
    On failure returns {"error": <message>, "type": <exception class name>}.
    """
    try:
        path = Path(csv_path).expanduser().resolve()
        if not path.exists():
            return {"error": f"File not found: {path}", "type": "FileNotFoundError"}
        if not path.is_file():
            return {"error": f"Not a regular file: {path}", "type": "NotAFileError"}
        if path.suffix.lower() not in {".csv", ".tsv", ".txt"}:
            return {
                "error": f"Unexpected file extension: {path.suffix}. Expected .csv/.tsv/.txt.",
                "type": "UnsupportedFileType",
            }

        try:
            df = pd.read_csv(path)
        except Exception as e:
            return {"error": str(e), "type": e.__class__.__name__}

        if df.empty:
            return {"error": "CSV has zero rows after loading.", "type": "EmptyDataError"}

        n_rows, n_cols = df.shape
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        null_counts = {col: int(df[col].isna().sum()) for col in df.columns}
        null_pct = {col: round(100.0 * v / n_rows, 2) for col, v in null_counts.items()}

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        numeric_stats: dict[str, dict[str, Any]] = {}
        for col in numeric_cols:
            desc = df[col].describe()
            numeric_stats[col] = {
                "mean": _to_native(desc.get("mean")),
                "std": _to_native(desc.get("std")),
                "min": _to_native(desc.get("min")),
                "p25": _to_native(desc.get("25%")),
                "p50": _to_native(desc.get("50%")),
                "p75": _to_native(desc.get("75%")),
                "max": _to_native(desc.get("max")),
            }

        categorical_summary: dict[str, dict[str, Any]] = {}
        for col in cat_cols:
            vc = df[col].value_counts(dropna=False).head(10)
            categorical_summary[col] = {
                "n_unique": int(df[col].nunique(dropna=True)),
                "top_10": {str(k): int(v) for k, v in vc.items()},
            }

        return {
            "path": str(path),
            "n_rows": int(n_rows),
            "n_cols": int(n_cols),
            "columns": list(df.columns),
            "dtypes": dtypes,
            "null_counts": null_counts,
            "null_pct": null_pct,
            "numeric_cols": numeric_cols,
            "categorical_cols": cat_cols,
            "numeric_stats": numeric_stats,
            "categorical_summary": categorical_summary,
        }
    except Exception as e:
        return {"error": str(e), "type": e.__class__.__name__}


def define_task_impl(
    csv_path: str,
    target_column: str,
    task_type: str = "classification",
    ignore_columns: list[str] | None = None,
    seed: int = 42,
) -> dict:
    """Register an ML task in the store and return its task_id.

    Idempotent: calling with the same (csv_path, target_column, task_type) returns
    the existing task_id with status="existing".
    """
    try:
        df = data.load_csv(csv_path)
        data.validate_task(df, target_column, task_type)
        schema = data.infer_schema(df, target_column, ignore_columns)
        _ = data.build_preprocessor(schema)
    except FileNotFoundError as e:
        return {"error": str(e), "type": "FileNotFoundError"}
    except ValueError as e:
        return {"error": str(e), "type": "ValidationError"}
    except Exception as e:
        return {"error": str(e), "type": e.__class__.__name__}

    try:
        task_id = data.generate_task_id(csv_path, target_column, task_type)
        abs_path = str(Path(csv_path).expanduser().resolve())

        with storage.get_session() as s:
            existing = s.get(storage.Task, task_id)
            if existing is not None:
                return {
                    "task_id": task_id,
                    "csv_path": existing.csv_path,
                    "target_column": existing.target_column,
                    "task_type": existing.task_type,
                    "schema": json.loads(existing.schema_json),
                    "n_rows": int(len(df)),
                    "status": "existing",
                }
            s.add(
                storage.Task(
                    id=task_id,
                    csv_path=abs_path,
                    target_column=target_column,
                    task_type=task_type,
                    schema_json=json.dumps(schema),
                    seed=seed,
                )
            )

        return {
            "task_id": task_id,
            "csv_path": abs_path,
            "target_column": target_column,
            "task_type": task_type,
            "schema": schema,
            "n_rows": int(len(df)),
            "status": "created",
        }
    except Exception as e:
        return {"error": str(e), "type": e.__class__.__name__}


def run_experiment_impl(
    task_id: str,
    models: list[str] | None = None,
    search_strategy: str = "default",
    time_budget_seconds: int = 60,
    n_trials_max: int = 100,
    n_splits: int = 5,
    params: dict | None = None,
) -> dict:
    """Run a multi-model classification experiment, optionally tuning with Optuna."""
    try:
        if search_strategy not in {"default", "optuna"}:
            return {
                "error": f"search_strategy must be 'default' or 'optuna', got {search_strategy!r}",
                "type": "ValidationError",
            }

        with storage.get_session() as s:
            task = s.get(storage.Task, task_id)
            if task is None:
                return {"error": f"task_id {task_id!r} not found", "type": "TaskNotFound"}
            csv_path = task.csv_path
            target_column = task.target_column
            task_type = task.task_type
            schema = json.loads(task.schema_json)
            seed = task.seed

        if task_type != "classification":
            return {
                "error": f"v0.1.0 supports classification only, got {task_type}",
                "type": "UnsupportedTaskType",
            }

        available = trainers.available_trainers()
        if models is None or models == []:
            model_list = available
        else:
            unknown = [m for m in models if m not in available]
            if unknown:
                return {
                    "error": f"Unknown model(s): {unknown}. Available: {available}",
                    "type": "ValidationError",
                }
            model_list = list(dict.fromkeys(models))

        df = data.load_csv(csv_path)
        y = df[target_column].to_numpy()
        feature_cols = schema["numeric"] + schema["categorical"]
        X = df[feature_cols]
        preprocessor = data.build_preprocessor(schema)
        n_classes = schema["n_classes"]

    except ValueError as e:
        return {"error": str(e), "type": "ValidationError"}
    except Exception as e:
        return {"error": str(e), "type": e.__class__.__name__}

    experiment_id = f"exp_{uuid.uuid4().hex[:10]}"
    with storage.get_session() as s:
        s.add(
            storage.Experiment(
                id=experiment_id,
                task_id=task_id,
                models=json.dumps(model_list),
                search_strategy=search_strategy,
                time_budget_s=time_budget_seconds if search_strategy == "optuna" else None,
                status="running",
            )
        )

    primary = primary_metric_name(n_classes, True)

    per_model_budget = (
        max(1, time_budget_seconds // len(model_list))
        if search_strategy == "optuna"
        else None
    )

    per_model_results: dict[str, dict] = {}
    overall_best_score = -float("inf")
    overall_best_model: str | None = None
    overall_best_params: dict | None = None
    total_trials = 0

    for model_name in model_list:
        trainer = trainers.get_trainer(model_name)

        def make_persister(model: str):
            def persist(trial_info: dict) -> None:
                with storage.get_session() as s:
                    s.add(
                        storage.Trial(
                            experiment_id=experiment_id,
                            model=model,
                            params_json=json.dumps(trial_info["params"]),
                            metrics_json=json.dumps(trial_info["metrics"]),
                            duration_s=trial_info["duration_s"],
                        )
                    )
            return persist

        persister = make_persister(model_name)

        try:
            if search_strategy == "default":
                effective_params = trainer.default_params()
                if params is not None and len(model_list) == 1:
                    effective_params = {**effective_params, **params}

                cv_result = search.cross_validate_classification(
                    trainer=trainer,
                    X=X,
                    y=y,
                    preprocessor=preprocessor,
                    params=effective_params,
                    n_splits=n_splits,
                    seed=seed,
                    n_classes=n_classes,
                )
                score = cv_result["aggregated"][primary]["mean"]
                persister(
                    {
                        "params": effective_params,
                        "metrics": cv_result["aggregated"],
                        "duration_s": cv_result["total_fit_seconds"],
                    }
                )
                per_model_results[model_name] = {
                    "best_score": score,
                    "best_params": effective_params,
                    "n_trials": 1,
                    "aggregated_metrics": cv_result["aggregated"],
                }
                total_trials += 1
            else:
                tune_result = search.tune(
                    trainer=trainer,
                    X=X,
                    y=y,
                    preprocessor=preprocessor,
                    n_classes=n_classes,
                    primary_metric=primary,
                    time_budget_seconds=per_model_budget,
                    n_trials_max=n_trials_max,
                    n_splits=n_splits,
                    seed=seed,
                    on_trial_complete=persister,
                )
                per_model_results[model_name] = {
                    "best_score": tune_result["best_score"],
                    "best_params": tune_result["best_params"],
                    "n_trials": tune_result["n_trials"],
                    "elapsed_seconds": tune_result["elapsed_seconds"],
                }
                total_trials += tune_result["n_trials"]

            if per_model_results[model_name]["best_score"] > overall_best_score:
                overall_best_score = per_model_results[model_name]["best_score"]
                overall_best_model = model_name
                overall_best_params = per_model_results[model_name]["best_params"]

        except Exception as e:
            per_model_results[model_name] = {
                "error": str(e),
                "type": e.__class__.__name__,
            }

    with storage.get_session() as s:
        exp = s.get(storage.Experiment, experiment_id)
        if exp is not None:
            exp.status = "complete" if overall_best_model else "failed"
            exp.best_model = overall_best_model
            exp.best_score = overall_best_score if overall_best_model else None
            exp.finished_at = datetime.utcnow()

    return {
        "experiment_id": experiment_id,
        "task_id": task_id,
        "search_strategy": search_strategy,
        "time_budget_seconds": time_budget_seconds if search_strategy == "optuna" else None,
        "primary_metric": primary,
        "models_run": model_list,
        "best_model": overall_best_model,
        "best_score": overall_best_score if overall_best_model else None,
        "best_params": overall_best_params,
        "total_trials": total_trials,
        "per_model": per_model_results,
    }


def get_results_impl(experiment_id: str) -> dict:
    """Return a full markdown report for one experiment, plus headline metrics."""
    try:
        report = reporting.generate_report(experiment_id)
        with storage.get_session() as s:
            exp = s.get(storage.Experiment, experiment_id)
            if exp is None:
                return {
                    "experiment_id": experiment_id,
                    "found": False,
                    "report": report,
                }
            return {
                "experiment_id": experiment_id,
                "found": True,
                "status": exp.status,
                "best_model": exp.best_model,
                "best_score": exp.best_score,
                "report": report,
            }
    except Exception as e:
        return {"error": str(e), "type": e.__class__.__name__}


def compare_runs_impl(experiment_ids: list[str]) -> dict:
    """Return a side-by-side comparison report across experiments."""
    try:
        if not experiment_ids:
            return {
                "error": "experiment_ids must be a non-empty list",
                "type": "ValidationError",
            }
        report = reporting.compare_runs_report(experiment_ids)
        return {
            "experiment_ids": experiment_ids,
            "report": report,
        }
    except Exception as e:
        return {"error": str(e), "type": e.__class__.__name__}
