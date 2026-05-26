from __future__ import annotations
import json
import time
import uuid
from datetime import datetime


from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mcp_ml_lab import data, search, storage, trainers
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

import json

from mcp_ml_lab import data, storage


def define_task_impl(
    csv_path: str,
    target_column: str,
    task_type: str = "classification",
    ignore_columns: list[str] | None = None,
    seed: int = 42,) -> dict:
    """Register an ML task in the store and return its task_id.

    Idempotent: calling with the same (csv_path, target_column, task_type) returns
    the existing task_id with status="existing".
    """
    try:
        df = data.load_csv(csv_path)
        data.validate_task(df, target_column, task_type)
        schema = data.infer_schema(df, target_column, ignore_columns)
        # construct the preprocessor as a smoke test — if it raises now,
        # better to surface here than mid-experiment on Day 3.
        _ = data.build_preprocessor(schema)
    except FileNotFoundError as e:
        return {"error": str(e), "type": "FileNotFoundError"}
    except ValueError as e:
        return {"error": str(e), "type": "ValidationError"}
    except Exception as e:
        return {"error": str(e), "type": e.__class__.__name__}

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

def run_experiment_impl(
    task_id: str,
    model_name: str = "xgboost",
    params: dict | None = None,
    n_splits: int = 5,) -> dict:
    """Train one model on the registered task using cross-validation.
    """
    try:
        # 1. Look up the task
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

        # 2. Build the trainer + merged params
        trainer = trainers.get_trainer(model_name)
        merged_params = {**trainer.default_params(), **(params or {})}

        # 3. Reload the CSV and trim to features the schema knows about
        df = data.load_csv(csv_path)
        y = df[target_column].to_numpy()
        feature_cols = schema["numeric"] + schema["categorical"]
        X = df[feature_cols]

        # 4. Rebuild the unfit preprocessor from the persisted schema
        preprocessor = data.build_preprocessor(schema)

        # 5. Run CV
        cv_result = search.cross_validate_classification(
            trainer=trainer,
            X=X,
            y=y,
            preprocessor=preprocessor,
            params=merged_params,
            n_splits=n_splits,
            seed=seed,
            n_classes=schema["n_classes"],
        )
    except ValueError as e:
        return {"error": str(e), "type": "ValidationError"}
    except Exception as e:
        return {"error": str(e), "type": e.__class__.__name__}

    # 6. Pick primary metric and persist
    has_proba = any("auc" in m or "auc_ovr" in m for m in cv_result["fold_metrics"])
    primary = primary_metric_name(schema["n_classes"], has_proba)
    best_score = cv_result["aggregated"][primary]["mean"]

    experiment_id = f"exp_{uuid.uuid4().hex[:10]}"

    with storage.get_session() as s:
        s.add(
            storage.Experiment(
                id=experiment_id,
                task_id=task_id,
                models=json.dumps([model_name]),
                search_strategy="default",
                time_budget_s=None,
                status="complete",
                best_model=model_name,
                best_score=best_score,
                finished_at=datetime.utcnow(),
            )
        )
        s.add(
            storage.Trial(
                experiment_id=experiment_id,
                model=model_name,
                params_json=json.dumps(merged_params),
                metrics_json=json.dumps(cv_result["aggregated"]),
                duration_s=cv_result["total_fit_seconds"],
            )
        )

    return {
        "experiment_id": experiment_id,
        "task_id": task_id,
        "model": model_name,
        "primary_metric": primary,
        "best_score": best_score,
        "aggregated_metrics": cv_result["aggregated"],
        "fold_metrics": cv_result["fold_metrics"],
        "mean_fit_seconds": cv_result["mean_fit_seconds"],
        "total_fit_seconds": cv_result["total_fit_seconds"],
        "n_splits": cv_result["n_splits"],
        "params_used": merged_params,
    }