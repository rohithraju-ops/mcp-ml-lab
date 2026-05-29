"""Data loading, validation, schema inference, preprocessor construction."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

VALID_TASK_TYPES = {"classification", "regression"}
# v0.1.0 trainers only support classification, but the data layer accepts both
SUPPORTED_TASK_TYPES = {"classification"}


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV from disk. Raises FileNotFoundError / ValueError on issues."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    if not p.is_file():
        raise ValueError(f"Not a regular file: {p}")
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError("CSV loaded but contains zero rows")
    return df


def validate_task(df: pd.DataFrame, target_column: str, task_type: str) -> None:
    """Raise ValueError if (df, target, task_type) isn't internally consistent."""
    if task_type not in VALID_TASK_TYPES:
        raise ValueError(
            f"task_type must be one of {sorted(VALID_TASK_TYPES)}, got {task_type!r}"
        )
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(
            f"task_type {task_type!r} is not supported in v0.1.0 "
            f"(supported: {sorted(SUPPORTED_TASK_TYPES)})"
        )
    if target_column not in df.columns:
        raise ValueError(
            f"target_column {target_column!r} not in CSV columns: {list(df.columns)}"
        )
    target = df[target_column]
    if target.isna().all():
        raise ValueError(f"target column {target_column!r} is entirely null")

    if task_type == "classification":
        n_unique = target.nunique(dropna=True)
        if n_unique < 2:
            raise ValueError(
                f"classification target {target_column!r} has only {n_unique} unique "
                "value(s); need at least 2"
            )
        # cheap heuristic: float target with many uniques is probably regression
        if pd.api.types.is_float_dtype(target) and n_unique > 20:
            raise ValueError(
                f"target {target_column!r} looks continuous ({n_unique} unique float "
                "values); did you mean task_type='regression'?"
            )


def infer_schema(
    df: pd.DataFrame,
    target_column: str,
    ignore_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Split feature columns into numeric / categorical / ignored.

    Rules:
      - target_column is excluded from features.
      - Anything in ignore_columns is moved to `ignored`.
      - Columns that are entirely null are moved to `ignored`.
      - Numeric dtype -> `numeric`. Object/category/bool dtype -> `categorical`.

    Returns a JSON-serializable dict that gets persisted as tasks.schema_json.
    """
    ignore_set = set(ignore_columns or [])

    numeric: list[str] = []
    categorical: list[str] = []
    ignored: list[str] = []

    for col in df.columns:
        if col == target_column:
            continue
        if col in ignore_set:
            ignored.append(col)
            continue
        if df[col].isna().all():
            ignored.append(col)
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        else:
            categorical.append(col)

    target = df[target_column]
    return {
        "target": target_column,
        "target_dtype": str(target.dtype),
        "n_classes": int(target.nunique(dropna=True)),
        "numeric": numeric,
        "categorical": categorical,
        "ignored": ignored,
    }


def build_preprocessor(schema: dict[str, Any]) -> ColumnTransformer:
    """Return an unfit ColumnTransformer from a schema dict.

    Intentionally unfit so callers can clone it per CV fold — the test fold
    never influences the scaler/encoder state.
    """

    transformers = []
    if schema["numeric"]:
        transformers.append(("num", StandardScaler(), schema["numeric"]))
    if schema["categorical"]:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                schema["categorical"],
            )
        )
    if not transformers:
        raise ValueError(
            "Schema produced zero feature columns (all ignored or all-null)."
        )
    return ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def train_test_split_stratified(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified split for classification, plain random split for regression."""
    y = df[target_column]
    X = df.drop(columns=[target_column])
    stratify = y if task_type == "classification" else None
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )


def generate_task_id(csv_path: str | Path, target_column: str, task_type: str) -> str:
    """Return a deterministic task id: task_<csv_stem>_<8-char md5 of (path|target|type)>."""
    p = Path(csv_path).expanduser().resolve()
    raw = f"{p}|{target_column}|{task_type}".encode()
    digest = hashlib.md5(raw).hexdigest()[:8]
    return f"task_{p.stem}_{digest}"
