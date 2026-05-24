# src/mcp_ml_lab/tools.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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