from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from pathlib import Path

from mcp_ml_lab.tools import (
    define_task_impl,
    inspect_data_impl,
    run_experiment_impl,
)

from mcp_ml_lab.trainers import available_trainers

mcp = FastMCP("mcp-ml-lab")


@mcp.tool()
def inspect_data(csv_path: str) -> dict:
    """Profile a CSV file before defining an ML task.

    Use this as the FIRST step whenever a user points you at a dataset you haven't seen.
    Returns shape, column dtypes, null counts (absolute + percentage), summary stats for
    numeric columns (mean/std/min/quartiles/max), and top-10 value counts for categoricals.

    Pair this with `define_task` (coming in v0.1.0): inspect first to confirm the target
    column exists and to understand class balance, then call `define_task`.

    Args:
        csv_path: Absolute path, or ~-relative path, to a .csv/.tsv/.txt file on disk.

    Returns:
        On success: dict with keys `path`, `n_rows`, `n_cols`, `columns`, `dtypes`,
        `null_counts`, `null_pct`, `numeric_cols`, `categorical_cols`, `numeric_stats`,
        `categorical_summary`.
        On failure: dict with `error` (human message) and `type` (exception class name).
    """
    return inspect_data_impl(csv_path)



@mcp.tool()
def define_task(
    csv_path: str,
    target_column: str,
    task_type: str = "classification",
    ignore_columns: list[str] | None = None,
    seed: int = 42,) -> dict:
    """Register an ML task so it can be referenced by run_experiment.

    Call this AFTER inspect_data has confirmed the dataset shape and the target
    column exists. Validates the (csv, target, task_type) combination, infers
    which feature columns are numeric vs categorical, and persists the task to
    SQLite. Returns a task_id you'll pass to run_experiment.

    The call is idempotent — re-calling with the same arguments returns the
    same task_id with status="existing" (no duplicate rows).

    Args:
        csv_path: Absolute or ~-relative path to the CSV.
        target_column: The column to predict. Must exist in the CSV.
        task_type: "classification" (v0.1.0 supports only this). "regression"
            is reserved for v0.2.0 and will error.
        ignore_columns: Optional list of feature column names to exclude
            (e.g. IDs, leaked-future columns).
        seed: Random seed stored with the task for reproducibility across
            experiments (default 42).

    Returns:
        On success: dict with `task_id`, `csv_path`, `target_column`,
        `task_type`, `schema` (numeric/categorical/ignored split + class count),
        `n_rows`, and `status` ("created" or "existing").
        On failure: dict with `error` and `type`.
    """
    return define_task_impl(
        csv_path=csv_path,
        target_column=target_column,
        task_type=task_type,
        ignore_columns=ignore_columns,
        seed=seed,
    )

@mcp.tool()
def run_experiment(
    task_id: str,
    model_name: str = "xgboost",
    params: dict | None = None,
    n_splits: int = 5,) -> dict:
    """Train one model on a registered task using k-fold cross-validation.

    Call this AFTER define_task has registered the task. Returns mean and
    std of every classification metric across folds, plus the per-fold scores
    so you can spot instability.

    v0.1.0 Day 3 ships without hyperparameter search — pass params=None to use
    sensible defaults, or pass a dict to override individual settings.
    Hyperparameter tuning (Optuna) lands in the next release.

    Args:
        task_id: Returned by define_task. Must already exist in the store.
        model_name: One of the available trainers. Currently: "xgboost", "lightgbm".
        params: Optional dict of hyperparameter overrides. Unknown keys are
            passed straight to the underlying library — typos will error there.
        n_splits: Number of CV folds (default 5).

    Returns:
        On success: dict with `experiment_id`, `primary_metric`, `best_score`,
        `aggregated_metrics` (mean/std per metric), `fold_metrics` (per-fold dicts),
        timing info, and the resolved `params_used`.
        On failure: dict with `error` and `type`.
    """
    return run_experiment_impl(
        task_id=task_id,
        model_name=model_name,
        params=params,
        n_splits=n_splits,
    )


@mcp.tool()
def list_trainers() -> dict:
    """Return the list of model names accepted by run_experiment.

    Useful for an LLM to discover what's available before calling run_experiment.
    """
    return {"trainers": available_trainers()}

def main() -> None:
    """Console-script entrypoint. Runs the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()

