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
    models: list[str] | None = None,
    search_strategy: str = "default",
    time_budget_seconds: int = 60,
    n_trials_max: int = 100,
    n_splits: int = 5,
    params: dict | None = None,) -> dict:
    """Train one or more models on a registered task, optionally tuning hyperparameters.

    Call this after define_task. The same call can train multiple models and
    return the overall winner.

    Two strategies:
      - search_strategy="default" (fast, ~5 seconds total): each model trained
        once with sensible defaults. `params` can override defaults only when
        exactly one model is listed.
      - search_strategy="optuna" (recommended for real use): each model gets
        its own Optuna TPE search, sharing the time budget equally. The trials
        table will hold one row per Optuna trial — typically 20-60 per model
        within a 60-second budget.

    Args:
        task_id: From define_task.
        models: List of trainer names. None or empty list = all available
            (currently ["lightgbm", "xgboost"]).
        search_strategy: "default" or "optuna".
        time_budget_seconds: Wall-clock budget for Optuna search, divided
            evenly across models. Ignored when search_strategy="default".
        n_trials_max: Hard cap on Optuna trial count per model. Acts as a
            circuit breaker — Optuna stops at whichever of (timeout, this) hits first.
        n_splits: CV folds per trial (default 5).
        params: Hyperparameter overrides. Honored only when
            search_strategy="default" AND len(models)==1.

    Returns:
        On success: dict with `experiment_id`, `best_model`, `best_score`,
        `best_params`, `total_trials`, and a `per_model` breakdown
        (best score, best params, trial count per model).
        On failure: dict with `error` and `type`.
    """
    return run_experiment_impl(
        task_id=task_id,
        models=models,
        search_strategy=search_strategy,
        time_budget_seconds=time_budget_seconds,
        n_trials_max=n_trials_max,
        n_splits=n_splits,
        params=params,
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

