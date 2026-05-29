from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from mcp_ml_lab.tools import (
    compare_runs_impl,
    define_task_impl,
    get_results_impl,
    inspect_data_impl,
    run_experiment_impl,
)

mcp = FastMCP("mcp-ml-lab")


@mcp.tool()
def inspect_data(csv_path: str) -> dict:
    """Profile a CSV before defining an ML task.

    Use this as the FIRST step when a user points you at an unfamiliar dataset.
    Returns dataset shape, column dtypes, null counts and percentages, summary
    stats for numeric columns, and top-10 value counts for categoricals — enough
    to confirm the target column exists, understand class balance, and pick
    reasonable arguments for define_task.

    Args:
        csv_path: Absolute path, or ~-relative path, to a .csv/.tsv/.txt file.

    Returns:
        On success: dict with keys n_rows, n_cols, columns, dtypes, null_counts,
        null_pct, numeric_cols, categorical_cols, numeric_stats, categorical_summary.
        On failure: {"error": str, "type": str}.

    Typical flow: inspect_data → define_task → run_experiment → get_results.
    """
    return inspect_data_impl(csv_path)


@mcp.tool()
def define_task(
    csv_path: str,
    target_column: str,
    task_type: str = "classification",
    ignore_columns: list[str] | None = None,
    seed: int = 42,
) -> dict:
    """Register an ML task so it can be referenced by run_experiment.

    Call this AFTER inspect_data has confirmed the dataset and target column.
    Validates the (csv, target, task_type) combination, infers which feature
    columns are numeric vs categorical, and persists the task. Returns a
    task_id for subsequent run_experiment calls.

    Idempotent — calling again with the same args returns the same task_id
    with status="existing" (no duplicates).

    Args:
        csv_path: Absolute or ~-relative path to the CSV.
        target_column: Column to predict. Must exist in the CSV.
        task_type: Currently only "classification" is supported. Regression is
            reserved for v0.2.0 and will return an error.
        ignore_columns: Optional list of feature columns to exclude (e.g. IDs,
            free-text fields, or columns containing leaked future information).
        seed: Random seed stored with the task for reproducible experiments.

    Returns:
        On success: task_id, csv_path, target_column, task_type, schema
        (numeric/categorical/ignored splits + n_classes), n_rows, status
        ("created" or "existing").
        On failure: {"error": str, "type": str}.
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
    params: dict | None = None,
) -> dict:
    """Train one or more models on a registered task, optionally tuning hyperparameters.

    Call this after define_task. The same call can train multiple models and
    return the overall winner.

    Two strategies:
      - search_strategy="default" (fast, ~5 seconds total): each model trained
        once with sensible library defaults.
      - search_strategy="optuna" (recommended): each model gets its own Optuna
        TPE search, sharing the time budget evenly. Typically lifts AUC by
        1-3 points over defaults.

    Args:
        task_id: From define_task.
        models: List of trainer names. None or empty list = all available
            (currently "xgboost" and "lightgbm").
        search_strategy: "default" or "optuna".
        time_budget_seconds: Wall-clock budget for Optuna, divided evenly
            across models. Ignored when search_strategy="default".
        n_trials_max: Hard cap on Optuna trials per model (circuit breaker).
        n_splits: CV folds per trial (default 5).
        params: Hyperparameter overrides. Honored ONLY when
            search_strategy="default" AND exactly one model is listed.

    Returns:
        On success: experiment_id, best_model, best_score, best_params,
        total_trials, plus a per_model breakdown.
        On failure: {"error": str, "type": str}.

    Follow up with get_results(experiment_id) for a full markdown report
    including feature importance and per-metric breakdown.
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
def get_results(experiment_id: str) -> dict:
    """Return a full markdown report for a single experiment.

    Call this AFTER run_experiment to see model comparison, best hyperparameters,
    per-metric mean ± std across CV folds, and top-10 feature importance from
    the winning model.

    The response also includes structured fields (best_model, best_score, status)
    so you can branch on outcomes without parsing markdown.

    Args:
        experiment_id: From run_experiment's response.

    Returns:
        On success: {experiment_id, found: True, status, best_model, best_score, report}.
        If not found: {experiment_id, found: False, report: "..."}.
        On failure: {"error": str, "type": str}.
    """
    return get_results_impl(experiment_id)


@mcp.tool()
def compare_runs(experiment_ids: list[str]) -> dict:
    """Compare multiple experiments side-by-side in a markdown table.

    Use this to answer questions like "did the second tuning run beat the first?"
    or "which task got the best score?" The report includes one row per experiment
    with task, strategy, winning model, best score, and trial count, plus an
    overall winner line.

    Args:
        experiment_ids: List of experiment_id strings (2+ recommended).

    Returns:
        On success: {experiment_ids, report (markdown)}.
        On failure: {"error": str, "type": str}.
    """
    return compare_runs_impl(experiment_ids)


def main() -> None:
    """Console-script entrypoint. Runs the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
