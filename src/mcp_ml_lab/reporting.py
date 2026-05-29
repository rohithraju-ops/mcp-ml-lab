# src/mcp_ml_lab/reporting.py
"""Markdown report generation for experiments and run comparisons.

Reads from SQLite (experiments + trials tables) and refits the winning model
for feature importance. Returns markdown strings ready to surface in any
MCP client.
"""
from __future__ import annotations

import json

from sqlalchemy import select

from mcp_ml_lab import data, storage, trainers


def _format_score(value: float | None, precision: int = 4) -> str:
    return "—" if value is None else f"{value:.{precision}f}"


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _best_trial_per_model(trials_data: list[dict], primary: str) -> dict[str, dict]:
    """Walk trials once, keep highest-scoring trial per model."""
    best: dict[str, dict] = {}
    for trial in trials_data:
        score = trial["metrics"].get(primary, {}).get("mean", -float("inf"))
        current = best.get(trial["model"])
        if current is None or score > current["score"]:
            best[trial["model"]] = {
                "score": score,
                "params": trial["params"],
                "metrics": trial["metrics"],
                "duration_s": trial["duration_s"],
            }
    return best


def _pick_primary_metric(sample_metrics: dict, n_classes: int) -> str:
    """Pick the metric to rank on, based on what's actually present in trial output."""
    candidates = (
        ["auc", "f1", "accuracy"] if n_classes == 2
        else ["auc_ovr", "f1_macro", "f1_weighted", "accuracy"]
    )
    for c in candidates:
        if c in sample_metrics:
            return c
    return "accuracy"


def _refit_and_get_importance(
    trainer_name: str,
    params: dict,
    task_data: dict,
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """Refit the best model on full data and return top-N (feature, importance) pairs."""
    df = data.load_csv(task_data["csv_path"])
    y = df[task_data["target_column"]].to_numpy()
    schema = task_data["schema"]
    feature_cols = schema["numeric"] + schema["categorical"]
    X = df[feature_cols]

    preprocessor = data.build_preprocessor(schema)
    X_t = preprocessor.fit_transform(X)
    if hasattr(X_t, "columns"):
        feature_names = list(X_t.columns)
    else:
        feature_names = list(preprocessor.get_feature_names_out())

    trainer = trainers.get_trainer(trainer_name)
    merged = {**trainer.default_params(), **params}
    model = trainer.fit(X_t, y, merged)

    importances = trainer.feature_importance(model)
    if importances is None:
        return []

    pairs = list(zip(feature_names, [float(v) for v in importances]))
    pairs.sort(key=lambda p: p[1], reverse=True)
    return pairs[:top_n]


def _load_experiment_data(experiment_id: str) -> dict | None:
    """Load everything needed for a report into plain dicts before session closes."""
    with storage.get_session() as s:
        exp = s.get(storage.Experiment, experiment_id)
        if exp is None:
            return None
        task = s.get(storage.Task, exp.task_id)
        if task is None:
            return None
        trials = list(
            s.execute(
                select(storage.Trial).where(storage.Trial.experiment_id == experiment_id)
            ).scalars()
        )
        return {
            "exp": {
                "id": exp.id,
                "status": exp.status,
                "best_model": exp.best_model,
                "best_score": exp.best_score,
                "search_strategy": exp.search_strategy,
                "time_budget_s": exp.time_budget_s,
                "models": json.loads(exp.models),
                "started_at": str(exp.started_at) if exp.started_at else None,
                "finished_at": str(exp.finished_at) if exp.finished_at else None,
            },
            "task": {
                "id": task.id,
                "csv_path": task.csv_path,
                "target_column": task.target_column,
                "task_type": task.task_type,
                "schema": json.loads(task.schema_json),
                "seed": task.seed,
            },
            "trials": [
                {
                    "model": t.model,
                    "params": json.loads(t.params_json),
                    "metrics": json.loads(t.metrics_json),
                    "duration_s": t.duration_s,
                }
                for t in trials
            ],
        }


def generate_report(experiment_id: str) -> str:
    """Build a markdown report for one experiment."""
    payload = _load_experiment_data(experiment_id)
    if payload is None:
        return f"# Report\n\nExperiment `{experiment_id}` not found.\n"

    exp = payload["exp"]
    task = payload["task"]
    trials_data = payload["trials"]
    n_classes = task["schema"]["n_classes"]

    if not trials_data:
        return (
            f"# Experiment Report: `{experiment_id}`\n\n"
            f"Status: {exp['status']}\n\nNo trials recorded.\n"
        )

    primary = _pick_primary_metric(trials_data[0]["metrics"], n_classes)
    per_model_best = _best_trial_per_model(trials_data, primary)
    trial_counts: dict[str, int] = {}
    for t in trials_data:
        trial_counts[t["model"]] = trial_counts.get(t["model"], 0) + 1

    parts: list[str] = []
    parts.append(f"# Experiment Report: `{experiment_id}`\n")
    parts.append(
        f"**Task:** `{task['id']}` "
        f"({task['task_type']}, {n_classes} classes)  "
    )
    parts.append(f"**Status:** {exp['status']}  ")
    strat_line = f"**Search strategy:** {exp['search_strategy']}"
    if exp["time_budget_s"]:
        strat_line += f" (budget {exp['time_budget_s']}s)"
    parts.append(strat_line + "  ")
    parts.append(
        f"**Winning model:** `{exp['best_model']}` "
        f"with {primary} = {_format_score(exp['best_score'])}\n"
    )

    # Model comparison table
    parts.append("## Model comparison\n")
    all_metric_keys = sorted(
        {k for info in per_model_best.values() for k in info["metrics"].keys()}
    )
    header = ["Model", "n_trials"] + [f"{m} (mean ± std)" for m in all_metric_keys]
    rows = []
    for model, info in per_model_best.items():
        row = [f"`{model}`", str(trial_counts.get(model, 1))]
        for m in all_metric_keys:
            vals = info["metrics"].get(m)
            if vals is None:
                row.append("—")
            else:
                row.append(f"{_format_score(vals['mean'])} ± {_format_score(vals['std'])}")
        rows.append(row)
    parts.append(_md_table(header, rows))

    # Best hyperparameters per model
    parts.append("## Best hyperparameters per model\n")
    for model, info in per_model_best.items():
        parts.append(f"### `{model}`\n")
        parts.append("```json")
        parts.append(json.dumps(info["params"], indent=2, default=str))
        parts.append("```\n")

    # Per-fold breakdown for the winner
    if exp["best_model"] and exp["best_model"] in per_model_best:
        parts.append(f"## Metrics breakdown — `{exp['best_model']}`\n")
        rows = []
        for metric, vals in per_model_best[exp["best_model"]]["metrics"].items():
            rows.append([metric, _format_score(vals["mean"]), _format_score(vals["std"])])
        parts.append(_md_table(["Metric", "Mean", "Std"], rows))

    # Feature importance
    if exp["best_model"] and exp["best_model"] in per_model_best:
        try:
            top = _refit_and_get_importance(
                trainer_name=exp["best_model"],
                params=per_model_best[exp["best_model"]]["params"],
                task_data=task,
                top_n=10,
            )
        except Exception as e:
            parts.append(f"## Feature importance\n\n_Could not compute: {e}_\n")
        else:
            if top:
                parts.append(f"## Top 10 feature importance — `{exp['best_model']}`\n")
                rows = [
                    [str(i), f"`{name}`", f"{imp:.4f}"]
                    for i, (name, imp) in enumerate(top, 1)
                ]
                parts.append(_md_table(["Rank", "Feature", "Importance"], rows))

    # Footer
    footer_bits = []
    if exp["started_at"]:
        footer_bits.append(f"_Started: {exp['started_at']}_")
    if exp["finished_at"]:
        footer_bits.append(f"_Finished: {exp['finished_at']}_")
    footer_bits.append(f"_Total trials: {len(trials_data)}_")
    parts.append("\n" + "  \n".join(footer_bits))

    return "\n".join(parts)


def compare_runs_report(experiment_ids: list[str]) -> str:
    """Build a side-by-side markdown comparison of multiple experiments."""
    if not experiment_ids:
        return "_No experiment IDs provided._\n"

    rows_data: dict[str, dict] = {}
    with storage.get_session() as s:
        for eid in experiment_ids:
            exp = s.get(storage.Experiment, eid)
            if exp is None:
                rows_data[eid] = {"error": "not found"}
                continue
            stmt = select(storage.Trial).where(storage.Trial.experiment_id == eid)
            n_trials = len(list(s.execute(stmt).scalars()))
            rows_data[eid] = {
                "task_id": exp.task_id,
                "search_strategy": exp.search_strategy,
                "best_model": exp.best_model,
                "best_score": exp.best_score,
                "n_trials": n_trials,
                "time_budget_s": exp.time_budget_s,
                "status": exp.status,
            }

    parts = ["# Comparison Report\n"]
    header = ["Experiment", "Task", "Strategy", "Best Model", "Best Score", "n_trials", "Status"]
    rows = []
    for eid in experiment_ids:
        r = rows_data[eid]
        if "error" in r:
            rows.append([f"`{eid}`", "—", "—", "—", "—", "—", r["error"]])
            continue
        rows.append([
            f"`{eid}`",
            f"`{r['task_id']}`",
            r["search_strategy"],
            f"`{r['best_model']}`" if r["best_model"] else "—",
            _format_score(r["best_score"]),
            str(r["n_trials"]),
            r["status"],
        ])
    parts.append(_md_table(header, rows))

    valid = [
        (eid, r) for eid, r in rows_data.items()
        if "error" not in r and r["best_score"] is not None
    ]
    if valid:
        winner_eid, winner = max(valid, key=lambda x: x[1]["best_score"])
        parts.append(
            f"\n**Overall winner:** `{winner_eid}` — `{winner['best_model']}` "
            f"with best score {_format_score(winner['best_score'])}\n"
        )

    return "\n".join(parts)
