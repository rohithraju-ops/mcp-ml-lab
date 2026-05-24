# Architecture — mcp-ml-lab v0.1.0

This document is the contract for v0.1.0. Every "should I add X?" question during
week 1 gets answered with "is it in this file?" If not, defer to v0.2.0.

## Tool contract

Five tools. No more for v0.1.0.

| Tool             | Inputs                                                                   | Returns                                       |
|------------------|--------------------------------------------------------------------------|-----------------------------------------------|
| `inspect_data`   | `csv_path: str`                                                          | profile dict (shape, dtypes, nulls, stats)    |
| `define_task`    | `csv_path: str`, `target_column: str`, `task_type: str`                  | `task_id: str` + summary                      |
| `run_experiment` | `task_id`, `models: list[str]`, `search_strategy`, `time_budget_seconds` | `experiment_id` + metrics summary             |
| `get_results`    | `experiment_id: str`                                                     | metrics, best params, per-trial details       |
| `compare_runs`   | `experiment_ids: list[str]`                                              | markdown comparison report                    |

## Storage schema (SQLite)

Single database at `~/.mcp-ml-lab/store.db`. Three tables, foreign-keyed.

```
tasks
  id              TEXT PRIMARY KEY   -- e.g. "task_breast_cancer_<short_hash>"
  csv_path        TEXT NOT NULL
  target_column   TEXT NOT NULL
  task_type       TEXT NOT NULL      -- "classification" | "regression" (v0.1.0: classification only)
  schema_json     TEXT NOT NULL      -- inferred column roles (numeric/categorical/ignored)
  seed            INTEGER NOT NULL   -- the "task seed" used for all randomness
  created_at      DATETIME NOT NULL

experiments
  id              TEXT PRIMARY KEY
  task_id         TEXT NOT NULL REFERENCES tasks(id)
  models          TEXT NOT NULL      -- JSON list of model names
  search_strategy TEXT NOT NULL      -- "default" | "optuna"
  time_budget_s   INTEGER
  status          TEXT NOT NULL      -- "running" | "complete" | "failed"
  best_model      TEXT
  best_score      REAL
  started_at      DATETIME NOT NULL
  finished_at     DATETIME

trials
  id              INTEGER PRIMARY KEY AUTOINCREMENT
  experiment_id   TEXT NOT NULL REFERENCES experiments(id)
  model           TEXT NOT NULL
  params_json     TEXT NOT NULL
  metrics_json    TEXT NOT NULL      -- {auc, f1, accuracy, fold_scores: [...]}
  duration_s      REAL NOT NULL
  created_at      DATETIME NOT NULL
```

## Trainer interface

Every model adapter implements this interface (defined in `trainers/base.py`):

```python
class BaseTrainer:
    name: str

    def default_params(self) -> dict: ...
    def params_space(self) -> dict: ...      # Optuna trial space
    def fit(self, X, y, params) -> Any: ...  # returns fitted model
    def predict(self, model, X) -> np.ndarray: ...
    def predict_proba(self, model, X) -> np.ndarray | None: ...
```

v0.1.0 ships two implementations: `XGBoostTrainer`, `LightGBMTrainer`.

## Execution flow

```
LLM calls tool
   |
server.py routes to tools.py implementation
   |
data.py: load CSV, validate, infer schema
   |
storage.py: write task / experiment / trial rows
   |
search.py: invoke trainers, run CV, optionally tune
   |
metrics.py: compute and persist
   |
reporting.py: assemble markdown for compare_runs
   |
return JSON-serializable dict to LLM
```

## Out of scope for v0.1.0

- Regression tasks (only classification ships)
- Time series
- Deep learning trainers
- Distributed training
- Authentication / multi-user storage
- Streaming progress updates over MCP (returning final results only)
- Reproducible artifacts (Docker, frozen environments)