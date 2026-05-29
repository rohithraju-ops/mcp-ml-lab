# mcp-ml-lab

**Let AI agents run real ML experiments end-to-end.**

[![PyPI](https://img.shields.io/pypi/v/mcp-ml-lab.svg)](https://pypi.org/project/mcp-ml-lab/)
[![Python](https://img.shields.io/pypi/pyversions/mcp-ml-lab.svg)](https://pypi.org/project/mcp-ml-lab/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An MCP server that gives Claude (or any MCP-aware AI agent) the ability to
profile a CSV, define an ML task, tune XGBoost and LightGBM with Optuna,
and produce a markdown report with feature importance — all from natural
language.

## Why this exists

The existing ML-related MCP servers wrap MLflow, ZenML, or Weights & Biases
and expose them as **read-only** — agents can browse experiment history but
can't actually *run* anything. `mcp-ml-lab` fills the gap: it lets agents
execute the full experimentation loop from a user's natural-language request.

A user typing "train a model on titanic.csv to predict survival" should not
need to know what XGBoost is, what cross-validation is, or how to write a
hyperparameter search. The agent handles all of that — `mcp-ml-lab` is the
tools layer that makes it possible.

## Quick start

```bash
pip install mcp-ml-lab
```

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "ml-lab": {
      "command": "mcp-ml-lab"
    }
  }
}
```

Restart Claude Desktop. The five tools below are now available.

## Example queries

Try these in Claude Desktop with `mcp-ml-lab` connected:

- *"Profile this CSV and tell me if there's class imbalance"*
- *"Compare XGBoost and LightGBM on titanic.csv with 60 seconds of tuning"*
- *"Show me the top 10 features the winning model used"*
- *"How did my last three experiments on the wine dataset compare?"*

## Tools

| Tool | What it does |
|---|---|
| `inspect_data` | Profile a CSV — shape, dtypes, nulls, summary stats, class balance |
| `define_task` | Register an ML task (CSV + target + classification/regression) |
| `run_experiment` | Train one or more models, optionally tuning with Optuna |
| `get_results` | Markdown report with metrics, hyperparameters, feature importance |
| `compare_runs` | Side-by-side comparison of multiple experiments |

Each tool's full signature is in its docstring; they self-document to the LLM.

## How it works

```
Claude Desktop  ───MCP/stdio───  mcp-ml-lab server
                                  │
                                  ├── data.py        CSV loading, schema inference, preprocessor
                                  ├── trainers/      Pluggable XGBoost + LightGBM adapters
                                  ├── search.py      Stratified CV + Optuna TPE tuning
                                  ├── metrics.py     Accuracy, F1, AUC, log loss
                                  ├── storage.py     SQLite via SQLAlchemy 2.0
                                  └── reporting.py   Markdown report generation
```

All experiments and trials are persisted to `~/.mcp-ml-lab/store.db` so an
agent can refer back to runs across sessions.

Full design notes in [ARCHITECTURE.md](ARCHITECTURE.md).

## Roadmap

v0.1.0 ships classification with XGBoost and LightGBM. Planned for v0.2.0+:

- Regression tasks
- Time series forecasting (sktime / darts integration)
- Deep learning baselines (pytorch-tabular)
- Optuna multi-objective search (accuracy × latency × model size)
- Persisted model artifacts with Docker reproducibility
- Permutation feature importance (bias-free alternative to gain importance)
- Notebook export — emit a Jupyter notebook that reproduces the winning run

Issues and PRs welcome.

## Development

```bash
git clone https://github.com/rohithraju-ops/mcp-ml-lab.git
cd mcp-ml-lab
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -v
```

Local debugging is easiest with the [MCP Inspector](https://github.com/modelcontextprotocol/inspector):

```bash
npx @modelcontextprotocol/inspector mcp-ml-lab
```

## License

MIT.

  <!-- mcp-name: io.github.rohithraju-ops/mcp-ml-lab -->