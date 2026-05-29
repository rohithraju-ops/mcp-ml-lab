"""Deliberate failure-trigger tests — each test passes bad input and asserts a
structured error response rather than a raised exception.

NOTE: shares ~/.mcp-ml-lab/store.db; idempotent task IDs prevent accumulation.
"""
from pathlib import Path

from mcp_ml_lab import tools

# ---------- inspect_data ----------

def test_inspect_data_missing_file(tmp_path: Path):
    result = tools.inspect_data_impl(str(tmp_path / "does_not_exist.csv"))
    assert "error" in result
    assert result["type"] == "FileNotFoundError"


def test_inspect_data_wrong_extension(tmp_path: Path):
    p = tmp_path / "data.xlsx"
    p.touch()
    result = tools.inspect_data_impl(str(p))
    assert "error" in result
    assert result["type"] == "UnsupportedFileType"


def test_inspect_data_empty_csv(tmp_path: Path):
    p = tmp_path / "empty.csv"
    p.write_text("col1,col2\n")  # header only, zero data rows
    result = tools.inspect_data_impl(str(p))
    assert "error" in result
    assert "Empty" in result["type"] or result["type"] == "EmptyDataError"


# ---------- define_task ----------

def test_define_task_missing_target(breast_cancer_csv: Path):
    result = tools.define_task_impl(
        csv_path=str(breast_cancer_csv),
        target_column="this_does_not_exist",
        task_type="classification",
    )
    assert "error" in result
    assert result["type"] == "ValidationError"


def test_define_task_unsupported_type(breast_cancer_csv: Path):
    result = tools.define_task_impl(
        csv_path=str(breast_cancer_csv),
        target_column="target",
        task_type="regression",
    )
    assert "error" in result
    assert result["type"] == "ValidationError"
    assert "v0.1.0" in result["error"]


def test_define_task_missing_file(tmp_path: Path):
    result = tools.define_task_impl(
        csv_path=str(tmp_path / "nope.csv"),
        target_column="target",
        task_type="classification",
    )
    assert "error" in result
    assert result["type"] == "FileNotFoundError"


# ---------- run_experiment ----------

def test_run_experiment_unknown_task():
    result = tools.run_experiment_impl(task_id="task_does_not_exist_xyz")
    assert "error" in result
    assert result["type"] == "TaskNotFound"


def test_run_experiment_unknown_model(breast_cancer_csv: Path):
    define_result = tools.define_task_impl(
        csv_path=str(breast_cancer_csv),
        target_column="target",
        task_type="classification",
    )
    assert "task_id" in define_result, f"define_task failed: {define_result}"
    task_id = define_result["task_id"]

    result = tools.run_experiment_impl(
        task_id=task_id,
        models=["fakemodel_does_not_exist"],
    )
    assert "error" in result
    assert result["type"] == "ValidationError"


def test_run_experiment_bad_strategy(breast_cancer_csv: Path):
    define_result = tools.define_task_impl(
        csv_path=str(breast_cancer_csv),
        target_column="target",
        task_type="classification",
    )
    assert "task_id" in define_result, f"define_task failed: {define_result}"
    task_id = define_result["task_id"]

    result = tools.run_experiment_impl(
        task_id=task_id,
        search_strategy="random",
    )
    assert "error" in result
    assert result["type"] == "ValidationError"


# ---------- get_results ----------

def test_get_results_unknown_experiment():
    # not-found → found=False, not an error key — a distinct, expected outcome
    result = tools.get_results_impl(experiment_id="exp_does_not_exist_xyz")
    assert result["found"] is False
    assert "report" in result


# ---------- compare_runs ----------

def test_compare_runs_empty_list():
    result = tools.compare_runs_impl(experiment_ids=[])
    assert "error" in result
    assert result["type"] == "ValidationError"


def test_compare_runs_unknown_ids():
    # unknown IDs → report with "not found" rows, not an error
    result = tools.compare_runs_impl(
        experiment_ids=["exp_nope_1", "exp_nope_2"]
    )
    assert "report" in result
    assert "not found" in result["report"]
