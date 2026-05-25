from pathlib import Path

import pytest
from sklearn.compose import ColumnTransformer

from mcp_ml_lab import data


def test_load_csv_success(breast_cancer_csv: Path) -> None:
    df = data.load_csv(breast_cancer_csv)
    assert len(df) == 569
    assert "target" in df.columns


def test_load_csv_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        data.load_csv(tmp_path / "does_not_exist.csv")


def test_validate_task_ok(breast_cancer_df) -> None:
    # No exception means pass
    data.validate_task(breast_cancer_df, "target", "classification")


def test_validate_task_missing_target(breast_cancer_df) -> None:
    with pytest.raises(ValueError, match="not in CSV columns"):
        data.validate_task(breast_cancer_df, "this_column_does_not_exist", "classification")


def test_validate_task_unsupported_type(breast_cancer_df) -> None:
    with pytest.raises(ValueError, match="not supported in v0.1.0"):
        data.validate_task(breast_cancer_df, "target", "regression")


def test_infer_schema_breast_cancer(breast_cancer_df) -> None:
    schema = data.infer_schema(breast_cancer_df, "target")
    # all 30 features are numeric floats
    assert len(schema["numeric"]) == 30
    assert schema["categorical"] == []
    assert schema["target"] == "target"
    assert schema["n_classes"] == 2


def test_build_preprocessor_returns_columntransformer(breast_cancer_df) -> None:
    schema = data.infer_schema(breast_cancer_df, "target")
    pre = data.build_preprocessor(schema)
    assert isinstance(pre, ColumnTransformer)
    # not yet fit — fitting happens inside CV folds
    assert not hasattr(pre, "transformers_")


def test_generate_task_id_is_deterministic(tmp_path: Path, breast_cancer_csv: Path) -> None:
    id_a = data.generate_task_id(breast_cancer_csv, "target", "classification")
    id_b = data.generate_task_id(breast_cancer_csv, "target", "classification")
    assert id_a == id_b
    # different target -> different id
    id_c = data.generate_task_id(breast_cancer_csv, "mean radius", "classification")
    assert id_a != id_c