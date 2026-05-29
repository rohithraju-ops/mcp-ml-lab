from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer


@pytest.fixture(scope="session")
def breast_cancer_df() -> pd.DataFrame:
    bunch = load_breast_cancer()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    df["target"] = bunch.target
    return df


@pytest.fixture()
def breast_cancer_csv(tmp_path: Path, breast_cancer_df: pd.DataFrame) -> Path:
    p = tmp_path / "breast_cancer.csv"
    breast_cancer_df.to_csv(p, index=False)
    return p
