# scripts/make_demo_data.py
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine

OUT_DIR = Path(__file__).resolve().parent.parent / "examples" / "data"


def dump(loader, name: str) -> None:
    bunch = loader()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    df["target"] = bunch.target
    out = OUT_DIR / f"{name}.csv"
    df.to_csv(out, index=False)
    print(f"  wrote {out}  ({len(df)} rows x {len(df.columns)} cols)")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing demo datasets to {OUT_DIR}/")
    dump(load_breast_cancer, "breast_cancer")
    dump(load_wine, "wine")
    print("Done.")