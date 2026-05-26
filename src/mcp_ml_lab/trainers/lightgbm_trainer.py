from __future__ import annotations

from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from sklearn import set_config
from mcp_ml_lab.trainers.base import BaseTrainer, ParamsSpace

set_config(transform_output="pandas")
class LightGBMTrainer(BaseTrainer):
    name = "lightgbm"

    def default_params(self) -> dict:
        return {
            "n_estimators": 200,
            "num_leaves": 31,    # LightGBM's complexity knob — leaf-wise growth, not depth-wise
            "max_depth": -1,     # -1 = unlimited; let num_leaves control complexity
            "learning_rate": 0.1,
            "subsample": 1.0,
            "subsample_freq": 0, # disable row subsampling unless > 0
            "colsample_bytree": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,       # suppress LightGBM's chatty stdout — important for MCP stdio
        }

    def params_space(self) -> ParamsSpace:
        return {
            "n_estimators": ("int", 100, 500),
            "num_leaves": ("int", 15, 127),
            "max_depth": ("int", 3, 12),
            "learning_rate": ("loguniform", 0.01, 0.3),
            "subsample": ("uniform", 0.6, 1.0),
            "colsample_bytree": ("uniform", 0.6, 1.0),
        }

    def fit(self, X, y, params: dict) -> Any:
        merged = {**self.default_params(), **params}
        # If subsample < 1.0 was passed in, also enable subsample_freq
        if merged.get("subsample", 1.0) < 1.0 and merged.get("subsample_freq", 0) == 0:
            merged["subsample_freq"] = 1
        model = LGBMClassifier(**merged)
        model.fit(X, y)
        return model

    def predict(self, model: Any, X) -> np.ndarray:
        return model.predict(X)

    def predict_proba(self, model: Any, X) -> np.ndarray:
        return model.predict_proba(X)