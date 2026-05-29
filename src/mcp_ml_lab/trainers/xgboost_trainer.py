from __future__ import annotations

from typing import Any

import numpy as np
from xgboost import XGBClassifier

from mcp_ml_lab.trainers.base import BaseTrainer, ParamsSpace


class XGBoostTrainer(BaseTrainer):
    name = "xgboost"

    def default_params(self) -> dict:
        return {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",  # 5-10x faster than exact on tabular data
            "eval_metric": "logloss",
        }

    def params_space(self) -> ParamsSpace:
        return {
            "n_estimators": ("int", 100, 500),
            "max_depth": ("int", 3, 10),
            "learning_rate": ("loguniform", 0.01, 0.3),
            "subsample": ("uniform", 0.6, 1.0),
            "colsample_bytree": ("uniform", 0.6, 1.0),
        }

    def fit(self, X, y, params: dict) -> Any:
        merged = {**self.default_params(), **params}
        model = XGBClassifier(**merged)
        model.fit(X, y)
        return model

    def predict(self, model: Any, X) -> np.ndarray:
        return model.predict(X)

    def predict_proba(self, model: Any, X) -> np.ndarray:
        return model.predict_proba(X)

    def feature_importance(self, model: Any) -> np.ndarray:
        return model.feature_importances_
