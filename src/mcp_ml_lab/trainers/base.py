# src/mcp_ml_lab/trainers/base.py
"""Abstract trainer interface. Each model library gets a thin adapter here so
the CV runner and Optuna search code can stay model-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

# Declarative param-space spec. Each entry is (kind, *args).
#   ("int", low, high)          -> integer in [low, high]
#   ("uniform", low, high)      -> float uniform in [low, high]
#   ("loguniform", low, high)   -> float log-uniform in [low, high]
#   ("categorical", [choices])  -> one of the listed values
# Day 4's search.py interprets this with Optuna's suggest_* methods.
ParamsSpace = dict[str, tuple]


class BaseTrainer(ABC):
    """Adapter contract every model implementation must satisfy."""

    name: str  # short identifier, e.g. "xgboost"

    @abstractmethod
    def default_params(self) -> dict:
        """Sensible defaults used when no tuning is requested."""

    @abstractmethod
    def params_space(self) -> ParamsSpace:
        """Declarative search space — see ParamsSpace docstring above."""

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        params: dict,) -> Any:
        """Train and return a fitted model. The returned object is opaque
        to callers — only this trainer's predict/predict_proba touch it."""

    @abstractmethod
    def predict(self, model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Hard class predictions, shape (n_samples,)."""

    def predict_proba(
        self, model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray | None:
        """Per-class probabilities, shape (n_samples, n_classes), or None.

        Override in subclasses that support it. Returning None disables AUC
        and log loss reporting for that trainer (accuracy/F1 still work).
        """
        return None

    def feature_importance(self, model: Any) -> np.ndarray | None:
        """Return per-feature importance scores (shape: (n_features,)) or None.

        Subclasses override when the underlying library exposes importances.
        Returning None lets reporting code skip the section gracefully.
        """
        return None
