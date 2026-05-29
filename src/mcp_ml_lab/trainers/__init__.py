"""Trainer registry — available model adapters."""
from mcp_ml_lab.trainers.base import BaseTrainer
from mcp_ml_lab.trainers.lightgbm_trainer import LightGBMTrainer
from mcp_ml_lab.trainers.xgboost_trainer import XGBoostTrainer

TRAINERS: dict[str, type[BaseTrainer]] = {
    "xgboost": XGBoostTrainer,
    "lightgbm": LightGBMTrainer,
}


def get_trainer(name: str) -> BaseTrainer:
    if name not in TRAINERS:
        raise ValueError(
            f"Unknown trainer {name!r}. Available: {sorted(TRAINERS)}"
        )
    return TRAINERS[name]()


def available_trainers() -> list[str]:
    return sorted(TRAINERS)


__all__ = ["BaseTrainer", "TRAINERS", "get_trainer", "available_trainers"]
