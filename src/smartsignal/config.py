from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the walk-forward Random Forest benchmark."""

    n_estimators: int = 400
    max_depth: int | None = 7
    min_samples_leaf: int = 8
    max_features: str | float = "sqrt"
    class_weight: str | None = "balanced_subsample"
    random_state: int = 42
    n_jobs: int = -1
    holdout_fraction: float = 0.20
    validation_folds: int = 5
    min_train_size: int = 252

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

