from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

from smartsignal.config import ModelConfig
from smartsignal.evaluation import classification_metrics
from smartsignal.features import TARGET_COLUMN


@dataclass
class ValidationResult:
    metrics: dict[str, object]
    fold_metrics: list[dict[str, object]]
    predictions: pd.DataFrame


def build_model(config: ModelConfig) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        class_weight=config.class_weight,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )


def walk_forward_validate(
    features: pd.DataFrame,
    columns: list[str],
    config: ModelConfig,
) -> ValidationResult:
    """Evaluate expanding-window folds in chronological order."""
    if len(features) <= config.min_train_size + config.validation_folds:
        raise ValueError("Not enough observations for walk-forward validation.")

    available = len(features) - config.min_train_size
    test_size = max(1, available // config.validation_folds)
    splitter = TimeSeriesSplit(
        n_splits=config.validation_folds,
        test_size=test_size,
        max_train_size=None,
    )

    prediction_parts: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, object]] = []
    for fold_number, (train_indices, test_indices) in enumerate(splitter.split(features), start=1):
        if len(train_indices) < config.min_train_size:
            continue
        train = features.iloc[train_indices]
        test = features.iloc[test_indices]
        model = build_model(config)
        model.fit(train[columns], train[TARGET_COLUMN])
        probability = model.predict_proba(test[columns])[:, 1]
        prediction = (probability >= 0.5).astype(int)
        metrics = classification_metrics(test[TARGET_COLUMN], prediction, probability)
        metrics["fold"] = fold_number
        metrics["train_start"] = train.index.min().date().isoformat()
        metrics["train_end"] = train.index.max().date().isoformat()
        metrics["test_start"] = test.index.min().date().isoformat()
        metrics["test_end"] = test.index.max().date().isoformat()
        fold_metrics.append(metrics)
        prediction_parts.append(
            pd.DataFrame(
                {
                    "actual": test[TARGET_COLUMN],
                    "predicted": prediction,
                    "probability_up": probability,
                    "fold": fold_number,
                },
                index=test.index,
            )
        )

    predictions = pd.concat(prediction_parts).sort_index()
    aggregate = classification_metrics(
        predictions["actual"],
        predictions["predicted"],
        predictions["probability_up"],
    )
    return ValidationResult(aggregate, fold_metrics, predictions)


def chronological_split(
    features: pd.DataFrame,
    holdout_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(features) * (1.0 - holdout_fraction))
    if split_index <= 0 or split_index >= len(features):
        raise ValueError("holdout_fraction must leave non-empty train and holdout sets.")
    return features.iloc[:split_index], features.iloc[split_index:]


def baseline_predictions(holdout: pd.DataFrame) -> np.ndarray:
    """Persistence baseline: tomorrow has the same direction as today."""
    return (holdout["return_1d"].to_numpy() > 0.0).astype(int)

