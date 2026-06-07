from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_probability: Sequence[float],
) -> dict[str, object]:
    truth = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    probability = np.asarray(y_probability)
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    try:
        roc_auc = float(roc_auc_score(truth, probability))
    except ValueError:
        roc_auc = None

    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "roc_auc": roc_auc,
        "brier_score": float(brier_score_loss(truth, probability)),
        "confusion_matrix": matrix.tolist(),
        "observations": int(len(truth)),
        "positive_rate": float(truth.mean()),
    }


def annualized_strategy_metrics(
    forward_returns: Sequence[float],
    predictions: Sequence[int],
) -> dict[str, float | None]:
    returns = np.asarray(forward_returns, dtype=float)
    positions = np.where(np.asarray(predictions) == 1, 1.0, -1.0)
    strategy_returns = positions * returns
    standard_deviation = strategy_returns.std(ddof=1)
    sharpe = (
        float(np.sqrt(252.0) * strategy_returns.mean() / standard_deviation)
        if standard_deviation > 0
        else None
    )
    equity = np.cumprod(1.0 + strategy_returns)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    return {
        "annualized_return": float((equity[-1] ** (252.0 / len(equity))) - 1.0),
        "annualized_volatility": float(standard_deviation * np.sqrt(252.0)),
        "sharpe_ratio": sharpe,
        "max_drawdown": float(drawdown.min()),
        "cumulative_return": float(equity[-1] - 1.0),
    }

