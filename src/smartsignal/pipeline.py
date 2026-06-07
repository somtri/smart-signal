from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from smartsignal.config import ModelConfig
from smartsignal.evaluation import annualized_strategy_metrics, classification_metrics
from smartsignal.features import (
    FORWARD_RETURN_COLUMN,
    TARGET_COLUMN,
    build_feature_frame,
    feature_columns,
)
from smartsignal.modeling import (
    baseline_predictions,
    build_model,
    chronological_split,
    walk_forward_validate,
)


@dataclass
class RunResult:
    metrics: dict[str, object]
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    artifact_dir: Path | None = None


class ForecastPipeline:
    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()

    def run(
        self,
        market_data: pd.DataFrame,
        ticker: str = "DEMO",
        data_source: str = "unknown",
        artifact_dir: str | Path | None = None,
    ) -> RunResult:
        frame = build_feature_frame(market_data)
        inference_frame = build_feature_frame(market_data, include_unlabeled=True)
        columns = feature_columns(frame)
        train, holdout = chronological_split(frame, self.config.holdout_fraction)

        validation = walk_forward_validate(train, columns, self.config)
        model = build_model(self.config)
        model.fit(train[columns], train[TARGET_COLUMN])

        probability = model.predict_proba(holdout[columns])[:, 1]
        prediction = (probability >= 0.5).astype(int)
        baseline = baseline_predictions(holdout)
        baseline_probability = np.full(len(holdout), train[TARGET_COLUMN].mean())

        model_metrics = classification_metrics(holdout[TARGET_COLUMN], prediction, probability)
        model_metrics["strategy"] = annualized_strategy_metrics(
            holdout[FORWARD_RETURN_COLUMN],
            prediction,
        )
        baseline_metrics = classification_metrics(
            holdout[TARGET_COLUMN],
            baseline,
            baseline_probability,
        )
        sentiment_columns = [
            column
            for column in columns
            if "sentiment" in column or column == "headline_count_log"
        ]
        sentiment_ablation = None
        if sentiment_columns:
            technical_columns = [
                column for column in columns if column not in sentiment_columns
            ]
            technical_model = build_model(self.config)
            technical_model.fit(train[technical_columns], train[TARGET_COLUMN])
            technical_probability = technical_model.predict_proba(
                holdout[technical_columns]
            )[:, 1]
            technical_prediction = (technical_probability >= 0.5).astype(int)
            technical_metrics = classification_metrics(
                holdout[TARGET_COLUMN],
                technical_prediction,
                technical_probability,
            )
            sentiment_ablation = {
                "technical_only_accuracy": technical_metrics["accuracy"],
                "technical_plus_sentiment_accuracy": model_metrics["accuracy"],
                "sentiment_lift_percentage_points": 100.0
                * (
                    float(model_metrics["accuracy"])
                    - float(technical_metrics["accuracy"])
                ),
                "sentiment_feature_count": len(sentiment_columns),
            }

        predictions = pd.DataFrame(
            {
                "actual": holdout[TARGET_COLUMN],
                "predicted": prediction,
                "probability_up": probability,
                "baseline_predicted": baseline,
                "forward_return": holdout[FORWARD_RETURN_COLUMN],
            },
            index=holdout.index,
        )
        predictions["strategy_return"] = np.where(
            predictions["predicted"] == 1,
            predictions["forward_return"],
            -predictions["forward_return"],
        )
        predictions["strategy_equity"] = (1.0 + predictions["strategy_return"]).cumprod()
        predictions["buy_hold_equity"] = (1.0 + predictions["forward_return"]).cumprod()

        importance = pd.DataFrame(
            {"feature": columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False, ignore_index=True)

        deployment_model = build_model(self.config)
        deployment_model.fit(frame[columns], frame[TARGET_COLUMN])
        latest_date = inference_frame.index.max()
        latest_probability = float(
            deployment_model.predict_proba(inference_frame.loc[[latest_date], columns])[0, 1]
        )
        latest_signal = {
            "as_of_date": latest_date.date().isoformat(),
            "direction": "UP" if latest_probability >= 0.5 else "DOWN",
            "probability_up": latest_probability,
            "confidence": abs(latest_probability - 0.5) * 2.0,
        }

        metrics: dict[str, object] = {
            "project": "SmartSignal",
            "ticker": ticker.upper(),
            "data_source": data_source,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "date_range": {
                "start": frame.index.min().date().isoformat(),
                "end": frame.index.max().date().isoformat(),
            },
            "feature_count": len(columns),
            "total_observations": len(frame),
            "train_observations": len(train),
            "holdout_observations": len(holdout),
            "holdout": model_metrics,
            "baseline": baseline_metrics,
            "sentiment_ablation": sentiment_ablation,
            "latest_signal": latest_signal,
            "accuracy_lift_percentage_points": 100.0
            * (float(model_metrics["accuracy"]) - float(baseline_metrics["accuracy"])),
            "walk_forward_validation": validation.metrics,
            "validation_folds": validation.fold_metrics,
            "model_config": self.config.to_dict(),
            "methodology": {
                "target": "Next trading day's close is above the current close.",
                "split": "Final 20% chronological holdout; no random shuffling.",
                "validation": "Expanding-window TimeSeriesSplit on pre-holdout data.",
                "baseline": "Persistence: next direction equals current daily direction.",
                "decision_threshold": 0.5,
                "strategy_note": "Illustrative long/short returns exclude costs and slippage.",
            },
        }

        output_path = Path(artifact_dir) if artifact_dir else None
        if output_path:
            self._save_artifacts(
                output_path,
                metrics,
                predictions,
                importance,
                deployment_model,
                columns,
            )
        return RunResult(metrics, predictions, importance, output_path)

    @staticmethod
    def _save_artifacts(
        output_path: Path,
        metrics: dict[str, object],
        predictions: pd.DataFrame,
        importance: pd.DataFrame,
        model: object,
        columns: list[str],
    ) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        with (output_path / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        with (output_path / "latest_signal.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics["latest_signal"], handle, indent=2)
        predictions.to_csv(output_path / "predictions.csv", index=True)
        importance.to_csv(output_path / "feature_importance.csv", index=False)
        joblib.dump(
            {"model": model, "feature_columns": columns},
            output_path / "model.joblib",
        )
