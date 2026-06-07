from __future__ import annotations

import numpy as np
import pandas as pd

from smartsignal.data import normalize_market_data

TARGET_COLUMN = "target_up"
FORWARD_RETURN_COLUMN = "forward_return_1d"


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1 / window, adjust=False).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + relative_strength))


def build_feature_frame(
    market_data: pd.DataFrame,
    include_unlabeled: bool = False,
) -> pd.DataFrame:
    """Engineer features available at close on day t to predict day t+1."""
    data = normalize_market_data(market_data)
    features = pd.DataFrame(index=data.index)
    close = data["close"]
    returns = close.pct_change()

    features["return_1d"] = returns
    for lag in range(2, 6):
        features[f"return_{lag}d"] = close.pct_change(lag)

    for window in (5, 10, 20, 50):
        moving_average = close.rolling(window).mean()
        features[f"close_to_sma_{window}"] = close / moving_average - 1.0

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    features["macd_pct"] = macd / close
    features["macd_signal_pct"] = macd.ewm(span=9, adjust=False).mean() / close
    features["rsi_14"] = _rsi(close, 14) / 100.0

    for window in (5, 10, 20):
        features[f"volatility_{window}"] = returns.rolling(window).std()

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    features["atr_14_pct"] = true_range.rolling(14).mean() / close

    rolling_mean = close.rolling(20).mean()
    rolling_std = close.rolling(20).std()
    features["bollinger_position"] = (close - rolling_mean) / (2.0 * rolling_std)

    log_volume = np.log1p(data["volume"])
    features["volume_change_1d"] = log_volume.diff()
    features["volume_zscore_20"] = (
        (log_volume - log_volume.rolling(20).mean()) / log_volume.rolling(20).std()
    )

    day_of_week = data.index.dayofweek
    features["day_sin"] = np.sin(2.0 * np.pi * day_of_week / 5.0)
    features["day_cos"] = np.cos(2.0 * np.pi * day_of_week / 5.0)

    if "sentiment" in data:
        sentiment = data["sentiment"].clip(-1.0, 1.0).fillna(0.0)
        features["sentiment"] = sentiment
        features["sentiment_mean_3d"] = sentiment.rolling(3, min_periods=1).mean()
        features["sentiment_mean_7d"] = sentiment.rolling(7, min_periods=1).mean()
        features["sentiment_change"] = sentiment.diff()
        if "headline_count" in data:
            features["headline_count_log"] = np.log1p(data["headline_count"].fillna(0.0))

    features[FORWARD_RETURN_COLUMN] = close.shift(-1) / close - 1.0
    features[TARGET_COLUMN] = (features[FORWARD_RETURN_COLUMN] > 0.0).astype(float)
    features.loc[features[FORWARD_RETURN_COLUMN].isna(), TARGET_COLUMN] = np.nan
    features = features.replace([np.inf, -np.inf], np.nan)
    predictors = feature_columns(features)
    features = features.dropna(subset=predictors)
    if include_unlabeled:
        features[TARGET_COLUMN] = features[TARGET_COLUMN].astype("Int64")
    else:
        features = features.dropna(subset=[FORWARD_RETURN_COLUMN, TARGET_COLUMN])
        features[TARGET_COLUMN] = features[TARGET_COLUMN].astype(int)
    return features


def feature_columns(feature_frame: pd.DataFrame) -> list[str]:
    excluded = {TARGET_COLUMN, FORWARD_RETURN_COLUMN}
    return [column for column in feature_frame.columns if column not in excluded]
