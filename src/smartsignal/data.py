from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


def normalize_market_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Return canonical daily OHLCV data sorted by date."""
    if frame.empty:
        raise ValueError("Market data is empty.")

    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = [str(column).strip().lower().replace(" ", "_") for column in data.columns]

    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce", utc=True)
        data = data.set_index("date")
    elif not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Input must have a DatetimeIndex or a 'date' column.")

    data.index = pd.to_datetime(data.index, errors="coerce", utc=True)
    data.index = data.index.tz_convert(None)
    data.index.name = "date"

    missing = [column for column in PRICE_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required market columns: {', '.join(missing)}")

    optional = [column for column in ("sentiment", "headline_count") if column in data.columns]
    data = data[list(PRICE_COLUMNS) + optional].apply(pd.to_numeric, errors="coerce")
    data = data[~data.index.duplicated(keep="last")].sort_index()
    data = data.dropna(subset=list(PRICE_COLUMNS))
    data = data[data["volume"] >= 0]

    if len(data) < 100:
        raise ValueError("At least 100 valid daily observations are required.")
    return data


def load_csv(path: str | Path) -> pd.DataFrame:
    return normalize_market_data(pd.read_csv(path))


def fetch_yahoo_data(
    ticker: str,
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """Download daily OHLCV history with yfinance."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install project dependencies before fetching market data.") from exc

    frame = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        actions=False,
        threads=False,
    )
    if frame.empty:
        raise ValueError(f"No market data returned for ticker '{ticker}'.")
    return normalize_market_data(frame)


def attach_daily_sentiment(
    market_data: pd.DataFrame,
    sentiment_data: pd.DataFrame,
) -> pd.DataFrame:
    """Join date-level sentiment values to market data without backfilling."""
    market = normalize_market_data(market_data)
    sentiment = sentiment_data.copy()
    sentiment.columns = [str(column).strip().lower() for column in sentiment.columns]
    if "date" not in sentiment.columns or "sentiment" not in sentiment.columns:
        raise ValueError("Sentiment CSV must include 'date' and 'sentiment' columns.")

    sentiment["date"] = pd.to_datetime(sentiment["date"], errors="coerce", utc=True).dt.tz_convert(None)
    sentiment["sentiment"] = pd.to_numeric(sentiment["sentiment"], errors="coerce")
    sentiment = sentiment.dropna(subset=["date", "sentiment"])
    if "headline_count" in sentiment:
        sentiment["headline_count"] = pd.to_numeric(
            sentiment["headline_count"], errors="coerce"
        ).fillna(0.0)
        daily = sentiment.groupby("date").agg(
            sentiment=("sentiment", "mean"),
            headline_count=("headline_count", "sum"),
        )
    else:
        daily = sentiment.groupby("date").agg(
            sentiment=("sentiment", "mean"),
            headline_count=("sentiment", "size"),
        )
    joined = market.drop(columns=["sentiment", "headline_count"], errors="ignore").join(daily)
    joined["sentiment"] = joined["sentiment"].fillna(0.0)
    joined["headline_count"] = joined["headline_count"].fillna(0.0)
    return joined


def generate_demo_data(
    periods: int = 1500,
    seed: int = 42,
    start: str = "2019-01-02",
) -> pd.DataFrame:
    """Create deterministic market-like data with a modest learnable signal.

    The generated series is for pipeline demonstration only. It contains
    volatility regimes, momentum, mean reversion, volume, and a noisy
    sentiment signal observable before the next simulated return.
    """
    if periods < 400:
        raise ValueError("Demo data requires at least 400 observations.")

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=periods)
    returns = np.zeros(periods)
    sentiment = np.zeros(periods)
    volatility = np.full(periods, 0.011)
    latent_regime = 1.0

    for day in range(1, periods):
        if rng.random() < 0.025:
            latent_regime *= -1.0
        volatility[day] = np.clip(
            0.92 * volatility[day - 1] + 0.08 * (0.007 + 0.009 * rng.random()),
            0.006,
            0.025,
        )
        recent = returns[max(0, day - 5) : day]
        momentum = recent.mean() if len(recent) else 0.0
        sentiment[day - 1] = np.clip(
            0.45 * latent_regime + 18.0 * momentum + rng.normal(0.0, 0.48),
            -1.0,
            1.0,
        )
        predictable = 0.0065 * sentiment[day - 1] + 0.08 * momentum
        mean_reversion = -0.03 * returns[day - 1]
        returns[day] = predictable + mean_reversion + rng.normal(0.0, volatility[day])

    sentiment[-1] = np.clip(rng.normal(0.0, 0.55), -1.0, 1.0)
    close = 100.0 * np.exp(np.cumsum(returns))
    overnight = rng.normal(0.0, 0.0025, periods)
    open_price = close * np.exp(overnight)
    intraday_range = np.abs(rng.normal(0.008, 0.003, periods))
    high = np.maximum(open_price, close) * (1.0 + intraday_range)
    low = np.minimum(open_price, close) * (1.0 - intraday_range)
    volume = (
        4_000_000
        * (1.0 + 13.0 * np.abs(returns))
        * rng.lognormal(mean=0.0, sigma=0.22, size=periods)
    ).astype(int)
    headline_count = rng.poisson(lam=4.0 + 3.0 * np.abs(sentiment), size=periods)

    return normalize_market_data(
        pd.DataFrame(
            {
                "date": dates,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "sentiment": sentiment,
                "headline_count": headline_count,
            }
        )
    )


def save_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=True)
    return output


def validate_columns(columns: Iterable[str]) -> None:
    missing = set(PRICE_COLUMNS).difference(columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
