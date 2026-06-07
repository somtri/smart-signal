from __future__ import annotations

import re

import pandas as pd

POSITIVE_WORDS = {
    "accelerate",
    "beat",
    "bullish",
    "buy",
    "exceed",
    "gain",
    "growth",
    "improve",
    "innovation",
    "outperform",
    "profit",
    "raise",
    "record",
    "rebound",
    "strong",
    "surge",
    "upgrade",
}

NEGATIVE_WORDS = {
    "bearish",
    "cut",
    "decline",
    "downgrade",
    "drop",
    "fall",
    "fraud",
    "investigation",
    "layoff",
    "loss",
    "miss",
    "risk",
    "sell",
    "slowdown",
    "weak",
    "warning",
    "worse",
}

NEGATIONS = {"not", "never", "no", "hardly", "without"}
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def _polarity(token: str) -> int:
    forms = {token}
    if token.endswith("ies"):
        forms.add(f"{token[:-3]}y")
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            forms.add(token[: -len(suffix)])
    if token.endswith("ed"):
        forms.add(token[:-1])
    return int(bool(forms & POSITIVE_WORDS)) - int(bool(forms & NEGATIVE_WORDS))


def score_headline(text: str) -> float:
    """Score a headline from -1 to 1 with a transparent finance lexicon."""
    tokens = TOKEN_PATTERN.findall(str(text).lower())
    if not tokens:
        return 0.0

    score = 0.0
    matched = 0
    for index, token in enumerate(tokens):
        polarity = _polarity(token)
        if polarity == 0:
            continue
        if any(word in NEGATIONS for word in tokens[max(0, index - 3) : index]):
            polarity *= -1
        score += polarity
        matched += 1
    return max(-1.0, min(1.0, score / max(1, matched)))


def aggregate_headlines(
    headlines: pd.DataFrame,
    date_column: str = "date",
    text_column: str = "headline",
) -> pd.DataFrame:
    """Convert timestamped headlines into daily sentiment features."""
    if date_column not in headlines or text_column not in headlines:
        raise ValueError(f"Headlines require '{date_column}' and '{text_column}' columns.")
    data = headlines[[date_column, text_column]].copy()
    data["date"] = pd.to_datetime(data[date_column], errors="coerce", utc=True).dt.tz_convert(None)
    data["sentiment"] = data[text_column].map(score_headline)
    data = data.dropna(subset=["date"])
    return data.groupby("date", as_index=False).agg(
        sentiment=("sentiment", "mean"),
        headline_count=("sentiment", "size"),
    )
