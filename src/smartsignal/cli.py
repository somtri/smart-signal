from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from smartsignal.data import (
    attach_daily_sentiment,
    fetch_yahoo_data,
    generate_demo_data,
    load_csv,
    save_csv,
)
from smartsignal.pipeline import ForecastPipeline
from smartsignal.sentiment import aggregate_headlines


def _print_summary(metrics: dict[str, object], output: str) -> None:
    holdout = metrics["holdout"]
    baseline = metrics["baseline"]
    validation = metrics["walk_forward_validation"]
    print("\nSmartSignal evaluation")
    print(f"Ticker: {metrics['ticker']} | Source: {metrics['data_source']}")
    print(
        f"Holdout accuracy: {holdout['accuracy']:.1%} | "
        f"Baseline: {baseline['accuracy']:.1%} | "
        f"Lift: {metrics['accuracy_lift_percentage_points']:+.1f} pp"
    )
    print(
        f"Walk-forward accuracy: {validation['accuracy']:.1%} | "
        f"ROC AUC: {holdout['roc_auc']:.3f}"
    )
    signal = metrics["latest_signal"]
    print(
        f"Latest signal ({signal['as_of_date']}): {signal['direction']} | "
        f"P(up): {signal['probability_up']:.1%}"
    )
    print(f"Artifacts: {Path(output).resolve()}")


def _run_training(
    data: pd.DataFrame,
    ticker: str,
    source: str,
    output: str,
) -> None:
    result = ForecastPipeline().run(
        data,
        ticker=ticker,
        data_source=source,
        artifact_dir=output,
    )
    _print_summary(result.metrics, output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smartsignal",
        description="Train and evaluate a next-day stock direction model.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run the reproducible offline benchmark.")
    demo.add_argument("--periods", type=int, default=1500)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--output", default="artifacts")

    train = subparsers.add_parser("train", help="Train from a local OHLCV CSV.")
    train.add_argument("--csv", required=True)
    train.add_argument("--ticker", default="CUSTOM")
    train.add_argument("--sentiment-csv")
    train.add_argument("--output", default="artifacts")

    fetch = subparsers.add_parser("fetch", help="Download Yahoo Finance data and train.")
    fetch.add_argument("--ticker", required=True)
    fetch.add_argument("--start", default="2018-01-01")
    fetch.add_argument("--end")
    fetch.add_argument("--output", default="artifacts")
    fetch.add_argument("--save-data", default="data/raw")

    sentiment = subparsers.add_parser(
        "score-headlines",
        help="Convert a headline CSV into daily sentiment features.",
    )
    sentiment.add_argument("--csv", required=True)
    sentiment.add_argument("--output", default="data/daily_sentiment.csv")

    inspect = subparsers.add_parser("show-metrics", help="Print a saved metrics file.")
    inspect.add_argument("--path", default="artifacts/metrics.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "demo":
        data = generate_demo_data(periods=args.periods, seed=args.seed)
        _run_training(data, "DEMO", "synthetic reproducible demo", args.output)
    elif args.command == "train":
        data = load_csv(args.csv)
        if args.sentiment_csv:
            data = attach_daily_sentiment(data, pd.read_csv(args.sentiment_csv))
        _run_training(data, args.ticker, f"CSV: {args.csv}", args.output)
    elif args.command == "fetch":
        data = fetch_yahoo_data(args.ticker, args.start, args.end)
        save_csv(data, Path(args.save_data) / f"{args.ticker.upper()}.csv")
        _run_training(data, args.ticker, "Yahoo Finance", args.output)
    elif args.command == "score-headlines":
        daily = aggregate_headlines(pd.read_csv(args.csv))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(output, index=False)
        print(f"Saved {len(daily)} daily sentiment rows to {output.resolve()}")
    elif args.command == "show-metrics":
        with Path(args.path).open(encoding="utf-8") as handle:
            print(json.dumps(json.load(handle), indent=2))


if __name__ == "__main__":
    main()
