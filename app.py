from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from smartsignal.data import generate_demo_data
from smartsignal.pipeline import ForecastPipeline

ARTIFACT_DIR = Path("artifacts")

st.set_page_config(page_title="SmartSignal", layout="wide")
st.title("SmartSignal")
st.caption("Leakage-aware next-day stock direction forecasting")

metrics_path = ARTIFACT_DIR / "metrics.json"
if not metrics_path.exists():
    with st.spinner("Building the reproducible demo benchmark..."):
        ForecastPipeline().run(
            generate_demo_data(),
            ticker="DEMO",
            data_source="synthetic reproducible demo",
            artifact_dir=ARTIFACT_DIR,
        )

with metrics_path.open(encoding="utf-8") as handle:
    metrics = json.load(handle)
predictions = pd.read_csv(ARTIFACT_DIR / "predictions.csv", parse_dates=["date"])
importance = pd.read_csv(ARTIFACT_DIR / "feature_importance.csv")

holdout = metrics["holdout"]
baseline = metrics["baseline"]
columns = st.columns(5)
columns[0].metric("Holdout Accuracy", f"{holdout['accuracy']:.1%}")
columns[1].metric(
    "Lift vs. Baseline",
    f"{metrics['accuracy_lift_percentage_points']:+.1f} pp",
)
columns[2].metric("ROC AUC", f"{holdout['roc_auc']:.3f}")
columns[3].metric("Walk-Forward Accuracy", f"{metrics['walk_forward_validation']['accuracy']:.1%}")
signal = metrics["latest_signal"]
columns[4].metric("Latest Signal", signal["direction"], f"P(up) {signal['probability_up']:.1%}")

st.info(
    f"Dataset: **{metrics['ticker']}** · {metrics['data_source']} · "
    f"{metrics['date_range']['start']} to {metrics['date_range']['end']}"
)

left, right = st.columns(2)
with left:
    st.subheader("Out-of-Sample Equity")
    equity = predictions.melt(
        id_vars="date",
        value_vars=["strategy_equity", "buy_hold_equity"],
        var_name="series",
        value_name="growth",
    )
    figure = px.line(equity, x="date", y="growth", color="series")
    figure.update_layout(legend_title_text="", yaxis_title="Growth of $1")
    st.plotly_chart(figure, width="stretch")

with right:
    st.subheader("Top Model Drivers")
    top = importance.head(12).sort_values("importance")
    figure = px.bar(top, x="importance", y="feature", orientation="h")
    figure.update_layout(xaxis_title="Random Forest importance", yaxis_title="")
    st.plotly_chart(figure, width="stretch")

st.subheader("Prediction Confidence")
confidence = go.Figure()
confidence.add_trace(
    go.Scatter(
        x=predictions["date"],
        y=predictions["probability_up"],
        mode="lines",
        name="P(up)",
    )
)
confidence.add_hline(y=0.5, line_dash="dash", line_color="gray")
confidence.update_layout(yaxis_range=[0, 1], yaxis_title="Probability", xaxis_title="")
st.plotly_chart(confidence, width="stretch")

st.subheader("Evaluation Design")
st.json(
    {
        **metrics["methodology"],
        "baseline_accuracy": baseline["accuracy"],
        "model_config": metrics["model_config"],
    },
    expanded=False,
)
st.caption("Research demonstration only. It is not investment advice.")
