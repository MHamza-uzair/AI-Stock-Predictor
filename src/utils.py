"""
Shared utility helpers: plotting, metrics formatting, and date handling.
"""
import logging
from datetime import timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
HISTORICAL_DAYS_TO_SHOW = 180
CHART_HISTORICAL_COLOR  = "#2563EB"                    # Electric blue
CHART_PRED_COLOR        = "#00C896"                    # Bullish green
CHART_PRED_BEARISH      = "#FF4444"                    # Bearish red
CHART_CI_COLOR          = "rgba(0, 200, 150, 0.15)"   # Bullish CI band
CHART_CI_BEARISH        = "rgba(255, 68, 68, 0.15)"   # Bearish CI band


def build_forecast_dates(
    last_known_date: pd.Timestamp,
    n_days: int,
) -> List[pd.Timestamp]:
    """
    Generate a list of future trading-day dates (Mon–Fri, skipping weekends).

    Parameters
    ----------
    last_known_date : pd.Timestamp
        The last date with real data (today or most recent trading day).
    n_days : int
        Number of future trading days to generate.

    Returns
    -------
    List[pd.Timestamp]
        Ordered list of future dates.
    """
    future_dates = []
    current = last_known_date + timedelta(days=1)
    while len(future_dates) < n_days:
        if current.weekday() < 5:      # 0=Mon … 4=Fri
            future_dates.append(current)
        current += timedelta(days=1)
    return future_dates


def build_prediction_chart(
    ticker: str,
    mode: str,
    historical_prices: pd.Series,
    predicted_prices: np.ndarray,
    future_dates: List[pd.Timestamp],
    confidence_std: float,
    is_bullish: bool = True,
) -> go.Figure:
    """
    Build a Plotly interactive chart showing historical price and forecast.

    Parameters
    ----------
    ticker : str
    mode : str  'short_term' or 'long_term'
    historical_prices : pd.Series
        Indexed by date, values in USD. Shows last HISTORICAL_DAYS_TO_SHOW days.
    predicted_prices : np.ndarray
        Shape (n_days,) of forecast prices in USD.
    future_dates : List[pd.Timestamp]
        Dates corresponding to each predicted price.
    confidence_std : float
        ±1σ band half-width in USD.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    # Trim historical data to the most recent HISTORICAL_DAYS_TO_SHOW days
    hist_trimmed = historical_prices.iloc[-HISTORICAL_DAYS_TO_SHOW:]

    mode_label = "Short-Term" if mode == "short_term" else "Long-Term"
    title = f"{ticker.upper()} - {mode_label} Price Prediction"
    today = historical_prices.index[-1]

    pred_color = CHART_PRED_COLOR if is_bullish else CHART_PRED_BEARISH
    ci_color   = CHART_CI_COLOR  if is_bullish else CHART_CI_BEARISH

    # Anchor the prediction line to the last historical close so there is no gap.
    last_close = float(historical_prices.iloc[-1])
    anchored_dates  = [today] + list(future_dates)
    anchored_prices = np.concatenate([[last_close], predicted_prices])
    n_days = len(predicted_prices)
    # Width grows as sqrt(t/n_days): zero at the anchor, full confidence_std at the horizon.
    t = np.arange(n_days + 1)          # 0, 1, …, n_days
    ci_width = confidence_std * np.sqrt(t / n_days)
    upper_band = anchored_prices + ci_width
    lower_band = anchored_prices - ci_width

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist_trimmed.index, y=hist_trimmed.values,
        mode="lines", name="Historical Price",
        line=dict(color=CHART_HISTORICAL_COLOR, width=2),
    ))

    fig.add_trace(go.Scatter(
        x=list(anchored_dates) + list(anchored_dates[::-1]),
        y=list(upper_band) + list(lower_band[::-1]),
        fill="toself", fillcolor=ci_color,
        line=dict(color="rgba(255,255,255,0)"),
        name="68% Confidence Interval", hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=anchored_dates, y=anchored_prices,
        mode="lines", name="Predicted Price",
        line=dict(color=pred_color, width=2.5),
    ))

    today_str = today.strftime("%Y-%m-%d")
    fig.add_shape(
        type="line", x0=today_str, x1=today_str, y0=0, y1=1, yref="paper",
        line=dict(dash="dash", color="#94A3B8", width=1),
    )
    fig.add_annotation(
        x=today_str, y=1, yref="paper", text="Today",
        showarrow=False, xanchor="left",
        font=dict(color="#94A3B8", size=11, family="Inter"),
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#F1F5F9", family="Inter"), x=0.01),
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(17,17,24,0.8)", bordercolor="#1E1E2E", borderwidth=1,
            font=dict(color="#94A3B8", size=11),
        ),
        paper_bgcolor="#111118",
        plot_bgcolor="#111118",
        font=dict(color="#94A3B8", family="Inter, sans-serif"),
        xaxis=dict(gridcolor="#1E1E2E", linecolor="#1E1E2E",
                   tickcolor="#94A3B8", tickfont=dict(color="#94A3B8")),
        yaxis=dict(gridcolor="#1E1E2E", linecolor="#1E1E2E",
                   tickcolor="#94A3B8", tickfont=dict(color="#94A3B8")),
        margin=dict(l=50, r=20, t=50, b=40),
    )

    return fig


def get_trend_signal(
    current_price: float,
    final_predicted_price: float,
) -> Tuple[str, str]:
    """
    Determine trend direction based on predicted vs current price.

    Parameters
    ----------
    current_price : float
    final_predicted_price : float

    Returns
    -------
    Tuple[str, str]
        (signal, color) where signal ∈ {'Bullish', 'Bearish'} and
        color ∈ {'green', 'red'}
    """
    if final_predicted_price >= current_price:
        return "Bullish", "green"
    return "Bearish", "red"


def format_metrics_for_display(metrics: dict) -> str:
    """
    Format evaluation metrics dict as a human-readable string.

    Parameters
    ----------
    metrics : dict  Output from train.evaluate_on_test()

    Returns
    -------
    str  Multi-line formatted text
    """
    lines = [
        f"RMSE:  ${metrics.get('rmse', 'N/A'):.2f}",
        f"MAE:   ${metrics.get('mae', 'N/A'):.2f}",
        f"MAPE:  {metrics.get('mape', 'N/A'):.1f}%",
        f"Directional Accuracy: {metrics.get('directional_accuracy', 'N/A'):.1f}%",
    ]
    return "\n".join(lines)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a consistent format for the application."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
