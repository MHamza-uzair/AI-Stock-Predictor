"""
Inference module: generates multi-step forecasts and computes confidence intervals.
"""
import logging
from typing import Tuple

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from src.feature_engineering import FEATURE_COLUMNS, apply_scaler
from src.model import (
    StockLSTM,
    ST_LOOKBACK, ST_PREDICTION_HORIZON,
    LT_LOOKBACK, LT_PREDICTION_HORIZON,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
CLOSE_COL_IDX = FEATURE_COLUMNS.index("Close")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rolling window for confidence interval std estimation
CI_WINDOW = 30


def _get_last_window(scaled_data: np.ndarray, lookback: int) -> np.ndarray:
    """
    Extract the most recent `lookback` rows from the scaled feature array.

    Parameters
    ----------
    scaled_data : np.ndarray  Shape (n_rows, n_features)
    lookback : int

    Returns
    -------
    np.ndarray  Shape (1, lookback, n_features) — batch of size 1 for model input
    """
    if len(scaled_data) < lookback:
        raise ValueError(
            f"Not enough data: need {lookback} rows but only {len(scaled_data)} available."
        )
    window = scaled_data[-lookback:]             # (lookback, n_features)
    return window[np.newaxis, ...]               # (1, lookback, n_features)


def _single_shot_forecast(
    model: StockLSTM,
    initial_window: np.ndarray,
) -> np.ndarray:
    """
    Generate a full-horizon forecast in a single forward pass (seq2seq).

    The model was trained to map a lookback window directly to a horizon-length
    output vector, so no iterative re-feeding is required.  This eliminates the
    mean-reversion artefact where auto-regressive predictions flatten out after
    a few steps because each re-fed prediction is increasingly noise-corrupted.

    Parameters
    ----------
    model : StockLSTM
    initial_window : np.ndarray  Shape (1, lookback, n_features)

    Returns
    -------
    np.ndarray  Shape (model.horizon,) — normalised predicted close prices
    """
    model.eval()
    model.to(DEVICE)

    x_tensor = torch.tensor(initial_window, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pred = model(x_tensor)          # shape: (1, horizon)

    return pred.cpu().numpy().flatten()  # shape: (horizon,)


def _inverse_transform_prices(
    normalised_prices: np.ndarray,
    close_scaler: MinMaxScaler,
) -> np.ndarray:
    """
    Inverse-transform normalised close prices back to real dollar values.

    Parameters
    ----------
    normalised_prices : np.ndarray  Shape (n,)
    close_scaler : MinMaxScaler  Fitted on the Close column only

    Returns
    -------
    np.ndarray  Shape (n,) in original dollar scale
    """
    return close_scaler.inverse_transform(
        normalised_prices.reshape(-1, 1)
    ).flatten()


def _compute_test_errors(
    model: StockLSTM,
    X_test: np.ndarray,
    y_test: np.ndarray,
    close_scaler: MinMaxScaler,
) -> np.ndarray:
    """
    Run the model on the test set and return per-step prediction errors in real scale.
    Used to estimate the confidence interval width.

    Parameters
    ----------
    model : StockLSTM
    X_test : np.ndarray  Shape (n, lookback, n_features)
    y_test : np.ndarray  Shape (n, horizon) — normalised close prices
    close_scaler : MinMaxScaler

    Returns
    -------
    np.ndarray  Shape (n * horizon,) — errors in dollar scale
    """
    model.eval()
    model.to(DEVICE)

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        preds_norm = model(X_tensor).cpu().numpy()   # (n_samples, horizon)

    preds_flat = preds_norm.flatten()
    actual_flat = y_test.flatten()                   # y_test is (n_samples, horizon)

    preds_real = _inverse_transform_prices(preds_flat, close_scaler)
    actual_real = _inverse_transform_prices(actual_flat, close_scaler)
    return preds_real - actual_real


def _estimate_confidence_std(errors: np.ndarray) -> float:
    """
    Estimate the confidence interval half-width as the mean rolling std of test errors.

    Rolling std is computed over a CI_WINDOW-day window, then averaged to get
    a single representative uncertainty estimate for future predictions.

    Parameters
    ----------
    errors : np.ndarray  Shape (n,)

    Returns
    -------
    float  Standard deviation estimate (in dollars)
    """
    import pandas as pd
    rolling_std = pd.Series(errors).rolling(CI_WINDOW).std().dropna()
    if rolling_std.empty:
        # Fall back to global std if not enough test data
        return float(np.std(errors))
    return float(rolling_std.mean())


def forecast(
    model: StockLSTM,
    feature_df,                     # pd.DataFrame with FEATURE_COLUMNS
    feature_scaler: MinMaxScaler,
    close_scaler: MinMaxScaler,
    mode: str,                      # 'short_term' or 'long_term'
    n_days: int,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Generate a multi-step forecast for a given mode.

    Parameters
    ----------
    model : StockLSTM
        Loaded and trained model (short or long term).
    feature_df : pd.DataFrame
        Full (un-scaled) feature matrix — the last `lookback` rows are used.
    feature_scaler : MinMaxScaler
        Pre-fitted scaler for all features.
    close_scaler : MinMaxScaler
        Pre-fitted scaler for the Close column.
    mode : str
        'short_term' (lookback=60) or 'long_term' (lookback=252).
    n_days : int
        Number of future days to predict.
    X_test, y_test : np.ndarray
        Test set sequences used to estimate confidence interval width.

    Returns
    -------
    Tuple[np.ndarray, float]
        - predicted_prices : shape (n_days,) in real dollar values
        - confidence_std   : estimated ±1σ band width in dollars
    """
    lookback = ST_LOOKBACK if mode == "short_term" else LT_LOOKBACK
    max_horizon = ST_PREDICTION_HORIZON if mode == "short_term" else LT_PREDICTION_HORIZON

    # Clamp n_days to the model's max horizon
    n_days = min(n_days, max_horizon)

    scaled_all = apply_scaler(feature_df, feature_scaler)
    initial_window = _get_last_window(scaled_all, lookback)

    logger.info("Generating %d-step %s forecast…", n_days, mode)
    predicted_norm = _single_shot_forecast(model, initial_window)   # (horizon,)
    predicted_prices = _inverse_transform_prices(predicted_norm, close_scaler)
    predicted_prices = predicted_prices[:n_days]                    # slice if fewer days requested

    # Estimate confidence interval from test-set errors
    errors = _compute_test_errors(model, X_test, y_test, close_scaler)
    confidence_std = _estimate_confidence_std(errors)

    logger.info(
        "Forecast complete: first=%.2f, last=%.2f, CI_std=%.2f",
        predicted_prices[0], predicted_prices[-1], confidence_std,
    )
    return predicted_prices, confidence_std


def predict_xgb_direction(ticker: str, feature_df) -> Tuple[str, float]:
    """
    Load the XGBoost model for `ticker` and predict next-day price direction.

    Parameters
    ----------
    ticker : str
    feature_df : pd.DataFrame  Full unscaled feature matrix (uses last row).

    Returns
    -------
    Tuple[str, float]
        (direction, confidence_pct) — 'Bullish'/'Bearish' and class probability %.

    Raises
    ------
    FileNotFoundError  If XGBoost model hasn't been trained yet.
    """
    from src.ensemble import load_xgb, predict_direction as _predict_direction
    xgb_model = load_xgb(ticker)
    return _predict_direction(xgb_model, feature_df, ticker)


def get_test_sequences(
    feature_df,
    feature_scaler: MinMaxScaler,
    mode: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct the test sequences from the feature matrix.
    Used at inference time to compute confidence intervals.

    Parameters
    ----------
    feature_df : pd.DataFrame
    feature_scaler : MinMaxScaler
    mode : str  'short_term' or 'long_term'
    train_ratio, val_ratio : float

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (X_test, y_test) — test sequences in scaled space
    """
    from src.feature_engineering import build_sequences_seq2seq, chronological_split

    lookback = ST_LOOKBACK if mode == "short_term" else LT_LOOKBACK
    horizon = ST_PREDICTION_HORIZON if mode == "short_term" else LT_PREDICTION_HORIZON
    scaled_all = apply_scaler(feature_df, feature_scaler)
    train_end, val_end = chronological_split(feature_df, train_ratio, val_ratio)

    X_all, y_all = build_sequences_seq2seq(scaled_all, lookback, horizon, CLOSE_COL_IDX)
    val_end_seq = val_end - lookback - horizon
    return X_all[val_end_seq:], y_all[val_end_seq:]
