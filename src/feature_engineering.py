"""
Feature engineering module: computes technical indicators and normalises features.
"""
import logging
import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# Rolling window sizes
SMA_SHORT = 20
SMA_MED = 50
SMA_LONG = 200
EMA_SHORT = 12
EMA_LONG = 26
EMA_SIGNAL = 9      # MACD signal line period
RSI_PERIOD = 14
ATR_PERIOD = 14
VOL_SMA = 20

# Rows to drop after feature computation to remove NaN from long rolling windows
NAN_TRIM_ROWS = 200

# Feature column order used by both models — must stay consistent
FEATURE_COLUMNS = [
    "Close",
    "Daily_Return",
    "Log_Return",
    "SMA_20",
    "SMA_50",
    "SMA_200",
    "EMA_12",
    "EMA_26",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "RSI_14",
    "BB_Upper",
    "BB_Lower",
    "BB_Width",
    "ATR_14",
    "Volume",
    "Volume_SMA_20",
    "Volume_Ratio",
    "Sentiment_Score",
]


# ---------------------------------------------------------------------------
# Technical indicator helpers
# ---------------------------------------------------------------------------

def _compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    Compute Relative Strength Index (RSI) without relying on ta-lib.

    Uses Wilder's smoothing (exponential weighted with alpha = 1/period).
    RSI output is in [0, 100].
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder EMA: span = 2*period - 1 gives alpha=1/period
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Avoid divide-by-zero
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                 period: int = ATR_PERIOD) -> pd.Series:
    """
    Compute Average True Range (ATR): a measure of intraday volatility.

    True Range = max(H-L, |H-C_prev|, |L-C_prev|)
    ATR = Wilder EMA of True Range over `period` days.
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr


# ---------------------------------------------------------------------------
# Main feature engineering function
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators and assemble the feature matrix.

    This operates on raw OHLCV data BEFORE normalisation. The first
    NAN_TRIM_ROWS rows are dropped after computation to eliminate NaNs
    produced by long rolling windows (e.g. SMA_200 needs 200 days).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned OHLCV DataFrame (from data_ingestion).

    Returns
    -------
    pd.DataFrame
        Feature DataFrame with FEATURE_COLUMNS columns and a DatetimeIndex.
        The 'Sentiment_Score' column is initialised to 0.0 (neutral).
    """
    feat = pd.DataFrame(index=df.index)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # --- Price features ---
    feat["Close"] = close
    feat["Daily_Return"] = close.pct_change()
    # log(0) is undefined: shift guarantees we use close[t-1] which is always >0
    feat["Log_Return"] = np.log(close / close.shift(1))

    # --- Simple Moving Averages ---
    feat["SMA_20"] = close.rolling(SMA_SHORT).mean()
    feat["SMA_50"] = close.rolling(SMA_MED).mean()
    feat["SMA_200"] = close.rolling(SMA_LONG).mean()

    # --- Exponential Moving Averages ---
    feat["EMA_12"] = close.ewm(span=EMA_SHORT, adjust=False).mean()
    feat["EMA_26"] = close.ewm(span=EMA_LONG, adjust=False).mean()

    # --- MACD family ---
    feat["MACD"] = feat["EMA_12"] - feat["EMA_26"]
    feat["MACD_Signal"] = feat["MACD"].ewm(span=EMA_SIGNAL, adjust=False).mean()
    feat["MACD_Histogram"] = feat["MACD"] - feat["MACD_Signal"]

    # --- RSI ---
    feat["RSI_14"] = _compute_rsi(close)

    # --- Bollinger Bands ---
    rolling_std = close.rolling(SMA_SHORT).std()
    feat["BB_Upper"] = feat["SMA_20"] + 2 * rolling_std
    feat["BB_Lower"] = feat["SMA_20"] - 2 * rolling_std
    # Width normalised by SMA to be price-scale-independent
    feat["BB_Width"] = (feat["BB_Upper"] - feat["BB_Lower"]) / feat["SMA_20"]

    # --- ATR (volatility) ---
    feat["ATR_14"] = _compute_atr(high, low, close)

    # --- Volume features ---
    feat["Volume"] = volume
    feat["Volume_SMA_20"] = volume.rolling(VOL_SMA).mean()
    feat["Volume_Ratio"] = feat["Volume"] / feat["Volume_SMA_20"].replace(0, np.finfo(float).eps)

    # --- Sentiment placeholder (filled at inference time if an article is provided) ---
    feat["Sentiment_Score"] = 0.0

    # Drop the first NAN_TRIM_ROWS rows to remove NaNs from long rolling windows
    feat = feat.iloc[NAN_TRIM_ROWS:].copy()

    # Ensure column order matches FEATURE_COLUMNS exactly
    feat = feat[FEATURE_COLUMNS]

    logger.info("Feature matrix shape after trimming: %s", feat.shape)
    _log_any_remaining_nans(feat)

    return feat


def _log_any_remaining_nans(feat: pd.DataFrame) -> None:
    """Log a warning if any NaN values survive after the trim."""
    nan_counts = feat.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if not nan_cols.empty:
        logger.warning("NaN values remain after feature computation: %s", nan_cols.to_dict())


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def fit_scalers(
    feat: pd.DataFrame,
    train_end_idx: int,
    ticker: str,
) -> Tuple[MinMaxScaler, MinMaxScaler]:
    """
    Fit MinMaxScalers exclusively on the training split.

    Two scalers are created:
    - `feature_scaler`: scales all FEATURE_COLUMNS to [0, 1]
    - `close_scaler`:   scales only the 'Close' column to [0, 1]
                        — required for inverse-transforming predictions back
                        to real dollar values

    Parameters
    ----------
    feat : pd.DataFrame
        Full feature matrix (all splits combined).
    train_end_idx : int
        The integer index (exclusive) of the last training row.
    ticker : str
        Used to name the persisted scaler files.

    Returns
    -------
    Tuple[MinMaxScaler, MinMaxScaler]
        (feature_scaler, close_scaler) both fitted on training data only.
    """
    train_data = feat.iloc[:train_end_idx]

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler.fit(train_data.values)

    close_scaler = MinMaxScaler(feature_range=(0, 1))
    # Reshape needed: MinMaxScaler expects 2-D input
    close_scaler.fit(train_data[["Close"]].values)

    _save_scalers(feature_scaler, close_scaler, ticker)
    logger.info("Scalers fitted on %d training rows and saved for %s", len(train_data), ticker)

    return feature_scaler, close_scaler


def _save_scalers(feature_scaler: MinMaxScaler, close_scaler: MinMaxScaler, ticker: str) -> None:
    """Persist both scalers to disk in the models directory."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    paths = {
        "feature_scaler": os.path.join(MODELS_DIR, f"{ticker}_feature_scaler.pkl"),
        "close_scaler": os.path.join(MODELS_DIR, f"{ticker}_close_scaler.pkl"),
    }
    for name, path in paths.items():
        scaler = feature_scaler if name == "feature_scaler" else close_scaler
        try:
            with open(path, "wb") as fh:
                pickle.dump(scaler, fh)
            logger.info("Saved %s to %s", name, path)
        except Exception as exc:
            logger.error("Failed to save %s: %s", name, exc)


def load_scalers(ticker: str) -> Tuple[MinMaxScaler, MinMaxScaler]:
    """
    Load previously fitted scalers from disk.

    Parameters
    ----------
    ticker : str
        Ticker symbol whose scalers to load.

    Returns
    -------
    Tuple[MinMaxScaler, MinMaxScaler]
        (feature_scaler, close_scaler)

    Raises
    ------
    FileNotFoundError
        If scaler files are not found (model has not been trained yet).
    """
    paths = {
        "feature_scaler": os.path.join(MODELS_DIR, f"{ticker}_feature_scaler.pkl"),
        "close_scaler": os.path.join(MODELS_DIR, f"{ticker}_close_scaler.pkl"),
    }
    scalers = {}
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Scaler '{name}' not found at {path}. "
                f"Train the model for {ticker} first."
            )
        try:
            with open(path, "rb") as fh:
                scalers[name] = pickle.load(fh)
        except Exception as exc:
            raise RuntimeError(f"Failed to load {name} from {path}: {exc}") from exc

    return scalers["feature_scaler"], scalers["close_scaler"]


def apply_scaler(feat: pd.DataFrame, scaler: MinMaxScaler) -> np.ndarray:
    """
    Transform the feature matrix using a pre-fitted scaler.

    Parameters
    ----------
    feat : pd.DataFrame
        Feature matrix to transform.
    scaler : MinMaxScaler
        Previously fitted scaler (never fit on val/test data).

    Returns
    -------
    np.ndarray
        Scaled values with same shape as feat.values.
    """
    return scaler.transform(feat.values)


# ---------------------------------------------------------------------------
# Train / Validation / Test split
# ---------------------------------------------------------------------------

def chronological_split(
    feat: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[int, int]:
    """
    Compute chronological split indices (train / val / test).

    Splits are time-ordered: oldest data → train, newest → test.
    Returns integer indices (exclusive upper bounds).

    Parameters
    ----------
    feat : pd.DataFrame
        Full feature DataFrame.
    train_ratio : float
        Fraction of data for training (default 70%).
    val_ratio : float
        Fraction for validation (default 15%). Remainder is test.

    Returns
    -------
    Tuple[int, int]
        (train_end, val_end) — use feat.iloc[:train_end] for train, etc.
    """
    n = len(feat)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    logger.info(
        "Data split — Train: %d rows, Val: %d rows, Test: %d rows",
        train_end, val_end - train_end, n - val_end
    )
    return train_end, val_end


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------

def build_sequences(
    scaled_data: np.ndarray,
    lookback: int,
    close_col_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window (lookback, num_features) sequences for LSTM input.

    X[i] = rows i … i+lookback-1  (shape: lookback × num_features)
    y[i] = the Close price at row i+lookback  (shape: scalar)

    Parameters
    ----------
    scaled_data : np.ndarray
        Normalised feature array of shape (n_rows, n_features).
    lookback : int
        Number of past trading days in each sequence window.
    close_col_idx : int
        Column index of 'Close' in scaled_data (default 0).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X of shape (n_samples, lookback, n_features),
        y of shape (n_samples,).
    """
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback: i])        # past lookback days
        y.append(scaled_data[i, close_col_idx])        # next-day close
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_sequences_seq2seq(
    scaled_data: np.ndarray,
    lookback: int,
    horizon: int,
    close_col_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window sequences for sequence-to-sequence LSTM training.

    Unlike build_sequences (which produces a single next-step target), this
    function produces a full future window as the target so the model can learn
    to predict the entire horizon in one forward pass — eliminating the mean-
    reversion artefact caused by auto-regressive re-feeding.

    X[i] = scaled_data[i : i+lookback]                        shape: (lookback, n_features)
    y[i] = scaled_data[i+lookback : i+lookback+horizon, close_col_idx]  shape: (horizon,)

    Split boundaries should be offset by (lookback + horizon) rather than just
    lookback so that no target window crosses a train/val/test boundary.

    Parameters
    ----------
    scaled_data : np.ndarray
        Normalised feature array of shape (n_rows, n_features).
    lookback : int
        Number of past trading days in each input window.
    horizon : int
        Number of future trading days to predict (length of target vector).
    close_col_idx : int
        Column index of 'Close' in scaled_data (default 0).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X of shape (n_samples, lookback, n_features),
        y of shape (n_samples, horizon)  — normalised close prices.
    """
    X, y = [], []
    n = len(scaled_data)
    for i in range(n - lookback - horizon + 1):
        X.append(scaled_data[i : i + lookback])
        y.append(scaled_data[i + lookback : i + lookback + horizon, close_col_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
