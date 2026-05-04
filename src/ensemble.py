"""
XGBoost ensemble classifier for next-day price direction prediction.
"""
import logging
import os
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.feature_engineering import FEATURE_COLUMNS, chronological_split

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# ---------------------------------------------------------------------------
# Sector ETF mapping
# ---------------------------------------------------------------------------
SECTOR_ETF: Dict[str, str] = {
    "AAPL": "XLK", "MSFT": "XLK", "GOOGL": "XLK",
    "AMZN": "XLK", "NVDA": "XLK", "META": "XLK",
    "JPM": "XLF", "V": "XLF",
    "NFLX": "XLC",
    "XOM": "XLE",
}
_DEFAULT_ETF = "SPY"

# In-session cache to avoid repeated yfinance requests for the same ETF/date range
_etf_cache: Dict[Tuple, pd.DataFrame] = {}


def _get_etf(ticker: str) -> str:
    """Return the sector ETF symbol for a given stock ticker."""
    return SECTOR_ETF.get(ticker.upper(), _DEFAULT_ETF)


def _xgb_path(ticker: str) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    return os.path.join(MODELS_DIR, f"{ticker}_xgb_model.pkl")


def xgb_exists(ticker: str) -> bool:
    """Return True if the XGBoost model file exists for this ticker."""
    return os.path.exists(_xgb_path(ticker))


def get_sector_momentum(ticker: str, feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch sector ETF data and compute 5-day and 20-day price momentum
    aligned to feature_df's date index.

    Momentum = (price_today / price_n_days_ago) - 1  (same as pct_change(n)).
    Results are cached in-session to avoid redundant network requests during
    training (called once for train/val/test, once at inference).

    Parameters
    ----------
    ticker : str
    feature_df : pd.DataFrame  Feature matrix with a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Two columns: Sector_Mom_5d, Sector_Mom_20d.
        Index matches feature_df.index exactly.
        Rows with insufficient history are filled with 0.0 (neutral).
    """
    import yfinance as yf

    etf = _get_etf(ticker)
    idx = feature_df.index

    # Strip timezone so yfinance accepts the timestamps
    start = idx[0].tz_localize(None) if idx[0].tzinfo else idx[0]
    end   = idx[-1].tz_localize(None) if idx[-1].tzinfo else idx[-1]

    cache_key = (etf, str(start.date()), str(end.date()))
    if cache_key in _etf_cache:
        return _etf_cache[cache_key]

    # Fetch 40 extra calendar days so the 20-day lookback is valid from row 0
    fetch_start = start - pd.Timedelta(days=40)
    try:
        raw = yf.download(
            etf,
            start=fetch_start,
            end=end + pd.Timedelta(days=1),
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )
        if raw.empty:
            raise ValueError(f"No data returned for ETF {etf}")
        close = raw["Close"]
    except Exception as exc:
        logger.warning(
            "Could not fetch ETF %s for %s: %s — using zero momentum.", etf, ticker, exc,
        )
        result = pd.DataFrame(
            {"Sector_Mom_5d": 0.0, "Sector_Mom_20d": 0.0}, index=idx,
        )
        return result

    mom_5d  = close.pct_change(5)
    mom_20d = close.pct_change(20)

    result = pd.DataFrame(index=idx)
    result["Sector_Mom_5d"]  = mom_5d.reindex(idx, method="ffill")
    result["Sector_Mom_20d"] = mom_20d.reindex(idx, method="ffill")
    result = result.fillna(0.0)

    _etf_cache[cache_key] = result
    logger.info("Sector momentum fetched for %s (%s), %d rows.", ticker, etf, len(result))
    return result


def _build_X(feature_df: pd.DataFrame, mom_df: pd.DataFrame) -> np.ndarray:
    """Concatenate the 20 FEATURE_COLUMNS with the 2 sector momentum columns (→ 22 features)."""
    return np.hstack([
        feature_df[FEATURE_COLUMNS].values,
        mom_df[["Sector_Mom_5d", "Sector_Mom_20d"]].values,
    ])


def train_xgb(ticker: str, feature_df: pd.DataFrame) -> float:
    """
    Train an XGBoost binary classifier for next-day price direction.

    Features: 20 FEATURE_COLUMNS + Sector_Mom_5d + Sector_Mom_20d = 22 total.
    Target:   1 if close[t+1] > close[t], else 0.
    Split:    same 70 / 15 / 15 chronological split as the LSTM models.

    Parameters
    ----------
    ticker : str
    feature_df : pd.DataFrame
        Full unscaled feature matrix from compute_features().

    Returns
    -------
    float
        Test-set directional accuracy in percent.
    """
    mom_df = get_sector_momentum(ticker, feature_df)

    close    = feature_df["Close"]
    y_series = (close.shift(-1) > close).astype(int)

    X_all = _build_X(feature_df, mom_df)[:-1]   # drop last row (no label available)
    y_all = y_series.values[:-1]

    train_end, val_end = chronological_split(feature_df)

    X_train, y_train = X_all[:train_end],       y_all[:train_end]
    X_val,   y_val   = X_all[train_end:val_end], y_all[train_end:val_end]
    X_test,  y_test  = X_all[val_end:],          y_all[val_end:]

    logger.info(
        "XGB split for %s — train: %d, val: %d, test: %d, features: %d",
        ticker, len(X_train), len(X_val), len(X_test), X_all.shape[1],
    )

    use_early_stopping = len(X_val) >= 20
    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        early_stopping_rounds=30 if use_early_stopping else None,
        random_state=42,
        n_jobs=-1,
    )

    if use_early_stopping:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train, verbose=False)

    test_acc = float(np.mean(model.predict(X_test) == y_test) * 100)
    logger.info(
        "XGBoost trained for %s — test directional accuracy: %.2f%%", ticker, test_acc,
    )

    with open(_xgb_path(ticker), "wb") as fh:
        pickle.dump(model, fh)
    logger.info("XGBoost model saved to %s", _xgb_path(ticker))

    return test_acc


def load_xgb(ticker: str) -> XGBClassifier:
    """Load the saved XGBoost model for `ticker`."""
    path = _xgb_path(ticker)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"XGBoost model not found at {path}. "
            f"Train the model for {ticker} first."
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


def predict_direction(
    xgb_model: XGBClassifier,
    feature_df: pd.DataFrame,
    ticker: str,
) -> Tuple[str, float]:
    """
    Predict next-day price direction using the last row of feature_df.

    Parameters
    ----------
    xgb_model : XGBClassifier
    feature_df : pd.DataFrame  Full unscaled feature matrix (uses last row).
    ticker : str  Used to fetch the correct sector ETF momentum.

    Returns
    -------
    Tuple[str, float]
        (direction, confidence_pct) — 'Bullish'/'Bearish' and class probability %.
    """
    mom_df = get_sector_momentum(ticker, feature_df)
    X = _build_X(feature_df, mom_df)[-1:]   # last row only, shape (1, 22)

    proba = xgb_model.predict_proba(X)[0]   # [prob_down, prob_up]
    prob_up = float(proba[1])
    direction = "Bullish" if prob_up >= 0.5 else "Bearish"
    confidence = max(prob_up, 1.0 - prob_up) * 100
    return direction, confidence
