"""
Live data module: fetches the most recent trading day's price and handles market hours.
"""
import logging
from datetime import datetime, time
from typing import Optional, Tuple

import pandas as pd
import pytz
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
NYSE_TZ = "America/New_York"
MARKET_OPEN = time(9, 30)       # 09:30 ET
MARKET_CLOSE = time(16, 0)      # 16:00 ET
LIVE_FETCH_PERIOD = "5d"        # Fetch last 5 trading days to guarantee a valid close
LIVE_FETCH_INTERVAL = "1d"


def is_market_open() -> bool:
    """
    Return True if NYSE is currently open (09:30–16:00 ET, Mon–Fri).

    Uses pytz for reliable Eastern Time conversion.

    Returns
    -------
    bool
    """
    eastern = pytz.timezone(NYSE_TZ)
    now_et = datetime.now(eastern)

    is_weekday = now_et.weekday() < 5           # 0=Mon … 4=Fri
    current_time = now_et.time().replace(tzinfo=None)
    during_hours = MARKET_OPEN <= current_time <= MARKET_CLOSE

    return is_weekday and during_hours


def fetch_live_price(ticker: str) -> Tuple[float, float, str]:
    """
    Fetch the most recent available closing price for a ticker.

    Queries the last 5 trading days from Yahoo Finance and uses the most
    recent row. This ensures we always get a valid close even on weekends
    or public holidays when markets are closed.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. 'AAPL').

    Returns
    -------
    Tuple[float, float, str]
        (latest_close, pct_change_from_prior_day, date_str)
        - latest_close : float — most recent adjusted close price in USD
        - pct_change   : float — % change relative to the previous trading day
        - date_str     : str  — date of the latest data point (YYYY-MM-DD)

    Raises
    ------
    RuntimeError
        If Yahoo Finance returns no data for the ticker.
    """
    ticker = ticker.upper().strip()
    logger.info("Fetching live price for %s…", ticker)

    try:
        raw = yf.download(
            ticker,
            period=LIVE_FETCH_PERIOD,
            interval=LIVE_FETCH_INTERVAL,
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch live data for {ticker}: {exc}") from exc

    if raw.empty:
        raise RuntimeError(f"No live data returned for ticker '{ticker}'. Check the symbol.")

    # Flatten MultiIndex columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.sort_index(ascending=True)

    latest_close = float(raw["Close"].iloc[-1])
    date_str = raw.index[-1].strftime("%B %d, %Y")

    # Compute % change relative to prior trading day
    if len(raw) >= 2:
        prior_close = float(raw["Close"].iloc[-2])
        pct_change = ((latest_close - prior_close) / prior_close) * 100
    else:
        pct_change = 0.0

    logger.info(
        "Live data for %s: $%.2f (%+.2f%%) as of %s",
        ticker, latest_close, pct_change, date_str,
    )
    return latest_close, pct_change, date_str


def append_live_row(
    feature_df: pd.DataFrame,
    ticker: str,
    sentiment_score: float = 0.0,
) -> pd.DataFrame:
    """
    Fetch the latest trading day's OHLCV data and append it to the feature DataFrame.

    This ensures that predictions always use the most recent closing price as the
    final input point in the lookback window. The new row's technical indicators
    are forward-filled from the previous row as a reasonable approximation
    (they will be recomputed when the full historical data is refreshed next day).

    Parameters
    ----------
    feature_df : pd.DataFrame
        Existing feature matrix (from compute_features).
    ticker : str
    sentiment_score : float
        Current sentiment score to inject into the new row.

    Returns
    -------
    pd.DataFrame
        feature_df with the live row appended (or returned unchanged if the
        latest date is already present in the DataFrame).
    """
    try:
        raw = yf.download(
            ticker.upper(),
            period=LIVE_FETCH_PERIOD,
            interval=LIVE_FETCH_INTERVAL,
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        logger.warning("Could not fetch live row for appending: %s. Using existing data.", exc)
        return feature_df

    if raw.empty:
        return feature_df

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.sort_index(ascending=True)
    latest_date = raw.index[-1]

    # Skip if we already have this date in the feature matrix
    if latest_date in feature_df.index:
        logger.info("Live row date %s already in feature matrix — no append needed.", latest_date.date())
        return feature_df

    # Forward-fill all feature values from the last known row, then override Close & Sentiment
    new_row = feature_df.iloc[-1].copy()
    new_row["Close"] = float(raw["Close"].iloc[-1])
    new_row["Sentiment_Score"] = sentiment_score

    new_df = pd.DataFrame([new_row], index=[latest_date])
    updated_df = pd.concat([feature_df, new_df])

    logger.info("Appended live row for %s: date=%s, close=%.2f", ticker, latest_date.date(), new_row["Close"])
    return updated_df


def get_market_status_message() -> Optional[str]:
    """
    Return a warning message if the market is currently open, else None.

    Returns
    -------
    Optional[str]
        Warning string if market is open; None if closed.
    """
    if is_market_open():
        return (
            "Market is open — using previous close price. "
            "Intraday data not supported by this model."
        )
    return None
