"""
Data ingestion module: fetches and caches historical OHLCV data from Yahoo Finance.
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CACHE_MAX_AGE_DAYS = 1          # Re-fetch if cached file is older than this
FETCH_PERIOD = "5y"             # 5 years of daily data
FETCH_INTERVAL = "1d"          # Daily bars
MIN_TRADING_DAYS = 252          # Minimum acceptable history (1 year)


def _cache_path(ticker: str) -> str:
    """Return the full path to the local CSV cache file for a given ticker."""
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f"{ticker.upper()}_historical.csv")


def _is_cache_stale(path: str) -> bool:
    """Return True if the cached file is missing or older than CACHE_MAX_AGE_DAYS."""
    if not os.path.exists(path):
        return True
    modified_time = datetime.fromtimestamp(os.path.getmtime(path))
    age = datetime.now() - modified_time
    return age > timedelta(days=CACHE_MAX_AGE_DAYS)


def fetch_historical_data(ticker: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch 5 years of daily OHLCV data for a stock ticker from Yahoo Finance.
    Results are cached as CSV and reused until more than CACHE_MAX_AGE_DAYS old.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. 'AAPL').
    force_refresh : bool
        If True, ignore the cache and re-download from Yahoo Finance.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with DatetimeIndex and OHLCV columns.

    Raises
    ------
    ValueError
        If fewer than MIN_TRADING_DAYS rows are available after cleaning.
    RuntimeError
        If the download from Yahoo Finance fails.
    """
    ticker = ticker.upper().strip()
    cache_file = _cache_path(ticker)

    if not force_refresh and not _is_cache_stale(cache_file):
        logger.info("Loading %s data from cache: %s", ticker, cache_file)
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            logger.info("Cache hit: %d rows loaded for %s", len(df), ticker)
            return df
        except Exception as exc:
            logger.warning("Failed to read cache file (%s). Re-fetching. %s", cache_file, exc)

    logger.info("Fetching %s historical data from Yahoo Finance (period=%s, interval=%s)…",
                ticker, FETCH_PERIOD, FETCH_INTERVAL)
    try:
        raw = yf.download(
            ticker,
            period=FETCH_PERIOD,
            interval=FETCH_INTERVAL,
            auto_adjust=True,   # Adjusts for splits & dividends
            progress=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Yahoo Finance download failed for {ticker}: {exc}") from exc

    if raw.empty:
        raise RuntimeError(f"No data returned from Yahoo Finance for ticker '{ticker}'. "
                           "Check that the ticker symbol is valid.")

    df = _clean_data(raw, ticker)

    # Persist to local cache
    try:
        df.to_csv(cache_file)
        logger.info("Cached %d rows to %s", len(df), cache_file)
    except Exception as exc:
        logger.warning("Could not write cache file %s: %s", cache_file, exc)

    return df


def _clean_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Apply all data cleaning steps in order.

    Steps:
    1. Flatten MultiIndex columns that yfinance may return
    2. Drop rows where ALL OHLCV columns are NaN simultaneously
    3. Forward-fill then backward-fill remaining NaNs
    4. Flag single-day price moves > 25% as potential outliers (logged, not removed)
    5. Sort by date ascending, remove duplicate dates (keep last)
    6. Enforce minimum length

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame returned by yfinance.
    ticker : str
        Ticker symbol (used only for log messages).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    # Flatten MultiIndex columns that yfinance v0.2+ returns (e.g. ('Close','AAPL') → 'Close')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
    # Keep only columns that exist in this download
    existing_cols = [c for c in ohlcv_cols if c in df.columns]

    # Step 1: Drop rows where ALL OHLCV columns are simultaneously NaN
    before = len(df)
    df = df.dropna(subset=existing_cols, how="all")
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d fully-NaN rows for %s", dropped, ticker)

    # Step 2: Forward-fill then backward-fill to handle non-trading days
    df = df.ffill().bfill()

    # Step 3: Detect and log large single-day price moves (>25%) — keep but warn
    if "Close" in df.columns:
        pct_change = df["Close"].pct_change().abs()
        outlier_dates = df.index[pct_change > 0.25]
        for date in outlier_dates:
            logger.warning(
                "Outlier detected for %s on %s: %.1f%% single-day move. "
                "Row retained (may be an earnings or split event).",
                ticker, date.date(), pct_change.loc[date] * 100
            )

    # Step 4: Sort ascending, remove duplicate dates
    df = df.sort_index(ascending=True)
    dupes = df.index.duplicated(keep="last")
    if dupes.any():
        logger.warning("Removed %d duplicate date rows for %s", dupes.sum(), ticker)
        df = df[~dupes]

    # Step 5: Enforce minimum data length
    if len(df) < MIN_TRADING_DAYS:
        raise ValueError(
            f"Only {len(df)} trading days available for {ticker}. "
            f"Minimum required is {MIN_TRADING_DAYS} (1 year). "
            "Try a different ticker or a longer date range."
        )

    logger.info("Data cleaning complete for %s: %d rows retained", ticker, len(df))
    return df


def load_or_fetch(ticker: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Public convenience wrapper: returns cleaned historical data for the ticker.
    Alias for fetch_historical_data.
    """
    return fetch_historical_data(ticker, force_refresh=force_refresh)
