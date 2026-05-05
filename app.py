"""
AI Stock Predictor — single-page app with tabs.
Entry point: `streamlit run app.py`
"""
import glob
import logging
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

from src.data_ingestion import fetch_historical_data
from src.feature_engineering import compute_features, load_scalers
from src.live_data import append_live_row, fetch_live_price, get_market_status_message
from src.model import (
    ST_LOOKBACK, ST_HIDDEN_SIZE, ST_NUM_LAYERS, ST_DROPOUT, ST_PREDICTION_HORIZON,
    LT_LOOKBACK, LT_HIDDEN_SIZE, LT_NUM_LAYERS, LT_DROPOUT, LT_PREDICTION_HORIZON,
    NUM_FEATURES, count_parameters, build_short_term_model, build_long_term_model,
)
from src.ensemble import train_xgb, xgb_exists
from src.news_fetcher import fetch_news
from src.predict import forecast, get_test_sequences, predict_xgb_direction
from src.sentiment import score_article
from src.train import load_trained_models, load_metrics, models_exist, train_models
from src.ui import (
    PLOTLY_CONFIG, THEME, apply_chart_theme,
    footer, inject_css, metric_card, section_header, signal_card,
)
from src.utils import (
    build_forecast_dates, build_prediction_chart, get_trend_signal, setup_logging,
)

setup_logging(logging.WARNING)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

SHORT_TERM_DEFAULT = 30
LONG_TERM_DEFAULT_MONTHS = 3

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<p style="background:linear-gradient(135deg,#2563EB,#60A5FA);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
        'background-clip:text;font-size:1.15rem;font-weight:800;margin:0 0 1.25rem;">📊 AI Stock Predictor</p>',
        unsafe_allow_html=True,
    )

    ticker = st.text_input(
        "Stock Ticker",
        value=st.session_state.get("ticker", "AAPL"),
        placeholder="AAPL, MSFT, GOOGL…",
    ).upper().strip()
    st.session_state.ticker = ticker

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#94A3B8;font-size:0.72rem;font-weight:600;letter-spacing:0.07em;'
        'text-transform:uppercase;margin:0 0 0.6rem;">Dashboard</p>',
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Prediction Mode",
        options=["short_term", "long_term"],
        format_func=lambda x: "Short-Term (7–30 days)" if x == "short_term" else "Long-Term (1–12 months)",
        index=0,
    )

    if mode == "short_term":
        n_days = st.slider("Horizon (days)", 7, 30, SHORT_TERM_DEFAULT)
    else:
        n_months = st.slider("Horizon (months)", 1, 12, LONG_TERM_DEFAULT_MONTHS)
        n_days = n_months * 21

    st.markdown(
        '<p style="color:#94A3B8;font-size:0.75rem;font-weight:600;'
        'letter-spacing:0.06em;text-transform:uppercase;margin:0.75rem 0 0.5rem;">News Article</p>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Upload .txt or .pdf", type=["txt", "pdf"], label_visibility="collapsed")
    article_text  = st.text_area("Or paste article text", height=80, placeholder="Paste financial news here…")

    fetch_news_btn = st.button("Fetch Latest News", type="secondary", use_container_width=True)
    if fetch_news_btn:
        _api_key = os.getenv("NEWS_API_KEY", "")
        if not ticker:
            st.warning("Enter a ticker first.")
        elif not _api_key:
            st.error("NEWS_API_KEY not found in .env")
        else:
            with st.spinner("Fetching headlines…"):
                try:
                    _articles = fetch_news(ticker, _api_key)
                except RuntimeError as _exc:
                    st.error(str(_exc))
                    _articles = []
            if _articles:
                with st.spinner(f"Scoring {min(len(_articles), 10)} headlines with FinBERT…"):
                    _scored = []
                    for _art in _articles[:10]:
                        _text = f"{_art['title']}. {_art['description']}".strip()
                        _s, _l, _ = score_article(text=_text)
                        _scored.append({"title": _art["title"], "url": _art["url"],
                                        "score": _s, "label": _l})
                    _avg = float(np.mean([h["score"] for h in _scored]))
                    st.session_state.news_headlines       = _scored[:5]
                    st.session_state.news_sentiment_score = _avg
                    st.session_state.news_ticker          = ticker
                st.success(f"{len(_articles)} articles · avg {_avg:+.3f} — click Run Prediction to apply")
            else:
                st.info("No articles found for this ticker.")
                st.session_state.pop("news_headlines", None)
                st.session_state.pop("news_sentiment_score", None)
                st.session_state.pop("news_ticker", None)

    # ── Sentiment status indicator ────────────────────────────────────────────
    _has_manual  = uploaded_file is not None or bool(article_text.strip())
    _news_active = (
        st.session_state.get("news_ticker") == ticker
        and bool(st.session_state.get("news_headlines"))
    )

    if _has_manual:
        st.markdown(
            '<div style="background:#0A2010;border:1px solid #00C89644;border-radius:6px;'
            'padding:0.45rem 0.75rem;margin:0.5rem 0 0;">'
            '<p style="color:#00C896;font-size:0.74rem;font-weight:600;margin:0;">'
            '✓ Manual article loaded</p></div>',
            unsafe_allow_html=True,
        )
    elif _news_active:
        _disp_avg = st.session_state.news_sentiment_score
        _disp_n   = len(st.session_state.news_headlines)
        _nc = THEME["bullish"] if _disp_avg > 0.1 else (
              THEME["bearish"] if _disp_avg < -0.1 else THEME["muted"])
        _bg = "#0A2010" if _disp_avg > 0.1 else ("#2A0A0A" if _disp_avg < -0.1 else "#111118")
        st.markdown(
            f'<div style="background:{_bg};border:1px solid {_nc}44;border-radius:6px;'
            f'padding:0.45rem 0.75rem;margin:0.5rem 0 0;">'
            f'<p style="color:{_nc};font-size:0.74rem;font-weight:600;margin:0;">'
            f'✓ News fetched: {_disp_n} headlines, score: {_disp_avg:+.3f}</p></div>',
            unsafe_allow_html=True,
        )
        if st.button("Clear Sentiment", type="secondary", use_container_width=True):
            st.session_state.pop("news_headlines", None)
            st.session_state.pop("news_sentiment_score", None)
            st.session_state.pop("news_ticker", None)
            st.rerun()
    else:
        st.markdown(
            '<div style="background:#111118;border:1px solid #1E1E2E;border-radius:6px;'
            'padding:0.45rem 0.75rem;margin:0.5rem 0 0;">'
            '<p style="color:#94A3B8;font-size:0.74rem;font-weight:600;margin:0;">'
            '○ No sentiment — neutral</p></div>',
            unsafe_allow_html=True,
        )

    run_btn = st.button("Run Prediction", type="primary", use_container_width=True)

    if models_exist(ticker):
        if st.button("Retrain Model", type="secondary", use_container_width=True):
            for f in glob.glob(os.path.join("models", f"{ticker}_*.pt")):
                os.remove(f)
            for f in glob.glob(os.path.join("models", f"{ticker}_*.pkl")):
                os.remove(f)
            st.success("Cache cleared. Click Run Prediction to retrain.")
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#94A3B8;font-size:0.72rem;font-weight:600;letter-spacing:0.07em;'
        'text-transform:uppercase;margin:0 0 0.6rem;">Analysis</p>',
        unsafe_allow_html=True,
    )
    days_to_show = st.slider("History (days)", 60, 504, 180)
    analyse_btn  = st.button("Load Analysis", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _train_with_progress(ticker: str, feature_df) -> dict:
    from src.train import ST_MAX_EPOCHS, LT_MAX_EPOCHS
    status = st.empty()
    bar    = st.progress(0)

    def cb(name, epoch, max_ep, tl, vl):
        label = "Short-Term" if name == "short_term" else "Long-Term"
        total = ST_MAX_EPOCHS + LT_MAX_EPOCHS
        step  = epoch if name == "short_term" else ST_MAX_EPOCHS + epoch
        bar.progress(min(step / total, 1.0))
        status.text(f"Training {label}… Epoch {epoch}/{max_ep} | Train {tl:.5f} | Val {vl:.5f}")

    metrics = train_models(ticker, feature_df, progress_callback=cb)
    bar.progress(1.0)
    status.text("Training complete!")
    return metrics


def _buy_sell_hold(feature_df: pd.DataFrame):
    rsi  = feature_df["RSI_14"].iloc[-1]
    macd = feature_df["MACD"].iloc[-1]
    sig  = feature_df["MACD_Signal"].iloc[-1]
    c    = feature_df["Close"].iloc[-1]
    bbu  = feature_df["BB_Upper"].iloc[-1]
    bbl  = feature_df["BB_Lower"].iloc[-1]
    rng  = bbu - bbl

    votes = []
    if rsi < 35:   votes.append(1)
    elif rsi > 65: votes.append(-1)
    else:          votes.append(0)
    votes.append(1 if macd > sig else (-1 if macd < sig else 0))
    pos = (c - bbl) / rng if rng > 0 else 0.5
    if pos < 0.2:   votes.append(1)
    elif pos > 0.8: votes.append(-1)
    else:           votes.append(0)

    total = sum(votes)
    if total >= 2:    return "BUY",  THEME["bullish"], "▲"
    elif total <= -2: return "SELL", THEME["bearish"],  "▼"
    else:             return "HOLD", THEME["warn"],     "◆"


def _metric_row(items):
    cols = st.columns(len(items))
    for col, (label, value, color) in zip(cols, items):
        with col:
            metric_card(label, value, value_color=color)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_dash, tab_analysis, tab_sentiment, tab_model = st.tabs(
    ["📊 Dashboard", "📈 Analysis", "🧠 Sentiment", "🤖 Model Info"]
)

# ── Tab 1: Dashboard ────────────────────────────────────────────────────────
with tab_dash:
    st.markdown(
        '<h1 style="background:linear-gradient(135deg,#2563EB,#60A5FA);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
        'background-clip:text;font-size:2rem;font-weight:800;margin:0 0 0.25rem;">AI Stock Predictor</h1>'
        '<p style="color:#94A3B8;font-size:0.9rem;margin:0 0 1.5rem;">'
        'Dual-mode LSTM · FinBERT Sentiment · XGBoost Direction</p>',
        unsafe_allow_html=True,
    )

    if not run_btn:
        st.markdown(
            '<div style="background:#111118;border:1px solid #1E1E2E;border-radius:12px;'
            'padding:2.5rem;text-align:center;margin-top:3rem;">'
            '<p style="font-size:2.5rem;margin:0 0 1rem;">📊</p>'
            '<p style="color:#F1F5F9;font-size:1.1rem;font-weight:600;margin:0 0 0.5rem;">'
            'Enter a ticker and click Run Prediction</p>'
            '<p style="color:#94A3B8;font-size:0.85rem;margin:0;">'
            'Supports any Yahoo Finance ticker symbol</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    elif not ticker:
        st.error("Please enter a valid stock ticker.")
    else:
        # Step 1 — fetch data
        with st.spinner(f"Fetching {ticker} data…"):
            try:
                hist_df    = fetch_historical_data(ticker)
                feature_df = compute_features(hist_df)
                st.session_state.feature_df    = feature_df
                st.session_state.cached_ticker = ticker
            except Exception as exc:
                st.error(f"Data error: {exc}")
                feature_df = None

        if feature_df is not None:
            # Step 2 — live price banner
            try:
                live_price, pct_change, live_date = fetch_live_price(ticker)
                price_color = THEME["bullish"] if pct_change >= 0 else THEME["bearish"]
                sign = "+" if pct_change >= 0 else ""
                st.markdown(
                    f'<div style="background:#111118;border:1px solid {price_color}44;border-radius:8px;'
                    f'padding:0.9rem 1.4rem;margin-bottom:1rem;display:flex;align-items:center;gap:1.5rem;">'
                    f'<span style="color:#F1F5F9;font-size:1.3rem;font-weight:700;">{ticker.upper()}</span>'
                    f'<span style="color:{price_color};font-size:1.6rem;font-weight:800;">'
                    f'${live_price:,.2f}</span>'
                    f'<span style="background:{price_color}22;color:{price_color};'
                    f'padding:0.2rem 0.7rem;border-radius:20px;font-size:0.9rem;font-weight:600;">'
                    f'{sign}{pct_change:.2f}%</span>'
                    f'<span style="color:#94A3B8;font-size:0.8rem;margin-left:auto;">{live_date}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            except Exception as exc:
                logger.warning("Live price failed: %s", exc)
                live_price = float(feature_df["Close"].iloc[-1])

            market_msg = get_market_status_message()
            if market_msg:
                st.markdown(
                    f'<p style="color:#F59E0B;font-size:0.78rem;margin:-0.5rem 0 0.75rem;">'
                    f'⚠ {market_msg}</p>',
                    unsafe_allow_html=True,
                )

            # Step 3 — sentiment
            sentiment_score   = 0.0
            sentiment_label   = "Neutral"
            sentiment_snippet = ""
            raw_article_text  = pdf_bytes = None

            if uploaded_file is not None:
                if uploaded_file.name.endswith(".pdf"):
                    pdf_bytes = uploaded_file.read()
                else:
                    raw_article_text = uploaded_file.read().decode("utf-8", errors="ignore")
            elif article_text.strip():
                raw_article_text = article_text.strip()

            _run_news_active = (
                st.session_state.get("news_ticker") == ticker
                and bool(st.session_state.get("news_headlines"))
            )
            if raw_article_text or pdf_bytes:
                with st.spinner("Analysing sentiment…"):
                    try:
                        sentiment_score, sentiment_label, sentiment_snippet = score_article(
                            text=raw_article_text, pdf_bytes=pdf_bytes
                        )
                    except Exception as exc:
                        st.warning(f"Sentiment failed: {exc}")
            elif _run_news_active:
                sentiment_score   = st.session_state.news_sentiment_score
                sentiment_label   = (
                    "Positive" if sentiment_score > 0.1
                    else "Negative" if sentiment_score < -0.1
                    else "Neutral"
                )
                sentiment_snippet = f"Average of {len(st.session_state.news_headlines)} fetched headlines"
            # else: sentiment_score stays 0.0, label "Neutral" — no adjustment applied

            feature_df = feature_df.copy()
            feature_df["Sentiment_Score"] = sentiment_score
            try:
                feature_df = append_live_row(feature_df, ticker, sentiment_score)
            except Exception as exc:
                logger.warning("append_live_row: %s", exc)

            # Step 4 — train or load
            if not models_exist(ticker):
                st.info(f"No trained model found for {ticker}. Training now — this may take a few minutes.")
                try:
                    _train_with_progress(ticker, feature_df)
                    st.success("Training complete!")
                except Exception as exc:
                    st.error(f"Training failed: {exc}")
                    feature_df = None

        if feature_df is not None and models_exist(ticker):
            try:
                feature_scaler, close_scaler = load_scalers(ticker)
                st_model, lt_model           = load_trained_models(ticker)
            except Exception as exc:
                st.error(f"Failed to load model: {exc}")
                feature_scaler = None

            if feature_scaler is not None:
                # Step 5 — forecast
                model = st_model if mode == "short_term" else lt_model
                with st.spinner("Generating forecast…"):
                    try:
                        X_test, y_test = get_test_sequences(feature_df, feature_scaler, mode)
                        predicted_prices, confidence_std = forecast(
                            model=model, feature_df=feature_df,
                            feature_scaler=feature_scaler, close_scaler=close_scaler,
                            mode=mode, n_days=n_days, X_test=X_test, y_test=y_test,
                        )
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")
                        predicted_prices = None

                if predicted_prices is not None:
                    # Step 5a — smooth long-term predictions (7-day rolling average)
                    if mode == "long_term":
                        predicted_prices = (
                            pd.Series(predicted_prices)
                            .rolling(window=7, min_periods=1)
                            .mean()
                            .to_numpy()
                        )

                    # Step 5b — historical return calibration (long-term only)
                    calibration_note = None
                    if mode == "long_term":
                        hist_close    = feature_df["Close"]
                        last_close_c  = float(hist_close.iloc[-1])
                        first_close_c = float(hist_close.iloc[0])
                        n_hist        = len(hist_close)
                        n_days_lt     = len(predicted_prices)
                        annual_return = (last_close_c / first_close_c) ** (252 / n_hist) - 1
                        if abs(annual_return) > 0.001 and n_days_lt > 0:
                            implied_return = (
                                float(predicted_prices[-1]) / last_close_c
                            ) ** (252 / n_days_lt) - 1
                            if abs(implied_return) > 1.5 * abs(annual_return):
                                target_implied = np.sign(implied_return) * 1.5 * abs(annual_return)
                                target_final   = last_close_c * (1 + target_implied) ** (n_days_lt / 252)
                                deviation      = float(predicted_prices[-1]) - last_close_c
                                if abs(deviation) > 1e-6:
                                    scale          = (target_final - last_close_c) / deviation
                                    predicted_prices = last_close_c + (predicted_prices - last_close_c) * scale
                                calibration_note = "Historical return calibration applied"

                    # Step 5c — XGBoost direction (auto-train if file missing)
                    if not xgb_exists(ticker):
                        with st.spinner("Training XGBoost direction model…"):
                            try:
                                train_xgb(ticker, feature_df)
                            except Exception as exc:
                                logger.warning("XGBoost auto-train failed: %s", exc)

                    xgb_direction = xgb_confidence = None
                    try:
                        xgb_direction, xgb_confidence = predict_xgb_direction(ticker, feature_df)
                    except FileNotFoundError:
                        pass
                    except Exception as exc:
                        logger.warning("XGBoost prediction: %s", exc)

                    # Step 5d — sentiment post-processing adjustment
                    sentiment_note = None
                    if abs(sentiment_score) > 0.1:
                        adjustment_pct = sentiment_score * 0.03
                        # Linear ramp: day 1 → 20% of full shift, day 15 → 100%, day 15+ → 100%
                        ramp = np.minimum(
                            1.0,
                            0.20 + (0.80 / 14) * np.arange(len(predicted_prices)),
                        )
                        predicted_prices = predicted_prices * (1.0 + adjustment_pct * ramp)
                        sentiment_note = (
                            f"Sentiment adjustment applied: {adjustment_pct * 100:+.1f}% by day 15"
                        )

                    # Step 6 — render
                    signal, _ = get_trend_signal(live_price, float(predicted_prices[-1]))
                    is_bullish = signal == "Bullish"
                    future_dates = build_forecast_dates(feature_df.index[-1], len(predicted_prices))

                    col_chart, col_info = st.columns([3, 1])
                    with col_chart:
                        fig = build_prediction_chart(
                            ticker=ticker, mode=mode,
                            historical_prices=feature_df["Close"],
                            predicted_prices=predicted_prices,
                            future_dates=future_dates,
                            confidence_std=confidence_std,
                            is_bullish=is_bullish,
                        )
                        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                        if calibration_note:
                            st.caption(f"ℹ {calibration_note}")
                        if sentiment_note:
                            st.caption(f"ℹ {sentiment_note}")
                        signal_card(
                            label="LSTM Trend Signal",
                            signal=f"{signal} — ${predicted_prices[-1]:,.2f} in {len(predicted_prices)} days",
                            subtitle="Based on LSTM multi-step forecast",
                            is_bullish=is_bullish,
                        )

                    with col_info:
                        section_header("Sentiment")
                        s_color = THEME["bullish"] if sentiment_label == "Positive" else (
                            THEME["bearish"] if sentiment_label == "Negative" else THEME["muted"]
                        )
                        metric_card("FinBERT Score", f"{sentiment_score:+.3f}",
                                    subtitle=sentiment_label, value_color=s_color)
                        if sentiment_snippet:
                            st.caption(f'"{sentiment_snippet[:160]}…"')
                        else:
                            st.caption("No article — neutral sentiment assumed.")

                        st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)
                        section_header("Confidence Band")
                        metric_card("±1σ Estimate", f"${confidence_std:.2f}",
                                    subtitle="68% CI from test-set errors")

                        st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)
                        section_header("Direction Signal (XGBoost)")
                        if xgb_direction is not None:
                            signal_card(
                                label="XGBoost Classifier",
                                signal=f"{xgb_direction}",
                                subtitle=f"Confidence: {xgb_confidence:.1f}% · next-day direction",
                                is_bullish=(xgb_direction == "Bullish"),
                            )
                        else:
                            st.info("XGBoost signal unavailable — retrain to generate it.")

                    # Step 7 — metrics
                    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
                    section_header("Model Performance (Test Set)")
                    saved_metrics = load_metrics(ticker)
                    if saved_metrics:
                        mc1, mc2, mc3 = st.columns(3)
                        for col, key, label in [
                            (mc1, "short_term", "Short-Term LSTM"),
                            (mc2, "long_term",  "Long-Term LSTM"),
                            (mc3, None,         "XGBoost"),
                        ]:
                            with col:
                                if key:
                                    m = saved_metrics.get(key, {})
                                    if "error" in m:
                                        st.info(m["error"])
                                    elif m:
                                        dir_label = "7-Day Directional Accuracy"
                                        metric_card(f"{label} — RMSE", f"${m.get('rmse', 0):.2f}")
                                        st.markdown("<div style='margin-top:0.5rem;'></div>",
                                                    unsafe_allow_html=True)
                                        metric_card(dir_label,
                                                    f"{m.get('directional_accuracy', 0):.1f}%")
                                else:
                                    xgb_acc = saved_metrics.get("xgb_directional_accuracy")
                                    if xgb_acc is not None:
                                        metric_card("XGBoost — Dir. Accuracy", f"{xgb_acc:.1f}%")
                    else:
                        st.info("Metrics will appear here after the first training run.")

# ── Tab 2: Analysis ──────────────────────────────────────────────────────────
with tab_analysis:
    st.markdown(
        '<h1 style="color:#F1F5F9;font-size:1.8rem;font-weight:800;margin:0 0 0.25rem;">'
        'Technical Analysis</h1>'
        '<p style="color:#94A3B8;font-size:0.85rem;margin:0 0 1.5rem;">'
        'RSI · MACD · Bollinger Bands · Volume · Sector comparison</p>',
        unsafe_allow_html=True,
    )

    cache_hit = (
        not analyse_btn
        and st.session_state.get("cached_ticker") == ticker
        and "feature_df" in st.session_state
    )

    if not analyse_btn and not cache_hit:
        st.markdown(
            '<div style="background:#111118;border:1px solid #1E1E2E;border-radius:12px;'
            'padding:2.5rem;text-align:center;margin-top:3rem;">'
            '<p style="font-size:2.5rem;margin:0 0 1rem;">📈</p>'
            '<p style="color:#F1F5F9;font-size:1.1rem;font-weight:600;margin:0 0 0.5rem;">'
            'Select a ticker and click Load Analysis</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        if not cache_hit:
            with st.spinner(f"Fetching {ticker} data…"):
                try:
                    hist_df    = fetch_historical_data(ticker)
                    a_feature_df = compute_features(hist_df)
                    st.session_state.feature_df    = a_feature_df
                    st.session_state.cached_ticker = ticker
                except Exception as exc:
                    st.error(f"Data error: {exc}")
                    a_feature_df = None
        else:
            a_feature_df = st.session_state.feature_df

        if a_feature_df is not None:
            df = a_feature_df.iloc[-days_to_show:]

            yr_slice  = a_feature_df["Close"].iloc[-min(252, len(a_feature_df)):]
            high_52w  = yr_slice.max()
            low_52w   = yr_slice.min()
            cur_price = a_feature_df["Close"].iloc[-1]

            bsh_label, bsh_color, bsh_arrow = _buy_sell_hold(a_feature_df)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                metric_card("Current Price", f"${cur_price:,.2f}")
            with c2:
                metric_card("52W High", f"${high_52w:,.2f}", value_color=THEME["bullish"])
            with c3:
                metric_card("52W Low",  f"${low_52w:,.2f}",  value_color=THEME["bearish"])
            with c4:
                metric_card("Combined Signal", f"{bsh_arrow} {bsh_label}",
                            value_color=bsh_color, subtitle="RSI + MACD + Bollinger")

            st.markdown("<div style='margin-top:1.25rem;'></div>", unsafe_allow_html=True)

            # Bollinger Bands
            section_header("Price & Bollinger Bands")
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(
                x=list(df.index) + list(df.index[::-1]),
                y=list(df["BB_Upper"]) + list(df["BB_Lower"][::-1]),
                fill="toself", fillcolor="rgba(37,99,235,0.08)",
                line=dict(color="rgba(255,255,255,0)"), name="Bollinger Band", hoverinfo="skip",
            ))
            fig_bb.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="Upper",
                line=dict(color="#2563EB", width=1, dash="dot")))
            fig_bb.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="Lower",
                line=dict(color="#2563EB", width=1, dash="dot")))
            fig_bb.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], name="SMA 20",
                line=dict(color="#94A3B8", width=1, dash="dash")))
            fig_bb.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                line=dict(color="#F1F5F9", width=2)))
            apply_chart_theme(fig_bb, f"{ticker} — Bollinger Bands")
            fig_bb.update_layout(yaxis_title="Price (USD)")
            st.plotly_chart(fig_bb, use_container_width=True, config=PLOTLY_CONFIG)

            # RSI
            section_header("RSI (14)")
            fig_rsi = go.Figure()
            fig_rsi.add_hrect(y0=70, y1=100, fillcolor="rgba(255,68,68,0.06)", line_width=0)
            fig_rsi.add_hrect(y0=0,  y1=30,  fillcolor="rgba(0,200,150,0.06)",  line_width=0)
            fig_rsi.add_hline(y=70, line_dash="dash", line_color=THEME["bearish"],
                              annotation_text="Overbought 70",
                              annotation_font_color=THEME["bearish"],
                              annotation_position="top left")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color=THEME["bullish"],
                              annotation_text="Oversold 30",
                              annotation_font_color=THEME["bullish"],
                              annotation_position="bottom left")
            fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI_14"], name="RSI 14",
                line=dict(color="#F59E0B", width=2)))
            apply_chart_theme(fig_rsi, "RSI (14-day)")
            fig_rsi.update_layout(yaxis=dict(range=[0, 100]), yaxis_title="RSI")
            st.plotly_chart(fig_rsi, use_container_width=True, config=PLOTLY_CONFIG)

            # MACD
            section_header("MACD")
            hist_colors = [
                THEME["bullish"] if v >= 0 else THEME["bearish"]
                for v in df["MACD_Histogram"]
            ]
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Bar(x=df.index, y=df["MACD_Histogram"], name="Histogram",
                marker_color=hist_colors, opacity=0.7))
            fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                line=dict(color="#2563EB", width=2)))
            fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal",
                line=dict(color="#F59E0B", width=1.5, dash="dot")))
            apply_chart_theme(fig_macd, "MACD")
            fig_macd.update_layout(yaxis_title="MACD")
            st.plotly_chart(fig_macd, use_container_width=True, config=PLOTLY_CONFIG)

            # Volume
            section_header("Volume")
            vol_colors = [
                THEME["bullish"] if df["Close"].iloc[i] >= df["Close"].iloc[i - 1]
                else THEME["bearish"]
                for i in range(len(df))
            ]
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                marker_color=vol_colors, opacity=0.8))
            fig_vol.add_trace(go.Scatter(x=df.index, y=df["Volume_SMA_20"], name="Vol SMA 20",
                line=dict(color="#94A3B8", width=1.5, dash="dash")))
            apply_chart_theme(fig_vol, "Volume")
            fig_vol.update_layout(yaxis_title="Volume")
            st.plotly_chart(fig_vol, use_container_width=True, config=PLOTLY_CONFIG)

            # Performance vs SPY
            section_header("Performance vs S&P 500")
            try:
                start_date = df.index[0]
                end_date   = df.index[-1]
                sd = start_date.tz_localize(None) if start_date.tzinfo else start_date
                ed = end_date.tz_localize(None)   if end_date.tzinfo   else end_date

                spy_raw = yf.download(
                    "SPY", start=sd, end=ed + pd.Timedelta(days=1),
                    auto_adjust=True, progress=False, multi_level_index=False,
                )
                if not spy_raw.empty:
                    spy_close  = spy_raw["Close"].reindex(df.index, method="ffill").dropna()
                    common_idx = df.index.intersection(spy_close.index)
                    stock_norm = (df["Close"].loc[common_idx] / df["Close"].loc[common_idx].iloc[0]) * 100
                    spy_norm   = (spy_close.loc[common_idx] / spy_close.loc[common_idx].iloc[0]) * 100

                    stock_ret = stock_norm.iloc[-1] - 100
                    spy_ret   = spy_norm.iloc[-1] - 100

                    fig_perf = go.Figure()
                    fig_perf.add_trace(go.Scatter(x=common_idx, y=stock_norm, name=ticker,
                        line=dict(color=THEME["accent"], width=2)))
                    fig_perf.add_trace(go.Scatter(x=common_idx, y=spy_norm, name="SPY",
                        line=dict(color=THEME["muted"], width=1.5, dash="dash")))
                    apply_chart_theme(fig_perf, f"{ticker} vs SPY — normalised to 100")
                    fig_perf.update_layout(yaxis_title="Indexed Value (start=100)")
                    st.plotly_chart(fig_perf, use_container_width=True, config=PLOTLY_CONFIG)

                    ret_col1, ret_col2 = st.columns(2)
                    with ret_col1:
                        rc = THEME["bullish"] if stock_ret >= 0 else THEME["bearish"]
                        metric_card(f"{ticker} Return (period)", f"{stock_ret:+.1f}%", value_color=rc)
                    with ret_col2:
                        rc = THEME["bullish"] if spy_ret >= 0 else THEME["bearish"]
                        metric_card("SPY Return (period)", f"{spy_ret:+.1f}%", value_color=rc)
                else:
                    st.info("SPY data unavailable.")
            except Exception as exc:
                st.warning(f"Could not fetch SPY data: {exc}")

# ── Tab 3: Sentiment ─────────────────────────────────────────────────────────
with tab_sentiment:
    st.markdown(
        '<h1 style="color:#F1F5F9;font-size:1.8rem;font-weight:800;margin:0 0 0.25rem;">'
        'Sentiment Analysis</h1>'
        '<p style="color:#94A3B8;font-size:0.85rem;margin:0 0 1.5rem;">'
        'FinBERT financial sentiment model — scores news from −1 (negative) to +1 (positive)</p>',
        unsafe_allow_html=True,
    )

    # Fetched headlines section
    _tab_news = st.session_state.get("news_headlines", [])
    _tab_news_ticker = st.session_state.get("news_ticker", "")
    if _tab_news and _tab_news_ticker:
        section_header(f"Latest Headlines — {_tab_news_ticker}", "Fetched via NewsAPI · scored with FinBERT")
        for _h in _tab_news:
            _hc = THEME["bullish"] if _h["label"] == "Positive" else (
                  THEME["bearish"] if _h["label"] == "Negative" else THEME["muted"])
            _icon = "▲" if _h["label"] == "Positive" else ("▼" if _h["label"] == "Negative" else "◆")
            st.markdown(
                f'<div style="background:#111118;border:1px solid #1E1E2E;border-radius:8px;'
                f'padding:0.75rem 1rem;margin-bottom:0.5rem;display:flex;align-items:center;gap:1rem;">'
                f'<span style="color:{_hc};font-size:0.9rem;font-weight:700;min-width:3.5rem;">'
                f'{_icon} {_h["score"]:+.3f}</span>'
                f'<span style="color:#F1F5F9;font-size:0.85rem;line-height:1.4;">{_h["title"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        _avg_score = st.session_state.get("news_sentiment_score", 0.0)
        _avg_color = THEME["bullish"] if _avg_score > 0.1 else (THEME["bearish"] if _avg_score < -0.1 else THEME["muted"])
        st.markdown(
            f'<p style="color:{_avg_color};font-size:0.82rem;font-weight:600;margin:0.25rem 0 1.5rem;">'
            f'Average sentiment: {_avg_score:+.4f} — used for prediction adjustment</p>',
            unsafe_allow_html=True,
        )
        st.markdown("<hr>", unsafe_allow_html=True)

    col_upload, col_paste = st.columns(2)
    with col_upload:
        section_header("Upload Article")
        sent_file = st.file_uploader(
            "Drag & drop or browse",
            type=["txt", "pdf"],
            label_visibility="collapsed",
            key="sent_uploader",
        )
    with col_paste:
        section_header("Paste Text")
        sent_text = st.text_area(
            "Article content",
            height=180,
            placeholder="Paste financial news article text here…",
            label_visibility="collapsed",
            key="sent_textarea",
        )

    analyse_sent_btn = st.button("Analyse Sentiment", type="primary", key="analyse_sent")

    s_raw_text = s_pdf_bytes = None
    if sent_file is not None:
        if sent_file.name.endswith(".pdf"):
            s_pdf_bytes = sent_file.read()
        else:
            s_raw_text = sent_file.read().decode("utf-8", errors="ignore")
    elif sent_text.strip():
        s_raw_text = sent_text.strip()

    has_input = s_raw_text is not None or s_pdf_bytes is not None

    if analyse_sent_btn and not has_input:
        st.warning("Please upload a file or paste article text first.")
    elif analyse_sent_btn and has_input:
        with st.spinner("Running FinBERT…"):
            try:
                score, label, snippet = score_article(text=s_raw_text, pdf_bytes=s_pdf_bytes)
            except Exception as exc:
                st.error(f"Sentiment analysis failed: {exc}")
                score = label = snippet = None

        if score is not None:
            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
            if label == "Positive":
                s_color = THEME["bullish"];  icon = "😊"
            elif label == "Negative":
                s_color = THEME["bearish"];  icon = "😟"
            else:
                s_color = THEME["muted"];    icon = "😐"

            r1, r2, r3 = st.columns([1, 1, 2])
            with r1:
                metric_card("FinBERT Score", f"{score:+.4f}", value_color=s_color,
                            subtitle="Range: −1.0 (negative) → +1.0 (positive)")
            with r2:
                metric_card("Sentiment Label", f"{icon} {label}", value_color=s_color)
            with r3:
                if snippet:
                    st.markdown(
                        f'<div style="background:#111118;border:1px solid #1E1E2E;border-radius:8px;'
                        f'padding:1rem 1.25rem;height:100%;">'
                        f'<p style="color:#94A3B8;font-size:0.72rem;font-weight:600;letter-spacing:0.07em;'
                        f'text-transform:uppercase;margin:0 0 0.5rem;">Article Snippet</p>'
                        f'<p style="color:#F1F5F9;font-size:0.88rem;line-height:1.6;margin:0;">'
                        f'"{snippet[:350]}{"…" if len(snippet) > 350 else ""}"</p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
            pct = int((score + 1) / 2 * 100)
            st.markdown(
                f'<div style="background:#111118;border:1px solid #1E1E2E;border-radius:8px;padding:1rem 1.25rem;">'
                f'<p style="color:#94A3B8;font-size:0.72rem;font-weight:600;letter-spacing:0.07em;'
                f'text-transform:uppercase;margin:0 0 0.6rem;">Sentiment Scale</p>'
                f'<div style="display:flex;align-items:center;gap:0.75rem;">'
                f'<span style="color:#FF4444;font-size:0.8rem;">Negative</span>'
                f'<div style="flex:1;background:#1E1E2E;border-radius:4px;height:8px;overflow:hidden;">'
                f'<div style="background:{s_color};width:{pct}%;height:100%;border-radius:4px;'
                f'transition:width 0.5s ease;"></div>'
                f'</div>'
                f'<span style="color:#00C896;font-size:0.8rem;">Positive</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div style="background:#111118;border:1px solid #1E1E2E;border-radius:12px;'
            'padding:2.5rem;text-align:center;margin-top:2rem;">'
            '<p style="font-size:2.5rem;margin:0 0 1rem;">🧠</p>'
            '<p style="color:#F1F5F9;font-size:1.05rem;font-weight:600;margin:0 0 0.5rem;">'
            'Upload or paste an article, then click Analyse Sentiment</p>'
            '<p style="color:#94A3B8;font-size:0.83rem;margin:0;">'
            'Supports .txt files, .pdf files, or plain pasted text</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:1.25rem;'></div>", unsafe_allow_html=True)
    with st.expander("How does FinBERT work?"):
        st.markdown(
            """
**FinBERT** is a BERT-based language model pre-trained on a large corpus of financial text
(Reuters, Bloomberg, financial filings). It was fine-tuned on the Financial PhraseBank dataset
to classify financial sentences into three classes: **Positive**, **Neutral**, and **Negative**.

**How scores are computed:**
1. The article is split into overlapping 512-token chunks (FinBERT's context limit).
2. Each chunk is scored independently, producing probabilities for all three classes.
3. The final score is a weighted average across chunks, mapped to the range **−1 to +1**:
   - Score = P(positive) − P(negative)
4. A score near +1 indicates strongly bullish language; near −1 indicates bearish language.

**Limitations:**
- FinBERT reads language, not facts. It won't catch sarcasm or complex financial arguments.
- Short snippets (< 50 words) produce less reliable scores.
- The model scores *current* article text — future sentiment is unknowable.
            """,
            unsafe_allow_html=False,
        )

# ── Tab 4: Model Info ────────────────────────────────────────────────────────
with tab_model:
    st.markdown(
        '<h1 style="color:#F1F5F9;font-size:1.8rem;font-weight:800;margin:0 0 0.25rem;">Model Info</h1>'
        '<p style="color:#94A3B8;font-size:0.85rem;margin:0 0 1.5rem;">'
        'Architecture, training configuration, and test-set performance metrics</p>',
        unsafe_allow_html=True,
    )

    section_header(f"Test-Set Performance — {ticker}")
    metrics = load_metrics(ticker) if models_exist(ticker) else None

    if metrics:
        st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
        for model_key, model_label in [("short_term", "Short-Term LSTM"), ("long_term", "Long-Term LSTM")]:
            m = metrics.get(model_key, {})
            if not m or "error" in m:
                st.info(f"{model_label}: {m.get('error', 'No metrics available.')}")
                continue
            st.markdown(
                f'<p style="color:#94A3B8;font-size:0.78rem;font-weight:600;letter-spacing:0.07em;'
                f'text-transform:uppercase;margin:1rem 0 0.6rem;">{model_label}</p>',
                unsafe_allow_html=True,
            )
            dir_label = "7-Day Directional Accuracy"
            _metric_row([
                ("RMSE",      f"${m.get('rmse', 0):.2f}",  THEME["text"]),
                ("MAE",       f"${m.get('mae',  0):.2f}",  THEME["text"]),
                ("MAPE",      f"{m.get('mape',  0):.1f}%", THEME["text"]),
                (dir_label,   f"{m.get('directional_accuracy', 0):.1f}%",
                 THEME["bullish"] if m.get("directional_accuracy", 0) > 52 else THEME["muted"]),
            ])

        xgb_acc = metrics.get("xgb_directional_accuracy")
        if xgb_acc is not None:
            st.markdown(
                '<p style="color:#94A3B8;font-size:0.78rem;font-weight:600;letter-spacing:0.07em;'
                'text-transform:uppercase;margin:1rem 0 0.6rem;">XGBoost Classifier</p>',
                unsafe_allow_html=True,
            )
            _metric_row([
                ("Directional Accuracy", f"{xgb_acc:.1f}%",
                 THEME["bullish"] if xgb_acc > 52 else THEME["muted"]),
                ("Features", "22 (20 technical + 2 sector momentum)", THEME["text"]),
                ("Target", "Next-day up/down", THEME["text"]),
            ])
    else:
        st.info(f"No saved metrics for {ticker}. Run a prediction on the Dashboard to train the model.")

    st.markdown("<hr>", unsafe_allow_html=True)
    section_header("Model Architecture")
    col_st, col_lt = st.columns(2)

    with col_st:
        st.markdown(
            '<div style="background:#111118;border:1px solid #1E1E2E;border-radius:8px;padding:1.25rem;">'
            '<p style="color:#2563EB;font-size:0.78rem;font-weight:700;letter-spacing:0.08em;'
            'text-transform:uppercase;margin:0 0 0.75rem;">Short-Term LSTM</p>'
            '<p style="color:#F1F5F9;font-size:0.88rem;line-height:1.7;margin:0 0 0.75rem;">'
            'Predicts the <strong>next 30 trading days</strong> in a single forward pass. '
            'Designed for swing-trade and earnings-window horizons.</p>'
            f'<table style="width:100%;border-collapse:collapse;font-size:0.82rem;">'
            f'<tr><td style="color:#94A3B8;padding:0.25rem 0;">Lookback window</td>'
            f'<td style="color:#F1F5F9;text-align:right;">{ST_LOOKBACK} days (3 months)</td></tr>'
            f'<tr><td style="color:#94A3B8;padding:0.25rem 0;">Forecast horizon</td>'
            f'<td style="color:#F1F5F9;text-align:right;">{ST_PREDICTION_HORIZON} days</td></tr>'
            f'<tr><td style="color:#94A3B8;padding:0.25rem 0;">LSTM layers</td>'
            f'<td style="color:#F1F5F9;text-align:right;">{ST_NUM_LAYERS}</td></tr>'
            f'<tr><td style="color:#94A3B8;padding:0.25rem 0;">Hidden units</td>'
            f'<td style="color:#F1F5F9;text-align:right;">{ST_HIDDEN_SIZE}</td></tr>'
            f'<tr><td style="color:#94A3B8;padding:0.25rem 0;">Dropout</td>'
            f'<td style="color:#F1F5F9;text-align:right;">{ST_DROPOUT}</td></tr>'
            f'<tr><td style="color:#94A3B8;padding:0.25rem 0;">Head</td>'
            f'<td style="color:#F1F5F9;text-align:right;">'
            f'Linear({ST_HIDDEN_SIZE}→64) → ReLU → Linear(64→{ST_PREDICTION_HORIZON})</td></tr>'
            f'<tr><td style="color:#94A3B8;padding:0.25rem 0;">Parameters</td>'
            f'<td style="color:#F1F5F9;text-align:right;">'
            f'{count_parameters(build_short_term_model()):,}</td></tr>'
            f'</table></div>',
            unsafe_allow_html=True,
        )

    with col_lt:
        st.markdown(
            '<div style="background:#111118;border:1px solid #1E1E2E;border-radius:8px;padding:1.25rem;">'
            '<p style="color:#2563EB;font-size:0.78rem;font-weight:700;letter-spacing:0.08em;'
            'text-transform:uppercase;margin:0 0 0.75rem;">Long-Term LSTM</p>'
            '<p style="color:#F1F5F9;font-size:0.88rem;line-height:1.7;margin:0 0 0.75rem;">'
            'Predicts up to <strong>252 trading days (1 year)</strong> ahead. Uses a larger lookback '
            'and deeper network to capture macro trends.</p>'
            f'<table style="width:100%;border-collapse:collapse;font-size:0.82rem;">'
            f'<tr><td style="color:#94A3B8;padding:0.25rem 0;">Lookback window</td>'
            f'<td style="color:#F1F5F9;text-align:right;">{LT_LOOKBACK} days (1 year)</td></tr>'
            f'<tr><td style="color:#94A3B8;padding:0.25rem 0;">Forecast horizon</td>'
            f'<td style="color:#F1F5F9;text-align:right;">{LT_PREDICTION_HORIZON} days</td></tr>'
            f'<tr><td style="color:#94A3B8;padding:0.25rem 0;">LSTM layers</td>'
            f'<td style="color:#F1F5F9;text-align:right;">{LT_NUM_LAYERS}</td></tr>'
            f'<tr><td style="color:#94A3B8;padding:0.25rem 0;">Hidden units</td>'
            f'<td style="color:#F1F5F9;text-align:right;">{LT_HIDDEN_SIZE}</td></tr>'
            f'<tr><td style="color:#94A3B8;padding:0.25rem 0;">Dropout</td>'
            f'<td style="color:#F1F5F9;text-align:right;">{LT_DROPOUT}</td></tr>'
            f'<tr><td style="color:#94A3B8;padding:0.25rem 0;">Head</td>'
            f'<td style="color:#F1F5F9;text-align:right;">'
            f'Linear({LT_HIDDEN_SIZE}→128) → ReLU → Dropout(0.2) → Linear(128→{LT_PREDICTION_HORIZON})</td></tr>'
            f'<tr><td style="color:#94A3B8;padding:0.25rem 0;">Parameters</td>'
            f'<td style="color:#F1F5F9;text-align:right;">'
            f'{count_parameters(build_long_term_model()):,}</td></tr>'
            f'</table></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div style="background:#111118;border:1px solid #1E1E2E;border-radius:8px;padding:1.25rem;">'
        '<p style="color:#2563EB;font-size:0.78rem;font-weight:700;letter-spacing:0.08em;'
        'text-transform:uppercase;margin:0 0 0.75rem;">XGBoost Direction Classifier</p>'
        '<p style="color:#F1F5F9;font-size:0.88rem;line-height:1.7;margin:0 0 0.75rem;">'
        'A gradient-boosted tree ensemble trained to predict whether the next-day close will '
        'be higher or lower than today. Complements the LSTM price forecast by providing '
        'an independent binary direction signal.</p>'
        '<table style="width:100%;border-collapse:collapse;font-size:0.82rem;">'
        '<tr><td style="color:#94A3B8;padding:0.25rem 0;">Algorithm</td>'
        '<td style="color:#F1F5F9;text-align:right;">XGBoost (gradient-boosted trees)</td></tr>'
        '<tr><td style="color:#94A3B8;padding:0.25rem 0;">Input features</td>'
        '<td style="color:#F1F5F9;text-align:right;">20 technical indicators + 2 sector ETF momentum</td></tr>'
        '<tr><td style="color:#94A3B8;padding:0.25rem 0;">Target</td>'
        '<td style="color:#F1F5F9;text-align:right;">Binary: up (1) or down (0) next day</td></tr>'
        '<tr><td style="color:#94A3B8;padding:0.25rem 0;">Max trees</td>'
        '<td style="color:#F1F5F9;text-align:right;">500 (early stopping on validation loss)</td></tr>'
        '<tr><td style="color:#94A3B8;padding:0.25rem 0;">Sector ETF</td>'
        '<td style="color:#F1F5F9;text-align:right;">XLK / XLF / XLC / XLE / SPY (by sector)</td></tr>'
        '</table></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    section_header("Training Configuration")
    st.markdown(
        '<div style="background:#111118;border:1px solid #1E1E2E;border-radius:8px;padding:1.25rem;">'
        '<table style="width:100%;border-collapse:collapse;font-size:0.84rem;">'
        '<tr style="border-bottom:1px solid #1E1E2E;">'
        '  <th style="color:#94A3B8;text-align:left;padding:0.4rem 0;font-weight:600;">Setting</th>'
        '  <th style="color:#94A3B8;text-align:right;padding:0.4rem 0;font-weight:600;">Short-Term</th>'
        '  <th style="color:#94A3B8;text-align:right;padding:0.4rem 0;font-weight:600;">Long-Term</th>'
        '</tr>'
        '<tr><td style="color:#94A3B8;padding:0.35rem 0;">Optimizer</td>'
        '  <td style="color:#F1F5F9;text-align:right;">Adam (lr=0.001)</td>'
        '  <td style="color:#F1F5F9;text-align:right;">Adam (lr=0.0005)</td></tr>'
        '<tr><td style="color:#94A3B8;padding:0.35rem 0;">Batch size</td>'
        '  <td style="color:#F1F5F9;text-align:right;">32</td>'
        '  <td style="color:#F1F5F9;text-align:right;">16</td></tr>'
        '<tr><td style="color:#94A3B8;padding:0.35rem 0;">Max epochs</td>'
        '  <td style="color:#F1F5F9;text-align:right;">100</td>'
        '  <td style="color:#F1F5F9;text-align:right;">150</td></tr>'
        '<tr><td style="color:#94A3B8;padding:0.35rem 0;">Early stopping patience</td>'
        '  <td style="color:#F1F5F9;text-align:right;">15 epochs</td>'
        '  <td style="color:#F1F5F9;text-align:right;">20 epochs</td></tr>'
        '<tr><td style="color:#94A3B8;padding:0.35rem 0;">LR scheduler</td>'
        '  <td style="color:#F1F5F9;text-align:right;" colspan="2">ReduceLROnPlateau (×0.5 after 5/7 epochs)</td></tr>'
        '<tr><td style="color:#94A3B8;padding:0.35rem 0;">Gradient clip (L2)</td>'
        '  <td style="color:#F1F5F9;text-align:right;" colspan="2">1.0</td></tr>'
        '<tr><td style="color:#94A3B8;padding:0.35rem 0;">Loss function</td>'
        '  <td style="color:#F1F5F9;text-align:right;" colspan="2">Mean Squared Error</td></tr>'
        '<tr><td style="color:#94A3B8;padding:0.35rem 0;">Data split</td>'
        '  <td style="color:#F1F5F9;text-align:right;" colspan="2">70% train / 15% val / 15% test (chronological)</td></tr>'
        '<tr><td style="color:#94A3B8;padding:0.35rem 0;">Input features</td>'
        f' <td style="color:#F1F5F9;text-align:right;" colspan="2">{NUM_FEATURES} (Close, returns, SMA/EMA, MACD, RSI, Bollinger, ATR, Volume, Sentiment)</td></tr>'
        '</table></div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
footer()
