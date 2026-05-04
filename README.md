# AI Stock Predictor
### Dual-Mode LSTM Neural Network with FinBERT Sentiment Analysis & XGBoost Direction Signal

**IBA Karachi Campus**
**Course: AI Project**

| Team Member | ERP ID |
|---|---|
| Muhammad Hamza Uzair | 30544 |
| Qurat ul Ain Siddique | 30565 |
| Ebad Ur Rehman Shaikh | 30523 |

## Overview
An AI-powered stock prediction system that combines deep learning, financial sentiment analysis, and ensemble methods to forecast stock prices and trend direction.

## Features
- Dual-mode LSTM: Short-term (30-day) and Long-term (252-day) seq2seq forecasting
- FinBERT sentiment analysis on news articles with prediction adjustment
- XGBoost direction classifier (Bullish/Bearish signal)
- Technical analysis dashboard: RSI, MACD, Bollinger Bands, Volume, S&P500 comparison
- Live price fetching via Yahoo Finance
- Historical return calibration for realistic long-term forecasts
- Pre-trained models for AAPL and MSFT (loads instantly)

## Tech Stack
- PyTorch (LSTM models)
- XGBoost (direction classifier)
- HuggingFace Transformers (FinBERT)
- Streamlit + Plotly (UI)
- Yahoo Finance / yfinance (data)

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Usage
1. Enter a stock ticker (AAPL or MSFT for instant results)
2. Select Short-Term or Long-Term prediction mode
3. Optionally upload or paste a news article for sentiment analysis
4. Click **Run Prediction**
5. Explore the **Analysis** tab for technical indicators

## Model Performance (AAPL)

| Model | RMSE | MAPE | 7-Day Directional Accuracy |
|---|---|---|---|
| Short-Term LSTM | $12.49 | 4.0% | 56.9% |
| Long-Term LSTM | $43.18 | 16.8% | 56.2% |

## Disclaimer
Not financial advice. For educational and research purposes only.
