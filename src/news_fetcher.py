"""Fetch recent news headlines for a ticker via NewsAPI, with per-day in-memory caching."""
import logging
from datetime import date, timedelta
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)

TICKER_TO_COMPANY: Dict[str, str] = {
    "AAPL":  "Apple",
    "MSFT":  "Microsoft",
    "GOOGL": "Google",
    "GOOG":  "Google",
    "TSLA":  "Tesla",
    "AMZN":  "Amazon",
    "META":  "Meta",
    "NVDA":  "NVIDIA",
    "NFLX":  "Netflix",
    "JPM":   "JPMorgan",
    "V":     "Visa",
    "AMD":   "AMD",
    "INTC":  "Intel",
    "BABA":  "Alibaba",
    "UBER":  "Uber",
    "DIS":   "Disney",
    "PYPL":  "PayPal",
    "ADBE":  "Adobe",
    "CRM":   "Salesforce",
    "QCOM":  "Qualcomm",
}

# {ticker: {"date": date, "articles": [...]}}
_cache: Dict[str, dict] = {}


def fetch_news(ticker: str, api_key: str, n_days: int = 7) -> List[Dict[str, str]]:
    """
    Fetch recent news articles for a ticker from NewsAPI.

    Results are cached per ticker per calendar day to stay within the free-tier
    limit of 100 requests/day.

    Parameters
    ----------
    ticker : str
    api_key : str  NewsAPI key
    n_days : int   How many past days of articles to request (default 7)

    Returns
    -------
    List[dict]  Each item has keys: title, description, url
    """
    today = date.today()
    cached = _cache.get(ticker.upper())
    if cached and cached["date"] == today:
        logger.info("News cache hit for %s", ticker)
        return cached["articles"]

    company = TICKER_TO_COMPANY.get(ticker.upper(), ticker)
    from_date = (today - timedelta(days=n_days)).isoformat()

    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q":        f'"{company}" stock',
                "from":     from_date,
                "sortBy":   "relevancy",
                "language": "en",
                "pageSize": 20,
                "apiKey":   api_key,
            },
            timeout=10,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Network error fetching news: {exc}") from exc

    if resp.status_code == 429:
        raise RuntimeError("NewsAPI rate limit reached (100 requests/day on free tier)")
    if resp.status_code != 200:
        msg = resp.json().get("message", "unknown error")
        raise RuntimeError(f"NewsAPI returned {resp.status_code}: {msg}")

    articles = [
        {
            "title":       a.get("title", "").strip(),
            "description": (a.get("description") or "").strip(),
            "url":         a.get("url", ""),
        }
        for a in resp.json().get("articles", [])
        if a.get("title") and "[Removed]" not in a.get("title", "")
    ]

    _cache[ticker.upper()] = {"date": today, "articles": articles}
    logger.info("Fetched %d articles for %s (%s)", len(articles), ticker, company)
    return articles
