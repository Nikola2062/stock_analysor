"""Polygon.io REST adapter — minimal, focused on what the pipeline needs.

Polygon's strength: clean US equity data, real-time + intraday. Weaker for
HK and futures, so the unified market.py routes US to Polygon (when keyed) and
HK/futures to yfinance.

No polygon-api-client dependency — direct requests.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

from src.config.loader import load_secrets

log = logging.getLogger(__name__)

BASE_URL = "https://api.polygon.io"


def _key() -> Optional[str]:
    sec = load_secrets()
    return sec.polygon.api_key if (sec.polygon and sec.polygon.api_key) else None


def is_available() -> bool:
    return _key() is not None


def fetch_daily_bars(symbol: str, lookback_days: int = 400) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV bars for a US ticker. Returns a DataFrame indexed by date.

    Schema: columns = [Open, High, Low, Close, Volume], index = DatetimeIndex (naive UTC).
    Returns None on any failure (caller falls back to yfinance).
    """
    key = _key()
    if not key:
        return None
    today = date.today()
    start = today - timedelta(days=lookback_days)
    url = f"{BASE_URL}/v2/aggs/ticker/{symbol}/range/1/day/{start.isoformat()}/{today.isoformat()}"
    params = {"adjusted": "true", "sort": "asc", "limit": 5000, "apiKey": key}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        log.info("Polygon daily bars failed for %s: %s", symbol, e)
        return None

    results = data.get("results") or []
    if not results:
        return None

    df = pd.DataFrame(results)
    # Polygon timestamps are milliseconds (UTC).
    df["date"] = pd.to_datetime(df["t"], unit="ms").dt.tz_localize(None)
    df = df.set_index("date").sort_index()
    df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]]


def fetch_last_trade(symbol: str) -> Optional[float]:
    """Last trade price for a US ticker (real-time on paid tier, 15-min delayed on free)."""
    key = _key()
    if not key:
        return None
    url = f"{BASE_URL}/v2/last/trade/{symbol}"
    try:
        r = requests.get(url, params={"apiKey": key}, timeout=10)
        r.raise_for_status()
        data = r.json()
        return float((data.get("results") or {}).get("p"))
    except (requests.RequestException, KeyError, TypeError, ValueError) as e:
        log.info("Polygon last_trade failed for %s: %s", symbol, e)
        return None


def fetch_ticker_details(symbol: str) -> Optional[dict]:
    """Company details: name, market cap, sector via SIC, etc. US tickers only."""
    key = _key()
    if not key:
        return None
    url = f"{BASE_URL}/v3/reference/tickers/{symbol}"
    try:
        r = requests.get(url, params={"apiKey": key}, timeout=10)
        r.raise_for_status()
        return r.json().get("results")
    except requests.RequestException as e:
        log.info("Polygon ticker_details failed for %s: %s", symbol, e)
        return None
