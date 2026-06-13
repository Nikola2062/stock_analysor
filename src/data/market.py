"""Unified market-data interface.

Picks the best backend per request:
  - US equity bars → Polygon (when key set) for clean adjusted data, else yfinance
  - HK equity bars → yfinance (Polygon free tier doesn't cover .HK)
  - Futures / FX / indices → yfinance (Polygon strength is US equities)

Keep the surface lean: callers see the same DataFrame schema regardless of backend.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import yfinance as yf

from src.data import polygon_client

log = logging.getLogger(__name__)


def _is_us_equity(symbol: str) -> bool:
    """US tickers don't have a market suffix and are alphanumeric (+ optional . for class shares)."""
    if symbol.endswith(".HK"):
        return False
    if symbol.endswith("=F"):  # futures
        return False
    if symbol.startswith("^"):  # indices
        return False
    if "=" in symbol:  # FX pairs
        return False
    return True


def fetch_bars(symbol: str, lookback_days: int = 365) -> Optional[pd.DataFrame]:
    """Returns daily OHLCV bars. Tries Polygon for US (when keyed), else yfinance.

    Columns: Open, High, Low, Close, Volume. Index: naive UTC dates.
    """
    if _is_us_equity(symbol) and polygon_client.is_available():
        df = polygon_client.fetch_daily_bars(symbol, lookback_days=lookback_days)
        if df is not None and not df.empty:
            log.debug("Used Polygon for %s.", symbol)
            return df
        log.info("Polygon returned no data for %s — falling back to yfinance.", symbol)

    # yfinance fallback (or primary for HK/futures/indices)
    try:
        # Convert lookback_days to a yfinance period string
        if lookback_days <= 5:
            period = "5d"
        elif lookback_days <= 30:
            period = "1mo"
        elif lookback_days <= 90:
            period = "3mo"
        elif lookback_days <= 180:
            period = "6mo"
        elif lookback_days <= 365:
            period = "1y"
        elif lookback_days <= 730:
            period = "2y"
        else:
            period = "5y"
        df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if df.empty:
            return None
        # Strip timezone for consistent comparisons with Polygon
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        log.warning("yfinance failed for %s: %s", symbol, e)
        return None
