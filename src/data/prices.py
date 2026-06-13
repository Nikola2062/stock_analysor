"""Price / return / volatility data via yfinance.

Handles US tickers (e.g. FIG, NVDA) and HK tickers (e.g. 0700.HK).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from src.data.market import fetch_bars as fetch_market_bars

log = logging.getLogger(__name__)


@dataclass
class PriceSnapshot:
    symbol: str
    current_price: float
    currency: str
    history: pd.DataFrame  # OHLCV indexed by date
    realized_vol_annualized_pct: float  # from daily log returns over `vol_window_days`
    cumulative_return_1m_pct: float
    cumulative_return_3m_pct: float
    cumulative_return_1y_pct: float
    timestamp_utc: datetime


def fetch_prices(
    symbol: str,
    period: str = "1y",
    vol_window_days: int = 30,
) -> PriceSnapshot:
    # Route through the unified market interface (Polygon for US when keyed, else yfinance)
    lookback = {"5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730}.get(period, 365)
    history = fetch_market_bars(symbol, lookback_days=lookback)
    if history is None or history.empty:
        raise ValueError(f"No price history returned for {symbol!r}.")
    # We still need yfinance for ticker .info (currency, etc.) — Polygon details are different
    ticker = yf.Ticker(symbol)

    # Ensure a clean datetime index without TZ for arithmetic, but report TS in UTC.
    closes = history["Close"].dropna()
    current = float(closes.iloc[-1])

    # Currency from .info (yfinance), fall back per market suffix.
    try:
        info = ticker.info or {}
    except Exception as e:  # yfinance can raise on rate limit
        log.warning("yfinance .info failed for %s: %s", symbol, e)
        info = {}
    currency = info.get("currency") or ("HKD" if symbol.endswith(".HK") else "USD")

    # Realized vol from daily log returns
    log_returns = np.log(closes / closes.shift(1)).dropna()
    window = log_returns.tail(vol_window_days)
    if len(window) >= 5:
        daily_std = float(window.std(ddof=1))
        ann_vol_pct = daily_std * np.sqrt(252) * 100.0
    else:
        ann_vol_pct = float("nan")

    def cum_return_pct(days: int) -> float:
        if len(closes) <= days:
            return float("nan")
        past = float(closes.iloc[-days - 1])
        return (current / past - 1.0) * 100.0

    return PriceSnapshot(
        symbol=symbol,
        current_price=current,
        currency=currency,
        history=history,
        realized_vol_annualized_pct=ann_vol_pct,
        cumulative_return_1m_pct=cum_return_pct(21),
        cumulative_return_3m_pct=cum_return_pct(63),
        cumulative_return_1y_pct=cum_return_pct(252),
        timestamp_utc=datetime.now(timezone.utc),
    )
