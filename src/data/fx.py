"""Live FX rates with caching.

Uses yfinance currency pairs (HKDUSD=X, EURUSD=X, etc.). Cached per-process
for the session so we don't refetch on every Portfolio Fit call.

Fallback: hardcoded fair-value rates if yfinance fails (HKD pegged at 7.8).
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import yfinance as yf

log = logging.getLogger(__name__)

# Fallback rates if yfinance fails. HKD is pegged at ~7.8 / USD.
_FALLBACK_TO_USD = {
    "USD": 1.0,
    "HKD": 1.0 / 7.8,
    "EUR": 1.05,
    "GBP": 1.27,
    "JPY": 1.0 / 150,
    "CNY": 1.0 / 7.2,
}


def _yf_pair_symbol(from_ccy: str, to_ccy: str) -> str:
    return f"{from_ccy.upper()}{to_ccy.upper()}=X"


@lru_cache(maxsize=64)
def get_rate(from_ccy: str, to_ccy: str = "USD") -> float:
    """Returns the multiplier such that amount_from_ccy * rate = amount_in_to_ccy."""
    from_ccy = from_ccy.upper()
    to_ccy = to_ccy.upper()
    if from_ccy == to_ccy:
        return 1.0

    pair = _yf_pair_symbol(from_ccy, to_ccy)
    try:
        df = yf.Ticker(pair).history(period="5d")["Close"].dropna()
        if not df.empty:
            return float(df.iloc[-1])
    except Exception as e:
        log.info("FX %s failed: %s — using fallback.", pair, e)

    # Fallback: triangulate via USD if both have entries
    if from_ccy in _FALLBACK_TO_USD and to_ccy in _FALLBACK_TO_USD:
        return _FALLBACK_TO_USD[from_ccy] / _FALLBACK_TO_USD[to_ccy]

    log.warning("No FX rate available for %s→%s — returning 1.0", from_ccy, to_ccy)
    return 1.0


def convert(amount: float, from_ccy: str, to_ccy: str = "USD") -> float:
    return amount * get_rate(from_ccy, to_ccy)


def reset_cache() -> None:
    get_rate.cache_clear()
