"""Hedge candidate pool + correlation computation against a position.

Pool selection is heuristic (sector / market based). Correlations use 90 trading
days of adjusted closes from yfinance. Higher |correlation| = better hedge
effectiveness when the candidate is shorted against a long position.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


@dataclass
class HedgeInstrument:
    ticker: str
    kind: Literal["future", "etf", "etf_short", "index"]
    notional_per_contract_usd: Optional[float]  # for futures; None for ETFs (use shares × price)
    description: str


# ---- Candidate pools ----

_US_TECH = [
    HedgeInstrument("NQ=F", "future", 20.0 * 21000, "E-mini Nasdaq-100 futures (~$20 × index value)"),
    HedgeInstrument("MNQ=F", "future", 2.0 * 21000, "Micro E-mini Nasdaq-100 (~$2 × index value)"),
    HedgeInstrument("QQQ", "etf_short", None, "Invesco QQQ — short as Nasdaq proxy"),
    HedgeInstrument("IGV", "etf_short", None, "iShares Software ETF — short as US software proxy"),
    HedgeInstrument("ES=F", "future", 50.0 * 5500, "E-mini S&P 500 futures (~$50 × index value)"),
    HedgeInstrument("VX=F", "future", 1000 * 18, "VIX futures (~$1000 × VIX) — long for vol spike protection"),
]

_US_BROAD = [
    HedgeInstrument("ES=F", "future", 50.0 * 5500, "E-mini S&P 500 futures (~$50 × index value)"),
    HedgeInstrument("MES=F", "future", 5.0 * 5500, "Micro E-mini S&P 500 (~$5 × index value)"),
    HedgeInstrument("SPY", "etf_short", None, "SPDR S&P 500 ETF — short as broad market proxy"),
    HedgeInstrument("VX=F", "future", 1000 * 18, "VIX futures — long for vol-spike protection"),
]

_US_HEALTHCARE = [
    HedgeInstrument("XLV", "etf_short", None, "Health Care Select Sector SPDR — short"),
    HedgeInstrument("ES=F", "future", 50.0 * 5500, "E-mini S&P 500 futures"),
]

_US_FINANCIAL = [
    HedgeInstrument("XLF", "etf_short", None, "Financial Select Sector SPDR — short"),
    HedgeInstrument("ES=F", "future", 50.0 * 5500, "E-mini S&P 500 futures"),
]

_US_CONSUMER = [
    HedgeInstrument("XLY", "etf_short", None, "Consumer Discretionary SPDR — short"),
    HedgeInstrument("XLP", "etf_short", None, "Consumer Staples SPDR — short (for staples names)"),
    HedgeInstrument("ES=F", "future", 50.0 * 5500, "E-mini S&P 500 futures"),
]

_HK_TECH = [
    HedgeInstrument("HSI=F", "future", 50.0 * 19000, "Hang Seng Index futures (HK$50 × index)"),
    HedgeInstrument("HHI=F", "future", 50.0 * 7000, "Hang Seng China Enterprises (H-shares) futures"),
    HedgeInstrument("KWEB", "etf_short", None, "KraneShares CSI China Internet ETF — short"),
    HedgeInstrument("CQQQ", "etf_short", None, "Invesco China Technology ETF — short"),
    HedgeInstrument("FXI", "etf_short", None, "iShares China Large-Cap ETF — short"),
]

_HK_BROAD = [
    HedgeInstrument("HSI=F", "future", 50.0 * 19000, "Hang Seng Index futures"),
    HedgeInstrument("HHI=F", "future", 50.0 * 7000, "H-shares futures"),
    HedgeInstrument("FXI", "etf_short", None, "iShares China Large-Cap ETF — short"),
]


def candidate_pool(market: str, sector: Optional[str]) -> list[HedgeInstrument]:
    sector_lc = (sector or "").lower()
    if market == "US":
        if "tech" in sector_lc or "communication" in sector_lc:
            return _US_TECH
        if "health" in sector_lc:
            return _US_HEALTHCARE
        if "financial" in sector_lc:
            return _US_FINANCIAL
        if "consumer" in sector_lc:
            return _US_CONSUMER
        return _US_BROAD
    if market == "HK":
        if "tech" in sector_lc or "communication" in sector_lc or "consumer" in sector_lc:
            return _HK_TECH
        return _HK_BROAD
    return _US_BROAD


# ---- Correlations ----

def compute_correlations(
    position_ticker: str,
    candidates: list[HedgeInstrument],
    period_days: int = 90,
) -> dict[str, Optional[float]]:
    """Pearson correlation of daily log returns between position and each candidate.

    Returns a dict mapping candidate ticker → correlation (None if data is missing).
    """
    period = f"{max(period_days + 10, 100)}d"  # buffer for non-trading days
    tickers = [position_ticker] + [c.ticker for c in candidates]

    try:
        df = yf.download(
            tickers,
            period=period,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception as e:
        log.warning("yf.download failed for correlation: %s", e)
        return {c.ticker: None for c in candidates}

    # Build per-ticker close series
    closes: dict[str, pd.Series] = {}
    for t in tickers:
        try:
            if isinstance(df.columns, pd.MultiIndex):
                s = df[t]["Close"].dropna()
            else:
                s = df["Close"].dropna()
            if not s.empty:
                closes[t] = s
        except Exception:
            continue

    if position_ticker not in closes or len(closes[position_ticker]) < 30:
        return {c.ticker: None for c in candidates}

    pos_ret = np.log(closes[position_ticker] / closes[position_ticker].shift(1)).dropna()
    pos_ret = pos_ret.tail(period_days)

    out: dict[str, Optional[float]] = {}
    for c in candidates:
        if c.ticker not in closes:
            out[c.ticker] = None
            continue
        cand_ret = np.log(closes[c.ticker] / closes[c.ticker].shift(1)).dropna()
        # Align on common dates
        joined = pd.concat([pos_ret, cand_ret], axis=1, join="inner").dropna()
        if len(joined) < 20:
            out[c.ticker] = None
            continue
        corr = joined.iloc[:, 0].corr(joined.iloc[:, 1])
        out[c.ticker] = float(corr) if pd.notna(corr) else None
    return out
