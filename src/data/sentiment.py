"""Sentiment + insider transactions data (Finnhub).

Finnhub free tier supports:
  - news_sentiment(symbol): bullish/bearish % from news mentions
  - stock_insider_transactions(symbol, _from, to): aggregated insider activity

Both US-only on free tier.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import finnhub

from src.config.loader import load_secrets

log = logging.getLogger(__name__)


@dataclass
class SentimentSnapshot:
    symbol: str
    market: str
    # News sentiment
    news_bullish_pct: Optional[float] = None
    news_bearish_pct: Optional[float] = None
    company_news_score: Optional[float] = None       # -1..+1
    sector_news_score: Optional[float] = None
    # Insider activity
    insider_mspr_recent: Optional[float] = None      # monthly share purchase ratio (Finnhub)
    insider_net_change_shares: Optional[int] = None  # net buys - net sells in shares
    # Notes from data source quirks
    notes: list[str] = field(default_factory=list)


def _fh() -> Optional[finnhub.Client]:
    key = load_secrets().finnhub.api_key
    return finnhub.Client(api_key=key) if key else None


def fetch_sentiment(symbol: str, market: str) -> SentimentSnapshot:
    snap = SentimentSnapshot(symbol=symbol, market=market)
    fh = _fh()
    if fh is None:
        snap.notes.append("Finnhub key not set — sentiment data unavailable.")
        return snap
    if market != "US":
        snap.notes.append(f"Finnhub free tier is US-only; no sentiment for {market} ticker.")
        return snap

    # News sentiment
    try:
        ns = fh.news_sentiment(symbol)
        if ns:
            comp = ns.get("companyNewsScore")
            sec = ns.get("sectorAverageNewsScore")
            buzz = ns.get("buzz", {})
            snap.company_news_score = float(comp) if comp is not None else None
            snap.sector_news_score = float(sec) if sec is not None else None
            sentiment = ns.get("sentiment", {})
            if sentiment:
                snap.news_bullish_pct = (
                    float(sentiment.get("bullishPercent", 0)) * 100 if sentiment.get("bullishPercent") is not None else None
                )
                snap.news_bearish_pct = (
                    float(sentiment.get("bearishPercent", 0)) * 100 if sentiment.get("bearishPercent") is not None else None
                )
    except Exception as e:
        snap.notes.append(f"Finnhub news_sentiment failed: {e}")

    # Insider transactions: last 90 days
    try:
        today = date.today()
        start = today - timedelta(days=90)
        ins = fh.stock_insider_transactions(symbol, _from=start.isoformat(), to=today.isoformat())
        rows = (ins or {}).get("data", []) or []
        net_change = 0
        for r in rows:
            try:
                change = int(r.get("change", 0))
                net_change += change
            except (TypeError, ValueError):
                continue
        snap.insider_net_change_shares = net_change
    except Exception as e:
        snap.notes.append(f"Finnhub insider_transactions failed: {e}")

    # MSPR (monthly share purchase ratio)
    try:
        mspr = fh.stock_insider_sentiment(symbol, _from=(date.today() - timedelta(days=120)).isoformat(), to=date.today().isoformat())
        data = (mspr or {}).get("data", [])
        if data:
            # Take the most recent month's MSPR
            latest = sorted(data, key=lambda x: (x.get("year", 0), x.get("month", 0)))[-1]
            v = latest.get("mspr")
            snap.insider_mspr_recent = float(v) if v is not None else None
    except Exception as e:
        snap.notes.append(f"Finnhub insider_sentiment failed: {e}")

    return snap
