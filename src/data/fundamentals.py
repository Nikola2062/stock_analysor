"""Fundamentals via yfinance + Finnhub.

yfinance gives us .info (key stats) and the financial statements.
Finnhub adds company description, peers, and recent news headlines.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import finnhub
import yfinance as yf

from src.config.loader import load_secrets
from src.storage import cache as kv_cache

log = logging.getLogger(__name__)

# Fundamentals don't change minute-to-minute. 24h cache so the 4 daily Telegram
# digests share one fetch per ticker, instead of burning 4 Finnhub quota-slots.
FUNDAMENTALS_CACHE_TTL_SECONDS = 24 * 60 * 60


@dataclass
class FundamentalsSnapshot:
    symbol: str
    name: Optional[str]
    sector: Optional[str]
    industry: Optional[str]
    market_cap: Optional[float]
    currency: str
    business_description: Optional[str]

    # Ratios / scalars
    pe_trailing: Optional[float]
    pe_forward: Optional[float]
    ev_to_ebitda: Optional[float]
    price_to_sales: Optional[float]
    price_to_book: Optional[float]
    gross_margin: Optional[float]
    operating_margin: Optional[float]
    profit_margin: Optional[float]
    return_on_equity: Optional[float]
    return_on_assets: Optional[float]
    debt_to_equity: Optional[float]
    current_ratio: Optional[float]
    quick_ratio: Optional[float]
    revenue_growth_yoy: Optional[float]
    earnings_growth_yoy: Optional[float]
    free_cash_flow: Optional[float]
    total_cash: Optional[float]
    total_debt: Optional[float]

    # Peers + recent news (Finnhub)
    peers: list[str] = field(default_factory=list)
    recent_headlines: list[dict[str, Any]] = field(default_factory=list)

    # Raw info dump for the LLM to inspect (truncated)
    raw_info: dict[str, Any] = field(default_factory=dict)


def _finnhub_client() -> Optional[finnhub.Client]:
    secrets = load_secrets()
    key = secrets.finnhub.api_key
    if not key:
        return None
    return finnhub.Client(api_key=key)


_INFO_KEEP_KEYS = {
    "longName", "shortName", "sector", "industry", "country",
    "currency", "marketCap", "enterpriseValue",
    "trailingPE", "forwardPE", "priceToSalesTrailing12Months",
    "priceToBook", "enterpriseToEbitda",
    "grossMargins", "operatingMargins", "profitMargins",
    "returnOnEquity", "returnOnAssets",
    "debtToEquity", "currentRatio", "quickRatio",
    "revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth",
    "freeCashflow", "operatingCashflow", "totalCash", "totalDebt",
    "dividendYield", "payoutRatio", "trailingEps", "forwardEps",
    "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
    "longBusinessSummary",
}


def fetch_fundamentals(
    symbol: str,
    headlines_days: int = 14,
    use_cache: bool = True,
) -> FundamentalsSnapshot:
    """Pull fundamentals + recent headlines. 24h TTL cache (per-symbol+headlines_days)
    avoids re-burning Finnhub quota on every scheduled run within a day."""
    if use_cache:
        cache_key = f"fundamentals:v1:{symbol}:hd={headlines_days}"
        return kv_cache.cached_call(
            cache_key,
            ttl_seconds=FUNDAMENTALS_CACHE_TTL_SECONDS,
            fn=lambda: _fetch_fundamentals_uncached(symbol, headlines_days),
        )
    return _fetch_fundamentals_uncached(symbol, headlines_days)


def _fetch_fundamentals_uncached(symbol: str, headlines_days: int = 14) -> FundamentalsSnapshot:
    ticker = yf.Ticker(symbol)
    try:
        info = ticker.info or {}
    except Exception as e:
        log.warning("yfinance .info failed for %s: %s", symbol, e)
        info = {}

    raw = {k: info.get(k) for k in _INFO_KEEP_KEYS if k in info}
    currency = info.get("currency") or ("HKD" if symbol.endswith(".HK") else "USD")

    # Finnhub: peers + recent headlines (best-effort)
    peers: list[str] = []
    headlines: list[dict[str, Any]] = []
    fh = _finnhub_client()
    if fh is not None:
        # Finnhub uses bare US tickers. For HK, Finnhub paid tier is needed — skip on free.
        if not symbol.endswith(".HK"):
            try:
                peers = fh.company_peers(symbol)[:10]
            except Exception as e:
                log.info("Finnhub peers failed for %s: %s", symbol, e)
            try:
                from datetime import date, timedelta
                today = date.today()
                start = today - timedelta(days=headlines_days)
                news = fh.company_news(symbol, _from=start.isoformat(), to=today.isoformat())
                headlines = [
                    {
                        "datetime": n.get("datetime"),
                        "headline": n.get("headline"),
                        "source": n.get("source"),
                        "summary": n.get("summary"),
                        "url": n.get("url"),
                    }
                    for n in (news or [])[:15]
                ]
            except Exception as e:
                log.info("Finnhub news failed for %s: %s", symbol, e)

    return FundamentalsSnapshot(
        symbol=symbol,
        name=info.get("longName") or info.get("shortName"),
        sector=info.get("sector"),
        industry=info.get("industry"),
        market_cap=info.get("marketCap"),
        currency=currency,
        business_description=info.get("longBusinessSummary"),
        pe_trailing=info.get("trailingPE"),
        pe_forward=info.get("forwardPE"),
        ev_to_ebitda=info.get("enterpriseToEbitda"),
        price_to_sales=info.get("priceToSalesTrailing12Months"),
        price_to_book=info.get("priceToBook"),
        gross_margin=info.get("grossMargins"),
        operating_margin=info.get("operatingMargins"),
        profit_margin=info.get("profitMargins"),
        return_on_equity=info.get("returnOnEquity"),
        return_on_assets=info.get("returnOnAssets"),
        debt_to_equity=info.get("debtToEquity"),
        current_ratio=info.get("currentRatio"),
        quick_ratio=info.get("quickRatio"),
        revenue_growth_yoy=info.get("revenueGrowth"),
        earnings_growth_yoy=info.get("earningsGrowth"),
        free_cash_flow=info.get("freeCashflow"),
        total_cash=info.get("totalCash"),
        total_debt=info.get("totalDebt"),
        peers=peers,
        recent_headlines=headlines,
        raw_info=raw,
    )
