"""Forward-looking catalyst data: upcoming earnings, news, economic calendar.

Sources:
  - Finnhub: earnings calendar, economic calendar, company news
  - NewsAPI: broader news coverage (English-language headlines)

HK tickers are limited on Finnhub's free tier — we degrade gracefully.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import finnhub
import requests

from src.config.loader import load_secrets
from src.storage import cache as kv_cache

log = logging.getLogger(__name__)

NEWSAPI_BASE = "https://newsapi.org/v2"

# Catalyst data (news headlines + earnings dates) shifts within a day but not
# hour-to-hour. 1h cache lets the 4 daily digests + intraday checks share a
# fetch instead of hitting NewsAPI / Finnhub on every run.
CATALYSTS_CACHE_TTL_SECONDS = 60 * 60


# ----- Data classes -----

@dataclass
class EarningsEvent:
    symbol: str
    event_date: date
    hour: Optional[str]            # 'bmo' = before market open, 'amc' = after market close
    eps_estimate: Optional[float]
    revenue_estimate: Optional[float]
    eps_actual: Optional[float] = None  # if already reported


@dataclass
class NewsItem:
    headline: str
    summary: Optional[str]
    source: str
    url: str
    timestamp_utc: datetime
    sentiment: Optional[float] = None  # -1..+1 if known, else None


@dataclass
class EconomicEvent:
    event_date: date
    country: str
    event: str
    impact: str                    # 'low' | 'medium' | 'high' | ''
    estimate: Optional[float]
    actual: Optional[float]
    prev: Optional[float] = None


@dataclass
class CatalystSnapshot:
    symbol: str
    market: str
    upcoming_earnings: list[EarningsEvent] = field(default_factory=list)
    recent_news: list[NewsItem] = field(default_factory=list)
    upcoming_economic_events: list[EconomicEvent] = field(default_factory=list)
    horizon_days: int = 30
    timestamp_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ----- Clients -----

def _finnhub() -> Optional[finnhub.Client]:
    key = load_secrets().finnhub.api_key
    return finnhub.Client(api_key=key) if key else None


def _newsapi_key() -> Optional[str]:
    nk = load_secrets().newsapi
    return nk.api_key if (nk and nk.api_key) else None


# ----- Fetchers -----

def _fetch_earnings(symbol: str, market: str, horizon_days: int) -> list[EarningsEvent]:
    fh = _finnhub()
    if fh is None or market != "US":
        # Finnhub free tier: US only. For HK, we have no good free source.
        return []
    today = date.today()
    end = today + timedelta(days=horizon_days)
    try:
        result = fh.earnings_calendar(_from=today.isoformat(), to=end.isoformat(), symbol=symbol)
    except Exception as e:
        log.info("Finnhub earnings calendar failed for %s: %s", symbol, e)
        return []
    events: list[EarningsEvent] = []
    for r in (result or {}).get("earningsCalendar", []) or []:
        try:
            events.append(
                EarningsEvent(
                    symbol=r.get("symbol", symbol),
                    event_date=date.fromisoformat(r["date"]),
                    hour=r.get("hour"),
                    eps_estimate=r.get("epsEstimate"),
                    revenue_estimate=r.get("revenueEstimate"),
                    eps_actual=r.get("epsActual"),
                )
            )
        except (KeyError, ValueError) as e:
            log.debug("Earnings row skipped: %s", e)
    return events


def _fetch_economic_calendar(horizon_days: int) -> list[EconomicEvent]:
    fh = _finnhub()
    if fh is None:
        return []
    today = date.today()
    end = today + timedelta(days=horizon_days)
    try:
        # Finnhub: economic_calendar; free tier may have limits
        result = fh.calendar_economic(_from=today.isoformat(), to=end.isoformat())
    except AttributeError:
        # The python-finnhub client API has shifted across versions; try alt name
        try:
            result = fh.economic_calendar(_from=today.isoformat(), to=end.isoformat())
        except Exception as e:
            log.info("Finnhub economic calendar failed: %s", e)
            return []
    except Exception as e:
        log.info("Finnhub economic calendar failed: %s", e)
        return []

    events: list[EconomicEvent] = []
    for r in (result or {}).get("economicCalendar", []) or []:
        try:
            events.append(
                EconomicEvent(
                    event_date=date.fromisoformat(r["time"][:10]) if r.get("time") else today,
                    country=r.get("country", ""),
                    event=r.get("event", ""),
                    impact=r.get("impact", ""),
                    estimate=r.get("estimate"),
                    actual=r.get("actual"),
                    prev=r.get("prev"),
                )
            )
        except (KeyError, ValueError) as e:
            log.debug("Economic event skipped: %s", e)
    # Filter to high+medium impact only, keep top 20
    events = [e for e in events if (e.impact or "").lower() in ("high", "medium")]
    return events[:20]


def _fetch_finnhub_news(symbol: str, market: str, days: int) -> list[NewsItem]:
    fh = _finnhub()
    if fh is None or market != "US":
        return []
    today = date.today()
    start = today - timedelta(days=days)
    try:
        news = fh.company_news(symbol, _from=start.isoformat(), to=today.isoformat()) or []
    except Exception as e:
        log.info("Finnhub news failed for %s: %s", symbol, e)
        return []
    items: list[NewsItem] = []
    for n in news[:25]:
        try:
            ts = datetime.fromtimestamp(int(n.get("datetime", 0)), tz=timezone.utc)
            items.append(
                NewsItem(
                    headline=n.get("headline", ""),
                    summary=n.get("summary"),
                    source=n.get("source", ""),
                    url=n.get("url", ""),
                    timestamp_utc=ts,
                )
            )
        except (TypeError, ValueError):
            continue
    return items


def _fetch_newsapi(symbol: str, name_hint: Optional[str], days: int) -> list[NewsItem]:
    key = _newsapi_key()
    if not key:
        return []
    today = date.today()
    start = today - timedelta(days=min(days, 28))  # NewsAPI free tier: 30-day window
    # Build query: prefer company name when available, else ticker
    q = name_hint or symbol
    params = {
        "q": q,
        "from": start.isoformat(),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 20,
        "apiKey": key,
    }
    try:
        r = requests.get(f"{NEWSAPI_BASE}/everything", params=params, timeout=15)
        r.raise_for_status()
        articles = r.json().get("articles", [])
    except requests.RequestException as e:
        log.info("NewsAPI request failed for %s: %s", symbol, e)
        return []
    items: list[NewsItem] = []
    for a in articles:
        try:
            ts = datetime.fromisoformat(a["publishedAt"].replace("Z", "+00:00"))
            items.append(
                NewsItem(
                    headline=a.get("title", "") or "",
                    summary=a.get("description"),
                    source=(a.get("source") or {}).get("name", ""),
                    url=a.get("url", ""),
                    timestamp_utc=ts,
                )
            )
        except (KeyError, ValueError):
            continue
    return items


# ----- Public API -----

def fetch_catalysts(
    symbol: str,
    market: str,
    name_hint: Optional[str] = None,
    horizon_days: int = 30,
    news_lookback_days: int = 14,
    use_cache: bool = True,
) -> CatalystSnapshot:
    """Aggregate forward-looking catalysts for a single symbol. 1h TTL cache
    keyed on all inputs that affect the result."""
    if use_cache:
        cache_key = (
            f"catalysts:v1:{symbol}:{market}:hz={horizon_days}:"
            f"nl={news_lookback_days}:nh={name_hint or ''}"
        )
        return kv_cache.cached_call(
            cache_key,
            ttl_seconds=CATALYSTS_CACHE_TTL_SECONDS,
            fn=lambda: _fetch_catalysts_uncached(
                symbol, market, name_hint, horizon_days, news_lookback_days
            ),
        )
    return _fetch_catalysts_uncached(
        symbol, market, name_hint, horizon_days, news_lookback_days
    )


def _fetch_catalysts_uncached(
    symbol: str,
    market: str,
    name_hint: Optional[str],
    horizon_days: int,
    news_lookback_days: int,
) -> CatalystSnapshot:
    earnings = _fetch_earnings(symbol, market, horizon_days)
    econ = _fetch_economic_calendar(horizon_days)
    news_fh = _fetch_finnhub_news(symbol, market, news_lookback_days)
    news_na = _fetch_newsapi(symbol, name_hint, news_lookback_days)

    # Merge + dedup news by headline
    seen_headlines: set[str] = set()
    merged: list[NewsItem] = []
    for n in (news_fh + news_na):
        key = (n.headline or "").strip().lower()
        if not key or key in seen_headlines:
            continue
        seen_headlines.add(key)
        merged.append(n)
    # Sort by recency desc, cap
    merged.sort(key=lambda x: x.timestamp_utc, reverse=True)
    merged = merged[:30]

    return CatalystSnapshot(
        symbol=symbol,
        market=market,
        upcoming_earnings=earnings,
        recent_news=merged,
        upcoming_economic_events=econ,
        horizon_days=horizon_days,
    )
