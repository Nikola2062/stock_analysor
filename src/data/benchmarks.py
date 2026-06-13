"""Sector ETF + market index lookups for the Relative Strength agent.

Pure data-layer: no LLM, no business logic. The sector→ETF map is loaded from
`config/technical.yaml` at call time so updates don't require a restart.
"""
from __future__ import annotations

import logging
from typing import Optional

from src.data.market import fetch_bars

log = logging.getLogger(__name__)


def get_sector_etf(market: str, sector: Optional[str]) -> Optional[str]:
    """Return the configured sector-ETF ticker for (market, sector), or None."""
    if not sector:
        return None
    # Lazy import to avoid a hard dependency cycle with the config layer at module load.
    from src.config.loader import load_technical
    tech_cfg = load_technical()
    market_map = tech_cfg.relative_strength.benchmarks.get(market, {})
    sector_etfs: dict = market_map.get("sector_etfs", {})
    # Case-insensitive substring match: "Technology" matches a sector field of
    # "Technology" or "Information Technology". This matches how yfinance returns sectors.
    sector_lc = sector.lower()
    for cfg_sector, etf in sector_etfs.items():
        if cfg_sector.lower() in sector_lc or sector_lc in cfg_sector.lower():
            return etf
    return None


def get_market_index(market: str) -> Optional[str]:
    """Return the configured market-index ticker for the given market."""
    from src.config.loader import load_technical
    tech_cfg = load_technical()
    market_map = tech_cfg.relative_strength.benchmarks.get(market, {})
    return market_map.get("market_index")


def fetch_returns(symbol: str, windows_days: list[int]) -> dict[int, Optional[float]]:
    """Return cumulative % returns over each window, keyed by window length.

    Returns None for any window that exceeds available history.
    """
    max_window = max(windows_days) if windows_days else 365
    # Add buffer for non-trading days
    bars = fetch_bars(symbol, lookback_days=int(max_window * 1.5) + 30)
    if bars is None or bars.empty or "Close" not in bars.columns:
        log.warning("benchmarks.fetch_returns: no bars for %s", symbol)
        return {w: None for w in windows_days}

    closes = bars["Close"].dropna()
    if closes.empty:
        return {w: None for w in windows_days}

    current = float(closes.iloc[-1])
    out: dict[int, Optional[float]] = {}
    for w in windows_days:
        if len(closes) <= w:
            out[w] = None
            continue
        past = float(closes.iloc[-w - 1])
        out[w] = (current / past - 1.0) * 100.0
    return out
