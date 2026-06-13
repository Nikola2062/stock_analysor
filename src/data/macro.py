"""Macro indicator snapshot.

FRED for US macro series (yield curve, unemployment, Fed funds).
Falls back to yfinance for VIX / cross-asset signals when FRED key is missing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import yfinance as yf

from src.config.loader import load_secrets

log = logging.getLogger(__name__)


@dataclass
class MacroSnapshot:
    yield_curve_10y3m_pct: Optional[float]      # T10Y3M; <0 historically signals recession risk
    yield_curve_10y2y_pct: Optional[float]      # T10Y2Y
    vix_level: Optional[float]                  # ^VIX or VIXCLS
    fed_funds_rate_pct: Optional[float]         # DFF
    unemployment_rate_pct: Optional[float]      # UNRATE
    cpi_yoy_pct: Optional[float]                # CPIAUCSL YoY
    sp500_drawdown_pct: Optional[float]         # current vs trailing 52w high
    hsi_drawdown_pct: Optional[float]           # Hang Seng vs trailing 52w high
    signals: list[str] = field(default_factory=list)  # human-readable triggers fired


def _fred_get(series_id: str) -> Optional[float]:
    secrets = load_secrets()
    key = (secrets.fred.api_key if secrets.fred else "") or ""
    if not key:
        return None
    try:
        from fredapi import Fred  # heavy import, lazy
        fred = Fred(api_key=key)
        s = fred.get_series(series_id).dropna()
        if s.empty:
            return None
        return float(s.iloc[-1])
    except Exception as e:
        log.info("FRED %s failed: %s", series_id, e)
        return None


def _fred_yoy(series_id: str) -> Optional[float]:
    secrets = load_secrets()
    key = (secrets.fred.api_key if secrets.fred else "") or ""
    if not key:
        return None
    try:
        from fredapi import Fred
        fred = Fred(api_key=key)
        s = fred.get_series(series_id).dropna()
        if len(s) < 13:
            return None
        latest = float(s.iloc[-1])
        year_ago = float(s.iloc[-13])
        return (latest / year_ago - 1.0) * 100.0
    except Exception as e:
        log.info("FRED %s YoY failed: %s", series_id, e)
        return None


def _yf_latest(symbol: str) -> Optional[float]:
    try:
        df = yf.Ticker(symbol).history(period="1mo")["Close"].dropna()
        if df.empty:
            return None
        return float(df.iloc[-1])
    except Exception as e:
        log.info("yfinance %s failed: %s", symbol, e)
        return None


def _yf_drawdown_from_52w_high(symbol: str) -> Optional[float]:
    try:
        df = yf.Ticker(symbol).history(period="1y")["Close"].dropna()
        if df.empty:
            return None
        peak = float(df.max())
        current = float(df.iloc[-1])
        return (current / peak - 1.0) * 100.0
    except Exception as e:
        log.info("yfinance drawdown %s failed: %s", symbol, e)
        return None


def fetch_macro() -> MacroSnapshot:
    yc_10y3m = _fred_get("T10Y3M")
    yc_10y2y = _fred_get("T10Y2Y")
    vix = _fred_get("VIXCLS") or _yf_latest("^VIX")
    ffr = _fred_get("DFF")
    unrate = _fred_get("UNRATE")
    cpi_yoy = _fred_yoy("CPIAUCSL")
    sp500_dd = _yf_drawdown_from_52w_high("^GSPC")
    hsi_dd = _yf_drawdown_from_52w_high("^HSI")

    signals: list[str] = []
    if yc_10y3m is not None and yc_10y3m < 0:
        signals.append(f"10Y-3M yield curve inverted ({yc_10y3m:.2f}%) — historic recession signal")
    if yc_10y2y is not None and yc_10y2y < 0:
        signals.append(f"10Y-2Y yield curve inverted ({yc_10y2y:.2f}%)")
    if vix is not None:
        if vix > 30:
            signals.append(f"VIX elevated at {vix:.1f} (stress regime)")
        elif vix > 20:
            signals.append(f"VIX above 20 ({vix:.1f}) — heightened uncertainty")
    if sp500_dd is not None and sp500_dd < -10:
        signals.append(f"S&P 500 down {sp500_dd:.1f}% from 52w high (correction territory)")
    if hsi_dd is not None and hsi_dd < -10:
        signals.append(f"Hang Seng down {hsi_dd:.1f}% from 52w high")
    if unrate is not None and unrate > 5:
        signals.append(f"US unemployment {unrate:.1f}% (elevated)")
    if cpi_yoy is not None and cpi_yoy > 4:
        signals.append(f"US CPI YoY {cpi_yoy:.1f}% (sticky inflation)")

    return MacroSnapshot(
        yield_curve_10y3m_pct=yc_10y3m,
        yield_curve_10y2y_pct=yc_10y2y,
        vix_level=vix,
        fed_funds_rate_pct=ffr,
        unemployment_rate_pct=unrate,
        cpi_yoy_pct=cpi_yoy,
        sp500_drawdown_pct=sp500_dd,
        hsi_drawdown_pct=hsi_dd,
        signals=signals,
    )
