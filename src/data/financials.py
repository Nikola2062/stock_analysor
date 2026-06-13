"""Financial statements via yfinance — annual + quarterly, most-recent 2 of each.

We pull the standard three statements (income, balance, cash flow) at both
cadences. Line items are sparse and yfinance label names drift, so we resolve
each canonical metric through a list of candidate row labels and take the first
hit. Missing values stay None — the LLM resolver tolerates gaps.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import yfinance as yf

from src.models.schemas import FinancialLine, FinancialPeriodSet, FinancialReport

log = logging.getLogger(__name__)


# Canonical label -> ordered yfinance row-name candidates.
_INCOME_ITEMS: list[tuple[str, list[str]]] = [
    ("Revenue",            ["Total Revenue", "Operating Revenue", "Revenue"]),
    ("Cost of Revenue",    ["Cost Of Revenue", "Reconciled Cost Of Revenue"]),
    ("Gross Profit",       ["Gross Profit"]),
    ("R&D",                ["Research And Development"]),
    ("SG&A",               ["Selling General And Administration", "Selling General Administrative"]),
    ("Operating Income",   ["Operating Income", "Total Operating Income As Reported"]),
    ("EBITDA",             ["EBITDA", "Normalized EBITDA"]),
    ("Interest Expense",   ["Interest Expense", "Net Interest Income"]),
    ("Net Income",         ["Net Income", "Net Income Common Stockholders", "Net Income Including Noncontrolling Interests"]),
    ("Diluted EPS",        ["Diluted EPS"]),
    ("Diluted Shares",     ["Diluted Average Shares"]),
]

_BALANCE_ITEMS: list[tuple[str, list[str]]] = [
    ("Cash & Equivalents", ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]),
    ("Current Assets",     ["Current Assets", "Total Current Assets"]),
    ("Total Assets",       ["Total Assets"]),
    ("Current Liabilities",["Current Liabilities", "Total Current Liabilities"]),
    ("Total Debt",         ["Total Debt"]),
    ("Net Debt",           ["Net Debt"]),
    ("Total Liabilities",  ["Total Liabilities Net Minority Interest", "Total Liabilities"]),
    ("Stockholders Equity",["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]),
    ("Working Capital",    ["Working Capital"]),
    ("Retained Earnings",  ["Retained Earnings"]),
    ("Shares Outstanding", ["Ordinary Shares Number", "Share Issued"]),
]

_CASHFLOW_ITEMS: list[tuple[str, list[str]]] = [
    ("Operating Cash Flow", ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]),
    ("CapEx",               ["Capital Expenditure"]),
    ("Free Cash Flow",      ["Free Cash Flow"]),
    ("Dividends Paid",      ["Cash Dividends Paid", "Common Stock Dividend Paid"]),
    ("Share Buybacks",      ["Repurchase Of Capital Stock", "Common Stock Payments"]),
    ("Debt Issued",         ["Issuance Of Debt", "Long Term Debt Issuance"]),
    ("Debt Repaid",         ["Repayment Of Debt", "Long Term Debt Payments"]),
    ("Net Change in Cash",  ["Changes In Cash"]),
]


def _val(df: Optional[pd.DataFrame], candidates: list[str], col) -> Optional[float]:
    """Resolve a row label by trying each candidate; return cell or None."""
    if df is None or df.empty or col not in df.columns:
        return None
    for name in candidates:
        if name in df.index:
            v = df.loc[name, col]
            try:
                if pd.isna(v):
                    return None
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def _build_period_set(
    cadence: str,
    income_df: Optional[pd.DataFrame],
    balance_df: Optional[pd.DataFrame],
    cashflow_df: Optional[pd.DataFrame],
    keep_n: int = 2,
) -> Optional[FinancialPeriodSet]:
    # Collect candidate period columns (a date is present in any statement).
    period_cols: list = []
    for df in (income_df, balance_df, cashflow_df):
        if df is not None and not df.empty:
            for c in df.columns:
                if c not in period_cols:
                    period_cols.append(c)
    if not period_cols:
        return None
    # Sort newest first, take N
    period_cols = sorted(period_cols, reverse=True)[:keep_n]

    period_strs = [
        (c.date().isoformat() if hasattr(c, "date") else str(c))
        for c in period_cols
    ]

    def _lines(df, items):
        out: list[FinancialLine] = []
        for label, candidates in items:
            values = [_val(df, candidates, c) for c in period_cols]
            if any(v is not None for v in values):
                out.append(FinancialLine(label=label, values=values))
        return out

    return FinancialPeriodSet(
        cadence=cadence,                                   # type: ignore[arg-type]
        periods=period_strs,
        income=_lines(income_df, _INCOME_ITEMS),
        balance=_lines(balance_df, _BALANCE_ITEMS),
        cashflow=_lines(cashflow_df, _CASHFLOW_ITEMS),
    )


def fetch_financials(symbol: str) -> FinancialReport:
    """Fetch annual + quarterly statements (last 2 periods each) and pack them up."""
    ticker = yf.Ticker(symbol)
    notes: list[str] = []
    currency = "USD"
    try:
        info = ticker.info or {}
        currency = info.get("financialCurrency") or info.get("currency") or currency
    except Exception as e:
        notes.append(f"info() failed: {e}")

    def _safe(getter, label):
        try:
            return getter()
        except Exception as e:
            notes.append(f"{label} failed: {e}")
            return None

    annual_inc = _safe(lambda: ticker.income_stmt,              "income_stmt")
    annual_bs  = _safe(lambda: ticker.balance_sheet,            "balance_sheet")
    annual_cf  = _safe(lambda: ticker.cashflow,                 "cashflow")
    q_inc      = _safe(lambda: ticker.quarterly_income_stmt,    "quarterly_income_stmt")
    q_bs       = _safe(lambda: ticker.quarterly_balance_sheet,  "quarterly_balance_sheet")
    q_cf       = _safe(lambda: ticker.quarterly_cashflow,       "quarterly_cashflow")

    annual = _build_period_set("annual",   annual_inc, annual_bs, annual_cf, keep_n=2)
    quart  = _build_period_set("quarterly", q_inc,     q_bs,      q_cf,      keep_n=2)

    if annual is None:
        notes.append("No annual statements available.")
    elif len(annual.periods) < 2:
        notes.append(f"Only {len(annual.periods)} annual period(s) available — PoP delta limited.")
    if quart is None:
        notes.append("No quarterly statements available.")
    elif len(quart.periods) < 2:
        notes.append(f"Only {len(quart.periods)} quarterly period(s) available — PoP delta limited.")

    return FinancialReport(
        currency=currency,
        annual=annual,
        quarterly=quart,
        deep_resolution=None,
        fetch_notes=notes,
    )
