"""Reference return series, computed directly from prices (no costs, no limits)."""
from __future__ import annotations

from datetime import date

import pandas as pd


def _window(s: pd.Series, start: date, end: date) -> pd.Series:
    """Restrict to [start, end) and index by plain dates, matching strategy returns."""
    d = s.index.date
    s = s[(d >= start) & (d < end)].copy()
    s.index = pd.to_datetime(s.index.date)
    return s


def buy_and_hold(close: pd.Series, start: date, end: date) -> pd.Series:
    """Daily returns of holding one series (e.g. Nifty 50)."""
    return _window(close.pct_change(), start, end).dropna()


def equal_weight(close_panel: pd.DataFrame, start: date, end: date) -> pd.Series:
    """Daily returns of an equal-weight portfolio of every listed stock, rebalanced daily."""
    rets = close_panel.pct_change(fill_method=None)
    return _window(rets.mean(axis=1, skipna=True), start, end).dropna()
