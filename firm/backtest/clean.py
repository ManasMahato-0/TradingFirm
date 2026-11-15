"""Data cleaning for daily bars.

1. Spikes: one-bar jumps that reverse next bar are bad ticks -> blanked.
2. Breaks: persistent one-day moves beyond a threshold are almost always corporate
   actions the source failed to adjust (demergers, splits, bonuses) -> earlier prices
   are scaled so the event shows zero return. Confirmed genuine moves are kept.
"""
from __future__ import annotations

import pandas as pd

from firm.backtest.data import MarketData

# (symbol, date) of verified real one-day moves beyond the break threshold.
KNOWN_REAL_MOVES = {
    ("PNB", "2017-10-25"),  # PSU bank recapitalisation rally
}
PRICE_FIELDS = ("open", "high", "low", "close")


def find_spikes(close: pd.DataFrame, threshold: float = 0.4, revert_tol: float = 0.15) -> pd.DataFrame:
    """Boolean mask of bars that jump more than `threshold` from the previous close
    and come back within `revert_tol` of it on the next bar."""
    prev, nxt = close.shift(1), close.shift(-1)
    jump = (close / prev - 1).abs() > threshold
    reverts = (nxt / prev - 1).abs() < revert_tol
    return jump & reverts


def remove_spikes(data: MarketData, threshold: float = 0.4,
                  revert_tol: float = 0.15) -> tuple[MarketData, list[tuple]]:
    """Blank out spike bars in every field. Repeats until none are left, since
    a multi-symbol glitch day can hide behind another."""
    panels = {f: p.copy() for f, p in data.panels.items()}
    removed: list[tuple] = []
    while True:
        mask = find_spikes(panels["close"].ffill(), threshold, revert_tol) & panels["close"].notna()
        if not mask.to_numpy().any():
            break
        for ts, sym in mask.stack()[lambda s: s].index:
            removed.append((ts, sym, float(panels["close"].at[ts, sym])))
        for p in panels.values():
            p[mask] = float("nan")
    return MarketData(panels), removed


def adjust_breaks(data: MarketData, threshold: float = 0.4,
                  keep: set[tuple[str, str]] = KNOWN_REAL_MOVES) -> tuple[MarketData, list[tuple]]:
    """Back-adjust prices before each unexplained one-day move beyond `threshold`."""
    panels = {f: p.copy() for f, p in data.panels.items()}
    close = panels["close"]
    rets = close.ffill().pct_change(fill_method=None).where(close.notna())
    breaks = rets.stack()[lambda s: s.abs() > threshold]
    adjusted: list[tuple] = []
    for (ts, sym), r in breaks.items():
        if (sym, ts.date().isoformat()) in keep:
            continue
        before = close.index < ts
        for f in PRICE_FIELDS:
            panels[f].loc[before, sym] *= 1 + r
        adjusted.append((ts, sym, float(r)))
    return MarketData(panels), adjusted


def clean(data: MarketData) -> tuple[MarketData, dict[str, list[tuple]]]:
    data, spikes = remove_spikes(data)
    data, breaks = adjust_breaks(data)
    return data, {"spikes_removed": spikes, "breaks_adjusted": breaks}
