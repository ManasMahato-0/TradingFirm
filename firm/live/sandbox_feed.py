"""Bar sources for the intraday sandbox. The runner only sees this interface, so a recorded
day can be replayed through exactly the code that runs live.

  ReplaySource  replays recorded 5-minute bars (tests, dry runs, shake-downs)
  FyersSource   (see sandbox_live.py) polls Fyers for 1-minute bars and builds 5-minute bars
                with the same `resample_5m` used for the training data
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

import numpy as np
import pandas as pd

from firm.backtest.data import MarketData

HISTORY_DAYS = 25  # prior trading days given to the features (20-day volume ratios need them)
BAR = pd.Timedelta(minutes=5)


@dataclass
class Bar:
    ts: pd.Timestamp            # bar START time (09:15 ... 15:25), tz Asia/Kolkata
    ohlcv: pd.DataFrame         # index = symbols, columns open high low close volume
    nifty_open: float
    nifty_close: float


class BarSource(Protocol):
    symbols: list[str]

    def history(self) -> tuple[MarketData, pd.Series, pd.Series]:
        """Prior trading days: stock bars, Nifty 5-minute close and open series."""
        ...

    def next_bar(self) -> Bar | None:
        """The next COMPLETED bar of the day (waits if live); None when the day is over."""
        ...

    def quotes(self, symbols: list[str]) -> dict[str, float]:
        """Best available price per symbol right now (decision-time quote)."""
        ...


class ReplaySource:
    """Replays one recorded day. The 'quote' at a decision is the next bar's open (no latency),
    so a replay isolates the strategy from feed delay."""

    def __init__(self, data: MarketData, nifty_close: pd.Series, nifty_open: pd.Series, day: date,
                 history_days: int = HISTORY_DAYS) -> None:
        self.symbols = list(data.symbols)
        days = sorted(set(data.index.date))
        if day not in days:
            raise ValueError(f"no recorded bars for {day}")
        before = [d for d in days if d < day][-history_days:]
        keep_hist = np.isin(data.index.date, before)
        self._hist = MarketData({f: p.loc[keep_hist] for f, p in data.panels.items()})
        self._hist_nifty = (nifty_close.reindex(self._hist.index), nifty_open.reindex(self._hist.index))
        today = data.index.date == day
        self._bars = [(t, i) for i, t in enumerate(data.index) if today[i]]
        self._data, self._nc, self._no = data, nifty_close, nifty_open
        self._n = 0

    def history(self):
        return self._hist, self._hist_nifty[0], self._hist_nifty[1]

    def _bar(self, t: pd.Timestamp) -> Bar:
        d = self._data
        frame = pd.DataFrame({f: d.panels[f].loc[t] for f in ("open", "high", "low", "close", "volume")})
        return Bar(t, frame, float(self._no.get(t, np.nan)), float(self._nc.get(t, np.nan)))

    def next_bar(self) -> Bar | None:
        if self._n >= len(self._bars):
            return None
        t, _ = self._bars[self._n]
        self._n += 1
        return self._bar(t)

    def quotes(self, symbols: list[str]) -> dict[str, float]:
        """Next bar's open (or the last close at the end of the day)."""
        d = self._data
        j = min(self._n, len(self._bars) - 1)
        t = self._bars[j][0]
        field = "open" if self._n < len(self._bars) else "close"
        row = d.panels[field].loc[t]
        return {s: float(row[s]) for s in symbols if s in row.index and pd.notna(row[s])}


def append_bar(today: dict[str, list], bar: Bar) -> None:
    """Accumulate completed bars of the day."""
    today.setdefault("index", []).append(bar.ts)
    for f in ("open", "high", "low", "close", "volume"):
        today.setdefault(f, []).append(bar.ohlcv[f])
    today.setdefault("nifty_close", []).append(bar.nifty_close)
    today.setdefault("nifty_open", []).append(bar.nifty_open)


def assemble(hist: tuple[MarketData, pd.Series, pd.Series], today: dict[str, list],
             symbols: list[str]) -> tuple[MarketData, pd.Series, pd.Series]:
    """History + today's completed bars as one MarketData with matching Nifty series."""
    h, hc, ho = hist
    if not today.get("index"):
        return h, hc, ho
    idx = pd.DatetimeIndex(today["index"], name="ts")
    panels = {}
    for f in ("open", "high", "low", "close", "volume"):
        new = pd.DataFrame([r.reindex(symbols) for r in today[f]], index=idx)
        panels[f] = pd.concat([h.panels[f], new]) if len(h.index) else new
    nc = pd.concat([hc, pd.Series(today["nifty_close"], index=idx)])
    no = pd.concat([ho, pd.Series(today["nifty_open"], index=idx)])
    return MarketData(panels), nc, no
