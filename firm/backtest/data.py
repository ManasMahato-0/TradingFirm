"""Market data panels and a point-in-time view that cannot see the future."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd

from firm.config import BARS_DIR
from firm.recorders.bars import _safe, load_bars

FIELDS = ("open", "high", "low", "close", "volume")


class MarketData:
    """Wide panels: one DataFrame per field, rows = timestamps, columns = symbols."""

    def __init__(self, panels: dict[str, pd.DataFrame]) -> None:
        self.panels = panels
        self.index: pd.DatetimeIndex = panels["close"].index
        self.symbols: list[str] = list(panels["close"].columns)

    @classmethod
    def from_frames(cls, frames: dict[str, pd.DataFrame]) -> MarketData:
        frames = {s: df for s, df in frames.items() if not df.empty}
        panels = {f: pd.DataFrame({s: df[f] for s, df in frames.items()}).sort_index() for f in FIELDS}
        return cls(panels)

    @classmethod
    def from_store(cls, dataset: str, symbols: list[str], start: date | None = None,
                   end: date | None = None, root: Path | None = None) -> MarketData:
        """Load from bar storage. A single-file cache of the whole dataset is rebuilt
        whenever any source file is newer, because reading thousands of small files is slow."""
        base = root or BARS_DIR
        cache = base.parent / "cache" / f"{dataset}.parquet"
        sources = list((base / dataset).glob("*/*.parquet"))
        newest = max((f.stat().st_mtime for f in sources), default=0.0)
        if cache.exists() and cache.stat().st_mtime >= newest:
            long = pd.read_parquet(cache)
        else:
            frames = []
            for sym_dir in sorted({f.parent for f in sources}):
                df = load_bars(dataset, sym_dir.name, root=root)
                if not df.empty:
                    frames.append(df.assign(symbol=sym_dir.name))
            long = pd.concat(frames) if frames else pd.DataFrame(columns=[*FIELDS, "symbol"])
            cache.parent.mkdir(parents=True, exist_ok=True)
            long.to_parquet(cache)
        long = long[long["symbol"].isin([_safe(s) for s in symbols])]
        if start:
            long = long[long.index.date >= start]
        if end:
            long = long[long.index.date < end]
        return cls.from_frames({s: g.drop(columns="symbol") for s, g in long.groupby("symbol")})

    def truncate(self, end: date) -> MarketData:
        """Copy containing only bars strictly before `end` (for fitting on a training window)."""
        mask = self.index.date < end
        return MarketData({f: p.loc[mask].copy() for f, p in self.panels.items()})

    def without(self, symbols: list[str]) -> MarketData:
        return MarketData({f: p.drop(columns=[s for s in symbols if s in p.columns])
                           for f, p in self.panels.items()})


class DataView:
    """What a strategy sees at bar i: everything up to and including bar i's close.

    Indicator helpers are computed once over the full panel with causal operations
    (rolling windows, shifts into the past) and then read only at row i.
    """

    def __init__(self, data: MarketData, next_trading_day: Callable[[date], date] | None = None) -> None:
        """next_trading_day: live trading passes the exchange calendar, so period ends
        are known on the latest bar (where there is no next bar in the data yet)."""
        self._data = data
        self._next_day = next_trading_day
        self._i = -1
        self._cache: dict[tuple, pd.DataFrame] = {}

    def _advance(self, i: int) -> None:  # engine only
        self._i = i

    @property
    def now(self) -> pd.Timestamp:
        return self._data.index[self._i]

    @property
    def bar_index(self) -> int:
        return self._i

    @property
    def symbols(self) -> list[str]:
        return self._data.symbols

    def current(self, field: str = "close") -> pd.Series:
        return self._data.panels[field].iloc[self._i].copy()

    def history(self, field: str = "close", n: int | None = None) -> pd.DataFrame:
        start = 0 if n is None else max(0, self._i - n + 1)
        return self._data.panels[field].iloc[start:self._i + 1].copy()

    def _at(self, key: tuple, build: Callable[[], pd.DataFrame]) -> pd.Series:
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key].iloc[self._i]

    def sma(self, window: int, field: str = "close") -> pd.Series:
        p = self._data.panels[field]
        return self._at(("sma", window, field), lambda: p.rolling(window, min_periods=window).mean())

    def pct_change(self, lookback: int, skip: int = 0) -> pd.Series:
        """Return from `lookback` bars ago to `skip` bars ago (e.g. 252, 21 = 12-1 momentum)."""
        c = self._data.panels["close"]
        return self._at(("ret", lookback, skip), lambda: c.shift(skip) / c.shift(lookback) - 1)

    def cached(self, key: str, build: Callable[[MarketData], object]) -> object:
        """Precompute something over the full data once (e.g. feature panels).
        `build` must be causal and callers must read only row `bar_index`;
        tests/test_features.py checks causality of the shipped feature builders."""
        if ("custom", key) not in self._cache:
            self._cache[("custom", key)] = build(self._data)
        return self._cache[("custom", key)]

    def is_period_end(self, freq: str) -> bool:
        """True on the last bar of a week ('W') or month ('M'). Looks only at the
        trading calendar of the next bar, never its prices."""
        idx = self._data.index
        if self._i >= len(idx) - 1:
            if self._next_day is None:
                return True
            a, b = idx[self._i].date(), self._next_day(idx[self._i].date())
        else:
            a, b = idx[self._i], idx[self._i + 1]
        if freq == "M":
            return (a.year, a.month) != (b.year, b.month)
        if freq == "W":
            return a.isocalendar()[:2] != b.isocalendar()[:2]
        raise ValueError(freq)

    def is_new_period(self, freq: str) -> bool:
        """True on the first bar of a new week ('W') or month ('M')."""
        if self._i == 0:
            return True
        a, b = self._data.index[self._i - 1], self.now
        if freq == "M":
            return (a.year, a.month) != (b.year, b.month)
        if freq == "W":
            return a.isocalendar()[:2] != b.isocalendar()[:2]
        raise ValueError(freq)
