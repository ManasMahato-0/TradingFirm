"""Live bar source for the intraday sandbox: polls Fyers 1-minute bars and builds 5-minute
bars with `resample_5m`, the same function that built the training data. The still-forming
bar is never used. Everything that touches the network or the clock is injectable, so the
whole loop is tested against a fake Fyers; the real thing is verified at the first live login.

Request budget: about 51 history calls per bar, paced under `max_per_minute` (default 40) and
started 45 seconds after each bar closes, away from the recorder's option-chain burst at the
bar boundary. Expect the bar to be ready about two minutes after it closes; the decision-time
quote is logged so that delay is measured, not guessed.
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from datetime import time as dtime
from typing import Callable

import numpy as np
import pandas as pd

from firm.backtest.data import MarketData
from firm.backtest.intraday import load_5m, resample_5m
from firm.config import IST
from firm.live.sandbox_feed import BAR, HISTORY_DAYS, Bar

OPEN = dtime(9, 15)
N_BARS = 75
START_DELAY = pd.Timedelta(seconds=45)
FIELDS = ("open", "high", "low", "close", "volume")


def probe_token(make_fyers: Callable[[], object]) -> tuple[bool, str]:
    """True only if Fyers answers a simple authenticated call. Any failure means 'please log in';
    an expired token can come back as a plain 'Bad request', not a clean auth error."""
    try:
        status = make_fyers().nse_cash_status()
        return True, f"Fyers is reachable (NSE cash market: {status})"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:160]}"


class FyersSource:
    def __init__(self, fyers, symbols: list[str], day: date, history: tuple[MarketData, pd.Series, pd.Series],
                 now: Callable[[], datetime] = lambda: datetime.now(IST),
                 sleep: Callable[[float], None] = time.sleep, max_per_minute: int = 40,
                 log: Callable[[str], None] = lambda m: None) -> None:
        self.fyers, self.symbols, self.day, self._hist = fyers, list(symbols), day, history
        self.now, self.sleep, self.log = now, sleep, log
        self.pause = 60.0 / max_per_minute
        self._k = 0
        self._cache: dict[str, pd.DataFrame] = {}
        self._fetch_started: datetime | None = None
        h = history[0]
        self._prev_close = h.panels["close"].ffill().iloc[-1] if len(h.index) else pd.Series(dtype=float)
        self.failed: set[str] = set()

    # ------------------------------------------------------------ history

    @classmethod
    def build_history(cls, fyers, symbols: list[str], day: date, nifty: str = "NIFTY50",
                      days: int = HISTORY_DAYS, log=lambda m: None, raw: MarketData | None = None,
                      calendar=None) -> tuple[MarketData, pd.Series, pd.Series]:
        """Prior trading days from the recorded store; days the recorder has not stored yet
        (it saves bars after 15:45) are fetched from Fyers and appended in memory.
        A weekend or holiday gap is not "missing": only the previous TRADING day counts."""
        from firm.calendar import TradingCalendar

        raw = raw if raw is not None else load_5m(symbols + [nifty])
        cal = calendar or TradingCalendar(refresh=False)
        have = sorted(d for d in set(raw.index.date) if d < day)
        stocks = raw.without([nifty])
        keep = np.isin(raw.index.date, have[-days:])
        h = MarketData({f: p.loc[keep] for f, p in stocks.panels.items()})
        nc, no = raw.panels["close"][nifty].loc[keep], raw.panels["open"][nifty].loc[keep]
        last = have[-1] if have else day - timedelta(days=days)
        prev_trading = day - timedelta(days=1)
        while not cal.is_trading_day(prev_trading):
            prev_trading -= timedelta(days=1)
        missing_from = last + timedelta(days=1)
        if last < prev_trading and fyers is not None:
            frames = {}
            for s in symbols + [nifty]:
                bars = resample_5m(fyers.history(s, "1m", missing_from, day))
                if len(bars):
                    frames[s] = bars
            if frames:
                log(f"history: patched {missing_from} to {day - timedelta(days=1)} from Fyers")
                add = MarketData.from_frames({s: f for s, f in frames.items() if s != nifty})
                h = MarketData({f: pd.concat([h.panels[f], add.panels[f].reindex(columns=h.panels[f].columns)])
                                for f in h.panels})
                if nifty in frames:
                    nc = pd.concat([nc, frames[nifty]["close"]])
                    no = pd.concat([no, frames[nifty]["open"]])
        return h, nc.reindex(h.index), no.reindex(h.index)

    def history(self):
        return self._hist

    # ------------------------------------------------------------ live bars

    def _start(self, k: int) -> pd.Timestamp:
        return pd.Timestamp(datetime.combine(self.day, OPEN), tz=IST) + k * BAR

    def _fetch(self) -> None:
        """One sweep: today's 1-minute bars per symbol (and Nifty) -> 5-minute bars."""
        tomorrow = self.day + timedelta(days=1)
        self._fetch_started = self.now()
        got: dict[str, pd.DataFrame] = {}
        for s in self.symbols + ["NIFTY50"]:
            for attempt in range(3):
                self.sleep(self.pause)
                try:
                    got[s] = resample_5m(self.fyers.history(s, "1m", self.day, tomorrow))
                    self.failed.discard(s)
                    break
                except Exception as e:  # transient error: retry, then carry the symbol forward flat
                    if attempt == 2:
                        self.failed.add(s)
                        self.log(f"no data for {s} this sweep: {type(e).__name__}: {str(e)[:80]}")
        self._cache = got

    def _complete(self, k: int) -> bool:
        """Bar k is usable only if the sweep that produced it STARTED after the bar closed;
        a sweep made earlier may hold a half-built, still-forming bar."""
        nifty = self._cache.get("NIFTY50")
        if nifty is None or self._fetch_started is None or self._start(k) not in nifty.index:
            return False
        return self._fetch_started >= (self._start(k) + BAR).to_pydatetime()

    def next_bar(self) -> Bar | None:
        if self._k >= N_BARS:
            return None
        t = self._start(self._k)
        ready = (t + BAR + START_DELAY).to_pydatetime()
        while self.now() < ready:
            self.sleep(min(20.0, max(1.0, (ready - self.now()).total_seconds())))
        if not self._complete(self._k):
            self._fetch()
        if not self._complete(self._k):                      # Nifty bar not there yet: wait a little, try once more
            self.sleep(15.0)
            self._fetch()
            if not self._complete(self._k):
                self.log(f"bar {t:%H:%M} not available from Fyers; skipping it")
                self._k += 1
                return self.next_bar()
        rows = {}
        for s in self.symbols:
            f = self._cache.get(s)
            if f is not None and t in f.index:
                rows[s] = f.loc[t, list(FIELDS)]
            else:                                            # no trades in this bar: flat bar at the last price
                px = self._last_price(s, t)
                rows[s] = pd.Series({"open": px, "high": px, "low": px, "close": px, "volume": 0.0})
        nf = self._cache["NIFTY50"]
        self._k += 1
        return Bar(t, pd.DataFrame(rows).T[list(FIELDS)], float(nf.loc[t, "open"]), float(nf.loc[t, "close"]))

    def _last_price(self, s: str, t: pd.Timestamp) -> float:
        f = self._cache.get(s)
        if f is not None:
            earlier = f[f.index < t]
            if len(earlier):
                return float(earlier["close"].iloc[-1])
        return float(self._prev_close.get(s, np.nan))

    def quotes(self, symbols: list[str]) -> dict[str, float]:
        try:
            return self.fyers.quotes(symbols)
        except Exception as e:
            self.log(f"quotes failed: {type(e).__name__}: {str(e)[:80]}")
            return {}
