from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from firm.backtest.data import MarketData
from firm.backtest.intraday import resample_5m
from firm.config import IST
from firm.live.sandbox_feed import BAR
from firm.live.sandbox_live import FyersSource, probe_token

DAY = date(2026, 3, 9)
SYMS = ["AAA", "BBB", "CCC"]


def one_minute_day(seed: int, day=DAY, base=100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(datetime.combine(day, datetime.min.time()) + timedelta(hours=9, minutes=15),
                        periods=375, freq="1min", tz=IST)
    c = base * np.exp(np.cumsum(rng.normal(0, 0.0007, len(idx))))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.0003, "low": np.minimum(o, c) * 0.9997,
                         "close": c, "volume": rng.integers(100, 900, len(idx)).astype(float)}, index=idx)


class Clock:
    def __init__(self, start):
        self.t = start

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += timedelta(seconds=s)


class FakeFyers:
    """Serves 1-minute bars as the clock advances. Includes the minute still forming, like Fyers may."""

    def __init__(self, clock, frames, fail=()):
        self.clock, self.frames, self.fail, self.calls = clock, frames, set(fail), 0

    def history(self, symbol, interval, start, end):
        self.calls += 1
        if symbol in self.fail:
            raise RuntimeError("Bad request")
        f = self.frames[symbol]
        now = pd.Timestamp(self.clock.now()) if self.clock else pd.Timestamp.max.tz_localize(IST)
        f = f[(f.index.date >= start) & (f.index.date < end) & (f.index <= now)]
        if f.empty:                                   # like the real client: a plain RangeIndex, no candles
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return f

    def quotes(self, symbols):
        return {s: 123.0 for s in symbols}

    def nse_cash_status(self):
        return "OPEN"


def empty_history():
    idx = pd.DatetimeIndex([], tz=IST, name="ts")
    h = MarketData({f: pd.DataFrame(columns=SYMS, index=idx, dtype=float) for f in
                    ("open", "high", "low", "close", "volume")})
    return h, pd.Series(dtype=float, index=idx), pd.Series(dtype=float, index=idx)


def make_source(fail=(), start_offset=timedelta(0)):
    frames = {s: one_minute_day(i) for i, s in enumerate(SYMS + ["NIFTY50"])}
    clock = Clock(datetime.combine(DAY, datetime.min.time(), IST) + timedelta(hours=9, minutes=15) + start_offset)
    fyers = FakeFyers(clock, frames, fail)
    src = FyersSource(fyers, SYMS, DAY, empty_history(), now=clock.now, sleep=clock.sleep, max_per_minute=400)
    return src, frames, clock, fyers


def test_live_bars_match_training_bars_and_never_use_the_forming_bar():
    src, frames, clock, fyers = make_source()
    got = []
    while (b := src.next_bar()) is not None:
        got.append(b)
        # A bar is handed over only after it closed.
        assert clock.now() >= (b.ts + BAR).to_pydatetime()
    assert len(got) == 75 and got[0].ts.time().hour == 9 and got[-1].ts.strftime("%H:%M") == "15:25"
    full = {s: resample_5m(frames[s]) for s in SYMS + ["NIFTY50"]}
    for b in got:
        for s in SYMS:
            pd.testing.assert_series_equal(b.ohlcv.loc[s], full[s].loc[b.ts], check_names=False, rtol=1e-9)
        assert b.nifty_close == pytest.approx(full["NIFTY50"].loc[b.ts, "close"])
        assert b.nifty_open == pytest.approx(full["NIFTY50"].loc[b.ts, "open"])


def test_request_budget_is_about_one_sweep_per_bar():
    src, _, clock, fyers = make_source()
    while src.next_bar() is not None:
        pass
    assert fyers.calls <= 75 * 4 * 1.2                  # 3 stocks + Nifty per bar, plus a few retries


def test_a_symbol_that_keeps_failing_is_carried_forward_flat_not_fatal():
    src, frames, _, _ = make_source(fail=["CCC"])
    src._prev_close = pd.Series({"AAA": 100.0, "BBB": 100.0, "CCC": 100.0})   # yesterday's close, as real history has
    b = None
    for _ in range(8):
        b = src.next_bar()
    assert "CCC" in src.failed
    row = b.ohlcv.loc["CCC"]
    assert row["volume"] == 0 and row["open"] == row["close"] == 100.0
    assert b.ohlcv.loc["AAA", "volume"] > 0


def test_starting_late_catches_up_on_missed_bars_in_order():
    src, _, clock, _ = make_source(start_offset=timedelta(minutes=47))
    first = src.next_bar()
    second = src.next_bar()
    assert first.ts.strftime("%H:%M") == "09:15" and second.ts.strftime("%H:%M") == "09:20"


def test_quotes_pass_through_and_survive_errors():
    src, _, _, fyers = make_source()
    assert src.quotes(["AAA"]) == {"AAA": 123.0}
    fyers.quotes = lambda s: (_ for _ in ()).throw(RuntimeError("boom"))
    assert src.quotes(["AAA"]) == {}


def test_token_probe_treats_any_failure_as_please_log_in():
    class Dead:
        def nse_cash_status(self):
            raise RuntimeError("Fyers history error for NIFTY50: Bad request")

    ok, msg = probe_token(lambda: Dead())
    assert not ok and "Bad request" in msg
    ok, msg = probe_token(lambda: FakeFyers(None, {}))
    assert ok and "OPEN" in msg

    def boom():
        raise RuntimeError("FYERS_CLIENT_ID and FYERS_ACCESS_TOKEN must be set in .env")
    assert not probe_token(boom)[0]


def stored_history(days_end: date, n_days=3):
    """Pretend the recorder store holds `n_days` days ending at `days_end` (5-minute bars)."""
    frames = {}
    for i, s_ in enumerate(SYMS + ["NIFTY50"]):
        parts = [resample_5m(one_minute_day(i, day=days_end - timedelta(days=k))) for k in range(n_days - 1, -1, -1)]
        frames[s_] = pd.concat(parts)
    stocks = MarketData.from_frames({s_: f for s_, f in frames.items() if s_ != "NIFTY50"})
    nifty = frames["NIFTY50"]
    raw = MarketData.from_frames({**{s_: frames[s_] for s_ in SYMS}, "NIFTY50": nifty})
    return raw


class CountingFyers(FakeFyers):
    pass


def test_history_does_not_fetch_when_only_a_weekend_separates_the_days():
    from firm.calendar import TradingCalendar
    cal = TradingCalendar(holidays={2026: set()})
    friday, monday = date(2026, 3, 6), date(2026, 3, 9)
    raw = stored_history(friday, n_days=3)
    clock = Clock(datetime.combine(monday, datetime.min.time(), IST) + timedelta(hours=8))
    fyers = FakeFyers(clock, {s_: one_minute_day(0, day=monday) for s_ in SYMS + ["NIFTY50"]})
    h, nc, no = FyersSource.build_history(fyers, SYMS, monday, raw=raw, calendar=cal)
    assert fyers.calls == 0                            # Friday is stored: Sat and Sun are not "missing"
    assert len(h.index) == len(nc) == len(no) > 0 and h.index[-1].date() == friday


def test_history_patches_a_missing_trading_day_from_fyers():
    from firm.calendar import TradingCalendar
    cal = TradingCalendar(holidays={2026: set()})
    thursday, friday, monday = date(2026, 3, 5), date(2026, 3, 6), date(2026, 3, 9)
    raw = stored_history(thursday, n_days=2)           # the recorder missed Friday
    clock = Clock(datetime.combine(monday, datetime.min.time(), IST) + timedelta(hours=8))
    fyers = FakeFyers(clock, {s_: pd.concat([one_minute_day(i, day=friday), one_minute_day(i, day=monday)])
                              for i, s_ in enumerate(SYMS + ["NIFTY50"])})
    h, nc, no = FyersSource.build_history(fyers, SYMS, monday, raw=raw, calendar=cal)
    assert fyers.calls == len(SYMS) + 1 and h.index[-1].date() == friday
    assert nc.notna().all() and len(nc) == len(h.index)


def test_empty_fyers_response_is_harmless():
    assert resample_5m(pd.DataFrame(columns=["open", "high", "low", "close", "volume"])).empty
