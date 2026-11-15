import pickle
from datetime import date
from datetime import time as dtime

import numpy as np
import pandas as pd
import pytest
from test_sandbox_strategies import CFG, LIMITS, Stub, five_min

from firm.activity import ActivityLog
from firm.backtest.engine import run_backtest
from firm.broker import Position
from firm.costs import Segment
from firm.features.intraday_model import model_features
from firm.live.sandbox import Sandbox, trade_from_date
from firm.live.sandbox_feed import ReplaySource
from firm.storage import Storage
from firm.strategies.intraday_sandbox import ModelStrategy


def setup(tmp_path, days=14, trade_from=None):
    data, nifty = five_min(days=days, k=8)
    nopen = nifty.shift(1).fillna(nifty.iloc[0])
    sb = Sandbox(tmp_path / "firm.db", tmp_path / "state", launch=sorted(set(data.index.date))[0],
                 trade_from=trade_from or date(2099, 1, 1))
    for name in ("sandbox_A", "sandbox_B"):
        sb.models.save(name, Stub(), {"trained_before": "2026-01-01", "rows": 1})
    return sb, data, nifty, nopen


def replay(sb, data, nifty, nopen, days, stop_after=None):
    all_days = sorted(set(data.index.date))
    for d in days:
        src = ReplaySource(data, nifty, nopen, d)
        if stop_after is not None and d == days[-1]:
            src = StopAfter(src, stop_after)
        sb.run_day(d, src, ensure=False)


class StopAfter:
    """Simulates the process dying mid-day."""

    def __init__(self, src, n):
        self.src, self.n, self.symbols = src, n, src.symbols

    def history(self):
        return self.src.history()

    def next_bar(self):
        if self.n == 0:
            raise KeyboardInterrupt("laptop died")
        self.n -= 1
        return self.src.next_bar()

    def quotes(self, symbols):
        return self.src.quotes(symbols)


def fills_of(db, book):
    c = Storage(db).conn
    return c.execute("SELECT ts, symbol, side, qty, price FROM fills WHERE team=? ORDER BY ts, symbol",
                     (book,)).fetchall()


def test_replayed_days_equal_a_backtest_of_the_same_days(tmp_path):
    sb, data, nifty, nopen = setup(tmp_path)
    days = sorted(set(data.index.date))[8:12]
    replay(sb, data, nifty, nopen, days)
    live = fills_of(tmp_path / "firm.db", "sandbox_A")
    # Backtest of the same four days (history before them is simply earlier bars in the data).
    strat = ModelStrategy(nifty, nopen, model=Stub(), label="A")
    res = run_backtest(strat, data, days[0], date(2099, 1, 1), LIMITS, CFG)
    res_days = [f for f in res.fills if f.ts.date() in days]
    want = [(f.ts.isoformat(), f.symbol, f.side.value, f.qty, f.price) for f in res_days]
    assert live and sorted(live) == sorted(want)
    assert any(r[2] == "SELL" for r in live) and any(r[2] == "BUY" for r in live)


def test_blank_account_is_flat_during_warmup_and_cash_account_never_trades(tmp_path):
    sb, data, nifty, nopen = setup(tmp_path)
    replay(sb, data, nifty, nopen, sorted(set(data.index.date))[8:10])
    assert not fills_of(tmp_path / "firm.db", "sandbox_B")      # warm-up: no trades yet
    assert not fills_of(tmp_path / "firm.db", "sandbox_D")
    assert fills_of(tmp_path / "firm.db", "sandbox_A")


def test_restart_mid_day_resumes_without_double_trading(tmp_path):
    full_dir = tmp_path / "full"
    full_dir.mkdir()
    sb, data, nifty, nopen = setup(full_dir)
    days = sorted(set(data.index.date))[8:10]
    replay(sb, data, nifty, nopen, days)
    want = fills_of(full_dir / "firm.db", "sandbox_A")

    cut_dir = tmp_path / "cut"
    cut_dir.mkdir()
    sb2, *_ = setup(cut_dir)
    with pytest.raises(KeyboardInterrupt):
        replay(sb2, data, nifty, nopen, days, stop_after=20)      # dies 20 bars into the second day
    sb3 = Sandbox(cut_dir / "firm.db", cut_dir / "state", launch=days[0], trade_from=date(2099, 1, 1))
    sb3.run_day(days[1], ReplaySource(data, nifty, nopen, days[1]), ensure=False)
    assert fills_of(cut_dir / "firm.db", "sandbox_A") == want


def test_overnight_guard_closes_stale_positions_with_an_alert(tmp_path):
    sb, data, nifty, nopen = setup(tmp_path)
    days = sorted(set(data.index.date))
    replay(sb, data, nifty, nopen, [days[8]])
    # Simulate a crash that left a position overnight.
    with open(sb._path("sandbox_C"), "rb") as f:
        acct = pickle.load(f)
    acct.session.account.positions["S1"] = Position("S1", Segment.EQ_INTRADAY, 10, 100.0)
    acct.session.account.cash -= 1000
    with open(sb._path("sandbox_C"), "wb") as f:
        pickle.dump(acct, f)
    sb2 = Sandbox(tmp_path / "firm.db", tmp_path / "state", launch=days[0], trade_from=date(2099, 1, 1))
    sb2.run_day(days[9], ReplaySource(data, nifty, nopen, days[9]), ensure=False)
    with open(sb2._path("sandbox_C"), "rb") as f:
        assert "S1" not in pickle.load(f).session.account.positions
    alerts = [r for r in ActivityLog(tmp_path / "firm.db").recent(500) if r[3] == "alert"]
    assert any("Overnight guard" in a[5] for a in alerts)


def test_every_account_flat_at_day_end_and_books_registered(tmp_path):
    sb, data, nifty, nopen = setup(tmp_path)
    replay(sb, data, nifty, nopen, sorted(set(data.index.date))[8:10])
    for acct in sb.accounts.values():
        assert not acct.session.account.positions
    books = Storage(tmp_path / "firm.db").conn.execute("SELECT name, status FROM books ORDER BY name").fetchall()
    assert books == [("sandbox_A", "sandbox"), ("sandbox_B", "sandbox"), ("sandbox_C", "sandbox"),
                     ("sandbox_D", "sandbox")]
    assert (tmp_path / "state" / "bars").exists() and list((tmp_path / "state" / "bars").glob("*.parquet"))
    q = Storage(tmp_path / "firm.db").conn.execute("SELECT COUNT(*) FROM sandbox_quotes").fetchone()[0]
    assert q > 0                                                # decision-time quotes are logged


def test_short_history_window_gives_the_same_features_as_full_history():
    data, nifty = five_min(days=45, k=8, seed=4)
    days = sorted(set(data.index.date))
    keep = np.isin(data.index.date, days[-26:])               # 25 prior days + the last day
    from firm.backtest.data import MarketData
    short = MarketData({f: p.loc[keep] for f, p in data.panels.items()})
    full = model_features(data, nifty)["X"][keep]
    part = model_features(short, nifty.loc[keep])["X"]
    last_day = np.asarray(short.index.date) == days[-1]
    np.testing.assert_allclose(full[last_day], part[last_day], equal_nan=True, rtol=1e-4, atol=1e-6)


def test_b_warmup_date_counts_ten_trading_days():
    from firm.calendar import TradingCalendar
    cal = TradingCalendar(holidays={2026: set()})
    d = trade_from_date(date(2026, 11, 23), cal)
    assert d == date(2026, 12, 7)                              # Mon 23 Nov + 10 trading days


def test_entry_threshold_comes_from_the_models_own_prediction_scale(tmp_path):
    from firm.live.sandbox import entry_threshold

    class Ladder:
        def predict(self, X):
            return np.linspace(-0.001, 0.001, len(X))

    th = entry_threshold(Ladder(), np.zeros((10_000, 3)))
    assert th == pytest.approx(0.001 * 0.999, rel=1e-3)             # 99.9th percentile of |prediction|


def test_run_day_applies_each_models_saved_threshold(tmp_path):
    sb, data, nifty, nopen = setup(tmp_path)
    days = sorted(set(data.index.date))
    sb.models.save("sandbox_A", Stub(), {"trained_before": "x", "rows": 1, "entry_threshold": 0.02})   # above Stub's 0.01
    sb.models.save("sandbox_B", Stub(), {"trained_before": "x", "rows": 1, "entry_threshold": 0.005})
    sb2 = Sandbox(tmp_path / "firm.db", tmp_path / "state", launch=days[0], trade_from=days[0])
    sb2.run_day(days[8], ReplaySource(data, nifty, nopen, days[8]), ensure=False)
    assert not fills_of(tmp_path / "firm.db", "sandbox_A")           # its bar is higher than any prediction
    assert fills_of(tmp_path / "firm.db", "sandbox_B")               # its bar is lower: it trades
