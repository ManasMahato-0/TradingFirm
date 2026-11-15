import importlib.util
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from test_sandbox_live import Clock, FakeFyers, one_minute_day

from firm.backtest.data import MarketData
from firm.backtest.intraday import resample_5m
from firm.config import IST
from firm.live.sandbox import TrainData
from firm.live.sandbox_live import FyersSource

SYMS = [f"S{i}" for i in range(8)]
DAY = date(2026, 3, 24)


def load_script():
    spec = importlib.util.spec_from_file_location(
        "sandbox_script", Path(__file__).resolve().parent.parent / "scripts" / "sandbox.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_script"] = mod
    spec.loader.exec_module(mod)
    return mod


def synthetic_world():
    days = list(pd.bdate_range("2026-03-02", "2026-03-23").date)
    frames = {}
    for i, s in enumerate(SYMS + ["NIFTY50"]):
        frames[s] = pd.concat([resample_5m(one_minute_day(i, day=d, base=100 + i)) for d in days])
    stocks = MarketData.from_frames({s: frames[s] for s in SYMS})
    train = TrainData(stocks, frames["NIFTY50"]["close"], frames["NIFTY50"]["open"])
    history = (stocks, frames["NIFTY50"]["close"].reindex(stocks.index), frames["NIFTY50"]["open"].reindex(stocks.index))
    today = {s: one_minute_day(i, day=DAY, base=100 + i) for i, s in enumerate(SYMS + ["NIFTY50"])}
    return train, history, today


def patch(monkeypatch, mod, tmp_path, train, history):
    monkeypatch.setattr("firm.live.sandbox.load_train_data", lambda universe=None: (train, SYMS))
    monkeypatch.setattr(FyersSource, "build_history", classmethod(lambda cls, fyers, symbols, day, **kw: history))
    monkeypatch.setattr(mod, "paths", lambda shakedown: (tmp_path / "firm.db", tmp_path / "state"))


def rows(db, sql, params=()):
    with sqlite3.connect(db) as c:
        return c.execute(sql, params).fetchall()


def test_full_live_day_with_retry_then_login(monkeypatch, tmp_path):
    mod = load_script()
    train, history, today = synthetic_world()
    patch(monkeypatch, mod, tmp_path, train, history)
    clock = Clock(datetime(2026, 3, 24, 8, 45, tzinfo=IST))
    fyers = FakeFyers(clock, today)
    attempts = {"n": 0}

    def make_fyers():
        attempts["n"] += 1
        if attempts["n"] <= 2:                           # the first two probes: not logged in yet
            raise RuntimeError("Fyers history error for NIFTY50: Bad request")
        return fyers

    status = mod.run_live_day(DAY, True, lambda m: None, now_fn=clock.now, sleep_fn=clock.sleep,
                              make_fyers=make_fyers)
    db = tmp_path / "firm.db"
    assert status == "done"
    assert [r[0] for r in rows(db, "SELECT name FROM books ORDER BY name")] == \
        ["sandbox_A", "sandbox_B", "sandbox_C", "sandbox_D"]
    assert rows(db, "SELECT COUNT(*) FROM equity WHERE team='sandbox_D'")[0][0] == 75
    log = [r[0] for r in rows(db, "SELECT text FROM activity")]
    assert sum("Fyers login needed" in t for t in log) == 1          # one alert, not one per retry
    assert any("Fyers login OK" in t for t in log)
    assert any("retrained" in t or "Opened sandbox account" in t for t in log)
    assert (tmp_path / "state" / "models" / "sandbox_A.pkl").exists()   # A trained BEFORE the first bar
    assert list((tmp_path / "state" / "bars").glob("2026-03-24.parquet"))
    assert clock.now().time() > datetime(2026, 3, 24, 15, 30).time()    # it waited for every bar to close
    # Equity per account is recorded each bar; nobody holds anything at the end.
    pos = rows(db, "SELECT COUNT(*) FROM positions")[0][0]
    assert pos == 0


def test_no_login_by_0930_skips_the_day_and_says_so(monkeypatch, tmp_path):
    mod = load_script()
    train, history, today = synthetic_world()
    patch(monkeypatch, mod, tmp_path, train, history)
    clock = Clock(datetime(2026, 3, 24, 9, 31, tzinfo=IST))

    def no_login():
        raise RuntimeError("FYERS_CLIENT_ID and FYERS_ACCESS_TOKEN must be set in .env")

    status = mod.run_live_day(DAY, True, lambda m: None, now_fn=clock.now, sleep_fn=clock.sleep, make_fyers=no_login)
    assert status == "skipped"
    log = [r[0] for r in rows(tmp_path / "firm.db", "SELECT text FROM activity")]
    assert any("is skipped" in t and "flat" in t for t in log)
    from firm.storage import Storage
    assert Storage(tmp_path / "firm.db").conn.execute("SELECT COUNT(*) FROM fills").fetchone()[0] == 0


def test_should_run_gate():
    mod = load_script()
    from firm.calendar import TradingCalendar
    cal = TradingCalendar(holidays={2026: {date(2026, 3, 25)}})
    at = lambda d, h, m: datetime(d.year, d.month, d.day, h, m, tzinfo=IST)  # noqa: E731
    launch = date(2026, 3, 23)
    d = date(2026, 3, 24)
    assert mod.should_run(d, at(d, 8, 50), cal, set(), False, launch)
    assert not mod.should_run(d, at(d, 8, 30), cal, set(), False, launch)        # too early
    assert not mod.should_run(d, at(d, 15, 45), cal, set(), False, launch)       # after the window
    assert not mod.should_run(d, at(d, 9, 0), cal, {d}, False, launch)           # already ran
    assert not mod.should_run(date(2026, 3, 25), at(date(2026, 3, 25), 9, 0), cal, set(), False, launch)  # holiday
    assert not mod.should_run(date(2026, 3, 22), at(date(2026, 3, 22), 9, 0), cal, set(), False, launch)  # Sunday
    before = date(2026, 3, 20)
    assert not mod.should_run(before, at(before, 9, 0), cal, set(), False, launch)   # before launch: idle
    assert mod.should_run(before, at(before, 9, 0), cal, set(), True, launch)        # but the shake-down may run
