import pickle
from datetime import date, datetime

import numpy as np
import pandas as pd

from firm.activity import ActivityLog
from firm.backtest.data import DataView, MarketData
from firm.backtest.engine import BacktestConfig, Session, run_backtest
from firm.calendar import TradingCalendar
from firm.config import IST, TEAMS
from firm.live.runner import BookSpec, LiveBook, LiveContext, run_books
from firm.storage import Storage
from firm.strategies.baselines import Momentum

LIMITS = TEAMS["swing_stocks"].limits
CAL = TradingCalendar(holidays={2025: set(), 2026: {date(2026, 1, 26)}})


def random_data(n=320, k=12, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-01", periods=n, tz="Asia/Kolkata", name="ts")
    frames = {}
    for j in range(k):
        c = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
        o = c * (1 + rng.normal(0, 0.003, n))
        frames[f"S{j}"] = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.01,
                                        "low": np.minimum(o, c) * 0.99, "close": c,
                                        "volume": 1000}, index=idx)
    return MarketData.from_frames(frames)


def test_calendar_month_end_and_holidays():
    assert not CAL.is_trading_day(date(2026, 1, 26))          # Republic Day
    assert CAL.next_trading_day(date(2026, 1, 23)) == date(2026, 1, 27)
    assert CAL.is_month_end(date(2026, 1, 30)) and not CAL.is_month_end(date(2026, 1, 29))


def test_period_end_on_latest_bar_uses_calendar():
    idx = pd.bdate_range("2026-01-26", "2026-01-28", tz="Asia/Kolkata")  # ends on a Wednesday
    data = MarketData.from_frames({"A": pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0,
                                                      "close": 1.0, "volume": 1}, index=idx)})
    plain, live = DataView(data), DataView(data, next_trading_day=CAL.next_trading_day)
    plain._advance(len(idx) - 1)
    live._advance(len(idx) - 1)
    assert plain.is_period_end("M")       # backtest: end of data counts as period end
    assert not live.is_period_end("M")    # live: Jan 28 is not the last trading day


def test_session_with_daily_pickle_matches_backtest():
    """Live = stepping one day per run with state saved in between. Must equal the backtest."""
    data = random_data()
    start, end = date(2026, 1, 1), date(2027, 1, 1)
    bt = run_backtest(Momentum(top_n=3, lookback=60, skip=5), data, start, end, LIMITS)

    session = Session(Momentum(top_n=3, lookback=60, skip=5), LIMITS, BacktestConfig())
    opens, marks = data.panels["open"], data.panels["close"].ffill()
    fills, equity = [], []
    for i, t in enumerate(data.index):
        if not start <= t.date() < end:
            continue
        view = DataView(data)  # fresh view each "evening", like the live runner
        r = session.step(view, i, opens, marks)
        fills += r.fills
        equity.append(r.equity)
        session = pickle.loads(pickle.dumps(session))
    assert len(fills) == len(bt.fills) > 0
    assert [(f.symbol, f.side, f.qty, f.price) for f in fills] == \
        [(f.symbol, f.side, f.qty, f.price) for f in bt.fills]
    assert np.allclose(equity, bt.equity.to_numpy())


def _spec(start):
    return BookSpec(name="test_mom", team="swing_stocks", label="Test momentum", status="shadow",
                    capital=100_000.0, start=start, agent="Momentum (test)",
                    factory=lambda ctx: Momentum(top_n=3, lookback=60, skip=5))


def test_live_book_catches_up_and_logs(tmp_path):
    data = random_data()
    ctx = LiveContext(data, pd.Series(dtype=float), None, {}, None, CAL)
    db = tmp_path / "firm.db"
    mid = data.index[260].date()
    run_books([_spec(date(2026, 1, 1))], db, tmp_path, ctx=ctx,
              now=datetime.combine(mid, datetime.min.time(), IST).replace(hour=17))
    out = run_books([_spec(date(2026, 1, 1))], db, tmp_path, ctx=ctx,
                    now=datetime(2027, 1, 1, 17, tzinfo=IST))
    assert out[0]["last_bar"] == data.index[-1].date()

    st = Storage(db)
    eq = st.conn.execute("SELECT COUNT(*) FROM equity WHERE team='test_mom'").fetchone()[0]
    n_days = sum(1 for t in data.index if t.date() >= date(2026, 1, 1))
    assert eq == n_days  # every day once, no gaps or repeats across runs
    log = ActivityLog(db)
    kinds = {k for _, _, _, k, _, _ in log.recent(1000)}
    assert {"think", "talk", "act"} <= kinds

    # Two runs equal one backtest over the same days.
    bt = run_backtest(Momentum(top_n=3, lookback=60, skip=5), data, date(2026, 1, 1), date(2027, 1, 1), LIMITS)
    n_fills = st.conn.execute("SELECT COUNT(*) FROM fills WHERE team='test_mom'").fetchone()[0]
    assert n_fills == len(bt.fills)


def test_live_book_skips_todays_bar_before_close(tmp_path):
    data = random_data(n=300)
    last = data.index[-1].date()
    ctx = LiveContext(data, pd.Series(dtype=float), None, {}, None, CAL)
    out = run_books([_spec(date(2026, 1, 1))], tmp_path / "f.db", tmp_path, ctx=ctx,
                    now=datetime.combine(last, datetime.min.time(), IST).replace(hour=11))
    assert out[0]["last_bar"] < last


def test_book_does_not_open_or_train_before_start(tmp_path):
    data = random_data(n=300)
    ctx = LiveContext(data, pd.Series(dtype=float), None, {}, None, CAL)
    future = data.index[-1].date() + pd.Timedelta(days=14)
    out = run_books([_spec(future)], tmp_path / "f.db", tmp_path, ctx=ctx,
                    now=datetime.combine(data.index[-1].date(), datetime.min.time(), IST).replace(hour=17))
    assert out[0]["days"] == 0 and not (tmp_path / "test_mom").exists()
