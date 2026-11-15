from datetime import date
from datetime import time as dtime

import numpy as np
import pandas as pd

from firm.backtest.data import MarketData
from firm.backtest.engine import BacktestConfig, run_backtest
from firm.backtest.intraday import resample_5m
from firm.config import TEAMS
from firm.costs import Segment
from firm.strategies.intraday import ORB, Gap, intraday_features

LIMITS = TEAMS["intraday_stocks"].limits
CFG = BacktestConfig(team="intraday_stocks", segment=Segment.EQ_INTRADAY, flatten_after=dtime(15, 10))


def five_min(days=3, k=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = []
    for d in pd.bdate_range("2026-03-02", periods=days):
        idx += list(pd.date_range(d + pd.Timedelta(hours=9, minutes=15), d + pd.Timedelta(hours=15, minutes=25),
                                  freq="5min"))
    idx = pd.DatetimeIndex(idx).tz_localize("Asia/Kolkata")
    frames = {}
    for j in range(k):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, len(idx))))
        o = np.r_[c[0], c[:-1]]
        frames[f"S{j}"] = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.001,
                                        "low": np.minimum(o, c) * 0.999, "close": c,
                                        "volume": rng.integers(1000, 5000, len(idx))}, index=idx)
    return MarketData.from_frames(frames)


class AlwaysLong:
    name, params = "always_long", {}

    def target_weights(self, view):
        return {"S0": 0.5, "S1": -0.3}


def test_resample_5m_labels_and_session():
    idx = pd.date_range("2026-03-02 09:15", "2026-03-02 15:29", freq="1min", tz="Asia/Kolkata")
    df = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10}, index=idx)
    b = resample_5m(df)
    assert b.index[0].time() == dtime(9, 15) and b.index[-1].time() == dtime(15, 25)
    assert len(b) == 75 and b.volume.iloc[0] == 50


def test_never_holds_overnight_and_squares_off_by_1515():
    data = five_min()
    res = run_backtest(AlwaysLong(), data, date(2026, 3, 2), date(2026, 3, 10), LIMITS, CFG)
    fills = pd.DataFrame([(f.ts, f.symbol, f.side.value, f.qty) for f in res.fills], columns=["ts", "sym", "side", "qty"])
    fills["signed"] = np.where(fills.side == "BUY", fills.qty, -fills.qty)
    for d, g in fills.groupby(fills.ts.dt.date):
        assert (g.groupby("sym").signed.sum() == 0).all()            # flat at day end
        exits = g[g.ts.dt.time >= dtime(15, 10)]
        assert len(exits) and exits.ts.dt.time.max() <= dtime(15, 15)  # squared off at 15:15 open
    # Each day re-enters at the second bar, never carrying the position overnight.
    assert fills[fills.ts.dt.time == dtime(9, 20)].groupby(fills.ts.dt.date).size().min() == 2


def test_opening_range_uses_only_first_30_minutes():
    data = five_min(days=2)
    f = intraday_features(data)
    k = f["k"]
    assert f["or_high"][k < 6].isna().all().all()                    # unknown until complete
    day1 = data.index.date == data.index[0].date()
    expected = data.panels["high"][day1].iloc[:6].max()
    pd.testing.assert_series_equal(f["or_high"][day1].iloc[6], expected, check_names=False)
    cut = 100
    short = MarketData({n: p.iloc[:cut] for n, p in data.panels.items()})
    part = intraday_features(short)
    for key in ("or_high", "vwap", "prev_close"):
        pd.testing.assert_frame_equal(f[key].iloc[:cut], part[key])


def test_orb_enters_each_stock_at_most_once_a_day():
    data = five_min(days=3, k=3, seed=3)
    res = run_backtest(ORB(vol_mult=0.0), data, date(2026, 3, 2), date(2026, 3, 10), LIMITS, CFG)
    opens = {}
    for f in res.fills:
        if f.ts.time() < dtime(15, 10):
            opens.setdefault((f.ts.date(), f.symbol), []).append(f)
    # Entry plus at most one exit before the square-off.
    assert all(len(v) <= 2 for v in opens.values())


def test_gap_needs_previous_close():
    data = five_min(days=2)
    s = Gap(None, mode="fade", min_gap=0.0)
    res = run_backtest(s, data, date(2026, 3, 2), date(2026, 3, 10), LIMITS, CFG)
    assert all(f.ts.date() != date(2026, 3, 2) for f in res.fills)   # day 1 has no prior close


def breakout_data(symbols=("A",), days=8, nifty_last=101.0):
    """Flat 100 every day; on the last day each stock jumps to 102 on 10x volume at 09:45
    (the first bar after the opening range). Returns (data, nifty)."""
    idx = []
    for d in pd.bdate_range("2026-03-02", periods=days):
        idx += list(pd.date_range(d + pd.Timedelta(hours=9, minutes=15), d + pd.Timedelta(hours=15, minutes=25),
                                  freq="5min"))
    idx = pd.DatetimeIndex(idx).tz_localize("Asia/Kolkata")
    last = idx[-1].date()
    frames = {}
    for sym in symbols:
        o = np.full(len(idx), 100.0)
        c = np.full(len(idx), 100.0)
        v = np.full(len(idx), 1000.0)
        for n, t in enumerate(idx):
            if t.date() == last and n >= len(idx) - 75 + 6:
                c[n] = 102.0
                o[n] = 100.0 if n == len(idx) - 75 + 6 else 102.0
        v[len(idx) - 75 + 6] = 10_000.0
        frames[sym] = pd.DataFrame({"open": o, "high": np.maximum(o, c) + 0.1, "low": np.minimum(o, c) - 0.1,
                                    "close": c, "volume": v}, index=idx)
    nifty = pd.Series(100.0, index=idx)
    nifty[nifty.index.date == last] = nifty_last
    return MarketData.from_frames(frames), nifty


def entry_fills(res):
    return [f for f in res.fills if f.ts.time() < dtime(15, 10)]


def run_orb(data, nifty, **kw):
    start, end = data.index[0].date(), date(2026, 12, 31)
    return run_backtest(ORB(nifty, **kw), data, start, end, LIMITS, CFG)


def test_orb_defaults_unchanged_and_params_not_polluted():
    data, nifty = breakout_data(("A", "B"))
    base = ORB(nifty)
    assert "market_filter" not in base.params and "max_entries_per_day" not in base.params
    assert len(entry_fills(run_orb(data, nifty))) == 2           # both stocks enter, no cap, no filter


def test_daily_entry_cap_is_across_all_stocks():
    data, nifty = breakout_data(("A", "B"))
    res = run_orb(data, nifty, max_entries_per_day=1)
    assert len(entry_fills(res)) == 1                            # two signals, one entry
    assert ORB(nifty, max_entries_per_day=1).params["max_entries_per_day"] == 1


def test_market_filter_blocks_longs_when_nifty_is_down():
    up, nifty_up = breakout_data(nifty_last=101.0)
    down, nifty_down = breakout_data(nifty_last=99.0)
    assert len(entry_fills(run_orb(up, nifty_up, market_filter=True))) == 1     # Nifty up: long allowed
    assert len(entry_fills(run_orb(down, nifty_down, market_filter=True))) == 0  # Nifty down: no long
    assert len(entry_fills(run_orb(down, nifty_down))) == 1                      # unfiltered still enters
