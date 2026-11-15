from datetime import date
from datetime import time as dtime

import numpy as np
import pandas as pd

from firm.backtest.data import DataView, MarketData
from firm.backtest.engine import BacktestConfig, run_backtest
from firm.config import TEAMS
from firm.costs import Segment
from firm.features.intraday_model import (
    FEATURES, HOLD_BARS, build_training_set, forward_labels, model_features, train_model,
)
from firm.strategies.intraday_sandbox import CashOnly, ModelStrategy, RandomTrader

LIMITS = TEAMS["intraday_stocks"].limits
CFG = BacktestConfig(team="intraday_stocks", segment=Segment.EQ_INTRADAY, flatten_after=dtime(15, 10))


def five_min(days=8, k=6, seed=0, start="2026-03-02"):
    rng = np.random.default_rng(seed)
    idx = []
    for d in pd.bdate_range(start, periods=days):
        idx += list(pd.date_range(d + pd.Timedelta(hours=9, minutes=15), d + pd.Timedelta(hours=15, minutes=25),
                                  freq="5min"))
    idx = pd.DatetimeIndex(idx).tz_localize("Asia/Kolkata")
    frames = {}
    for j in range(k):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, len(idx))))
        o = np.r_[c[0], c[:-1]]
        frames[f"S{j}"] = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.001, "low": np.minimum(o, c) * 0.999,
                                        "close": c, "volume": rng.integers(1000, 5000, len(idx))}, index=idx)
    nifty = pd.Series(10_000 * np.exp(np.cumsum(rng.normal(0, 0.001, len(idx)))), index=idx)
    return MarketData.from_frames(frames), nifty


class Stub:
    """Predicts a fixed ladder from most negative (first stock) to most positive (last)."""

    def predict(self, X, pred_contrib=False):
        if pred_contrib:
            return np.zeros((len(X), len(FEATURES) + 1))
        return np.linspace(-0.01, 0.01, len(X))


def test_features_are_causal():
    data, nifty = five_min(days=8)
    full = model_features(data, nifty)["X"]
    cut = 300
    short = MarketData({n: p.iloc[:cut] for n, p in data.panels.items()})
    part = model_features(short, nifty.iloc[:cut])["X"]
    np.testing.assert_allclose(full[:cut], part, equal_nan=True, rtol=1e-5)


def test_labels_never_cross_days_and_match_hand_calculation():
    data, nifty = five_min(days=3)
    y = forward_labels(data, nifty)
    k = pd.Series(1, index=data.index).groupby(np.asarray(data.index.date)).cumsum().to_numpy() - 1
    assert np.isnan(y[k < 6]).all() and np.isnan(y[k > 62]).all()      # outside the entry window
    i = 10
    o = data.panels["open"].to_numpy()
    expect = o[i + 7] / o[i + 1] - 1 - (nifty.iloc[i + 6] / nifty.iloc[i] - 1)
    np.testing.assert_allclose(y[i], np.clip(expect, -0.05, 0.05))
    assert np.isfinite(y[i]).all()


def test_training_set_respects_dates_and_trains():
    data, nifty = five_min(days=12, k=8)
    days = sorted(set(data.index.date))
    X, y = build_training_set(data, nifty, start=days[4], end=days[8])
    assert len(X) == len(y) > 0 and X.shape[1] == len(FEATURES)
    X_all, _ = build_training_set(data, nifty)
    assert len(X) < len(X_all)
    model = train_model(X, y)
    assert len(model.predict(X[:5])) == 5


def run(strategy, data, **kw):
    return run_backtest(strategy, data, data.index[0].date(), date(2026, 12, 31), LIMITS, CFG, **kw)


def entries(res):
    return [f for f in res.fills if f.ts.time() < dtime(15, 10)]


def test_model_strategy_limits_threshold_and_exits():
    data, nifty = five_min(days=3)
    res = run(ModelStrategy(nifty, model=Stub()), data)
    day1 = [f for f in res.fills if f.ts.date() == data.index[0].date()]
    buys_sells = [(f.ts.time(), f.symbol, f.side.value) for f in day1]
    times = [t for t, _, _ in buys_sells]
    # Decisions on bars 09:45 and 09:50 enter 2 each (fills one bar later); 6 bars later they leave.
    assert times[:2] == [dtime(9, 50)] * 2 and times[2:4] == [dtime(9, 55)] * 2
    assert times[4:8] == [dtime(10, 20)] * 2 + [dtime(10, 25)] * 2
    assert len(day1) == 8                                         # 4 entries + 4 exits, one entry per stock
    assert {s for _, s, side in buys_sells[:2]} == {"S0", "S5"}    # the two largest predictions


def test_model_strategy_threshold_blocks_weak_predictions():
    class Weak(Stub):
        def predict(self, X, pred_contrib=False):
            return np.zeros((len(X), len(FEATURES) + 1)) if pred_contrib else np.full(len(X), 0.001)
    data, nifty = five_min(days=3)
    assert not run(ModelStrategy(nifty, model=Weak()), data).fills       # zero trades is valid


def test_warmup_blocks_trading_until_trade_from():
    data, nifty = five_min(days=6)
    days = sorted(set(data.index.date))
    res = run(ModelStrategy(nifty, model=Stub(), trade_from=days[3]), data)
    assert res.fills and min(f.ts.date() for f in res.fills) >= days[3]
    assert not run(ModelStrategy(nifty, model=None), data).fills          # no model, no trades


def test_model_strategy_explains_entries():
    data, nifty = five_min(days=2)
    s = ModelStrategy(nifty, model=Stub())
    view = DataView(data)
    for i in range(7):
        view._advance(i)
        s.target_weights(view)
    assert any("model expects" in r for r in s.last_reason)


def test_random_trader_is_reproducible_and_near_its_rate():
    data, nifty = five_min(days=40)
    a = run(RandomTrader(nifty), data)
    b = run(RandomTrader(nifty), data)
    assert [(f.ts, f.symbol, f.side) for f in a.fills] == [(f.ts, f.symbol, f.side) for f in b.fills]
    per_day = len(entries(a)) / 2 / 40
    assert 0.4 < per_day < 1.8                                           # about 0.02 x 57 bars a day


def test_cash_account_trades_nothing_but_earns_interest():
    data, nifty = five_min(days=5)
    res = run(CashOnly(), data)
    assert not res.fills and res.equity.iloc[-1] > res.equity.iloc[0]


def expensive(data, symbol="S5", factor=300.0):
    """The same data with one stock priced far above a position slot (about Rs 30,000 a share)."""
    return MarketData({f: p.assign(**{symbol: p[symbol] * (factor if f != "volume" else 1.0)})
                       for f, p in data.panels.items()})


def test_unaffordable_stocks_are_never_picked_so_no_phantom_positions():
    data, nifty = five_min(days=3)
    data = expensive(data)                                          # S5 is the stub's top pick but costs ~Rs 30,000
    res = run(ModelStrategy(nifty, model=Stub()), data)
    assert res.fills and all(f.symbol != "S5" for f in res.fills)   # skipped; the next best is used instead
    day1 = [f for f in res.fills if f.ts.date() == data.index[0].date()]
    assert len(day1) == 8                                            # still 4 real entries and 4 exits
    s = ModelStrategy(nifty, model=Stub())
    view = DataView(data)
    for i in range(7):
        view._advance(i)
        s.target_weights(view)
    assert "S5" not in s._pos and "S5" not in s._entered             # the strategy's own books agree with the broker


def test_random_trader_skips_unaffordable_stocks_too():
    data, nifty = five_min(days=40)
    res = run(RandomTrader(nifty), expensive(data))
    assert res.fills and all(f.symbol != "S5" for f in res.fills)
