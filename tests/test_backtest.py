from datetime import date

import numpy as np
import pandas as pd
import pytest

from firm.backtest.data import MarketData
from firm.backtest.engine import BacktestConfig, run_backtest
from firm.backtest.lab import Lab, LockedPeriodError, make_folds
from firm.backtest.metrics import compute_metrics, deflated_sharpe
from firm.backtest.registry import TrialRegistry
from firm.broker import PaperBroker
from firm.costs import Segment
from firm.risk import RiskLimits

LIMITS = RiskLimits(
    allowed_segments=frozenset({Segment.EQ_DELIVERY}), allow_short=False,
    max_position_pct=1.0, max_gross_exposure_pct=1.0, max_daily_loss_pct=0.5,
    max_orders_per_day=100, max_data_age_seconds=10 * 86400,
)


def make_data(n=10, symbols=("A", "B")):
    idx = pd.date_range("2026-01-01", periods=n, freq="B", tz="Asia/Kolkata", name="ts")
    frames = {}
    for k, s in enumerate(symbols):
        base = 100.0 * (k + 1) + np.arange(n) * 10  # opens 100, 110, ... ; closes = open + 5
        frames[s] = pd.DataFrame({"open": base, "high": base + 6, "low": base - 1,
                                  "close": base + 5, "volume": 1000}, index=idx)
    return MarketData.from_frames(frames)


def no_slip():
    return PaperBroker(slippage_bps={s: 0.0 for s in Segment})


class Scripted:
    """Returns pre-set weights at given bar numbers; records what it saw."""

    name = "scripted"
    params: dict = {}

    def __init__(self, plan):
        self.plan, self.seen = plan, []

    def target_weights(self, view):
        h = view.history()
        self.seen.append((view.bar_index, len(h), h.index[-1] == view.now))
        return self.plan.get(view.bar_index)


def test_strategy_never_sees_future():
    data = make_data()
    s = Scripted({})
    run_backtest(s, data, date(2026, 1, 1), date(2027, 1, 1), LIMITS)
    assert s.seen and all(n == i + 1 and last_is_now for i, n, last_is_now in s.seen)


def test_fills_at_next_open():
    data = make_data()
    res = run_backtest(Scripted({0: {"A": 0.5}}), data, date(2026, 1, 1), date(2027, 1, 1),
                       LIMITS, broker=no_slip())
    f = res.fills[0]
    assert f.symbol == "A" and f.price == 110.0  # bar 1 open, not bar 0 close (105)
    assert f.ts.date() == data.index[1].date()
    assert f.qty == int(0.5 * 100_000 / 105)  # sized at decision close


def test_sells_before_buys_same_bar():
    data = make_data()
    res = run_backtest(Scripted({0: {"A": 0.9}, 2: {"B": 0.9}}), data, date(2026, 1, 1),
                       date(2027, 1, 1), LIMITS, broker=no_slip())
    bar3 = [f for f in res.fills if f.ts.date() == data.index[3].date()]
    assert [f.side.value for f in bar3] == ["SELL", "BUY"]


def test_idle_cash_earns_interest():
    data = make_data(n=253)
    res = run_backtest(Scripted({}), data, date(2026, 1, 1), date(2028, 1, 1), LIMITS,
                       BacktestConfig(cash_rate=0.065))
    assert res.equity.iloc[-1] == pytest.approx(100_000 * (1 + 0.065 / 252) ** 252, rel=1e-6)
    flat = run_backtest(Scripted({}), data, date(2026, 1, 1), date(2028, 1, 1), LIMITS,
                        BacktestConfig(cash_rate=0.0))
    assert flat.equity.iloc[-1] == 100_000


def test_small_rebalances_skipped():
    data = make_data()
    cfg = BacktestConfig(min_trade_value=50_000)
    res = run_backtest(Scripted({0: {"A": 0.3}}), data, date(2026, 1, 1), date(2027, 1, 1), LIMITS, cfg)
    assert res.fills == []


def test_equity_tracks_holdings():
    data = make_data()
    res = run_backtest(Scripted({0: {"A": 1.0}}), data, date(2026, 1, 1), date(2027, 1, 1),
                       LIMITS, broker=no_slip())
    # Fully invested in a rising stock: equity must end above start despite costs.
    assert res.equity.iloc[-1] > 100_000
    assert res.exposure.iloc[-1] > 0.9


def test_metrics_drawdown_and_return():
    rets = pd.Series([0.10, -0.50, 0.20], index=pd.date_range("2026-01-01", periods=3))
    m = compute_metrics(rets)
    assert m["total_return"] == pytest.approx(1.1 * 0.5 * 1.2 - 1)
    assert m["max_drawdown"] == pytest.approx(-0.5)


def test_turnover_uses_each_runs_own_capital():
    from datetime import datetime

    from firm.broker import Fill
    from firm.costs import Side

    idx_a = pd.date_range("2026-01-01", periods=252, freq="B")
    idx_b = pd.date_range("2027-01-01", periods=252, freq="B")
    eq_a = pd.Series(100_000.0, index=idx_a)
    eq_b = pd.Series(100_000.0, index=idx_b)
    fill = lambda d: Fill(datetime.combine(d, datetime.min.time()), "t", "A",  # noqa: E731
                          Segment.EQ_DELIVERY, Side.BUY, 1000, 100.0, 0.0, 0.0, "")
    fills = [fill(idx_a[5].date()), fill(idx_b[5].date())]  # Rs 1L traded in each year
    rets = pd.Series(0.0, index=idx_a.append(idx_b))
    # Pretend the stitched curve grew 10x: turnover must still be 1x per year.
    m = compute_metrics(rets, fills, equity_by_run=[eq_a, eq_b * 1.0])
    assert m["annual_turnover"] == pytest.approx(1.0)


def test_deflated_sharpe_penalises_many_trials():
    one = deflated_sharpe(0.1, 500, 0.0, 3.0, n_trials=1)
    many = deflated_sharpe(0.1, 500, 0.0, 3.0, n_trials=100, trial_sr_var=0.002)
    assert 0 <= many < one <= 1


def test_make_folds_roll_forward():
    folds = make_folds(date(2010, 1, 1), date(2014, 1, 1), train_years=3, test_months=6)
    assert folds[0] == (date(2010, 1, 1), date(2013, 1, 1), date(2013, 1, 1), date(2013, 7, 1))
    assert folds[-1][3] == date(2014, 1, 1)
    assert all(f[1] == f[2] for f in folds)  # test starts where training ends


def _single(closes, sym="X"):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="B", tz="Asia/Kolkata", name="ts")
    c = np.array(closes, dtype=float)
    return MarketData.from_frames({sym: pd.DataFrame(
        {"open": c, "high": c, "low": c, "close": c, "volume": 1}, index=idx)})


def test_clean_removes_spike_and_adjusts_demerger():
    from firm.backtest.clean import clean

    data, report = clean(_single([100, 101, 900, 102, 103, 40, 41]))
    c = data.panels["close"]["X"]
    assert np.isnan(c.iloc[2]) and len(report["spikes_removed"]) == 1
    assert len(report["breaks_adjusted"]) == 1
    rets = c.ffill().pct_change().dropna()
    assert rets.abs().max() < 0.05  # demerger day now shows ~zero return
    assert c.iloc[-1] == 41  # latest prices untouched


def test_clean_keeps_known_real_moves():
    from firm.backtest.clean import adjust_breaks

    idx_day = pd.date_range("2026-01-01", periods=3, freq="B")[2].date().isoformat()
    _, adjusted = adjust_breaks(_single([100, 100, 150], "PNB"), keep={("PNB", idx_day)})
    assert adjusted == []


def test_lab_enforces_lock_and_single_final_exam():
    data = make_data(n=30)
    lab = Lab(data, "swing", LIMITS, registry=TrialRegistry(":memory:"), lock_start=date(2026, 1, 20))
    with pytest.raises(LockedPeriodError):
        lab.backtest(Scripted({}), date(2026, 1, 1), date(2026, 2, 1))
    lab.backtest(Scripted({0: {"A": 0.5}}), date(2026, 1, 1))  # ends at lock by default
    lab.final_exam(Scripted({}))
    with pytest.raises(LockedPeriodError):
        lab.final_exam(Scripted({}))
