from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from firm.backtest.options_overnight import (
    AlwaysCall, Context, DipCall, TrendCallPut, pick_contract, run_overnight, years_beating_cash,
)
from firm.costs import Segment, Side, compute_costs

D = [date(2026, 3, 2) + timedelta(days=i) for i in range(10) if (date(2026, 3, 2) + timedelta(days=i)).weekday() < 5]


def row(day, kind, strike, expiry, close=5.0, open_=5.0, volume=1000):
    return {"date": day, "symbol": "NIFTY", "expiry": expiry, "strike": float(strike), "opt_type": kind,
            "open": open_, "high": close, "low": open_, "close": close, "settle": close, "volume": volume,
            "oi": 100, "fmt": "old"}


def spot_series(values, days=D):
    return pd.Series(values[: len(days)], index=pd.to_datetime(days))


def chain_for(days, expiry, per_day):
    rows = []
    for d, kw in zip(days, per_day):
        for kind in ("CE", "PE"):
            for strike in (90, 100, 110):
                rows.append(row(d, kind, strike, expiry, **kw))
    return pd.DataFrame(rows)


def test_pick_contract_uses_nearest_valid_expiry_and_nearest_strike():
    d = date(2026, 3, 2)
    rows = pd.DataFrame([row(d, "CE", 100, d + timedelta(days=1)),      # too close to expiry
                         row(d, "CE", 100, d + timedelta(days=5)), row(d, "CE", 110, d + timedelta(days=5)),
                         row(d, "CE", 100, d + timedelta(days=12)),
                         row(d, "CE", 90, d + timedelta(days=5), volume=0)])    # never traded
    c = pick_contract(rows, "CE", 104.0, d)
    assert c["strike"] == 100 and c["expiry"] == d + timedelta(days=5)
    assert pick_contract(rows, "PE", 104.0, d) is None


def test_one_night_pnl_is_entry_close_to_next_open_with_slippage_and_costs():
    exp = D[0] + timedelta(days=6)
    chain = chain_for(D[:3], exp, [dict(close=5.0, open_=5.0), dict(close=5.0, open_=6.0), dict(close=5.0, open_=5.0)])
    res = run_overnight(AlwaysCall(), chain, spot_series([100] * 10), D[0], D[2] + timedelta(days=1), cash_rate=0.0)
    tr = res.trades[0]
    assert tr.entry == pytest.approx(5.0 * 1.005) and tr.exit == pytest.approx(6.0 * 0.995)
    notional = 0.10 * 100_000
    units = notional / tr.entry
    fees = (compute_costs(Segment.OPTIONS, Side.BUY, notional).total
            + compute_costs(Segment.OPTIONS, Side.SELL, units * tr.exit).total)
    assert tr.pnl_net == pytest.approx(units * (tr.exit - tr.entry) - fees)
    assert res.equity.iloc[1] == pytest.approx(100_000 + tr.pnl_net)         # booked the NEXT morning, not at entry
    assert res.equity.iloc[0] == pytest.approx(100_000)
    assert len(res.fills) >= 2 and res.fills[1].realized_pnl == pytest.approx(tr.pnl_net)


def test_no_costs_run_has_no_fees_or_slippage():
    exp = D[0] + timedelta(days=6)
    chain = chain_for(D[:3], exp, [dict(close=5.0, open_=5.0), dict(close=5.0, open_=6.0), dict(close=5.0, open_=5.0)])
    res = run_overnight(AlwaysCall(), chain, spot_series([100] * 10), D[0], D[2] + timedelta(days=1),
                        cash_rate=0.0, with_costs=False)
    tr = res.trades[0]
    assert tr.costs == 0 and tr.entry == 5.0 and tr.exit == 6.0
    assert tr.pnl_net == pytest.approx(0.10 * 100_000 * (6.0 / 5.0 - 1))


def test_skips_when_the_next_open_did_not_trade_and_counts_the_skip():
    exp = D[0] + timedelta(days=6)
    chain = chain_for(D[:3], exp, [dict(close=5.0), dict(close=5.0, open_=0.0), dict(close=5.0)])
    res = run_overnight(AlwaysCall(), chain, spot_series([100] * 10), D[0], D[2] + timedelta(days=1))
    assert res.skipped >= 1 and all(t.day != D[0] for t in res.trades)


def test_the_end_date_is_exclusive_even_for_the_exit_price():
    exp = D[0] + timedelta(days=6)
    chain = chain_for(D[:4], exp, [dict(close=5.0, open_=5.0)] * 4)
    res = run_overnight(AlwaysCall(), chain, spot_series([100] * 10), D[0], D[3])     # D[3] itself is not allowed
    assert res.trades and all(t.exit_day < D[3] for t in res.trades)
    assert max(res.equity.index).date() < D[3]


def test_idle_cash_earns_interest_when_there_is_no_trade():
    exp = D[0] + timedelta(days=6)
    chain = chain_for(D[:4], exp, [dict(close=5.0)] * 4)
    res = run_overnight(DipCall(), chain, spot_series([100, 100, 100, 100], D[:4]), D[0], D[3] + timedelta(days=1))
    assert not res.trades and res.equity.iloc[-1] == pytest.approx(100_000 * (1 + 0.065 / 252) ** 3, rel=1e-9)


def test_rules_use_only_the_past():
    exp = D[0] + timedelta(days=6)
    days = D[:5]
    chain = chain_for(days, exp, [dict(close=5.0, open_=5.0)] * 5)
    spot = spot_series([100, 98, 99, 100, 101], days)                                 # day 2 falls 2%
    dip = run_overnight(DipCall(), chain, spot, days[0], days[-1] + timedelta(days=1))
    assert [t.day for t in dip.trades] == [days[1]]                                   # trades the night AFTER the drop
    up = spot_series([100 + i for i in range(60)], [date(2026, 1, 1) + timedelta(days=i) for i in range(60)])
    rule = TrendCallPut()
    ctx = Context(up)
    assert rule.signal(up.index[10], ctx) is None                                     # not enough history yet
    assert rule.signal(up.index[55], ctx) == "CE"
    down = Context(pd.Series(up.to_numpy()[::-1], index=up.index))
    assert rule.signal(down.spot.index[55], down) == "PE"


def test_years_beating_cash():
    idx = pd.bdate_range("2023-01-02", "2024-12-31")
    eq = pd.Series(np.r_[np.linspace(100, 120, (idx.year == 2023).sum()), np.linspace(120, 121, (idx.year == 2024).sum())],
                   index=idx)
    share, detail = years_beating_cash(eq)
    assert detail[2023][2] and not detail[2024][2] and share == 0.5
