from datetime import datetime, timedelta

from firm.broker import Order, PaperAccount, PaperBroker
from firm.config import TEAMS
from firm.costs import Segment, Side
from firm.risk import RiskEngine

NOW = datetime(2026, 11, 13, 10, 0)


def engine():
    return RiskEngine({k: t.limits for k, t in TEAMS.items()})


def acct(team):
    return PaperAccount(team=team, starting_capital=100_000)


def o(team, side, qty, seg, symbol="ABC", lot=1):
    return Order(team=team, symbol=symbol, segment=seg, side=side, qty=qty, lot_size=lot)


def test_ok_order_approved():
    d = engine().check(o("swing_stocks", Side.BUY, 10, Segment.EQ_DELIVERY),
                       acct("swing_stocks"), {"ABC": 100}, NOW, NOW)
    assert d.approved and d.qty == 10


def test_position_cap_resizes():
    # 20% of 1L = 20k -> max 200 shares at 100
    d = engine().check(o("swing_stocks", Side.BUY, 500, Segment.EQ_DELIVERY),
                       acct("swing_stocks"), {"ABC": 100}, NOW, NOW)
    assert d.approved and d.qty == 200


def test_wrong_segment_rejected():
    d = engine().check(o("swing_stocks", Side.BUY, 10, Segment.OPTIONS),
                       acct("swing_stocks"), {"ABC": 100}, NOW, NOW)
    assert not d.approved


def test_stale_data_rejected():
    d = engine().check(o("intraday_stocks", Side.BUY, 10, Segment.EQ_INTRADAY),
                       acct("intraday_stocks"), {"ABC": 100}, NOW - timedelta(minutes=10), NOW)
    assert not d.approved and "stale" in d.reason


def test_short_blocked_for_swing_but_close_trimmed():
    e, a, b = engine(), acct("swing_stocks"), PaperBroker()
    d = e.check(o("swing_stocks", Side.SELL, 10, Segment.EQ_DELIVERY), a, {"ABC": 100}, NOW, NOW)
    assert not d.approved
    b.execute(a, o("swing_stocks", Side.BUY, 10, Segment.EQ_DELIVERY), 100)
    d = e.check(o("swing_stocks", Side.SELL, 25, Segment.EQ_DELIVERY), a, {"ABC": 100}, NOW, NOW)
    assert d.approved and d.qty == 10


def test_short_allowed_intraday():
    d = engine().check(o("intraday_stocks", Side.SELL, 10, Segment.EQ_INTRADAY),
                       acct("intraday_stocks"), {"ABC": 100}, NOW, NOW)
    assert d.approved


def test_options_buy_only_and_lot_rounding():
    e, a = engine(), acct("intraday_options")
    sell = e.check(o("intraday_options", Side.SELL, 75, Segment.OPTIONS, lot=75),
                   a, {"ABC": 100}, NOW, NOW)
    assert not sell.approved
    # 10% of 1L = 10k premium cap; at ₹100 premium -> 100 qty -> rounds down to 75 (one lot)
    buy = e.check(o("intraday_options", Side.BUY, 300, Segment.OPTIONS, lot=75),
                  a, {"ABC": 100}, NOW, NOW)
    assert buy.approved and buy.qty == 75


def test_daily_loss_halts_team_but_allows_exit():
    e, a, b = engine(), acct("swing_stocks"), PaperBroker()
    b.execute(a, o("swing_stocks", Side.BUY, 200, Segment.EQ_DELIVERY), 100)
    crashed = {"ABC": 80, "XYZ": 50}  # -4k on 1L = -4%, limit is 3%
    d = e.check(o("swing_stocks", Side.BUY, 10, Segment.EQ_DELIVERY, symbol="XYZ"),
                a, crashed, NOW, NOW)
    assert not d.approved and a.halted
    exit_ = e.check(o("swing_stocks", Side.SELL, 200, Segment.EQ_DELIVERY), a, crashed, NOW, NOW)
    assert exit_.approved


def test_firm_halt_blocks_new_risk():
    e = engine()
    e.halt_firm("manual kill switch")
    d = e.check(o("swing_stocks", Side.BUY, 10, Segment.EQ_DELIVERY),
                acct("swing_stocks"), {"ABC": 100}, NOW, NOW)
    assert not d.approved and "kill switch" in d.reason
