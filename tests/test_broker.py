import pytest

from firm.broker import Order, OrderRejected, PaperAccount, PaperBroker, round_to_tick
from firm.costs import Segment, Side


def acct():
    return PaperAccount(team="t", starting_capital=100_000)


def order(side, qty, symbol="ABC", seg=Segment.EQ_DELIVERY, lot=1):
    return Order(team="t", symbol=symbol, segment=seg, side=side, qty=qty, lot_size=lot)


def no_slip():
    return PaperBroker(slippage_bps={s: 0.0 for s in Segment})


def test_tick_rounding_is_adverse():
    assert round_to_tick(100.01, Side.BUY) == 100.05
    assert round_to_tick(100.04, Side.SELL) == 100.00
    assert round_to_tick(100.05, Side.BUY) == 100.05


def test_buy_then_sell_round_trip():
    a, b = acct(), no_slip()
    f1 = b.execute(a, order(Side.BUY, 10), 100)
    assert a.position_qty("ABC") == 10
    assert a.cash == pytest.approx(100_000 - 1000 - f1.costs)
    f2 = b.execute(a, order(Side.SELL, 10), 110)
    assert f2.realized_pnl == pytest.approx(100)
    assert "ABC" not in a.positions
    assert a.equity({}) == pytest.approx(100_000 + 100 - f1.costs - f2.costs)
    assert a.costs_paid == pytest.approx(f1.costs + f2.costs)


def test_average_price_on_add():
    a, b = acct(), no_slip()
    b.execute(a, order(Side.BUY, 10), 100)
    b.execute(a, order(Side.BUY, 10), 110)
    assert a.positions["ABC"].avg_price == pytest.approx(105)


def test_partial_close_keeps_avg():
    a, b = acct(), no_slip()
    b.execute(a, order(Side.BUY, 10), 100)
    f = b.execute(a, order(Side.SELL, 4), 90)
    assert f.realized_pnl == pytest.approx(-40)
    assert a.positions["ABC"].qty == 6
    assert a.positions["ABC"].avg_price == pytest.approx(100)


def test_short_and_flip():
    a, b = acct(), no_slip()
    seg = Segment.EQ_INTRADAY
    b.execute(a, order(Side.SELL, 10, seg=seg), 100)
    assert a.position_qty("ABC") == -10
    # Short profits when price falls.
    assert a.equity({"ABC": 90}) > a.equity({"ABC": 100})
    f = b.execute(a, order(Side.BUY, 15, seg=seg), 90)
    assert f.realized_pnl == pytest.approx(100)
    assert a.positions["ABC"].qty == 5
    assert a.positions["ABC"].avg_price == pytest.approx(90)


def test_slippage_hurts_both_sides():
    a, b = acct(), PaperBroker()
    buy = b.execute(a, order(Side.BUY, 1), 100)
    sell = b.execute(a, order(Side.SELL, 1), 100)
    assert buy.price > 100 > sell.price


def test_insufficient_cash_rejected():
    a, b = acct(), no_slip()
    with pytest.raises(OrderRejected):
        b.execute(a, order(Side.BUY, 1000), 100)
    assert a.cash == 100_000 and not a.positions


def test_lot_size_enforced():
    a, b = acct(), no_slip()
    with pytest.raises(OrderRejected):
        b.execute(a, order(Side.BUY, 50, seg=Segment.OPTIONS, lot=75), 100)
