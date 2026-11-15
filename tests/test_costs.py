import pytest

from firm.costs import Segment, Side, compute_costs


def test_zero_turnover_costs_nothing():
    assert compute_costs(Segment.EQ_DELIVERY, Side.BUY, 0).total == 0


def test_delivery_buy_breakdown():
    c = compute_costs(Segment.EQ_DELIVERY, Side.BUY, 100_000)
    assert c.brokerage == 20  # min(20, 0.3% of 1L = 300)
    assert c.stt == pytest.approx(100)
    assert c.stamp == pytest.approx(15)
    assert c.gst == pytest.approx((20 + 2.97 + 0.1) * 0.18)


def test_intraday_stt_only_on_sell():
    buy = compute_costs(Segment.EQ_INTRADAY, Side.BUY, 50_000)
    sell = compute_costs(Segment.EQ_INTRADAY, Side.SELL, 50_000)
    assert buy.stt == 0
    assert sell.stt == pytest.approx(12.5)
    assert sell.stamp == 0


def test_small_intraday_brokerage_uses_pct():
    c = compute_costs(Segment.EQ_INTRADAY, Side.BUY, 10_000)
    assert c.brokerage == pytest.approx(3.0)


def test_options_flat_brokerage():
    c = compute_costs(Segment.OPTIONS, Side.SELL, 5_000)
    assert c.brokerage == 20
    assert c.stt == pytest.approx(5)
