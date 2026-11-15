"""Indian brokerage and statutory charges.

Rates are approximate defaults and fully configurable. Verify them against
the Fyers charges page and current NSE/SEBI circulars before trusting absolute
numbers: STT in particular changes with Union Budgets.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Segment(str, Enum):
    EQ_DELIVERY = "EQ_DELIVERY"
    EQ_INTRADAY = "EQ_INTRADAY"
    OPTIONS = "OPTIONS"


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class CostRates:
    brokerage_flat: float
    # If set, brokerage = min(brokerage_flat, brokerage_pct * turnover).
    brokerage_pct: float | None
    stt_buy_pct: float
    stt_sell_pct: float
    exchange_pct: float
    stamp_buy_pct: float
    sebi_per_crore: float = 10.0
    gst_pct: float = 0.18


DEFAULT_RATES: dict[Segment, CostRates] = {
    Segment.EQ_DELIVERY: CostRates(
        brokerage_flat=20.0, brokerage_pct=0.003,
        stt_buy_pct=0.001, stt_sell_pct=0.001,
        exchange_pct=0.0000297, stamp_buy_pct=0.00015,
    ),
    Segment.EQ_INTRADAY: CostRates(
        brokerage_flat=20.0, brokerage_pct=0.0003,
        stt_buy_pct=0.0, stt_sell_pct=0.00025,
        exchange_pct=0.0000297, stamp_buy_pct=0.00003,
    ),
    # Options: all percentages apply to premium turnover.
    Segment.OPTIONS: CostRates(
        brokerage_flat=20.0, brokerage_pct=None,
        stt_buy_pct=0.0, stt_sell_pct=0.001,
        exchange_pct=0.0003503, stamp_buy_pct=0.00003,
    ),
}


@dataclass(frozen=True)
class CostBreakdown:
    brokerage: float = 0.0
    stt: float = 0.0
    exchange: float = 0.0
    sebi: float = 0.0
    stamp: float = 0.0
    gst: float = 0.0

    @property
    def total(self) -> float:
        return self.brokerage + self.stt + self.exchange + self.sebi + self.stamp + self.gst


def compute_costs(
    segment: Segment,
    side: Side,
    turnover: float,
    rates: dict[Segment, CostRates] | None = None,
) -> CostBreakdown:
    """Charges for one order leg with the given turnover (price * qty)."""
    if turnover <= 0:
        return CostBreakdown()
    r = (rates or DEFAULT_RATES)[segment]
    if r.brokerage_pct is None:
        brokerage = r.brokerage_flat
    else:
        brokerage = min(r.brokerage_flat, r.brokerage_pct * turnover)
    stt = turnover * (r.stt_buy_pct if side is Side.BUY else r.stt_sell_pct)
    exchange = turnover * r.exchange_pct
    sebi = turnover * r.sebi_per_crore / 1e7
    stamp = turnover * r.stamp_buy_pct if side is Side.BUY else 0.0
    gst = (brokerage + exchange + sebi) * r.gst_pct
    return CostBreakdown(brokerage, stt, exchange, sebi, stamp, gst)
