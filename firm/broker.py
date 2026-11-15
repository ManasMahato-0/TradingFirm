"""Paper broker: simulated fills, positions and cash per team.

This module has no network access and no way to place real orders.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from firm.costs import CostRates, Segment, Side, compute_costs

TICK = 0.05

DEFAULT_SLIPPAGE_BPS: dict[Segment, float] = {
    Segment.EQ_DELIVERY: 5.0,
    Segment.EQ_INTRADAY: 3.0,
    Segment.OPTIONS: 50.0,  # option spreads are wide
}


class OrderRejected(Exception):
    pass


@dataclass
class Order:
    team: str
    symbol: str
    segment: Segment
    side: Side
    qty: int
    reason: str = ""
    lot_size: int = 1


@dataclass
class Position:
    symbol: str
    segment: Segment
    qty: int = 0  # signed: negative = short
    avg_price: float = 0.0


@dataclass(frozen=True)
class Fill:
    ts: datetime
    team: str
    symbol: str
    segment: Segment
    side: Side
    qty: int
    price: float
    costs: float
    realized_pnl: float
    reason: str


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def round_to_tick(price: float, side: Side) -> float:
    """Round against us: buys round up, sells round down."""
    steps = price / TICK
    steps = math.ceil(steps - 1e-9) if side is Side.BUY else math.floor(steps + 1e-9)
    return round(max(steps, 1) * TICK, 2)


@dataclass
class PaperAccount:
    team: str
    starting_capital: float
    cash: float = field(init=False)
    positions: dict[str, Position] = field(default_factory=dict)
    realized_pnl: float = 0.0
    costs_paid: float = 0.0
    day_start_equity: float = field(init=False)
    orders_today: int = 0
    halted: bool = False

    def __post_init__(self) -> None:
        self.cash = self.starting_capital
        self.day_start_equity = self.starting_capital

    def position_qty(self, symbol: str) -> int:
        pos = self.positions.get(symbol)
        return pos.qty if pos else 0

    def _price(self, pos: Position, prices: dict[str, float]) -> float:
        return prices.get(pos.symbol, pos.avg_price)

    def market_value(self, prices: dict[str, float]) -> float:
        return sum(p.qty * self._price(p, prices) for p in self.positions.values())

    def gross_exposure(self, prices: dict[str, float]) -> float:
        return sum(abs(p.qty) * self._price(p, prices) for p in self.positions.values())

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + self.market_value(prices)

    def start_new_day(self, prices: dict[str, float]) -> None:
        self.day_start_equity = self.equity(prices)
        self.orders_today = 0
        self.halted = False


class PaperBroker:
    def __init__(
        self,
        slippage_bps: dict[Segment, float] | None = None,
        cost_rates: dict[Segment, CostRates] | None = None,
    ) -> None:
        self.slippage_bps = slippage_bps or DEFAULT_SLIPPAGE_BPS
        self.cost_rates = cost_rates

    def execute(
        self,
        account: PaperAccount,
        order: Order,
        ref_price: float,
        ts: datetime | None = None,
    ) -> Fill:
        if order.qty <= 0:
            raise OrderRejected("qty must be positive")
        if order.qty % order.lot_size != 0:
            raise OrderRejected(f"qty {order.qty} not a multiple of lot size {order.lot_size}")
        if ref_price <= 0:
            raise OrderRejected("reference price must be positive")

        slip = self.slippage_bps[order.segment] / 1e4
        raw = ref_price * (1 + slip) if order.side is Side.BUY else ref_price * (1 - slip)
        price = round_to_tick(raw, order.side)
        turnover = price * order.qty
        costs = compute_costs(order.segment, order.side, turnover, self.cost_rates).total
        signed = order.qty if order.side is Side.BUY else -order.qty

        pos = account.positions.get(order.symbol) or Position(order.symbol, order.segment)
        old = pos.qty
        opening_long = signed > 0 and old >= 0
        if opening_long and account.cash < turnover + costs:
            raise OrderRejected(
                f"insufficient cash: need {turnover + costs:.2f}, have {account.cash:.2f}"
            )

        realized = 0.0
        new_qty = old + signed
        if old == 0 or _sign(old) == _sign(signed):
            pos.avg_price = (abs(old) * pos.avg_price + order.qty * price) / abs(new_qty)
        else:
            closing = min(abs(old), order.qty)
            realized = closing * (price - pos.avg_price) * _sign(old)
            if new_qty == 0:
                pos.avg_price = 0.0
            elif _sign(new_qty) != _sign(old):
                pos.avg_price = price  # flipped through zero
        pos.qty = new_qty

        account.cash -= signed * price + costs
        account.realized_pnl += realized
        account.costs_paid += costs
        account.orders_today += 1
        if pos.qty == 0:
            account.positions.pop(order.symbol, None)
        else:
            account.positions[order.symbol] = pos

        return Fill(
            ts=ts or datetime.now(),
            team=order.team,
            symbol=order.symbol,
            segment=order.segment,
            side=order.side,
            qty=order.qty,
            price=price,
            costs=costs,
            realized_pnl=realized,
            reason=order.reason,
        )
