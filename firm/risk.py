"""Risk engine: every order passes through here before the broker.

Plain deterministic rules. No AI can override a rejection.
Orders that only reduce an existing position are always allowed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime

from firm.broker import Order, PaperAccount
from firm.costs import Segment, Side


@dataclass(frozen=True)
class RiskLimits:
    allowed_segments: frozenset[Segment]
    allow_short: bool
    max_position_pct: float       # one position's value / team equity
    max_gross_exposure_pct: float # sum of all |position values| / team equity
    max_daily_loss_pct: float     # halt team when equity falls this far from day start
    max_orders_per_day: int
    max_data_age_seconds: float


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    qty: int
    reason: str


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


class RiskEngine:
    def __init__(self, limits_by_team: dict[str, RiskLimits]) -> None:
        self.limits = limits_by_team
        self.firm_halted = False
        self.firm_halt_reason = ""

    def halt_firm(self, reason: str) -> None:
        self.firm_halted = True
        self.firm_halt_reason = reason

    def resume_firm(self) -> None:
        self.firm_halted = False
        self.firm_halt_reason = ""

    def check(
        self,
        order: Order,
        account: PaperAccount,
        prices: dict[str, float],
        data_ts: datetime,
        now: datetime,
    ) -> RiskDecision:
        def reject(reason: str) -> RiskDecision:
            return RiskDecision(False, 0, reason)

        limits = self.limits.get(order.team)
        if limits is None:
            return reject(f"unknown team {order.team!r}")
        if order.qty <= 0:
            return reject("qty must be positive")
        price = prices.get(order.symbol)
        if price is None or price <= 0:
            return reject(f"no valid price for {order.symbol}")

        cur = account.position_qty(order.symbol)
        signed = order.qty if order.side is Side.BUY else -order.qty
        new = cur + signed

        # Cutting risk is always allowed, even when halted.
        if cur != 0 and abs(new) < abs(cur) and new * cur >= 0:
            return RiskDecision(True, order.qty, "reduces existing position")

        if self.firm_halted:
            return reject(f"firm halted: {self.firm_halt_reason}")
        if account.halted:
            return reject("team halted for the day")
        if order.segment not in limits.allowed_segments:
            return reject(f"segment {order.segment.value} not allowed for this team")
        age = (now - data_ts).total_seconds()
        if age > limits.max_data_age_seconds:
            return reject(f"stale data: {age:.0f}s old")
        if account.orders_today >= limits.max_orders_per_day:
            return reject("max orders per day reached")

        equity = account.equity(prices)
        if equity <= account.day_start_equity * (1 - limits.max_daily_loss_pct):
            account.halted = True
            return reject("daily loss limit hit, team halted")

        if new < 0 and not limits.allow_short:
            if cur > 0:  # selling more than held: trim to a plain close
                return RiskDecision(True, cur, "trimmed to close; shorting not allowed")
            return reject("shorting not allowed for this team")

        # Size caps on the resulting position.
        lot = order.lot_size
        caps = [
            equity * limits.max_position_pct,
            equity * limits.max_gross_exposure_pct
            - (account.gross_exposure(prices) - abs(cur) * price),
        ]
        if new > 0:
            caps.append(account.cash * 0.995 + max(cur, 0) * price)  # no leverage on longs
        max_abs = max(0, math.floor(min(caps) / price / lot) * lot)

        qty = order.qty
        if abs(new) > max_abs:
            target = _sign(new) * max_abs
            qty = abs(target - cur)
            qty = (qty // lot) * lot
            if qty <= 0 or _sign(target - cur) != _sign(signed):
                return reject("position or exposure limit reached")
            return RiskDecision(True, qty, f"resized {order.qty} -> {qty} to fit limits")
        return RiskDecision(True, qty, "ok")

    def apply(self, order: Order, decision: RiskDecision) -> Order:
        """Return the order with the approved quantity."""
        return replace(order, qty=decision.qty)
