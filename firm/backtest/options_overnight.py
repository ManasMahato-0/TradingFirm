"""Overnight index-option backtest on NSE daily bhavcopy (pre-registered in docs/PLAN.md).

Decide at day t's close, buy one at-the-money Nifty option at that contract's day-t close (plus
slippage), sell at the next trading day's open (minus slippage). One position, sized at a fixed
share of equity (fractional lots); the rest earns the idle-cash rate. Costs use the firm's option
cost model on each leg's premium. Nothing after the end date is ever read, including the exit
price of the last trade.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from firm.broker import Fill
from firm.config import IST
from firm.costs import Segment, Side, compute_costs

MIN_DAYS_TO_EXPIRY = 2
GRID = 50.0


@dataclass
class Trade:
    day: date
    exit_day: date
    kind: str
    strike: float
    expiry: date
    entry: float
    exit: float
    units: float
    pnl_net: float
    costs: float
    premium_return: float   # exit / entry - 1, after slippage
    spot_move: float        # Nifty close(t) -> close(t+1), for reference only


@dataclass
class OvernightResult:
    rule: str
    equity: pd.Series
    trades: list[Trade]
    fills: list[Fill]
    exposure: pd.Series
    nights: int = 0
    skipped: int = 0


class Context:
    """What a rule may look at on day t (never anything later)."""

    def __init__(self, spot: pd.Series) -> None:
        self.spot = spot
        self.sma50 = spot.rolling(50, min_periods=50).mean()
        self.ret1 = spot.pct_change()


class Rule:
    name = "rule"
    params: dict = {}

    def signal(self, t: pd.Timestamp, ctx: Context) -> str | None:  # 'CE' | 'PE' | None
        raise NotImplementedError


class AlwaysCall(Rule):
    name, params = "o1_always_call", {"side": "CE"}

    def signal(self, t, ctx):
        return "CE"


class TrendCallPut(Rule):
    name, params = "o2_trend_call_put", {"sma": 50}

    def signal(self, t, ctx):
        sma = ctx.sma50.get(t)
        if sma is None or not np.isfinite(sma):
            return None
        return "CE" if ctx.spot[t] > sma else "PE"


class DipCall(Rule):
    name, params = "o3_call_after_1pct_drop", {"drop": 0.01}

    def signal(self, t, ctx):
        r = ctx.ret1.get(t)
        return "CE" if r is not None and np.isfinite(r) and r <= -0.01 else None


def pick_contract(day_rows: pd.DataFrame, kind: str, spot: float, today: date) -> pd.Series | None:
    """Nearest expiry with at least 2 days left; strike nearest to spot; must have traded with a price."""
    g = day_rows[(day_rows["opt_type"] == kind) & (day_rows["expiry"] >= today + timedelta(days=MIN_DAYS_TO_EXPIRY))]
    if g.empty:
        return None
    g = g[g["expiry"] == g["expiry"].min()]
    g = g[(g["close"] > 0) & (g["volume"] > 0)]
    if g.empty:
        return None
    g = g.sort_values("strike")                          # ties go to the lower strike
    return g.loc[(g["strike"] - spot).abs().idxmin()]


def run_overnight(rule: Rule, chain: pd.DataFrame, spot: pd.Series, start: date, end: date,
                  capital: float = 100_000.0, pos_pct: float = 0.10, slip: float = 0.005,
                  cash_rate: float = 0.065, with_costs: bool = True) -> OvernightResult:
    """end is EXCLUSIVE: a trade is allowed only if its exit day is before `end`."""
    nifty = chain[chain["symbol"] == "NIFTY"]
    by_day = {d: g for d, g in nifty.groupby("date")}
    days = sorted(d for d in by_day if start <= d < end)
    spot = spot.copy()
    spot.index = pd.to_datetime(spot.index.date)          # plain dates, tz-free
    ctx = Context(spot)
    slip = slip if with_costs else 0.0

    equity, exposure = {}, {}
    eq = capital
    held_notional = 0.0
    trades: list[Trade] = []
    fills: list[Fill] = []
    nights = skipped = 0
    pending: Trade | None = None
    for i, d in enumerate(days):
        t = pd.Timestamp(d)
        eq += (eq - held_notional) * cash_rate / 252 if i else 0.0
        if pending is not None and pending.exit_day == d:
            eq += pending.pnl_net
            pending = None
            held_notional = 0.0
        equity[t], exposure[t] = eq, 0.0
        if i + 1 >= len(days) or t not in spot.index or not np.isfinite(spot[t]):
            continue
        kind = rule.signal(t, ctx)
        if kind is None:
            continue
        nights += 1
        nxt = days[i + 1]
        row = pick_contract(by_day[d], kind, float(spot[t]), d)
        if row is None:
            skipped += 1
            continue
        match = by_day[nxt]
        match = match[(match["opt_type"] == kind) & (match["expiry"] == row["expiry"]) & (match["strike"] == row["strike"])]
        if match.empty or not (match["open"].iloc[0] > 0):
            skipped += 1
            continue
        entry = float(row["close"]) * (1 + slip)
        exit_ = float(match["open"].iloc[0]) * (1 - slip)
        notional = pos_pct * eq
        units = notional / entry
        costs = 0.0
        if with_costs:
            costs = (compute_costs(Segment.OPTIONS, Side.BUY, notional).total
                     + compute_costs(Segment.OPTIONS, Side.SELL, units * exit_).total)
        pnl = units * (exit_ - entry) - costs
        tr = Trade(d, nxt, kind, float(row["strike"]), row["expiry"], entry, exit_, units, pnl, costs,
                   exit_ / entry - 1, float(spot[pd.Timestamp(nxt)] / spot[t] - 1) if pd.Timestamp(nxt) in spot.index else np.nan)
        trades.append(tr)
        pending, held_notional = tr, notional
        exposure[t] = notional / eq
        sym = f"NIFTY{row['expiry']:%d%b%y}{int(row['strike'])}{kind}".upper()
        ts_in = datetime.combine(d, time(15, 25), IST)
        ts_out = datetime.combine(nxt, time(9, 15), IST)
        qty = max(1, int(round(units)))
        buy_costs = compute_costs(Segment.OPTIONS, Side.BUY, notional).total if with_costs else 0.0
        fills.append(Fill(ts_in, "overnight_options", sym, Segment.OPTIONS, Side.BUY, qty, entry, buy_costs, 0.0, rule.name))
        fills.append(Fill(ts_out, "overnight_options", sym, Segment.OPTIONS, Side.SELL, qty, exit_,
                          costs - buy_costs, pnl, rule.name))
    idx = pd.DatetimeIndex(list(equity), name="ts")
    return OvernightResult(rule.name, pd.Series(list(equity.values()), index=idx, name="equity"), trades, fills,
                           pd.Series(list(exposure.values()), index=idx, name="exposure"), nights, skipped)


def years_beating_cash(equity: pd.Series, cash_rate: float = 0.065) -> tuple[float, dict]:
    """Share of calendar years in which equity growth beat cash pro rata, and the per-year detail."""
    out = {}
    for year, e in equity.groupby(equity.index.year):
        if len(e) < 20:
            continue
        start_val = equity[equity.index < e.index[0]].iloc[-1] if (equity.index < e.index[0]).any() else e.iloc[0]
        ret = e.iloc[-1] / start_val - 1
        cash = cash_rate * (e.index[-1] - e.index[0]).days / 365
        out[int(year)] = (float(ret), float(cash), bool(ret > cash))
    return (sum(v[2] for v in out.values()) / len(out) if out else float("nan")), out


for _rule in (Rule, AlwaysCall, TrendCallPut, DipCall):
    _rule.depends_on = (sys.modules[__name__],)
