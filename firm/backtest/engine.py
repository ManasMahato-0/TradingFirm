"""Event-driven backtest engine.

Each bar:
  1. at the open: execute orders decided at the previous close (sells first)
  2. at the close: mark positions, record equity
  3. strategy sees data up to this close and returns target weights
Orders pass through the real RiskEngine and PaperBroker.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from datetime import time as dtime
from typing import Callable, Protocol

import pandas as pd

from firm.backtest.data import DataView, MarketData
from firm.broker import Fill, Order, OrderRejected, PaperAccount, PaperBroker
from firm.costs import Segment, Side
from firm.risk import RiskEngine, RiskLimits


class Strategy(Protocol):
    name: str
    params: dict

    def target_weights(self, view: DataView) -> dict[str, float] | None:
        """Full target portfolio as fractions of equity. Missing symbols = 0.
        Return None to keep current positions unchanged."""
        ...


@dataclass
class BacktestConfig:
    team: str = "swing_stocks"
    segment: Segment = Segment.EQ_DELIVERY
    initial_capital: float = 100_000.0
    min_trade_value: float = 2_000.0  # skip tiny rebalances; full exits always allowed
    lot_sizes: dict[str, int] = field(default_factory=dict)
    # Idle cash parked in a liquid fund / LIQUIDBEES earns roughly this per year.
    cash_rate: float = 0.065
    # Intraday: from the bar starting at this time, targets are forced to flat (orders
    # fill at the next bar's open), and orders never carry over to the next day.
    flatten_after: dtime | None = None


@dataclass
class BacktestResult:
    strategy: str
    params: dict
    start: date
    end: date
    initial_capital: float
    equity: pd.Series
    exposure: pd.Series  # gross exposure / equity at each close
    fills: list[Fill]
    rejections: list[dict]
    metrics: dict = field(default_factory=dict)


def _valid(row: pd.Series) -> dict[str, float]:
    return {k: float(v) for k, v in row.items() if pd.notna(v) and v > 0}


EventFn = Callable[..., None]  # (kind, agent, text, to=None, details=None, ts=None)


@dataclass
class StepResult:
    fills: list[Fill]
    rejections: list[dict]
    equity: float
    exposure: float
    decided: bool  # strategy returned new targets at this close


class Session:
    """One account run bar by bar. The backtest and the live paper runner both use
    this, so live trades follow exactly the logic that was researched.

    Picklable (the event callback is dropped), so live state survives restarts.
    """

    def __init__(self, strategy: Strategy, limits: RiskLimits, config: BacktestConfig | None = None,
                 broker: PaperBroker | None = None, agent: str = "Strategy") -> None:
        self.strategy = strategy
        self.cfg = config or BacktestConfig()
        self.broker = broker or PaperBroker()
        self.risk = RiskEngine({self.cfg.team: limits})
        self.account = PaperAccount(team=self.cfg.team, starting_capital=self.cfg.initial_capital)
        self.agent = agent
        self.pending: dict[str, int] | None = None
        self.last_day: date | None = None
        self.on_event: EventFn | None = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["on_event"] = None
        return state

    def _emit(self, kind: str, agent: str, text: str, ts=None, **kw) -> None:
        if self.on_event is not None:
            self.on_event(kind, agent, text, ts=ts, **kw)

    def step(self, view: DataView, i: int, opens: pd.DataFrame, marks: pd.DataFrame) -> StepResult:
        """Bar i: fill pending orders at the open, mark at the close, then ask the
        strategy for new targets (filled at the next bar's open)."""
        cfg, account, risk = self.cfg, self.account, self.risk
        t = view._data.index[i]
        ts = t.to_pydatetime()
        view._advance(i)
        prev_marks = _valid(marks.iloc[i - 1]) if i > 0 else {}
        if t.date() != self.last_day:
            if self.last_day is not None and cfg.cash_rate and account.cash > 0:
                account.cash += account.cash * cfg.cash_rate / 252
            account.start_new_day(prev_marks)
            self.last_day = t.date()
            if cfg.flatten_after is not None:
                self.pending = None  # intraday: yesterday's unfilled orders expire

        fills: list[Fill] = []
        rejections: list[dict] = []
        if self.pending is not None:
            open_px = _valid(opens.iloc[i])
            risk_prices = {**prev_marks, **open_px}
            orders = []
            for sym, target in self.pending.items():
                delta = target - account.position_qty(sym)
                if delta == 0:
                    continue
                if sym not in open_px:
                    rejections.append({"ts": t, "symbol": sym, "reason": "no open price"})
                    continue
                if abs(delta) * open_px[sym] < cfg.min_trade_value and target != 0:
                    continue
                orders.append(Order(cfg.team, sym, cfg.segment, Side.BUY if delta > 0 else Side.SELL,
                                    abs(delta), reason=self.strategy.name,
                                    lot_size=cfg.lot_sizes.get(sym, 1)))
            orders.sort(key=lambda o: o.side is Side.BUY)  # sells first free up cash
            for o in orders:
                d = risk.check(o, account, risk_prices, data_ts=t, now=t)
                ask = f"{o.side.value} {o.qty} {o.symbol}"
                if not d.approved:
                    rejections.append({"ts": t, "symbol": o.symbol, "reason": d.reason})
                    self._emit("talk", "Risk engine", f"Rejected {ask}: {d.reason}", ts=ts, to=self.agent)
                    continue
                if d.qty != o.qty:
                    self._emit("talk", "Risk engine", f"Approved {ask} as {d.qty}: {d.reason}", ts=ts,
                               to=self.agent)
                try:
                    f = self.broker.execute(account, risk.apply(o, d), open_px[o.symbol], ts)
                    fills.append(f)
                    self._emit("act", "Broker", f"Filled {f.side.value} {f.qty} {f.symbol} @ Rs {f.price:,.2f}"
                               f" (open {open_px[o.symbol]:,.2f}, costs Rs {f.costs:,.2f})", ts=ts,
                               details={"symbol": f.symbol, "qty": f.qty, "price": f.price,
                                        "costs": f.costs, "realized_pnl": f.realized_pnl})
                except OrderRejected as e:
                    rejections.append({"ts": t, "symbol": o.symbol, "reason": str(e)})
                    self._emit("alert", "Broker", f"Could not fill {ask}: {e}", ts=ts)
            if orders:
                bad = sum(1 for r in rejections if r["reason"] != "no open price")
                self._emit("talk", "Risk engine", f"Checked {len(orders)} orders against limits: "
                           f"{len(orders) - bad} approved, {bad} rejected.", ts=ts, to=self.agent)
            self.pending = None

        close_marks = _valid(marks.iloc[i])
        equity = account.equity(close_marks)
        exposure = account.gross_exposure(close_marks) / equity if equity > 0 else 0.0

        weights = self.strategy.target_weights(view)
        if cfg.flatten_after is not None and t.time() >= cfg.flatten_after:
            weights = {} if (account.positions or weights) else None  # square off, no new entries
        if weights is not None:
            self.pending = {}
            for sym, w in weights.items():
                px = close_marks.get(sym)
                if px is None or not math.isfinite(w):
                    continue
                lot = cfg.lot_sizes.get(sym, 1)
                self.pending[sym] = int(w * equity / px / lot) * lot  # truncates toward zero
            for sym in account.positions:
                self.pending.setdefault(sym, 0)
        return StepResult(fills, rejections, equity, exposure, weights is not None)


def run_backtest(strategy: Strategy, data: MarketData, start: date, end: date,
                 limits: RiskLimits, config: BacktestConfig | None = None,
                 broker: PaperBroker | None = None) -> BacktestResult:
    cfg = config or BacktestConfig()
    session = Session(strategy, limits, cfg, broker)
    view = DataView(data)
    opens = data.panels["open"]
    marks = data.panels["close"].ffill()  # value holdings at last known close

    fills: list[Fill] = []
    rejections: list[dict] = []
    eq_pts, ex_pts = [], []
    for i, t in enumerate(data.index):
        if not start <= t.date() < end:
            continue
        r = session.step(view, i, opens, marks)
        fills += r.fills
        rejections += r.rejections
        eq_pts.append((t, r.equity))
        ex_pts.append((t, r.exposure))

    idx = pd.DatetimeIndex([t for t, _ in eq_pts], name="ts")
    return BacktestResult(
        strategy=strategy.name, params=dict(strategy.params), start=start, end=end,
        initial_capital=cfg.initial_capital,
        equity=pd.Series([e for _, e in eq_pts], index=idx, name="equity"),
        exposure=pd.Series([x for _, x in ex_pts], index=idx, name="exposure"),
        fills=fills, rejections=rejections,
    )
