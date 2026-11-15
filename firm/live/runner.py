"""Live paper runner: advances each paper book through new daily bars.

Each evening, for every book and every new trading day since its last run:
  1. fill orders decided at the previous close at this day's open (real open price)
  2. mark positions at the close, credit interest on idle cash, save equity
  3. let the strategy decide; new orders wait for the next day's open
This is the same `Session.step` the backtests use, so live follows research exactly.
Missed days (laptop off) are caught up in order on the next run.

State per book is pickled under <state_dir>/<book>/session.pkl; fills, equity,
positions and the activity feed go to SQLite for the dashboard.
"""
from __future__ import annotations

import json
import os
import pickle
import socket
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from typing import Callable

import pandas as pd

from firm.activity import ActivityLog
from firm.backtest.clean import clean
from firm.backtest.data import DataView, MarketData
from firm.backtest.engine import BacktestConfig, Session
from firm.backtest.registry import strategy_version
from firm.calendar import TradingCalendar
from firm.config import DATA_DIR, IST, TEAMS
from firm.live.explain import explain_decision
from firm.storage import Storage

BAR_READY = dtime(16, 0)  # today's daily bar is trusted only after this (IST)
OPEN_TIME, CLOSE_TIME = dtime(9, 15), dtime(15, 30)
STRATEGY_INPUTS = ("nifty", "vix", "meetings", "sectors")  # refreshed every run, not pickled


@dataclass
class LiveContext:
    data: MarketData
    nifty: pd.Series
    vix: pd.Series | None
    sectors: dict[str, str]
    meetings: pd.DataFrame | None
    calendar: TradingCalendar


@dataclass
class BookSpec:
    name: str                       # unique id, also the account name in the DB
    team: str                       # key in TEAMS (risk limits)
    label: str                      # shown on the dashboard
    status: str                     # probation | active | shadow
    capital: float
    start: date                     # first bar processed
    agent: str                      # who "thinks" for this book on the Trading Floor
    factory: Callable[[LiveContext], object]
    needs_meetings: bool = False
    note: str = ""


@dataclass
class BookState:
    session: Session
    version: str
    trained_through: date | None = None
    last_bar: date | None = None
    created: str = field(default_factory=lambda: datetime.now(IST).isoformat(timespec="seconds"))


def _at(day: date, t: dtime) -> datetime:
    return datetime.combine(day, t, tzinfo=IST)


def load_context(start: date = date(2005, 1, 1), with_meetings: bool = True) -> LiveContext:
    from firm.features.results import load_results_meetings
    from firm.universe import INDEXES, nifty100, sectors

    raw = MarketData.from_store("1d_yahoo", nifty100() + INDEXES, start=start)
    nifty = raw.panels["close"]["NIFTY50"].dropna()
    vix = raw.panels["close"]["INDIAVIX"].dropna() if "INDIAVIX" in raw.symbols else None
    data, _ = clean(raw.without(INDEXES))
    return LiveContext(data, nifty, vix, sectors(), load_results_meetings() if with_meetings else None,
                       TradingCalendar())


class LiveBook:
    def __init__(self, spec: BookSpec, store: Storage, activity: ActivityLog, state_dir: Path,
                 control=None) -> None:
        self.spec, self.store, self.activity = spec, store, activity
        self.control = control      # owner's switches; None = no switches (tests, replays)
        self.dir = state_dir / spec.name
        self.path = self.dir / "session.pkl"

    # ------------------------------------------------------------ helpers

    def say(self, kind: str, agent: str, text: str, ts=None, to=None, details=None) -> None:
        self.activity.log(kind, agent, text, team=self.spec.team, to=to, details=details, ts=ts,
                          book=self.spec.name)

    def _sync_halt(self, session, day) -> None:
        """Owner's pause/halt blocks NEW entries only. The book still marks to market, records equity and
        counts its days, so nothing is replayed on resume. Set both ways before every step (the flag is
        pickled with the session)."""
        from firm.control import OWNER_HALT_REASON

        halted = bool(self.control and self.control.halted(self.spec.name, str(day)))
        session.risk.firm_halted = halted
        session.risk.firm_halt_reason = OWNER_HALT_REASON if halted else ""

    def _on_event(self, kind, agent, text, ts=None, to=None, details=None):
        # Engine events at a daily bar's timestamp happen at the open (fills, risk checks).
        if isinstance(ts, datetime) and ts.time() == dtime(0, 0):
            ts = _at(ts.date(), OPEN_TIME)
        self.say(kind, agent, text, ts=ts, to=to, details=details)

    def _refresh_inputs(self, strategy, ctx: LiveContext) -> None:
        for attr in STRATEGY_INPUTS:
            if hasattr(strategy, attr):
                setattr(strategy, attr, getattr(ctx, attr))

    def load(self) -> BookState | None:
        if not self.path.exists():
            return None
        with open(self.path, "rb") as f:
            return pickle.load(f)

    def save(self, state: BookState) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        strategy = state.session.strategy
        saved = {a: getattr(strategy, a) for a in STRATEGY_INPUTS if hasattr(strategy, a)}
        for a in saved:
            setattr(strategy, a, None)  # big inputs are reloaded fresh each run
        try:
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(state, f)
            os.replace(tmp, self.path)
        finally:
            for a, v in saved.items():
                setattr(strategy, a, v)
        meta = {"version": state.version, "last_bar": str(state.last_bar),
                "trained_through": str(state.trained_through), "created": state.created}
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))

    def _fit(self, state: BookState, ctx: LiveContext, train_end: date, first: bool) -> None:
        strategy = state.session.strategy
        if not hasattr(strategy, "fit"):
            return
        self.activity.status(self.spec.agent, "working", f"Training on data before {train_end}", self.spec.team)
        t0 = time.time()
        strategy.fit(ctx.data.truncate(train_end), train_end - timedelta(days=3 * 365), train_end,
                     **({} if first else {"reset": False}))
        state.trained_through = train_end
        self.say("act", self.spec.agent,
                 f"{'Trained' if first else 'Yearly retrain:'} model on all history before {train_end} "
                 f"({time.time() - t0:.0f}s). Version {state.version}.", ts=datetime.now(IST))

    def _init(self, ctx: LiveContext) -> BookState:
        strategy = self.spec.factory(ctx)
        team = TEAMS[self.spec.team]
        cfg = BacktestConfig(team=self.spec.name, initial_capital=self.spec.capital)
        session = Session(strategy, team.limits, cfg, agent=self.spec.agent)
        state = BookState(session, strategy_version(strategy))
        self.say("act", "Runner", f"Opened paper book '{self.spec.label}' with Rs {self.spec.capital:,.0f} "
                 f"({self.spec.status}). First trading day {self.spec.start}.", ts=datetime.now(IST))
        self._fit(state, ctx, self.spec.start, first=True)
        return state

    # ------------------------------------------------------------ main

    def advance(self, ctx: LiveContext, now: datetime | None = None) -> dict:
        now = now or datetime.now(IST)
        cutoff = now.date() if now.time() >= BAR_READY else now.date() - timedelta(days=1)
        state = self.load()
        if state is None:
            ready = [t for t in ctx.data.index if self.spec.start <= t.date() <= cutoff]
            if not ready:  # open (and train) only when the first trading day's bar is in
                self.store.upsert_book(
                    name=self.spec.name, team=self.spec.team, label=self.spec.label, status=self.spec.status,
                    capital=self.spec.capital, start=str(self.spec.start), last_run=now.isoformat(timespec="seconds"),
                    note=self.spec.note)
                self.activity.status(self.spec.agent, "idle",
                                     f"Book opens {self.spec.start}: trains then, first decision at that close",
                                     self.spec.team)
                return {"book": self.spec.name, "days": 0, "last_bar": None, "equity": self.spec.capital,
                        "data_last": ctx.data.index[-1].date()}
            state = self._init(ctx)
        session = state.session
        session.on_event = self._on_event
        strategy = session.strategy
        self._refresh_inputs(strategy, ctx)

        data = ctx.data
        todo = [i for i, t in enumerate(data.index)
                if self.spec.start <= t.date() <= cutoff
                and (state.last_bar is None or t.date() > state.last_bar)]
        view = DataView(data, next_trading_day=ctx.calendar.next_trading_day)
        opens, closes = data.panels["open"], data.panels["close"]
        marks = closes.ffill()
        team = self.spec.team

        for i in todo:
            day = data.index[i].date()
            if state.trained_through and day >= date(state.trained_through.year + 1, 1, 1):
                self._fit(state, ctx, date(day.year, 1, 1), first=False)
            self.activity.status("Runner", "working", f"{self.spec.label}: processing {day}")
            held_before = list(getattr(strategy, "_held", None) or session.account.positions)
            self._sync_halt(session, day)
            r = session.step(view, i, opens, marks)
            for f in r.fills:
                self.store.record_fill(f)
            self.store.record_equity(_at(day, CLOSE_TIME), self.spec.name, r.equity, session.account.cash)
            if r.decided:
                weights = {s: q for s, q in (session.pending or {}).items() if q > 0}
                close_ts = _at(day, CLOSE_TIME)
                for thought in explain_decision(getattr(strategy, "last_decision", None),
                                                weights, held_before):
                    self.say("think", self.spec.agent, thought, ts=close_ts)
                pricey = [s for s, q in (session.pending or {}).items()
                          if q == 0 and s not in session.account.positions and s in (getattr(
                              strategy, "_weights", None) or {})]
                if pricey:
                    slot = r.equity * next(iter(strategy._weights.values()))
                    self.say("think", self.spec.agent,
                             f"Can't afford {', '.join(f'{s} (Rs {marks[s].iloc[i]:,.0f}/share)' for s in pricey)}: "
                             f"one share costs more than a slot of Rs {slot:,.0f}. That slot stays in cash.",
                             ts=close_ts)
                if weights or session.account.positions:
                    orders = [f"{s} {q}" for s, q in sorted(weights.items())]
                    self.say("talk", self.spec.agent,
                             f"New targets for {ctx.calendar.next_trading_day(day)} open "
                             f"({len(weights)} stocks, shares): {', '.join(orders) or 'all cash'}",
                             ts=close_ts, to="Risk engine", details={"targets": session.pending})
            state.last_bar = day

        prices = {s: float(marks[s].iloc[-1]) for s in session.account.positions if s in marks}
        self.store.record_positions(self.spec.name, now, [
            (s, p.qty, p.avg_price, prices.get(s)) for s, p in session.account.positions.items()])
        last = data.index[-1].date()
        self.store.upsert_book(
            name=self.spec.name, team=team, label=self.spec.label, status=self.spec.status,
            capital=self.spec.capital, start=str(self.spec.start), strategy=strategy.name,
            version=state.version, last_bar=str(state.last_bar), last_run=now.isoformat(timespec="seconds"),
            note=self.spec.note)
        self.save(state)

        equity = session.account.equity(prices)
        if todo:
            self.say("act", "Runner", f"{self.spec.label}: processed {len(todo)} day(s) up to {state.last_bar}. "
                     f"Equity Rs {equity:,.0f} ({equity / self.spec.capital - 1:+.2%} since start), "
                     f"{len(session.account.positions)} positions.", ts=now)
        next_day = ctx.calendar.next_trading_day(max(cutoff, last))
        self.activity.status(self.spec.agent, "idle",
                             f"Next look at {next_day}" + (" (month-end decision)"
                                                           if ctx.calendar.is_month_end(next_day) else ""),
                             team)
        return {"book": self.spec.name, "days": len(todo), "last_bar": state.last_bar, "equity": equity,
                "data_last": last}


class RunnerBusy(RuntimeError):
    pass


LOCK_PORT = 47654


def run_books(specs: list[BookSpec], db_path: Path, state_dir: Path, ctx: LiveContext | None = None,
              now: datetime | None = None, control=None) -> list[dict]:
    """Advance every book. Only one run at a time on this machine (state files)."""
    lock = socket.socket()
    try:
        lock.bind(("127.0.0.1", LOCK_PORT))
    except OSError as e:
        lock.close()
        raise RunnerBusy("another runner is already running") from e
    try:
        return _run_books(specs, db_path, state_dir, ctx, now, control)
    finally:
        lock.close()


def _run_books(specs, db_path, state_dir, ctx, now, control=None) -> list[dict]:
    now = now or datetime.now(IST)
    store, activity = Storage(db_path), ActivityLog(db_path)
    try:
        activity.status("Runner", "working", "Loading market data")
        ctx = ctx or load_context(with_meetings=any(s.needs_meetings for s in specs))
        cal = ctx.calendar
        last = ctx.data.index[-1].date()
        expected = now.date() if now.time() >= BAR_READY and cal.is_trading_day(now.date()) else None
        if expected is None:
            d = now.date() - timedelta(days=1)
            while not cal.is_trading_day(d):
                d -= timedelta(days=1)
            expected = d
        if last < expected:
            activity.log("alert", "Runner", f"Price data ends {last}, expected {expected}. "
                         "Yahoo may be late; the runner will catch up next time.", ts=now)
        if not cal.known(now.year):
            activity.log("alert", "Runner", f"NSE holiday list for {now.year} unavailable; "
                         "assuming every weekday trades.", ts=now)
        activity.status("Risk engine", "idle", "Checks every order. No AI can override it.")
        activity.status("Broker", "idle", "Paper fills at the real open price, with Indian costs")
        out = []
        for spec in specs:
            try:
                out.append(LiveBook(spec, store, activity, state_dir, control).advance(ctx, now))
            except Exception as e:
                activity.log("alert", "Runner", f"{spec.label} failed: {type(e).__name__}: {e}",
                             team=spec.team, book=spec.name, ts=now)
                activity.status(spec.agent, "error", f"{type(e).__name__}: {e}"[:200], spec.team)
                out.append({"book": spec.name, "error": f"{type(e).__name__}: {e}"})
        activity.status("Runner", "idle", f"Last run {now:%d %b %H:%M}, data up to {last}")
        return out
    except Exception as e:
        activity.status("Runner", "error", f"{type(e).__name__}: {e}"[:200])
        activity.log("alert", "Runner", f"Run failed: {type(e).__name__}: {e}", ts=now)
        raise
    finally:
        store.close()
        activity.close()


LIVE_DIR = DATA_DIR / "live"
DRYRUN_DIR = DATA_DIR / "dryrun"
