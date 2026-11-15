"""Intraday sandbox runner: four paper accounts (docs/PLAN.md, pre-registered).

  A  model trained on history       C  random trader (cost floor)
  B  blank learner (live data only) D  cash

Each completed 5-minute bar goes through the same `Session.step` the backtests use, so a
replayed day equals a backtest of that day (decide at a bar's close, fill at the next
bar's open). The quote at each decision is logged separately to measure feed latency.
State is saved after every bar, so a restart resumes mid-day.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

from firm.activity import ActivityLog
from firm.backtest.data import DataView, MarketData
from firm.backtest.engine import BacktestConfig, Session
from firm.backtest.registry import strategy_version
from firm.broker import Order
from firm.calendar import TradingCalendar
from firm.config import DATA_DIR, IST, TEAMS
from firm.costs import Segment, Side
from firm.features.intraday_model import build_training_set, train_model
from firm.live.sandbox_feed import BAR, BarSource, append_bar, assemble
from firm.storage import Storage
from firm.strategies.intraday_sandbox import THRESHOLD, CashOnly, ModelStrategy, RandomTrader

SANDBOX_DIR = DATA_DIR / "sandbox"
LAUNCH = date(2026, 11, 23)       # pre-registered
WARMUP_DAYS = 10                  # B may not trade for its first 10 trading days
MIN_B_ROWS = 5_000
QUOTE_WINDOW = timedelta(minutes=5)
ENTRY_PERCENTILE = 99.9   # amendment of 2026-11-15: enter on a model's most extreme 0.1% of predictions
TEAM = "intraday_stocks"
CAPITAL = 100_000.0

SPECS = {
    "sandbox_A": ("Sandbox A: trained on history", "Sandbox A (history)"),
    "sandbox_B": ("Sandbox B: blank learner", "Sandbox B (blank)"),
    "sandbox_C": ("Sandbox C: random trader", "Sandbox C (random)"),
    "sandbox_D": ("Sandbox D: cash", "Sandbox D (cash)"),
}
NOTE = ("Exploratory live forward test, not a proven team. Works only if it beats D and C with Sharpe "
        "> 0.5 at day 60 and 120. Rules: docs/PLAN.md")


@dataclass
class TrainData:
    stocks: MarketData
    nifty_close: pd.Series
    nifty_open: pd.Series


def load_train_data(universe: list[str] | None = None) -> tuple[TrainData, list[str]]:
    from firm.backtest.intraday import liquid_universe, load_5m
    from firm.universe import INDEXES, nifty100

    raw = load_5m(nifty100() + INDEXES)
    stocks = raw.without(INDEXES)
    universe = universe or liquid_universe(stocks, 50)
    data = stocks.without([s for s in stocks.symbols if s not in universe])
    return TrainData(data, raw.panels["close"]["NIFTY50"].dropna(), raw.panels["open"]["NIFTY50"].dropna()), universe


# ---------------------------------------------------------------- models

class ModelStore:
    """Trained models for A and B on disk, with what each was trained on."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def load(self, account: str):
        meta_path, model_path = self.root / f"{account}.json", self.root / f"{account}.pkl"
        if not meta_path.exists() or not model_path.exists():
            return None, None
        with open(model_path, "rb") as f:
            return pickle.load(f), json.loads(meta_path.read_text())

    def save(self, account: str, model, meta: dict) -> None:
        with open(self.root / f"{account}.pkl", "wb") as f:
            pickle.dump(model, f)
        (self.root / f"{account}.json").write_text(json.dumps(meta, indent=1))


def entry_threshold(model, X: np.ndarray, percentile: float = ENTRY_PERCENTILE) -> float:
    """A model's entry bar: the `percentile` of |prediction| over its own training sample. Set from
    prediction scale only (never from returns), because these models' forecasts are far smaller than
    the 0.25% the first rule asked for (see docs/PLAN.md, amendment of 2026-11-15)."""
    return float(np.percentile(np.abs(model.predict(X)), percentile))


def ensure_models(store: ModelStore, day: date, train: TrainData, launch: date, say=None) -> dict[str, dict]:
    """A: retrained on the first trading day of each month on all data before `day`.
    B: retrained each trading day on data from launch up to (not including) `day`."""
    said = say or (lambda *_: None)
    out = {}
    _, meta_a = store.load("sandbox_A")
    if meta_a is None or tuple(map(int, meta_a["trained_before"].split("-")[:2])) != (day.year, day.month):
        X, y = build_training_set(train.stocks, train.nifty_close, train.nifty_open, end=day)
        model = train_model(X, y)
        meta_a = {"trained_before": day.isoformat(), "rows": int(len(X)), "trained_at": datetime.now(IST).isoformat(),
                  "entry_threshold": entry_threshold(model, X)}
        store.save("sandbox_A", model, meta_a)
        said(f"A retrained on all data before {day}: {len(X):,} samples; it enters when it expects at least "
             f"{meta_a['entry_threshold']:.2%} over 30 minutes.")
    out["A"] = meta_a
    _, meta_b = store.load("sandbox_B")
    if day > launch and (meta_b is None or meta_b["trained_before"] != day.isoformat()):
        X, y = build_training_set(train.stocks, train.nifty_close, train.nifty_open, start=launch, end=day)
        if len(X) >= MIN_B_ROWS:
            model_b = train_model(X, y)
            store.save("sandbox_B", model_b, {"trained_before": day.isoformat(), "rows": int(len(X)),
                                              "trained_at": datetime.now(IST).isoformat(),
                                              "entry_threshold": entry_threshold(model_b, X)})
            said(f"B retrained on live days {launch} to {day - timedelta(days=1)}: {len(X):,} samples.")
        else:
            said(f"B still learning: {len(X):,} samples so far, needs {MIN_B_ROWS:,} before it may trade.")
    out["B"] = store.load("sandbox_B")[1]
    return out


def trade_from_date(launch: date, calendar: TradingCalendar | None = None, warmup: int = WARMUP_DAYS) -> date:
    """First trading day B may trade: after `warmup` full trading days starting at launch."""
    cal = calendar or TradingCalendar(refresh=False)
    d = launch
    for _ in range(warmup):
        d = cal.next_trading_day(d)
    return d


# ---------------------------------------------------------------- accounts

@dataclass
class Account:
    name: str
    label: str
    agent: str
    session: Session
    version: str = ""
    last_ts: pd.Timestamp | None = None
    fills_today: int = 0
    entries_today: int = 0
    start_equity: float = CAPITAL


def _config(name: str) -> BacktestConfig:
    return BacktestConfig(team=name, segment=Segment.EQ_INTRADAY, initial_capital=CAPITAL,
                          flatten_after=dtime(15, 10))


def make_strategy(name: str, model, trade_from: date | None):
    if name == "sandbox_A":
        return ModelStrategy(model=model, label="A")
    if name == "sandbox_B":
        return ModelStrategy(model=model, label="B", trade_from=trade_from)
    if name == "sandbox_C":
        return RandomTrader()
    return CashOnly()


class StopRequested(Exception):
    """The owner stopped the sandbox service. State is saved every bar, so a restart resumes."""


class Sandbox:
    def __init__(self, db_path: Path | str, state_dir: Path = SANDBOX_DIR, launch: date = LAUNCH,
                 calendar: TradingCalendar | None = None, trade_from: date | None = None,
                 control=None) -> None:
        self.control = control      # owner's switches; None = no switches (tests, replays)
        self.db_path, self.dir, self.launch = Path(db_path), state_dir, launch
        self.cal = calendar or TradingCalendar(refresh=False)
        self.trade_from = trade_from or trade_from_date(launch, self.cal)
        self.store, self.activity = Storage(db_path), ActivityLog(db_path)
        self.models = ModelStore(state_dir / "models")
        self.accounts: dict[str, Account] = {}

    # ------------------------------------------------------------ state

    def _path(self, name: str) -> Path:
        return self.dir / name / "session.pkl"

    def _load(self, name: str) -> Account:
        label, agent = SPECS[name]
        path = self._path(name)
        if path.exists():
            with open(path, "rb") as f:
                acct = pickle.load(f)
        else:
            model = self.models.load(name)[0]
            session = Session(make_strategy(name, model, self.trade_from), TEAMS[TEAM].limits, _config(name),
                              agent=agent)
            acct = Account(name, label, agent, session)
            self.say(name, "act", f"Opened sandbox account '{label}' with Rs {CAPITAL:,.0f} (paper).")
        acct.session.on_event = self._events(acct)
        return acct

    def _save(self, acct: Account) -> None:
        s = acct.session.strategy
        held = {a: getattr(s, a) for a in ("model", "nifty", "nifty_open") if hasattr(s, a)}
        for a in held:
            setattr(s, a, None)
        try:
            self._path(acct.name).parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path(acct.name).with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(acct, f)
            tmp.replace(self._path(acct.name))
        finally:
            for a, v in held.items():
                setattr(s, a, v)

    def _events(self, acct: Account):
        def emit(kind, agent, text, ts=None, to=None, details=None):
            self.activity.log(kind, agent, text, team=TEAM, to=to, details=details, ts=ts, book=acct.name)
            if kind == "act" and text.startswith("Filled"):
                acct.fills_today += 1
        return emit

    def say(self, name: str, kind: str, text: str, ts=None) -> None:
        self.activity.log(kind, SPECS[name][1], text, team=TEAM, book=name, ts=ts)

    def _sync_halt(self, acct: Account, day: date) -> None:
        """Owner's pause, halt or skip-day blocks NEW entries only: the account stays flat, still records
        equity every bar and still counts as a recorded session (the day-60/120 count depends on it)."""
        from firm.control import OWNER_HALT_REASON

        halted = bool(self.control and self.control.halted(acct.name, day.isoformat()))
        acct.session.risk.firm_halted = halted
        acct.session.risk.firm_halt_reason = OWNER_HALT_REASON if halted else ""

    # ------------------------------------------------------------ overnight guard

    def force_flat(self, acct: Account, prices: pd.Series, ts: datetime, why: str) -> int:
        """Close any position left from an earlier day at that day's last close (paper)."""
        closed = 0
        account = acct.session.account
        for sym, pos in list(account.positions.items()):
            px = float(prices.get(sym, np.nan))
            if not np.isfinite(px) or px <= 0:
                px = pos.avg_price
            side = Side.SELL if pos.qty > 0 else Side.BUY
            fill = acct.session.broker.execute(
                account, Order(acct.name, sym, Segment.EQ_INTRADAY, side, abs(pos.qty), reason=why), px, ts)
            self.store.record_fill(fill)
            closed += 1
        acct.session.pending = None
        if closed:
            self.say(acct.name, "alert", f"{why}: closed {closed} position(s) left from an earlier day at that "
                     "day's last close. This should never happen; check that the runner finished the last day.", ts=ts)
        return closed

    # ------------------------------------------------------------ day

    def run_day(self, day: date, source: BarSource, train: TrainData | None = None,
                ensure: bool = True, clock=None) -> dict:
        """clock: wall-clock function for live runs. A decision-time quote is logged only if the
        decision happens within QUOTE_WINDOW of the bar's close (a late start catching up on
        old bars would otherwise log quotes taken hours after the decision)."""
        hist = source.history()
        symbols = source.symbols
        if ensure and train is not None:
            ensure_models(self.models, day, train, self.launch, say=lambda t: self.say("sandbox_A", "act", t))
        self.accounts = {n: self._load(n) for n in SPECS}
        last_close = hist[0].panels["close"].ffill().iloc[-1] if len(hist[0].index) else pd.Series(dtype=float)
        prev_end = (hist[0].index[-1] + BAR).to_pydatetime() if len(hist[0].index) else datetime.now(IST)

        for name, acct in self.accounts.items():
            s = acct.session
            if s.account.positions and s.last_day is not None and s.last_day < day:
                self.force_flat(acct, last_close, prev_end, "Overnight guard")
            model, meta = self.models.load(name) if name in ("sandbox_A", "sandbox_B") else (None, None)
            if hasattr(s.strategy, "model"):
                s.strategy.model = model
                s.strategy.trade_from = self.trade_from if name == "sandbox_B" else None
                s.strategy.threshold = (meta or {}).get("entry_threshold", THRESHOLD)
                s.strategy.params["threshold"] = s.strategy.threshold
            if acct.last_ts is None or acct.last_ts.date() != day:
                acct.fills_today = acct.entries_today = 0
                acct.start_equity = s.account.equity({}) if not s.account.positions else acct.start_equity
            acct.version = strategy_version(s.strategy)
            if name == "sandbox_B":
                self.say(name, "act", "Blank learner: " + (
                    f"warm-up until {self.trade_from} (it observes and learns, no trades)." if day < self.trade_from
                    else "warm-up over, trading with a model trained on live days only."
                    if model is not None else "warm-up over but not enough live data yet; staying flat."))

        today: dict[str, list] = {}
        n_bars = 0
        while (bar := source.next_bar()) is not None:
            if self.control is not None:
                self.control.beat("sandbox")
                if not self.control.service_should_run("sandbox"):
                    raise StopRequested()
            append_bar(today, bar)
            data, nc, no = assemble(hist, today, symbols)
            i = len(data.index) - 1
            opens, marks = data.panels["open"], data.panels["close"].ffill()
            view = DataView(data)
            end_ts = (bar.ts + BAR).to_pydatetime()
            wanted: dict[tuple[str, str], str] = {}
            for name, acct in self.accounts.items():
                if acct.last_ts is not None and bar.ts <= acct.last_ts:
                    continue                                   # restart mid-day: already processed
                s = acct.session
                for a in ("nifty", "nifty_open"):
                    if hasattr(s.strategy, a):
                        setattr(s.strategy, a, nc if a == "nifty" else no)
                self._sync_halt(acct, day)
                r = s.step(view, i, opens, marks)
                for f in r.fills:
                    self.store.record_fill(f)
                self.store.record_equity(end_ts, name, r.equity, s.account.cash)
                if r.decided:
                    close_ts = end_ts
                    for line in getattr(s.strategy, "last_reason", []):
                        if line.startswith(("LONG", "SHORT")):
                            acct.entries_today += 1
                        self.say(name, "think", line, ts=close_ts)
                    for sym, tgt in (s.pending or {}).items():
                        delta = tgt - s.account.position_qty(sym)
                        if delta:
                            wanted[(name, sym)] = "BUY" if delta > 0 else "SELL"
                acct.last_ts = bar.ts
                self.activity.status(acct.agent, "working", f"Watching bar {bar.ts:%H:%M}", TEAM)
                self._save(acct)
            timely = clock is None or (clock() - end_ts) <= QUOTE_WINDOW
            if wanted and timely:
                q = source.quotes(sorted({sym for _, sym in wanted}))
                self.store.record_quotes([(n, bar.ts.isoformat(), sym, side, q[sym])
                                          for (n, sym), side in wanted.items() if sym in q])
            n_bars += 1
        return self._finish(day, today, symbols, n_bars)

    def _finish(self, day: date, today: dict, symbols: list[str], n_bars: int) -> dict:
        out = {}
        end = datetime.combine(day, dtime(15, 30), IST)
        close = pd.Series({s: today["close"][-1].get(s, np.nan) for s in symbols}) if today.get("close") else pd.Series()
        for name, acct in self.accounts.items():
            s = acct.session
            if s.account.positions:                           # engine flattens at 15:15; this is the backstop
                self.force_flat(acct, close, end, "End-of-day guard")
            prices = {k: float(v) for k, v in close.items() if pd.notna(v)}
            equity = s.account.equity(prices)
            self.store.record_positions(name, end, [])
            self.store.upsert_book(name=name, team=TEAM, label=acct.label, status="sandbox", capital=CAPITAL,
                                   start=self.launch.isoformat(), strategy=s.strategy.name, version=acct.version,
                                   last_bar=day.isoformat(), last_run=datetime.now(IST).isoformat(timespec="seconds"),
                                   note=NOTE)
            moved = equity - acct.start_equity
            self.say(name, "act", f"Day done {day}: equity Rs {equity:,.0f} ({moved:+,.0f} today), "
                     f"{acct.entries_today} entries, {acct.fills_today} fills.", ts=end)
            self.activity.status(acct.agent, "idle", f"Done for {day}; next session tomorrow", TEAM)
            self._save(acct)
            out[name] = {"equity": equity, "entries": acct.entries_today, "fills": acct.fills_today}
        self._save_bars(day, today, symbols)
        out["bars"] = n_bars
        return out

    def _save_bars(self, day: date, today: dict, symbols: list[str]) -> None:
        """Keep exactly the bars the runner saw, so any day can be replayed from them."""
        if not today.get("index"):
            return
        frames = []
        for t, k in zip(today["index"], range(len(today["index"]))):
            f = pd.DataFrame({fld: today[fld][k] for fld in ("open", "high", "low", "close", "volume")})
            f["ts"], f["symbol"] = t, f.index
            frames.append(f.reset_index(drop=True))
        path = self.dir / "bars" / f"{day.isoformat()}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames).to_parquet(path, index=False)

    def close(self) -> None:
        self.store.close()
        self.activity.close()
