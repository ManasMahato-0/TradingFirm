"""Intraday Stocks team, step 1: the five baselines, walk-forward, after real costs.

    .venv\\Scripts\\python scripts\\research_intraday.py           # logged research trials
    .venv\\Scripts\\python scripts\\research_intraday.py --quick   # smoke test, not logged

Data: our recorded 1-minute bars -> 5-minute bars, the 50 most traded Nifty 100 stocks
(picked from the first 3 months only). Research window: first data to the intraday lock
(14 Aug 2026). Walk-forward: 1 month warm-up, then 1-month test windows rolling forward.

Each strategy is also run with zero costs (not logged) to show whether there is any edge
before costs at all. The bar to beat is cash: the team is flat every night.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from datetime import date, datetime
from datetime import time as dtime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from firm.agents.researcher import ROUTINE  # noqa: E402
from firm.backtest.engine import BacktestConfig  # noqa: E402
from firm.backtest.intraday import liquid_universe, load_5m  # noqa: E402
from firm.backtest.lab import LOCK_START, Lab  # noqa: E402
from firm.backtest.registry import TrialRegistry  # noqa: E402
from firm.broker import PaperBroker  # noqa: E402
from firm.config import DATA_DIR, IST, RECORDER_DB_PATH, TEAMS  # noqa: E402
from firm.costs import CostRates, Segment  # noqa: E402
from firm.strategies.intraday import ORB, FilingReaction, Gap, NiftyMomentum, VWAPRevert  # noqa: E402
from firm.universe import INDEXES, nifty100  # noqa: E402

PASS = {"sharpe": 0.5, "max_drawdown": -0.15, "month_win_rate": 0.60, "deflated_sharpe": 0.95}


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def zero_cost_broker() -> PaperBroker:
    """No fees, no slippage: shows whether a strategy has any edge before costs."""
    zero = CostRates(brokerage_flat=0.0, brokerage_pct=None, stt_buy_pct=0.0, stt_sell_pct=0.0,
                     exchange_pct=0.0, stamp_buy_pct=0.0, sebi_per_crore=0.0, gst_pct=0.0)
    return PaperBroker(slippage_bps={s: 0.0 for s in Segment}, cost_rates={s: zero for s in Segment})


def load_events(symbols: list[str], start: date, end: date) -> pd.DataFrame:
    conn = sqlite3.connect(str(RECORDER_DB_PATH))
    ev = pd.read_sql_query("SELECT symbol, category, exchange_time FROM announcements"
                           " WHERE exchange_time >= ? AND exchange_time < ?", conn,
                           params=(start.isoformat(), end.isoformat()))
    ev = ev[ev.symbol.isin(symbols) & ~ev.category.isin(ROUTINE)].dropna(subset=["exchange_time"])
    ev["ts"] = pd.to_datetime(ev.exchange_time, utc=True).dt.tz_convert("Asia/Kolkata")
    t = ev.ts.dt.time
    ev = ev[(t >= dtime(9, 15)) & (t < dtime(14, 30))]
    return ev[["symbol", "category", "ts"]].sort_values("ts").reset_index(drop=True)


def month_win_rate(wf, cash_rate: float = 0.065) -> float:
    wins = [r.equity.iloc[-1] / r.initial_capital - 1 > cash_rate / 12 for r in wf.fold_results]
    return float(np.mean(wins)) if wins else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--strategies", nargs="+", default=None,
                    help="which strategies to run and LOG as trials (default: the six baselines). "
                         "Variants: gap_fade_big orb_filtered. Pass only what you intend to log.")
    args = ap.parse_args()

    raw = load_5m(nifty100() + INDEXES)
    nifty = raw.panels["close"]["NIFTY50"].dropna()
    stocks = raw.without(INDEXES)
    universe = liquid_universe(stocks, 50)
    data = stocks.without([s for s in stocks.symbols if s not in universe])
    start, end = data.index[0].date(), LOCK_START["intraday"]
    if args.quick:
        end = date(2026, 2, 17)
    events = load_events(universe, start, end)
    log(f"data: {len(universe)} stocks, {len(set(data.index.date))} days, {start} to {end}; "
        f"{len(events)} in-session filings")

    team = TEAMS["intraday_stocks"]
    cfg = BacktestConfig(team="intraday_stocks", segment=Segment.EQ_INTRADAY, flatten_after=dtime(15, 10))
    registry = TrialRegistry(":memory:") if args.quick else None
    lab = Lab(data, "intraday", team.limits, config=cfg, benchmark=nifty, registry=registry)
    gross_lab = Lab(data, "intraday", team.limits, config=cfg, registry=TrialRegistry(":memory:"),
                    broker=zero_cost_broker())

    def named(strategy, name):
        strategy.name = name
        return strategy

    strategies = {
        "orb": lambda: ORB(nifty),
        "nifty_momentum": lambda: NiftyMomentum(nifty, basket=universe[:4]),
        "gap_fade": lambda: Gap(nifty, mode="fade"),
        "gap_follow": lambda: Gap(nifty, mode="follow"),
        "vwap_revert": lambda: VWAPRevert(nifty),
        "filing_reaction": lambda: FilingReaction(nifty, events=events),
        # Pre-registered 2026-11-15 in docs/PLAN.md (settings fixed before any run).
        "gap_fade_big": lambda: named(Gap(nifty, mode="fade", min_gap=0.03), "gap_fade_big"),
        "orb_filtered": lambda: named(ORB(nifty, market_filter=True, max_entries_per_day=1), "orb_filtered"),
    }
    baselines = ["orb", "nifty_momentum", "gap_fade", "gap_follow", "vwap_revert", "filing_reaction"]
    chosen = args.strategies or baselines
    unknown = [n for n in chosen if n not in strategies]
    if unknown:
        raise SystemExit(f"unknown strategies {unknown}; choose from {list(strategies)}")
    strategies = {n: strategies[n] for n in chosen}
    log(f"running and logging: {list(strategies)}")
    results = {}
    for name, factory in strategies.items():
        t0 = time.time()
        wf = lab.walk_forward(factory, start, end, train_months=1, test_months=1)
        gross = gross_lab.walk_forward(factory, start, end, train_months=1, test_months=1)
        m = dict(wf.metrics)
        days = len(set(wf.returns.index))
        m["month_win_rate"] = month_win_rate(wf)
        m["round_trips"] = sum(len(r.fills) for r in wf.fold_results) / 2
        m["trades_per_day"] = m["round_trips"] / max(days, 1)
        m["gross_cagr"] = gross.metrics["cagr"]
        m["gross_sharpe"] = gross.metrics["sharpe"]
        results[name] = m
        log(f"{name}: {time.time() - t0:.0f}s | net CAGR {m['cagr']:.1%} Sharpe {m['sharpe']:.2f} | "
            f"before costs CAGR {m['gross_cagr']:.1%} Sharpe {m['gross_sharpe']:.2f} | "
            f"{m['trades_per_day']:.1f} round trips/day")

    rows = [("gross_cagr", "Yearly, before costs", "{:.1%}"), ("gross_sharpe", "Sharpe, before costs", "{:.2f}"),
            ("cagr", "Yearly, after costs", "{:.1%}"), ("sharpe", "Sharpe, after costs", "{:.2f}"),
            ("max_drawdown", "Max drawdown", "{:.1%}"), ("month_win_rate", "Months beating cash", "{:.0%}"),
            ("round_trips", "Round trips (total)", "{:.0f}"), ("trades_per_day", "Round trips/day", "{:.2f}"), ("win_rate", "Winning trades", "{:.0%}"),
            ("costs_paid", "Costs (Rs)", "{:,.0f}"), ("deflated_sharpe", "Deflated Sharpe", "{:.2f}")]
    names = list(results)
    print(f"\nIntraday walk-forward {start} to {end} (1-month tests), Rs 1L, 50 most traded stocks\n")
    print(f"{'':22}" + "".join(f"{n[:15]:>16}" for n in names))
    for key, label, fmt in rows:
        cells = []
        for n in names:
            v = results[n].get(key)
            cells.append("-" if v is None or (isinstance(v, float) and not math.isfinite(v)) else fmt.format(v))
        print(f"{label:22}" + "".join(f"{c:>16}" for c in cells))
    print(f"\nPass: Sharpe > {PASS['sharpe']}, max DD > {PASS['max_drawdown']:.0%}, months beating cash >= "
          f"{PASS['month_win_rate']:.0%}, deflated Sharpe > {PASS['deflated_sharpe']}")
    for n in names:
        m = results[n]
        checks = [m["sharpe"] > PASS["sharpe"], m["max_drawdown"] > PASS["max_drawdown"],
                  m["month_win_rate"] >= PASS["month_win_rate"], m["deflated_sharpe"] > PASS["deflated_sharpe"]]
        enough = m["round_trips"] >= 30
        # Promotion (pre-registered 2026-11-15): the first three gates AND at least 30 round trips.
        # Deflated Sharpe is reported but excluded, as in the swing probation precedent.
        promote = all(checks[:3]) and enough
        print(f"  {n:16} {'PROMOTE' if promote else 'no'}  " + " ".join("Y" if c else "n" for c in checks)
              + f"   enough trades ({m['round_trips']:.0f} >= 30): {'Y' if enough else 'n'}")

    out = DATA_DIR / "research" / f"intraday_{datetime.now(IST):%Y%m%d_%H%M}{'_quick' if args.quick else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"start": str(start), "end": str(end), "universe": universe,
                               "results": results}, indent=2, default=str))
    log(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
