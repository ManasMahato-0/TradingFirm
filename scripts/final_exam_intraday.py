"""One-shot final exam on the locked intraday window for gap_fade_big.

    .venv\Scripts\python scripts\final_exam_intraday.py

Each strategy version can take this exactly once (enforced by the trial registry).
Pass criteria were fixed in docs/PLAN.md before the run:
  1. net return over the window beats cash (6.5%/yr pro rata)
  2. max drawdown no worse than -5%
  3. round trips per day at most 2x the research rate
  4. at least 10 round trips, otherwise INCONCLUSIVE
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import timedelta
from datetime import time as dtime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from firm.backtest.engine import BacktestConfig  # noqa: E402
from firm.backtest.intraday import liquid_universe, load_5m  # noqa: E402
from firm.backtest.lab import LOCK_START, RESEARCH_DB, Lab  # noqa: E402
from firm.backtest.registry import strategy_version  # noqa: E402
from firm.config import TEAMS  # noqa: E402
from firm.costs import Segment  # noqa: E402
from firm.strategies.intraday import Gap  # noqa: E402
from firm.universe import INDEXES, nifty100  # noqa: E402

CASH_RATE = 0.065
MAX_DD = -0.05
TRADE_RATE_MULTIPLE = 2.0
MIN_ROUND_TRIPS = 10
RESEARCH_TRADES_PER_DAY = 78 / 162  # 78 round trips over the research days (logged 2026-11-15)


def main() -> None:
    raw = load_5m(nifty100() + INDEXES)
    nifty = raw.panels["close"]["NIFTY50"].dropna()
    stocks = raw.without(INDEXES)
    universe = liquid_universe(stocks, 50)  # chosen from the first 3 months only, as in research
    data = stocks.without([s for s in stocks.symbols if s not in universe])

    strategy = Gap(nifty, mode="fade", min_gap=0.03)
    strategy.name = "gap_fade_big"
    version = strategy_version(strategy)
    with sqlite3.connect(RESEARCH_DB) as c:
        row = c.execute("SELECT metrics FROM trials WHERE version=? AND kind='walkforward' ORDER BY id DESC LIMIT 1",
                        (version,)).fetchone()
    if row is None:
        raise SystemExit(f"no walk-forward research for version {version}; refusing to run the exam")
    ref = json.loads(row[0])

    cfg = BacktestConfig(team="intraday_stocks", segment=Segment.EQ_INTRADAY, flatten_after=dtime(15, 10))
    lab = Lab(data, "intraday", TEAMS["intraday_stocks"].limits, config=cfg, benchmark=nifty)
    res = lab.final_exam(strategy)  # raises if this version already took the exam
    m = res.metrics

    days = len(set(res.equity.index.date))
    cal_days = (res.equity.index[-1].date() - LOCK_START["intraday"]).days + 1
    round_trips = len(res.fills) / 2
    rate = round_trips / max(days, 1)
    cash_ret = CASH_RATE * cal_days / 365
    limit_rate = TRADE_RATE_MULTIPLE * RESEARCH_TRADES_PER_DAY

    print(f"FINAL EXAM: {strategy.name} (version {version}), locked window {LOCK_START['intraday']} "
          f"to {res.equity.index[-1].date()} ({days} trading days)\n")
    print(f"Return {m['total_return']:.2%} | cash over same period {cash_ret:.2%} | max drawdown {m['max_drawdown']:.2%}")
    print(f"Sharpe {m['sharpe']:.2f} (not a gate) | round trips {round_trips:.0f} ({rate:.2f}/day) | "
          f"costs Rs {m['costs_paid']:,.0f} | win rate {m.get('win_rate', float('nan')):.0%}\n")
    if round_trips < MIN_ROUND_TRIPS:
        print(f"RESULT: INCONCLUSIVE (only {round_trips:.0f} round trips, need {MIN_ROUND_TRIPS}); exam consumed")
        return
    checks = {
        f"return {m['total_return']:.2%} > cash {cash_ret:.2%}": m["total_return"] > cash_ret,
        f"max drawdown {m['max_drawdown']:.2%} >= {MAX_DD:.0%}": m["max_drawdown"] >= MAX_DD,
        f"round trips {rate:.2f}/day <= {limit_rate:.2f}/day (2x research)": rate <= limit_rate,
    }
    for text, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {text}")
    print(f"\nRESULT: {'PASSED' if all(checks.values()) else 'FAILED'}")
    last = sorted({(f.ts.date(), f.symbol, f.side.value) for f in res.fills})[-6:]
    print("Most recent fills:", last)


if __name__ == "__main__":
    main()
