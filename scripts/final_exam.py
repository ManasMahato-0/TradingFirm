"""One-shot final exam on the locked period for a swing strategy.

    .venv\\Scripts\\python scripts\\final_exam.py momentum

Each strategy version can take this exactly once (enforced by the trial registry).
Pass criteria are fixed here, before the run:
  1. max drawdown no worse than the worst drawdown seen in research (walk-forward)
  2. total return within 5 percentage points of equal-weight Nifty 100 over the same period
  3. annual turnover at most 2x the research turnover
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from firm.backtest.benchmarks import buy_and_hold, equal_weight  # noqa: E402
from firm.backtest.clean import clean  # noqa: E402
from firm.backtest.data import MarketData  # noqa: E402
from firm.backtest.lab import LOCK_START, RESEARCH_DB, Lab  # noqa: E402
from firm.backtest.metrics import compute_metrics  # noqa: E402
from firm.backtest.registry import strategy_version  # noqa: E402
from firm.config import TEAMS  # noqa: E402
from firm.strategies.baselines import Momentum  # noqa: E402
from firm.universe import INDEXES, nifty100  # noqa: E402

STRATEGIES = {"momentum": Momentum}
RETURN_TOLERANCE = 0.05
TURNOVER_MULTIPLE = 2.0


def research_reference(version: str) -> dict:
    with sqlite3.connect(RESEARCH_DB) as c:
        row = c.execute("SELECT metrics FROM trials WHERE version = ? AND kind = 'walkforward'"
                        " ORDER BY id DESC LIMIT 1", (version,)).fetchone()
    if row is None:
        raise SystemExit("no walk-forward research result for this strategy version; run research first")
    return json.loads(row[0])


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "momentum"
    strategy = STRATEGIES[name]()
    version = strategy_version(strategy)
    ref = research_reference(version)

    raw = MarketData.from_store("1d_yahoo", nifty100() + INDEXES, start=date(2005, 1, 1))
    nifty = raw.panels["close"]["NIFTY50"].dropna()
    data, _ = clean(raw.without(INDEXES))
    lab = Lab(data, "swing", TEAMS["swing_stocks"].limits, benchmark=nifty)

    res = lab.final_exam(strategy)
    m = res.metrics
    start, end = LOCK_START["swing"], data.index[-1].date() + timedelta(days=1)
    ew = compute_metrics(equal_weight(data.panels["close"], start, end))
    nf = compute_metrics(buy_and_hold(nifty, start, end))

    checks = {
        f"max drawdown {m['max_drawdown']:.1%} >= research worst {ref['max_drawdown']:.1%}":
            m["max_drawdown"] >= ref["max_drawdown"],
        f"return {m['total_return']:.1%} >= equal-weight {ew['total_return']:.1%} - {RETURN_TOLERANCE:.0%}":
            m["total_return"] >= ew["total_return"] - RETURN_TOLERANCE,
        f"turnover {m['annual_turnover']:.2f}/yr <= {TURNOVER_MULTIPLE}x research {ref['annual_turnover']:.2f}":
            m["annual_turnover"] <= TURNOVER_MULTIPLE * ref["annual_turnover"],
    }
    print(f"FINAL EXAM: {strategy.name} (version {version}), locked period {start} to {end}\n")
    print(f"{'':22}{'strategy':>12}{'equal-wt':>12}{'nifty 50':>12}")
    for key, label in [("total_return", "Return"), ("max_drawdown", "Max drawdown"),
                       ("volatility", "Volatility (yr)"), ("sharpe", "Sharpe")]:
        print(f"{label:22}{m[key]:>12.2f}{ew[key]:>12.2f}{nf[key]:>12.2f}" if key == "sharpe" else
              f"{label:22}{m[key]:>12.1%}{ew[key]:>12.1%}{nf[key]:>12.1%}")
    print(f"{'Trades closed':22}{m['trades']:>12}\n{'Costs (Rs)':22}{m['costs_paid']:>12,.0f}\n")
    for text, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {text}")
    print(f"\nRESULT: {'PASSED' if all(checks.values()) else 'FAILED'}")
    holdings = sorted({f.symbol for f in res.fills if f.side.value == 'BUY' and f.ts.date() >= end - timedelta(days=40)})
    print(f"Most recent buys: {', '.join(holdings)}")


if __name__ == "__main__":
    main()
