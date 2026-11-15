"""Run baseline strategies on the swing research period and compare with benchmarks.

    .venv\\Scripts\\python scripts\\backtest_baselines.py
"""
from __future__ import annotations

import math
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from firm.backtest.benchmarks import buy_and_hold, equal_weight  # noqa: E402
from firm.backtest.clean import clean  # noqa: E402
from firm.backtest.data import MarketData  # noqa: E402
from firm.backtest.lab import LOCK_START, Lab  # noqa: E402
from firm.backtest.metrics import compute_metrics  # noqa: E402
from firm.config import TEAMS  # noqa: E402
from firm.strategies.baselines import Momentum, TrendFollow  # noqa: E402
from firm.universe import INDEXES, nifty100  # noqa: E402

START = date(2008, 1, 1)
ROWS = [
    ("total_return", "Total return", "pct"), ("cagr", "Yearly growth (CAGR)", "pct"),
    ("volatility", "Volatility", "pct"), ("sharpe", "Sharpe", "num"), ("sortino", "Sortino", "num"),
    ("max_drawdown", "Max drawdown", "pct"), ("calmar", "Calmar", "num"),
    ("excess_cagr", "CAGR vs Nifty", "pct"), ("trades", "Closed trades", "int"),
    ("win_rate", "Win rate", "pct"), ("profit_factor", "Profit factor", "num"),
    ("costs_paid", "Costs paid (Rs)", "int"), ("annual_turnover", "Turnover / year", "num"),
    ("exposure", "Time invested", "pct"), ("deflated_sharpe", "Deflated Sharpe", "num"),
]


def fmt(v, kind):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    if kind == "pct":
        return f"{v:.1%}"
    if kind == "int":
        return f"{v:,.0f}"
    return f"{v:.2f}"


def main() -> None:
    t0 = time.time()
    end = LOCK_START["swing"]
    data = MarketData.from_store("1d_yahoo", nifty100() + INDEXES, start=date(2005, 1, 1))
    nifty = data.panels["close"]["NIFTY50"]
    trade_data = data.without(INDEXES)
    print(f"Loaded {len(trade_data.symbols)} stocks, {len(data.index)} days "
          f"({data.index[0].date()} to {data.index[-1].date()}) in {time.time() - t0:.1f}s")

    trade_data, report = clean(trade_data)
    print(f"Cleaning: {len(report['spikes_removed'])} bad-tick bars removed, "
          f"{len(report['breaks_adjusted'])} corporate-action breaks adjusted\n")

    lab = Lab(trade_data, "swing", TEAMS["swing_stocks"].limits, benchmark=nifty)
    cols = {
        "Nifty 50 (hold)": compute_metrics(buy_and_hold(nifty, START, end), benchmark_rets=lab.bench_rets),
        "Equal-wt N100*": compute_metrics(equal_weight(trade_data.panels["close"], START, end),
                                          benchmark_rets=lab.bench_rets),
    }
    for strat in (Momentum(), TrendFollow()):
        t1 = time.time()
        res = lab.backtest(strat, START, end)
        cols[strat.name] = res.metrics
        print(f"{strat.name}: {len(res.fills)} fills, {len(res.rejections)} risk rejections, "
              f"{time.time() - t1:.0f}s")
        if res.rejections:
            reasons = pd.Series([r["reason"].split(":")[0] for r in res.rejections]).value_counts()
            print("   top rejection reasons:", dict(reasons.head(3)))

    print(f"\nSwing research period {START} to {end} (after that is locked)")
    print("Benchmarks have no costs or limits; strategies pay real costs under team risk limits.\n")
    names = list(cols)
    print(f"{'':22}" + "".join(f"{n:>18}" for n in names))
    for key, label, kind in ROWS:
        print(f"{label:22}" + "".join(f"{fmt(cols[n].get(key), kind):>18}" for n in names))
    print("\n* Equal-weight of today's Nifty 100 members: survivorship bias inflates it.")


if __name__ == "__main__":
    main()
