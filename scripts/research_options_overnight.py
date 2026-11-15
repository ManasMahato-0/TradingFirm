"""Overnight Options, phase 2: three baselines on real NSE option prices (docs/PLAN.md, pre-registered).

    .venv\\Scripts\\python scripts\\research_options_overnight.py           # 3 logged trials, 2019 to 13 May 2026
    .venv\\Scripts\\python scripts\\research_options_overnight.py --quick   # smoke test, not logged

Pass rule (research window): CAGR > 6.5% cash; Sharpe > 0.5; max drawdown better than -25%;
at least 60% of calendar years beating cash; at least 100 trades. Deflated Sharpe is reported, not a gate.
The locked window (14 May 2026 on) is never read here.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from firm.backtest.data import MarketData  # noqa: E402
from firm.backtest.lab import RESEARCH_DB  # noqa: E402
from firm.backtest.metrics import compute_metrics, daily_returns  # noqa: E402
from firm.backtest.options_overnight import (  # noqa: E402
    AlwaysCall, DipCall, TrendCallPut, run_overnight, years_beating_cash,
)
from firm.backtest.registry import TrialRegistry, strategy_version  # noqa: E402
from firm.config import DATA_DIR, IST  # noqa: E402
from firm.recorders.fo import load_options  # noqa: E402

TRACK = "options_overnight"
START, LOCK = date(2019, 1, 1), date(2026, 5, 14)
CASH = 0.065
PASS = {"cagr": CASH, "sharpe": 0.5, "max_drawdown": -0.25, "years_beating_cash": 0.60, "trades": 100}


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def summarize(res, capital: float = 100_000.0) -> dict:
    rets = daily_returns(res.equity, capital)
    m = compute_metrics(rets, res.fills, res.exposure, None, initial_capital=capital, equity_by_run=[res.equity])
    share, per_year = years_beating_cash(res.equity)
    tr = res.trades
    m["years_beating_cash"] = share
    m["per_year"] = {y: {"return": v[0], "cash": v[1], "beat": v[2]} for y, v in per_year.items()}
    m["n_trades"] = len(tr)
    m["nights_signalled"], m["nights_skipped"] = res.nights, res.skipped
    m["avg_premium_return_pct"] = 100 * float(np.mean([t.premium_return for t in tr])) if tr else float("nan")
    m["avg_spot_move_pct"] = 100 * float(np.nanmean([t.spot_move for t in tr])) if tr else float("nan")
    m["win_rate_gross"] = float(np.mean([t.premium_return > 0 for t in tr])) if tr else float("nan")
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    start, end = (date(2019, 3, 1), date(2019, 11, 1)) if args.quick else (START, LOCK)

    chain = load_options(start, end)
    if chain.empty:
        raise SystemExit("no option data: run scripts/fo_backfill.py first")
    log(f"options: {len(chain):,} contract-days, {chain['date'].nunique()} trading days, "
        f"{chain['date'].min()} to {chain['date'].max()}")
    spot = MarketData.from_store("1d_yahoo", ["NIFTY50"], start=date(2018, 1, 1)).panels["close"]["NIFTY50"].dropna()

    registry = TrialRegistry(":memory:" if args.quick else RESEARCH_DB)
    rules = [AlwaysCall(), TrendCallPut(), DipCall()]
    results = {}
    for rule in rules:
        net = run_overnight(rule, chain, spot, start, end)
        gross = run_overnight(rule, chain, spot, start, end, with_costs=False, cash_rate=CASH)
        n_trials, var = registry.trial_stats(TRACK)
        m = summarize(net)
        gm = summarize(gross)
        m["gross_cagr"], m["gross_sharpe"] = gm["cagr"], gm["sharpe"]
        registry.log(TRACK, rule.name, strategy_version(rule), rule.params, "research", start, end,
                     {k: v for k, v in m.items() if not isinstance(v, dict)})
        results[rule.name] = m
        log(f"{rule.name}: net CAGR {m['cagr']:.1%} Sharpe {m['sharpe']:.2f} | before costs CAGR {gm['cagr']:.1%} "
            f"Sharpe {gm['sharpe']:.2f} | {m['n_trades']} trades")

    names = list(results)
    rows = [("n_trades", "Trades", "{:.0f}"), ("nights_skipped", "Nights skipped (no price)", "{:.0f}"),
            ("gross_cagr", "Yearly, before costs", "{:.1%}"), ("gross_sharpe", "Sharpe, before costs", "{:.2f}"),
            ("cagr", "Yearly, after costs", "{:.1%}"), ("sharpe", "Sharpe, after costs", "{:.2f}"),
            ("max_drawdown", "Max drawdown", "{:.1%}"), ("years_beating_cash", "Years beating cash", "{:.0%}"),
            ("win_rate_gross", "Nights the option rose", "{:.0%}"), ("avg_premium_return_pct", "Avg premium move/night %", "{:.2f}"),
            ("avg_spot_move_pct", "Avg Nifty move on those nights %", "{:.3f}"), ("costs_paid", "Costs paid (Rs)", "{:,.0f}"),
            ("deflated_sharpe", "Deflated Sharpe", "{:.2f}")]
    print(f"\nOvernight Nifty options, {start} to {end}, Rs 1L, 10% per trade\n")
    print(f"{'':32}" + "".join(f"{n[:22]:>24}" for n in names))
    for key, label, fmt in rows:
        cells = []
        for n in names:
            v = results[n].get(key)
            cells.append("-" if v is None or (isinstance(v, float) and not math.isfinite(v)) else fmt.format(v))
        print(f"{label:32}" + "".join(f"{c:>24}" for c in cells))
    print(f"\nPass rule: CAGR > {PASS['cagr']:.1%}, Sharpe > {PASS['sharpe']}, max DD better than "
          f"{PASS['max_drawdown']:.0%}, years beating cash >= {PASS['years_beating_cash']:.0%}, trades >= {PASS['trades']}")
    for n in names:
        m = results[n]
        checks = [m["cagr"] > PASS["cagr"], m["sharpe"] > PASS["sharpe"], m["max_drawdown"] > PASS["max_drawdown"],
                  m["years_beating_cash"] >= PASS["years_beating_cash"], m["n_trades"] >= PASS["trades"]]
        print(f"  {n:26} {'PASS' if all(checks) else 'fail'}  " + " ".join("Y" if c else "n" for c in checks)
              + "   (CAGR, Sharpe, DD, years, trades)")

    out = DATA_DIR / "research" / f"options_overnight_{datetime.now(IST):%Y%m%d_%H%M}{'_quick' if args.quick else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"start": str(start), "end": str(end), "results": results}, indent=2, default=str))
    log(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
