"""Walk-forward training and evaluation for Team D (swing stocks).

    .venv\\Scripts\\python scripts\\train_team_d.py               # full run (hours)
    .venv\\Scripts\\python scripts\\train_team_d.py --quick       # smoke test (minutes)
    .venv\\Scripts\\python scripts\\train_team_d.py --configs ppo_rule lgbm_none

Each config is "<picker>_<dial>[+res]": picker in {ppo, lgbm}, dial in {none, rule, ppo};
"+res" adds results-calendar features (days to/since results, past reactions).
Retrains every year on all data before the test year; tests 2011 to the lock date.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from firm.backtest.benchmarks import buy_and_hold, equal_weight  # noqa: E402
from firm.backtest.clean import clean  # noqa: E402
from firm.backtest.data import MarketData  # noqa: E402
from firm.backtest.lab import LOCK_START, Lab  # noqa: E402
from firm.backtest.metrics import compute_metrics  # noqa: E402
from firm.backtest.registry import TrialRegistry  # noqa: E402
from firm.config import DATA_DIR, TEAMS  # noqa: E402
from firm.features.results import load_results_meetings  # noqa: E402
from firm.strategies.baselines import Momentum  # noqa: E402
from firm.strategies.momentum_plus import MomentumPlus  # noqa: E402
from firm.strategies.swing_rl import SwingStrategy  # noqa: E402
from firm.universe import INDEXES, nifty100, sectors  # noqa: E402

START = date(2008, 1, 1)  # first test year = START + train_years
PASS = {"max_drawdown": -0.45, "fold_win_rate": 0.60, "deflated_sharpe": 0.95}


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def fold_win_rate(wf, ew_rets: pd.Series) -> float:
    wins = 0
    for res in wf.fold_results:
        d = pd.to_datetime(pd.Index(res.equity.index.date).unique())
        strat = res.equity.iloc[-1] / res.initial_capital - 1
        ew = float((1 + ew_rets.reindex(d).fillna(0)).prod() - 1)
        wins += strat > ew
    return wins / len(wf.fold_results) if wf.fold_results else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=[
        "mp_none_none", "mp_none_ddrule", "mp_lgbm_none", "mp_lgbm_ddrule",
        "mp_ppo_none", "mp_ppo_ddrule", "mp_ppo_ppo"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--picker-steps", type=int, default=200_000)
    ap.add_argument("--mp-picker-steps", type=int, default=100_000)
    ap.add_argument("--dial-steps", type=int, default=60_000)
    ap.add_argument("--quick", action="store_true", help="2 folds, tiny training")
    args = ap.parse_args()

    end = LOCK_START["swing"]
    start = START
    if args.quick:
        start, end = date(2019, 1, 1), date(2024, 1, 1)
        args.seeds, args.picker_steps, args.dial_steps, args.mp_picker_steps = 1, 8_000, 4_000, 8_000

    raw = MarketData.from_store("1d_yahoo", nifty100() + INDEXES, start=date(2005, 1, 1))
    nifty = raw.panels["close"]["NIFTY50"].dropna()
    vix = raw.panels["close"]["INDIAVIX"].dropna() if "INDIAVIX" in raw.symbols else None
    data, report = clean(raw.without(INDEXES))
    log(f"data: {len(data.symbols)} stocks, {len(data.index)} days; cleaned "
        f"{len(report['spikes_removed'])} spikes, {len(report['breaks_adjusted'])} breaks; "
        f"VIX from {vix.index[0].date() if vix is not None else 'n/a'}")

    # Smoke tests must not count as research trials.
    registry = TrialRegistry(":memory:") if args.quick else None
    lab = Lab(data, "swing", TEAMS["swing_stocks"].limits, benchmark=nifty, registry=registry)
    secs = sectors()
    results = {}

    def evaluate(name, factory):
        t0 = time.time()
        log(f"{name}: walk-forward {start} to {end} ...")
        wf = lab.walk_forward(factory, start, end, train_years=3, test_months=12)
        m = dict(wf.metrics)
        m["fold_win_rate"] = fold_win_rate(wf, ew_rets)
        m["minutes"] = (time.time() - t0) / 60
        results[name] = m
        log(f"{name}: done in {m['minutes']:.1f} min | CAGR {m['cagr']:.1%} Sharpe {m['sharpe']:.2f} "
            f"maxDD {m['max_drawdown']:.1%} folds won {m['fold_win_rate']:.0%}")

    test_start = (pd.Timestamp(start) + pd.DateOffset(years=3)).date()
    ew_rets = equal_weight(data.panels["close"], test_start, end)
    results["Nifty 50 (hold)"] = compute_metrics(buy_and_hold(nifty, test_start, end), benchmark_rets=lab.bench_rets)
    results["Equal-wt N100"] = compute_metrics(ew_rets, benchmark_rets=lab.bench_rets)
    evaluate("momentum_12_1", Momentum)

    meetings = None
    for cfg in args.configs:
        base, _, extra = cfg.partition("+")
        if extra == "res" and meetings is None:
            meetings = load_results_meetings()
            log(f"results calendar: {len(meetings)} results meetings")
        mt = meetings if extra == "res" else None
        seeds = tuple(range(args.seeds))
        if base.startswith("mp_"):
            _, picker, dial = base.split("_")
            evaluate(cfg, lambda p=picker, d=dial, mt=mt: MomentumPlus(
                secs, nifty, vix, picker=p, dial=d, seeds=seeds,
                picker_steps=args.mp_picker_steps, dial_steps=args.dial_steps, meetings=mt))
        else:
            picker, dial = base.split("_")
            evaluate(cfg, lambda p=picker, d=dial, mt=mt: SwingStrategy(
                secs, nifty, vix, picker=p, dial=d, seeds=seeds,
                picker_steps=args.picker_steps, dial_steps=args.dial_steps, meetings=mt))

    ew_sharpe = results["Equal-wt N100"]["sharpe"]
    rows = [("cagr", "CAGR", "{:.1%}"), ("sharpe", "Sharpe", "{:.2f}"), ("max_drawdown", "Max drawdown", "{:.1%}"),
            ("excess_cagr", "CAGR vs Nifty", "{:.1%}"), ("fold_win_rate", "Years beating EW", "{:.0%}"),
            ("deflated_sharpe", "Deflated Sharpe", "{:.2f}"), ("costs_paid", "Costs (Rs)", "{:,.0f}"),
            ("annual_turnover", "Turnover/yr", "{:.1f}"), ("exposure", "Time invested", "{:.0%}")]
    names = list(results)
    print(f"\nWalk-forward test {test_start} to {end}, retrained yearly\n")
    print(f"{'':18}" + "".join(f"{n[:15]:>16}" for n in names))
    for key, label, f in rows:
        cells = []
        for n in names:
            v = results[n].get(key)
            cells.append("-" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f.format(v))
        print(f"{label:18}" + "".join(f"{c:>16}" for c in cells))

    print(f"\nPass criteria: Sharpe > {ew_sharpe:.2f} (EW), max DD > {PASS['max_drawdown']:.0%}, "
          f"years beating EW >= {PASS['fold_win_rate']:.0%}, deflated Sharpe > {PASS['deflated_sharpe']}")
    for n in names[2:]:  # momentum baseline and every config
        m = results[n]
        checks = [m["sharpe"] > ew_sharpe, m["max_drawdown"] > PASS["max_drawdown"],
                  m["fold_win_rate"] >= PASS["fold_win_rate"], m["deflated_sharpe"] > PASS["deflated_sharpe"]]
        print(f"  {n:16} {'PASS' if all(checks) else 'fail'}  "
              + " ".join("Y" if c else "n" for c in checks) + "   (Sharpe, DD, years, deflated)")

    out = DATA_DIR / "research" / f"team_d_{datetime.now():%Y%m%d_%H%M}{'_quick' if args.quick else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"test_start": str(test_start), "end": str(end), "args": vars(args),
                               "results": results}, indent=2, default=str))
    log(f"saved {out}")


if __name__ == "__main__":
    main()
