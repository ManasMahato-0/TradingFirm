"""Step 2 of the fundamentals study: do results numbers predict next month's return?

    .venv\\Scripts\\python scripts\\fundamentals_sort.py

DESCRIPTIVE, not a logged trial. The rule below was fixed in docs/PLAN.md before any look:

  At each month-end decision from Oct 2019 to the swing lock (14 May 2026), sort the stocks
  into 5 equal groups by each measure and compare next month's return (open-to-open) minus
  the equal-weight average. A measure PASSES if ALL of:
    1. top group minus bottom group >= +0.30%/month
    2. that spread is positive in BOTH halves of the sample
    3. rank correlation of the 5 group means with group number >= 0.7
  PROCEED to a model only if at least 2 of the 5 measures pass.

A quarter's numbers count from the first trading day after the filing was public.
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from firm.backtest.clean import clean  # noqa: E402
from firm.backtest.data import MarketData  # noqa: E402
from firm.backtest.lab import LOCK_START  # noqa: E402
from firm.config import DATA_DIR, IST, RECORDER_DB_PATH  # noqa: E402
from firm.features.fundamentals import (  # noqa: E402
    FUNDAMENTAL_FEATURES, fundamental_panel, load_results, quarter_measures,
)
from firm.features.swing import decision_rows, forward_returns  # noqa: E402
from firm.universe import INDEXES, nifty100  # noqa: E402

START = date(2019, 10, 1)
MIN_SPREAD = 0.003
MIN_RANK_CORR = 0.7
MIN_PER_MONTH = 25     # need this many stocks with a value to form 5 meaningful groups
PROCEED_IF = 2


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def sort_measure(vals: pd.DataFrame, excess: pd.DataFrame) -> dict:
    """vals, excess: [months x stocks]. Returns group means, spreads and the verdict."""
    groups = []
    months = []
    for d in vals.index:
        v, e = vals.loc[d], excess.loc[d]
        ok = v.notna() & e.notna()
        if ok.sum() < MIN_PER_MONTH:
            continue
        q = pd.qcut(v[ok].rank(method="first"), 5, labels=False)
        groups.append(e[ok].groupby(q).mean())
        months.append(d)
    if not groups:
        return {"months": 0, "pass": False}
    g = pd.DataFrame(groups, index=months)              # month x group 0..4
    spread = g[4] - g[0]
    half = len(g) // 2
    means = g.mean().to_numpy()
    out = {
        "months": len(g), "group_means_pct": [round(100 * m, 3) for m in means],
        "spread_pct_per_month": round(100 * float(spread.mean()), 3),
        "spread_first_half_pct": round(100 * float(spread.iloc[:half].mean()), 3),
        "spread_second_half_pct": round(100 * float(spread.iloc[half:].mean()), 3),
        "spread_t_stat": round(float(spread.mean() / (spread.std() / np.sqrt(len(spread)))), 2),
        "rank_corr": round(spearman(np.arange(5), means), 2),
        "stocks_per_month": round(float(vals.notna().sum(axis=1).loc[months].mean()), 1),
    }
    out["pass_spread"] = out["spread_pct_per_month"] >= 100 * MIN_SPREAD
    out["pass_halves"] = out["spread_first_half_pct"] > 0 and out["spread_second_half_pct"] > 0
    out["pass_monotone"] = out["rank_corr"] >= MIN_RANK_CORR
    out["pass"] = bool(out["pass_spread"] and out["pass_halves"] and out["pass_monotone"])
    return out


def main() -> int:
    lock = LOCK_START["swing"]
    raw = MarketData.from_store("1d_yahoo", nifty100() + INDEXES, start=date(2018, 1, 1))
    data, _ = clean(raw.without(INDEXES))
    data = data.truncate(lock)                           # nothing at or after the lock is ever seen
    symbols = list(data.symbols)

    res = load_results(RECORDER_DB_PATH)
    res = res[res.symbol.isin(symbols)]
    print(f"results: {len(res)} filings, {res.symbol.nunique()} stocks, {res.to_date.min().date()} to "
          f"{res.to_date.max().date()}; {len(symbols)} stocks in price data")
    measures = quarter_measures(res)
    panel = fundamental_panel(data.index, measures, symbols)

    rows = decision_rows(data.index, "M")
    dates = data.index[rows]
    fwd = forward_returns(data.panels["open"], rows)
    excess = fwd.sub(fwd.mean(axis=1, skipna=True), axis=0)
    keep = (dates.date >= START) & fwd.notna().any(axis=1).to_numpy()
    excess = excess[keep]
    print(f"{len(excess)} monthly decisions, {excess.index[0].date()} to {excess.index[-1].date()}")

    results = {}
    for name in FUNDAMENTAL_FEATURES:
        vals = panel[name].loc[excess.index]
        results[name] = sort_measure(vals, excess)

    print(f"\nNext-month return vs equal-weight, by group (1 = lowest measure ... 5 = highest), % per month\n")
    print(f"{'measure':14}{'months':>7}{'stocks':>8}" + "".join(f"{'G' + str(i + 1):>8}" for i in range(5))
          + f"{'top-bot':>9}{'1st half':>10}{'2nd half':>10}{'t':>6}{'rankcorr':>9}  verdict")
    for name, r in results.items():
        if not r["months"]:
            print(f"{name:14} no usable months")
            continue
        print(f"{name:14}{r['months']:>7}{r['stocks_per_month']:>8}" + "".join(f"{m:>8.2f}" for m in r["group_means_pct"])
              + f"{r['spread_pct_per_month']:>9.2f}{r['spread_first_half_pct']:>10.2f}"
              + f"{r['spread_second_half_pct']:>10.2f}{r['spread_t_stat']:>6.1f}{r['rank_corr']:>9.2f}  "
              + ("PASS" if r["pass"] else "fail ") + f" (spread {'Y' if r['pass_spread'] else 'n'}, "
              f"halves {'Y' if r['pass_halves'] else 'n'}, monotone {'Y' if r['pass_monotone'] else 'n'})")
    n_pass = sum(r.get("pass", False) for r in results.values())
    print(f"\nPre-registered rule: proceed to a model if at least {PROCEED_IF} of 5 measures pass. "
          f"{n_pass} passed -> {'PROCEED' if n_pass >= PROCEED_IF else 'STOP'}")

    out = DATA_DIR / "research" / f"fundamentals_sort_{datetime.now(IST):%Y%m%d_%H%M}.json"
    out.write_text(json.dumps({"start": str(START), "lock": str(lock), "results": results, "n_pass": n_pass},
                              indent=2, default=str))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
