"""Facts for the weekly AI Trader review (used by the /weekly-review command in Claude Code).

    .venv\\Scripts\\python scripts\\weekly_report.py            # print the facts
    .venv\\Scripts\\python scripts\\weekly_report.py --post FILE # post a memo to the Trading Floor

Prints: each paper book's week and since-start vs equal-weight, drawdown, trades;
Researcher scoreboard; AI Trader guard proposals and how they turned out; Manager flags;
LLM health; research trial count.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from firm.activity import ActivityLog  # noqa: E402
from firm.agents.scoreboard import score_notes, summary  # noqa: E402
from firm.backtest.benchmarks import equal_weight  # noqa: E402
from firm.backtest.data import MarketData  # noqa: E402
from firm.config import DATA_DIR, IST  # noqa: E402
from firm.live.runner import LIVE_DIR  # noqa: E402
from firm.universe import nifty100  # noqa: E402

DB = LIVE_DIR / "firm.db"


def q(sql: str, params=(), db=DB) -> pd.DataFrame:
    if not Path(db).exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(str(db)) as c:
            return pd.read_sql_query(sql, c, params=params)
    except Exception:
        return pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", type=Path, help="markdown memo to post to the Trading Floor as the AI Trader")
    args = ap.parse_args()
    if args.post:
        text = args.post.read_text(encoding="utf-8").strip()
        ActivityLog(DB).log("think", "AI Trader", f"Weekly review memo ({args.post.name}): {text[:1500]}",
                            team="swing_stocks", details={"file": str(args.post)})
        print("posted")
        return 0

    today = datetime.now(IST).date()
    week_ago = today - timedelta(days=7)
    data = MarketData.from_store("1d_yahoo", nifty100(), start=today - timedelta(days=400))
    closes, opens = data.panels["close"], data.panels["open"]
    print(f"# Weekly facts, {today}\n")

    books = q("SELECT * FROM books")
    print("## Paper books")
    if books.empty or books.last_bar.isin(["None", None]).all():
        print("No book has traded yet.")
    for b in books.itertuples():
        eq = q("SELECT substr(ts,1,10) d, equity FROM equity WHERE team=? ORDER BY ts", (b.name,))
        if eq.empty:
            print(f"- {b.label} ({b.status}): opens {b.start}, no days yet")
            continue
        s = pd.Series(eq.equity.values, index=pd.to_datetime(eq.d))
        start = date.fromisoformat(b.start)
        ew = equal_weight(closes, start + timedelta(days=1), today + timedelta(days=1))
        wk = s[s.index.date <= week_ago]
        wk_ret = s.iloc[-1] / (wk.iloc[-1] if len(wk) else b.capital) - 1
        ew_wk = float((1 + ew[ew.index.date > week_ago]).prod() - 1)
        print(f"- {b.label} ({b.status}): equity Rs {s.iloc[-1]:,.0f}; week {wk_ret:+.2%} vs EW {ew_wk:+.2%}; "
              f"since start {s.iloc[-1] / b.capital - 1:+.2%} vs EW {float((1 + ew).prod() - 1):+.2%}; "
              f"max DD {(s / s.cummax() - 1).min():.1%}; days {len(s)}")
        fills = q("SELECT ts, side, qty, symbol, price FROM fills WHERE team=? AND ts >= ?", (b.name, str(week_ago)))
        for f in fills.itertuples():
            print(f"    {f.ts[:10]} {f.side} {f.qty} {f.symbol} @ {f.price:,.2f}")

    print("\n## Researcher scoreboard (5 trading days after the next open, vs equal-weight)")
    sc = score_notes(DB, opens, closes)
    print(json.dumps(summary(sc), indent=1))
    n = q("SELECT importance, COUNT(*) n FROM research_notes WHERE created >= ? GROUP BY importance", (str(week_ago),))
    print("Notes this week by importance:", dict(zip(n.importance, n.n)) if not n.empty else {})

    print("\n## AI Trader guard (shadow)")
    g = q("SELECT date, book, symbol, action, reason, price FROM guard_proposals ORDER BY date")
    if g.empty:
        print("No proposals yet.")
    for r in g.itertuples():
        now_px = closes[r.symbol].dropna().iloc[-1] if r.symbol in closes else np.nan
        after = (now_px / r.price - 1) if r.price else np.nan
        print(f"- {r.date} {r.action} {r.symbol}: {r.reason} | stock since: {after:+.2%}")

    print("\n## Manager flags this week")
    fl = q("SELECT date, book, flags FROM daily_notes WHERE date >= ?", (str(week_ago),))
    for r in fl.itertuples():
        for f in json.loads(r.flags or "[]"):
            print(f"- {r.date} {r.book}: [{f['level']}] {f['text']}")
    unchecked = q("SELECT COUNT(*) n FROM daily_notes WHERE checked=0 AND date >= ?", (str(week_ago),))
    if not unchecked.empty:
        print(f"Notes that failed the number check: {int(unchecked.n.iloc[0])}")

    print("\n## LLM health this week")
    h = q("SELECT agent, COUNT(*) calls, SUM(1-ok) failed, AVG(ms) avg_ms FROM llm_calls WHERE ts >= ?"
          " GROUP BY agent", (str(week_ago),))
    print(h.to_string(index=False) if not h.empty else "No calls.")

    t = q("SELECT kind, COUNT(*) n FROM trials GROUP BY kind", db=DATA_DIR / "research.db")
    print("\n## Research trials so far:", dict(zip(t.kind, t.n)) if not t.empty else {})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
