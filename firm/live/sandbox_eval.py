"""Sandbox checkpoint evaluation, exactly as pre-registered in docs/PLAN.md.

An account WORKS at a checkpoint (day 60 or 120) only if ALL hold, on ledger prices:
  1. final equity beats D (cash)        2. final equity beats C (random trader)
  3. Sharpe of daily excess over D > 0.5
  4. excess over D positive in BOTH halves of the period
  5. at least 30 round trips
and the latency rule: still beats D when every fill is re-priced at the quote taken at the
decision (the ledger fills at the next bar's open, live polling adds delay).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from firm.broker import DEFAULT_SLIPPAGE_BPS
from firm.costs import Segment

CHECKPOINTS = (60, 120)
MIN_ROUND_TRIPS = 30
MIN_QUOTE_COVERAGE = 0.9   # share of fills that must have a decision-time quote for the latency check to count
MIN_SHARPE = 0.5
BAR = pd.Timedelta(minutes=5)
BOOKS = ("sandbox_A", "sandbox_B", "sandbox_C", "sandbox_D")


def _q(db: Path | str, sql: str, params: tuple = ()) -> pd.DataFrame:
    if not Path(db).exists():
        return pd.DataFrame()
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10) as c:
        return pd.read_sql_query(sql, c, params=params)


def daily_equity(db, book: str) -> pd.Series:
    df = _q(db, "SELECT ts, equity FROM equity WHERE team=? ORDER BY ts", (book,))
    if df.empty:
        return pd.Series(dtype=float)
    s = pd.Series(df.equity.values, index=pd.to_datetime(df.ts.str[:10]))
    return s[~s.index.duplicated(keep="last")]


def latency_adjustment(db, book: str) -> tuple[float, float]:
    """(change in final equity if every fill had happened at the decision-time quote, same adverse
    slippage and costs; share of fills that had a quote). Fills without a logged quote are left as they were."""
    fills = _q(db, "SELECT ts, symbol, side, qty, price FROM fills WHERE team=?", (book,))
    quotes = _q(db, "SELECT decision_ts, symbol, side, quote FROM sandbox_quotes WHERE book=?", (book,))
    if fills.empty:
        return 0.0, 1.0                                  # nothing traded: nothing to re-price
    if quotes.empty:
        return 0.0, 0.0
    slip = DEFAULT_SLIPPAGE_BPS[Segment.EQ_INTRADAY] / 1e4
    q = {(pd.Timestamp(r.decision_ts), r.symbol, r.side): r.quote for r in quotes.itertuples()}
    delta, matched = 0.0, 0
    for f in fills.itertuples():
        quote = q.get((pd.Timestamp(f.ts) - BAR, f.symbol, f.side))
        if quote is None:
            continue
        matched += 1
        alt = quote * (1 + slip) if f.side == "BUY" else quote * (1 - slip)
        delta += -f.qty * (alt - f.price) if f.side == "BUY" else f.qty * (alt - f.price)
    return float(delta), matched / len(fills)


def evaluate(db, books=BOOKS, checkpoints=CHECKPOINTS) -> dict:
    eq = {b: daily_equity(db, b) for b in books}
    if any(s.empty for s in eq.values()):
        return {"days": 0, "accounts": {}}
    days = min(len(s) for s in eq.values())
    d = eq["sandbox_D"]
    ret_d = d.pct_change().dropna()
    out = {"days": days, "next_checkpoint": next((c for c in checkpoints if c > days), None), "accounts": {}}
    for b in books:
        s = eq[b]
        ret = s.pct_change().dropna()
        excess = (ret - ret_d).dropna()
        half = len(excess) // 2
        n_fills = int(_q(db, "SELECT COUNT(*) n FROM fills WHERE team=?", (b,)).n.iloc[0])
        round_trips = n_fills / 2
        sharpe = (float(excess.mean() / excess.std() * np.sqrt(252))
                  if len(excess) > 2 and excess.std() > 1e-12 else float("nan"))
        adj, coverage = latency_adjustment(db, b)
        final, d_final, c_final = s.iloc[-1], d.iloc[-1], eq["sandbox_C"].iloc[-1]
        row = {
            "equity": float(final), "return": float(final / s.iloc[0] - 1) if len(s) else 0.0,
            "vs_D": float(final - d_final), "vs_C": float(final - c_final),
            "sharpe_excess": sharpe, "round_trips": round_trips,
            "first_half_excess": float(excess.iloc[:half].mean()) if half else float("nan"),
            "second_half_excess": float(excess.iloc[half:].mean()) if half else float("nan"),
            "latency_adjusted_vs_D": float(final + adj - d_final), "latency_adjustment": adj,
            "quote_coverage": coverage,
        }
        row["checks"] = {
            "beats_D": row["vs_D"] > 0, "beats_C": row["vs_C"] > 0,
            "sharpe_above_0.5": bool(np.isfinite(sharpe) and sharpe > MIN_SHARPE),
            "both_halves_positive": bool(half and row["first_half_excess"] > 0 and row["second_half_excess"] > 0),
            "at_least_30_round_trips": round_trips >= MIN_ROUND_TRIPS,
            "beats_D_after_latency": row["latency_adjusted_vs_D"] > 0,
            "decision_quotes_logged": coverage >= MIN_QUOTE_COVERAGE,
        }
        row["works"] = bool(b in ("sandbox_A", "sandbox_B") and all(row["checks"].values()))
        out["accounts"][b] = row
    out["checkpoint_reached"] = max((c for c in checkpoints if c <= days), default=None)
    return out
