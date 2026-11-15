"""Forward scoring of LLM research and guard proposals.

A research note is "right" if, from the next open after it became public (the first
moment we could act), the stock beat the equal-weight average of our universe over the
next 5 trading days in the noted direction. Notes need 5 trading days to age before scoring.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

HOLD_DAYS = 5


def _entry_index(index: pd.DatetimeIndex, ts: str) -> int | None:
    t = pd.Timestamp(ts)
    t = t.tz_convert(index.tz) if t.tzinfo else t.tz_localize(index.tz)
    after_open = t.time() >= pd.Timestamp("09:15").time()
    day = t.normalize()
    pos = index.searchsorted(day, side="right" if after_open else "left")
    return int(pos) if pos < len(index) else None


def score_notes(db_path: Path | str, opens: pd.DataFrame, closes: pd.DataFrame,
                min_importance: int = 1) -> pd.DataFrame:
    """One row per scored note: excess return vs equal-weight and whether the direction was right."""
    try:
        with sqlite3.connect(str(db_path)) as conn:
            notes = pd.read_sql_query(
                "SELECT id, ts, symbol, source, event_type, direction, importance, summary FROM research_notes"
                " WHERE direction != 0 AND importance >= ?", conn, params=(min_importance,))
    except Exception:
        return pd.DataFrame()
    rows = []
    for n in notes.itertuples():
        if n.symbol not in closes:
            continue
        i = _entry_index(closes.index, n.ts)
        if i is None or i + HOLD_DAYS - 1 >= len(closes.index):
            continue
        j = i + HOLD_DAYS - 1
        stock = closes[n.symbol].iloc[j] / opens[n.symbol].iloc[i] - 1
        ew = float(np.nanmean(closes.iloc[j] / opens.iloc[i] - 1))
        if not np.isfinite(stock):
            continue
        excess = float(stock - ew)
        rows.append({**n._asdict(), "excess_5d": excess, "right": bool(np.sign(n.direction) * excess > 0)})
    return pd.DataFrame(rows)


def summary(scored: pd.DataFrame) -> dict:
    if scored.empty:
        return {"scored": 0}
    pos, neg = scored[scored.direction > 0], scored[scored.direction < 0]
    return {
        "scored": len(scored), "hit_rate": float(scored.right.mean()),
        "avg_excess_positive_notes": float(pos.excess_5d.mean()) if len(pos) else None,
        "avg_excess_negative_notes": float(neg.excess_5d.mean()) if len(neg) else None,
        "hit_rate_importance_3": float(scored[scored.importance == 3].right.mean())
        if (scored.importance == 3).any() else None,
    }
