"""Results-calendar features from NSE board meeting notices. Causal by construction:
a meeting is known only from its announcement time; a reaction only after it happened."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from firm.config import RECORDER_DB_PATH

RESULTS_FEATURES = ["days_to_results", "days_since_results", "results_reaction"]
HORIZON_DAYS = 30


def load_results_meetings(path: Path | str = RECORDER_DB_PATH) -> pd.DataFrame:
    """One row per (symbol, results meeting date), with the earliest announcement time."""
    with sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True) as c:
        df = pd.read_sql_query(
            "SELECT symbol, meeting_date, MIN(intimated_at) AS intimated_at FROM board_meetings"
            " WHERE is_results = 1 AND meeting_date IS NOT NULL AND intimated_at IS NOT NULL"
            " GROUP BY symbol, meeting_date", c)
    df["meeting_date"] = pd.to_datetime(df["meeting_date"]).dt.date
    df["intimated_date"] = pd.to_datetime(df["intimated_at"].str[:10]).dt.date
    return df[["symbol", "meeting_date", "intimated_date"]]


def results_features(index: pd.DatetimeIndex, close: pd.DataFrame,
                     meetings: pd.DataFrame) -> dict[str, pd.DataFrame]:
    days = np.array(index.date)
    n, syms = len(days), list(close.columns)
    to_res = np.full((n, len(syms)), np.inf)
    since = np.full((n, len(syms)), np.inf)
    reaction = np.full((n, len(syms)), np.nan)

    c = close.ffill().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        daily = c[1:] / c[:-1] - 1
    ew = np.concatenate([[np.nan], np.nanmean(np.where(np.isfinite(daily), daily, np.nan), axis=1)])
    ew_curve = np.nancumprod(np.nan_to_num(ew) + 1)

    by_symbol = {s: g for s, g in meetings.groupby("symbol")}
    day_ord = np.array([d.toordinal() for d in days])
    for j, sym in enumerate(syms):
        g = by_symbol.get(sym)
        if g is None:
            continue
        past_reactions: list[tuple[int, float]] = []
        for meet, known in zip(g["meeting_date"], g["intimated_date"]):
            m_ord = meet.toordinal()
            # Days to results: from the announcement until the meeting.
            lo, hi = np.searchsorted(day_ord, known.toordinal()), np.searchsorted(day_ord, m_ord, side="right")
            if lo < hi:
                to_res[lo:hi, j] = np.minimum(to_res[lo:hi, j], m_ord - day_ord[lo:hi])
            # Days since results: from the meeting onward.
            start = np.searchsorted(day_ord, m_ord)
            if start < n:
                since[start:, j] = np.minimum(since[start:, j], day_ord[start:] - m_ord)
            # Reaction: close before the meeting -> close of the next trading day after it,
            # minus the equal-weight move. Known from that next day's close.
            before = np.searchsorted(day_ord, m_ord) - 1
            after = np.searchsorted(day_ord, m_ord, side="right")
            if before >= 0 and after < n and np.isfinite(c[before, j]) and np.isfinite(c[after, j]):
                stock = c[after, j] / c[before, j] - 1
                market = ew_curve[after] / ew_curve[before] - 1
                past_reactions.append((after, stock - market))
        past_reactions.sort()
        for k, (pos, _) in enumerate(past_reactions):
            last4 = [r for _, r in past_reactions[max(0, k - 3): k + 1]]
            end = past_reactions[k + 1][0] if k + 1 < len(past_reactions) else n
            reaction[pos:end, j] = np.mean(last4)

    listed = close.notna()
    df = lambda a: pd.DataFrame(a, index=index, columns=syms)  # noqa: E731
    return {
        "days_to_results": (df(np.minimum(to_res, HORIZON_DAYS)) / HORIZON_DAYS - 0.5).where(listed),
        "days_since_results": (df(np.minimum(since, HORIZON_DAYS)) / HORIZON_DAYS - 0.5).where(listed),
        "results_reaction": (df(reaction).rank(axis=1, pct=True) - 0.5).fillna(0.0).where(listed),
    }
