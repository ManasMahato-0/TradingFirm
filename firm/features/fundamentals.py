"""Fundamentals from quarterly results (NSE XBRL): growth measures, a point-in-time panel,
and plain-English descriptions of a new result.

A quarter's numbers are usable only from the first trading day AFTER the day the filing
became public, and only the first filing of a quarter is used (no later revisions).
Growth is measured against the same quarter a year earlier on the same basis
(consolidated vs consolidated, standalone vs standalone).
"""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

FUNDAMENTAL_FEATURES = ["rev_yoy", "pat_yoy", "eps_yoy", "margin_chg", "pat_surprise"]
STALE_DAYS = 200  # a quarter's numbers stop counting if no newer quarter has appeared by then
CLIP = 3.0        # growth beyond +/-300% is treated as +/-300% (tiny bases make huge ratios)


def load_results(db: Path | str | sqlite3.Connection) -> pd.DataFrame:
    conn = db if isinstance(db, sqlite3.Connection) else sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query("SELECT * FROM results", conn)
    finally:
        if not isinstance(db, sqlite3.Connection):
            conn.close()
    df["to_date"] = pd.to_datetime(df["to_date"])
    df["broadcast"] = pd.to_datetime(df["broadcast"], utc=True).dt.tz_convert("Asia/Kolkata")
    return df


def _growth(now: float, before: float) -> float:
    if before is None or now is None or not np.isfinite(before) or not np.isfinite(now) or before == 0:
        return np.nan
    return float(np.clip((now - before) / abs(before), -CLIP, CLIP))


def quarter_measures(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (symbol, quarter, basis): YoY growth measures against the year-ago quarter
    of the same basis. Columns: symbol, to_date, consolidated, broadcast + FUNDAMENTAL_FEATURES."""
    out = []
    for (sym, cons), g in df.sort_values("to_date").groupby(["symbol", "consolidated"]):
        g = g.drop_duplicates("to_date").set_index("to_date")
        yoy_pat: list[float] = []
        for d, r in g.iterrows():
            prev_dates = g.index[(g.index >= d - pd.Timedelta(days=380)) & (g.index <= d - pd.Timedelta(days=350))]
            if len(prev_dates):
                p = g.loc[prev_dates[-1]]
                pat_g = _growth(r.pat, p.pat)
                margin = (r.pat / r.revenue - p.pat / p.revenue) if (r.revenue and p.revenue and r.revenue > 0
                                                                      and p.revenue > 0) else np.nan
                row = {"rev_yoy": _growth(r.revenue, p.revenue), "pat_yoy": pat_g,
                       "eps_yoy": _growth(r.eps, p.eps), "margin_chg": margin}
            else:
                pat_g = np.nan
                row = {"rev_yoy": np.nan, "pat_yoy": np.nan, "eps_yoy": np.nan, "margin_chg": np.nan}
            hist = [x for x in yoy_pat[-8:] if np.isfinite(x)]
            if np.isfinite(pat_g) and len(hist) >= 4 and np.std(hist) > 1e-9:
                row["pat_surprise"] = float(np.clip((pat_g - np.mean(hist)) / np.std(hist), -5, 5))
            else:
                row["pat_surprise"] = np.nan
            yoy_pat.append(pat_g)
            out.append({"symbol": sym, "to_date": d, "consolidated": cons, "broadcast": r.broadcast, **row})
    return pd.DataFrame(out)


def best_basis(m: pd.DataFrame) -> pd.DataFrame:
    """One row per (symbol, quarter): consolidated numbers when they exist, else standalone."""
    m = m.sort_values(["symbol", "to_date", "consolidated"], ascending=[True, True, False])
    return m.drop_duplicates(["symbol", "to_date"], keep="first")


def fundamental_panel(index: pd.DatetimeIndex, measures: pd.DataFrame,
                      symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Per measure, a [dates x symbols] frame of the latest quarter KNOWN at each date
    (available from the first trading day after the filing's public date)."""
    days = pd.DatetimeIndex(index.date).tz_localize(None) if index.tz is not None else index
    out = {}
    m = best_basis(measures)
    avail = m["broadcast"].dt.tz_localize(None).dt.normalize() + pd.Timedelta(days=1)
    for name in FUNDAMENTAL_FEATURES:
        frame = pd.DataFrame(np.nan, index=days, columns=symbols)
        age = pd.DataFrame(np.nan, index=days, columns=symbols)
        part = pd.DataFrame({"symbol": m["symbol"], "avail": avail, "val": m[name]}).dropna()
        part = part[part.symbol.isin(symbols)].sort_values("avail")
        wide = part.pivot_table(index="avail", columns="symbol", values="val", aggfunc="last")
        seen = part.pivot_table(index="avail", columns="symbol", values="avail", aggfunc="last")
        wide = wide.reindex(wide.index.union(days)).ffill().reindex(days)
        seen = seen.reindex(seen.index.union(days)).ffill().reindex(days)
        frame[wide.columns] = wide
        age_days = (pd.DataFrame({c: days for c in seen.columns}, index=days) - seen).apply(
            lambda col: col.dt.days)
        age[age_days.columns] = age_days
        out[name] = frame.where(age <= STALE_DAYS)
        out[name].index = index
    return out


# ---------------------------------------------------------------- plain English

def quarter_label(to_date: date | pd.Timestamp) -> str:
    """Indian fiscal quarter: Apr-Jun is Q1 of the FY ending next March ('Q2 FY27' = Jul-Sep 2026)."""
    d = pd.Timestamp(to_date)
    q = {6: 1, 9: 2, 12: 3, 3: 4}.get(d.month)
    fy = d.year + (1 if d.month > 3 else 0)
    return f"Q{q} FY{str(fy)[-2:]}" if q else d.strftime("%b %Y")


def _cr(x) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"Rs {x:,.0f} cr"


def _pct(x) -> str:
    return "" if x is None or not np.isfinite(x) else f" ({x:+.1%} vs a year ago)"


def describe_result(conn: sqlite3.Connection, rec: dict) -> str:
    """One plain-English line for a new result, with growth vs the same quarter last year."""
    d = pd.Timestamp(rec["to_date"])
    row = conn.execute(
        "SELECT revenue, pat, eps FROM results WHERE symbol=? AND consolidated=? AND to_date BETWEEN ? AND ?"
        " ORDER BY to_date DESC LIMIT 1",
        (rec["symbol"], rec["consolidated"], (d - pd.Timedelta(days=380)).date().isoformat(),
         (d - pd.Timedelta(days=350)).date().isoformat())).fetchone()
    prev_rev, prev_pat, prev_eps = row if row else (None, None, None)
    basis = "consolidated" if rec["consolidated"] else "standalone"
    g_rev, g_pat = _growth(rec["revenue"], prev_rev), _growth(rec["pat"], prev_pat)
    text = (f"{rec['symbol']} {quarter_label(d)} results ({basis}): revenue {_cr(rec['revenue'])}{_pct(g_rev)}, "
            f"net profit {_cr(rec['pat'])}{_pct(g_pat)}")
    if rec.get("eps") is not None:
        text += f", EPS Rs {rec['eps']:.2f}"
    if row is None:
        text += ". No year-ago quarter on file to compare."
    return text + " (from the filed XBRL numbers)."
