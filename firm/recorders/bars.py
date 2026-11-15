"""OHLCV bar storage: one parquet file per dataset / symbol / period.

Intraday datasets are split by month, daily datasets by year (fewer, bigger files).
Datasets: "1m" and "1d" (Fyers, unadjusted), "1d_yahoo" (Yahoo, split/dividend adjusted).
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd

from firm.config import BARS_DIR
from firm.data.fyers import FyersAuthError


def _safe(symbol: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", symbol)


def period_format(dataset: str) -> str:
    return "%Y" if dataset.startswith("1d") else "%Y-%m"


def bars_dir(dataset: str, symbol: str, root: Path | None = None) -> Path:
    return (root or BARS_DIR) / dataset / _safe(symbol)


def save_bars(df: pd.DataFrame, dataset: str, symbol: str, root: Path | None = None) -> int:
    """Merge bars into storage, newest copy wins on duplicates. Returns rows added."""
    if df.empty:
        return 0
    folder = bars_dir(dataset, symbol, root)
    folder.mkdir(parents=True, exist_ok=True)
    added = 0
    for period, part in df.groupby(df.index.strftime(period_format(dataset))):
        path = folder / f"{period}.parquet"
        if path.exists():
            old = pd.read_parquet(path)
            merged = pd.concat([old, part])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            added += len(merged) - len(old)
        else:
            merged = part[~part.index.duplicated(keep="last")].sort_index()
            added += len(merged)
        merged.to_parquet(path)
    return added


def load_bars(dataset: str, symbol: str, start: date | None = None, end: date | None = None,
              root: Path | None = None) -> pd.DataFrame:
    folder = bars_dir(dataset, symbol, root)
    files = sorted(folder.glob("*.parquet")) if folder.exists() else []
    fmt = period_format(dataset)
    if start:
        files = [f for f in files if f.stem >= start.strftime(fmt)]
    if end:
        files = [f for f in files if f.stem <= end.strftime(fmt)]
    if not files:
        return pd.DataFrame()
    df = pd.concat(pd.read_parquet(f) for f in files).sort_index()
    if start:
        df = df[df.index.date >= start]
    if end:
        df = df[df.index.date < end]
    return df


def record_bars(source, dataset: str, interval: str, symbols: list[str],
                start: date, end: date, root: Path | None = None) -> dict:
    """Fetch and store bars for many symbols. Auth errors abort; others are collected."""
    total, ok, errors = 0, 0, {}
    for sym in symbols:
        try:
            total += save_bars(source.history(sym, interval, start, end), dataset, sym, root)
            ok += 1
        except FyersAuthError:
            raise
        except Exception as e:  # one bad symbol must not stop the rest
            errors[sym] = f"{type(e).__name__}: {e}"[:200]
    return {"rows": total, "symbols_ok": ok, "errors": errors}
