"""Intraday data: 5-minute bars built from our recorded 1-minute bars, and the
liquid universe the intraday team trades."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from firm.backtest.data import MarketData
from firm.config import DATA_DIR
from firm.recorders.bars import bars_dir, load_bars

CACHE = DATA_DIR / "cache" / "5m.parquet"
SESSION_START, SESSION_END = "09:15", "15:29"


def resample_5m(df: pd.DataFrame) -> pd.DataFrame:
    """1-minute OHLCV -> 5-minute bars labelled by their start (09:15, 09:20, ... 15:25).
    Empty input (e.g. a weekend with no candles, which Fyers returns with a plain RangeIndex)
    gives an empty frame of the right shape instead of an error."""
    if df is None or len(df) == 0 or not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"], dtype=float,
                            index=pd.DatetimeIndex([], tz="Asia/Kolkata", name="ts"))
    df = df.between_time(SESSION_START, SESSION_END)
    out = df.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return out.dropna(subset=["open"])


def _newest_source(symbols: list[str]) -> float:
    return max((f.stat().st_mtime for s in symbols for f in bars_dir("1m", s).glob("*.parquet")), default=0.0)


def load_5m(symbols: list[str], start: date | None = None, end: date | None = None) -> MarketData:
    """5-minute MarketData for `symbols`. Cached in one parquet, rebuilt when 1m files change."""
    cached = None
    if CACHE.exists() and CACHE.stat().st_mtime >= _newest_source(symbols):
        cached = pd.read_parquet(CACHE)
        if not set(symbols) <= set(cached["symbol"].unique()):
            cached = None
    if cached is None:
        frames = []
        for s in symbols:
            df = load_bars("1m", s)
            if df.empty:
                continue
            b = resample_5m(df)
            b["symbol"] = s
            frames.append(b)
        cached = pd.concat(frames)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        cached.to_parquet(CACHE)
    long = cached[cached["symbol"].isin(symbols)]
    if start:
        long = long[long.index.date >= start]
    if end:
        long = long[long.index.date < end]
    return MarketData.from_frames({s: g.drop(columns="symbol") for s, g in long.groupby("symbol")})


def liquid_universe(data: MarketData, n: int = 50, months: int = 3) -> list[str]:
    """Top-n stocks by average daily traded value over the FIRST `months` of data only,
    so the choice does not use information from the test periods."""
    first = data.index[0].date()
    cut = data.truncate((pd.Timestamp(first) + pd.DateOffset(months=months)).date())
    value = (cut.panels["close"] * cut.panels["volume"]).groupby(cut.index.date).sum()
    return list(value.mean().nlargest(n).index)


def trading_days(data: MarketData) -> list[date]:
    return sorted(set(data.index.date))


def research_window(data: MarketData, lock: date) -> tuple[date, date]:
    return data.index[0].date(), lock - timedelta(days=0)
