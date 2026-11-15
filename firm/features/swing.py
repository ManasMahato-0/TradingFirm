"""Features for the swing team: per-stock (cross-sectionally ranked) and market-wide.

All panels are daily (rows = trading days, columns = symbols) and causal.
Weekly decision rows are the last trading day of each ISO week.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from firm.backtest.data import MarketData

STOCK_FEATURES = [
    "ret_5", "ret_21", "ret_63", "ret_126", "mom_12_1",
    "dist_sma50", "dist_sma200", "vol_21", "vol_63",
    "volume_trend", "dist_52w_high", "sector_rel_63",
]
MARKET_FEATURES = [
    "mkt_ret_21", "mkt_ret_63", "mkt_dist_sma200", "mkt_vol_21",
    "mkt_dist_52w_high", "breadth_200", "vix_z",
    "ew_dist_sma200", "ew_drawdown",  # equal-weight index of the universe (broad market)
]


def stock_features(close: pd.DataFrame, volume: pd.DataFrame,
                   sectors: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Raw per-stock features, then ranked across stocks each day to [-0.5, 0.5]."""
    c = close.ffill(limit=5)
    r = c.pct_change(fill_method=None)
    f: dict[str, pd.DataFrame] = {}
    for n in (5, 21, 63, 126):
        f[f"ret_{n}"] = c / c.shift(n) - 1
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1
    f["dist_sma50"] = c / c.rolling(50, min_periods=50).mean() - 1
    f["dist_sma200"] = c / c.rolling(200, min_periods=200).mean() - 1
    f["vol_21"] = r.rolling(21, min_periods=15).std()
    f["vol_63"] = r.rolling(63, min_periods=45).std()
    v = volume.where(volume > 0)
    f["volume_trend"] = v.rolling(5, min_periods=3).mean() / v.rolling(63, min_periods=40).mean()
    f["dist_52w_high"] = c / c.rolling(252, min_periods=200).max() - 1
    sector = pd.Series({s: sectors.get(s, "Other") for s in c.columns})
    sec_median = f["ret_63"].T.groupby(sector).transform("median").T
    f["sector_rel_63"] = f["ret_63"] - sec_median
    listed = close.notna()
    return {k: (v.rank(axis=1, pct=True) - 0.5).where(listed) for k, v in f.items()}


def market_features(index: pd.DatetimeIndex, close: pd.DataFrame, nifty: pd.Series,
                    vix: pd.Series | None = None) -> pd.DataFrame:
    """Market-wide features, roughly scaled to [-2, 2]."""
    n = nifty.reindex(index).ffill()
    out = pd.DataFrame(index=index)
    out["mkt_ret_21"] = (n / n.shift(21) - 1) * 10
    out["mkt_ret_63"] = (n / n.shift(63) - 1) * 5
    out["mkt_dist_sma200"] = (n / n.rolling(200, min_periods=200).mean() - 1) * 5
    out["mkt_vol_21"] = n.pct_change().rolling(21, min_periods=15).std() * math.sqrt(252) * 3
    out["mkt_dist_52w_high"] = (n / n.rolling(252, min_periods=200).max() - 1) * 3
    c = close.ffill(limit=5)
    sma = c.rolling(200, min_periods=200).mean()
    has = sma.notna() & c.notna()
    out["breadth_200"] = ((c > sma) & has).sum(axis=1) / has.sum(axis=1).replace(0, np.nan) * 2 - 1
    if vix is not None:
        x = vix.reindex(index).ffill()
        out["vix_z"] = ((x - x.rolling(252, min_periods=120).mean())
                        / x.rolling(252, min_periods=120).std()).clip(-3, 3) / 1.5
    else:
        out["vix_z"] = 0.0
    ew_idx = (1 + c.pct_change(fill_method=None).mean(axis=1).fillna(0)).cumprod()
    out["ew_dist_sma200"] = (ew_idx / ew_idx.rolling(200, min_periods=200).mean() - 1) * 5
    out["ew_drawdown"] = (ew_idx / ew_idx.rolling(252, min_periods=200).max() - 1) * 3
    return out


def decision_rows(index: pd.DatetimeIndex, freq: str = "W") -> np.ndarray:
    """Positions of the last trading day of each ISO week ('W') or month ('M').

    Uses only the trading calendar, not prices: on a Friday we know the next
    trading day is Monday."""
    if freq == "W":
        iso = index.isocalendar()
        key = (iso["year"].to_numpy() * 100 + iso["week"].to_numpy()).astype(np.int64)
    elif freq == "M":
        key = (index.year * 100 + index.month).to_numpy().astype(np.int64)
    else:
        raise ValueError(freq)
    last = np.append(key[:-1] != key[1:], True)
    return np.flatnonzero(last)


def forward_returns(open_: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    """Return from the open after decision k to the open after decision k+1
    (we decide at the close, fill at the next open, hold one week).
    Rows whose exit lies beyond the data are NaN, so no future leaks into training."""
    entry_pos = rows + 1
    out = pd.DataFrame(np.nan, index=open_.index[rows], columns=open_.columns)
    o = open_.to_numpy()
    for k in range(len(rows) - 1):
        a, b = entry_pos[k], entry_pos[k + 1]
        if b < len(o):
            out.iloc[k] = o[b] / o[a] - 1
    return out


@dataclass
class WeeklyDataset:
    dates: pd.DatetimeIndex     # decision dates
    symbols: list[str]
    X: np.ndarray               # [weeks, stocks, n_stock + n_market] (NaN -> 0)
    valid: np.ndarray           # [weeks, stocks] usable sample (features and label present)
    fwd: np.ndarray             # [weeks, stocks] forward weekly return
    excess: np.ndarray          # [weeks, stocks] forward return minus equal-weight average
    market: np.ndarray          # [weeks, n_market]
    ew_fwd: np.ndarray          # [weeks] equal-weight forward return


def feature_names(with_results: bool) -> list[str]:
    from firm.features.results import RESULTS_FEATURES

    return STOCK_FEATURES + (RESULTS_FEATURES if with_results else [])


def all_stock_features(data: MarketData, sectors: dict[str, str],
                       meetings: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    feats = stock_features(data.panels["close"], data.panels["volume"], sectors)
    if meetings is not None:
        from firm.features.results import results_features

        feats.update(results_features(data.index, data.panels["close"], meetings))
    return feats


def feature_tensor(feats: dict[str, pd.DataFrame], mkt: pd.DataFrame, rows: np.ndarray,
                   names: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X [weeks, stocks, k] with NaN kept, has_features mask, market [weeks, m])."""
    stock = np.stack([feats[k].to_numpy()[rows] for k in (names or STOCK_FEATURES)], axis=-1)
    m = mkt[MARKET_FEATURES].to_numpy()[rows]
    has = ~np.isnan(stock).any(axis=-1)
    full = np.concatenate([stock, np.repeat(m[:, None, :], stock.shape[1], axis=1)], axis=-1)
    return full, has, m


def build_weekly_dataset(data: MarketData, sectors: dict[str, str], nifty: pd.Series,
                         vix: pd.Series | None = None, min_week: int = 0,
                         meetings: pd.DataFrame | None = None, freq: str = "W") -> WeeklyDataset:
    """Samples at each decision date (weekly or monthly) with the return to the next one."""
    feats = all_stock_features(data, sectors, meetings)
    mkt = market_features(data.index, data.panels["close"], nifty, vix)
    rows = decision_rows(data.index, freq)
    X, has, m = feature_tensor(feats, mkt, rows, feature_names(meetings is not None))
    fwd = forward_returns(data.panels["open"], rows).to_numpy()
    valid = has & ~np.isnan(fwd) & ~np.isnan(m).any(axis=-1)[:, None]
    fwd_valid = np.where(valid, fwd, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # weeks with no valid stock -> NaN
        ew = np.nanmean(fwd_valid, axis=1)
    excess = fwd - ew[:, None]
    sl = slice(min_week, None)
    return WeeklyDataset(
        dates=data.index[rows][sl], symbols=list(data.symbols),
        X=np.nan_to_num(X[sl]).astype(np.float32), valid=valid[sl],
        fwd=np.nan_to_num(fwd[sl]), excess=np.nan_to_num(excess[sl]),
        market=np.nan_to_num(m[sl]).astype(np.float32), ew_fwd=np.nan_to_num(ew[sl]),
    )
