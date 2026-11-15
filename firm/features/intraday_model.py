"""Features, labels and training for the intraday sandbox models (accounts A and B).

All 14 features are causal at a bar's close. The label is the stock's return from the open
of the next bar to the open 7 bars later (a 6-bar hold, filled at next-bar opens like the
backtest) minus Nifty's return over the same window. Samples never cross the day boundary.
Pre-registered in docs/PLAN.md; do not change a setting without a new pre-registration.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from firm.backtest.data import MarketData
from firm.strategies.intraday import OR_BARS, intraday_features

FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_12",        # last n bars, minus Nifty's
    "vwap_dev", "vwap_z",                        # % from VWAP; in units of recent bar volatility
    "day_ret", "gap_rel",                        # day return and opening gap, minus Nifty's
    "or_pos",                                    # inside the opening range, -5..+5
    "volr", "sd",                                # log volume ratio; recent bar volatility (%)
    "tod", "nifty_day", "nifty_ret_6",           # time of day; Nifty's day and 6-bar return
]
FIRST_K = OR_BARS            # first decision: the bar starting 09:45 (opening range complete)
LAST_K = 62                  # last decision: the bar starting 14:25
HOLD_BARS = 6
LABEL_CLIP = 0.05

LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=1000,
                   subsample=0.7, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)


def _aligned(series: pd.Series | None, index: pd.DatetimeIndex) -> pd.Series | None:
    return None if series is None else series.reindex(index).ffill()


def model_features(data: MarketData, nifty_close: pd.Series,
                   nifty_open: pd.Series | None = None) -> dict:
    """Returns {"X": [bars, stocks, 14] float32, "k", "day", "symbols"}.
    nifty_open: Nifty's 5-minute OPEN series (for the opening gap); first-bar close if absent."""
    f = intraday_features(data, nifty_close)
    idx, c = data.index, f["close"]
    k, day = f["k"], f["day"]
    nc = _aligned(nifty_close, idx)
    no = _aligned(nifty_open, idx) if nifty_open is not None else nc
    n_first_open = no.groupby(day).transform("first")
    n_first_close = nc.groupby(day).transform("first")
    n_prev_close = pd.Series(f["nifty_prev_close"], index=idx)

    def ret(n: int) -> pd.DataFrame:
        r = c / c.shift(n) - 1
        nr = nc / nc.shift(n) - 1
        out = r.sub(nr, axis=0)
        return out.where(pd.DataFrame(np.repeat((k >= n)[:, None], c.shape[1], axis=1), index=idx, columns=c.columns))

    vw, sd = f["vwap"], f["sd"]
    or_hi, or_lo = f["or_high"], f["or_low"]
    day_ret = (c / f["day_open"] - 1).sub(nc / n_first_open - 1, axis=0)
    gap = (f["day_open"] / f["prev_close"] - 1).sub(n_first_open / n_prev_close - 1, axis=0)
    or_pos = ((c - (or_hi + or_lo) / 2) / (or_hi - or_lo).replace(0, np.nan)).clip(-5, 5)

    def bcast(x: pd.Series | np.ndarray) -> pd.DataFrame:
        v = np.asarray(x, dtype=float)
        return pd.DataFrame(np.repeat(v[:, None], c.shape[1], axis=1), index=idx, columns=c.columns)

    feats = [ret(1), ret(3), ret(6), ret(12),
             (c / vw - 1) * 100, (c / vw - 1) / sd,
             day_ret, gap, or_pos,
             np.log1p(f["vol_ratio"].clip(0, 50)), sd * 100,
             bcast(k / 74.0), bcast(nc / n_prev_close - 1),
             bcast((nc / nc.shift(6) - 1).where(k >= 6))]
    X = np.stack([x.to_numpy(dtype=np.float32) for x in feats], axis=-1)
    X[~np.isfinite(X)] = np.nan
    return {"X": X, "k": k, "day": day, "symbols": list(c.columns)}


def forward_labels(data: MarketData, nifty_close: pd.Series) -> np.ndarray:
    """[bars, stocks]: stock return open[i+1] -> open[i+7] minus Nifty close[i] -> close[i+6].
    NaN unless the decision bar is in the entry window and the exit is on the same day."""
    o = data.panels["open"].to_numpy(dtype=float)
    idx = data.index
    day = np.asarray(idx.date)
    k = pd.Series(1, index=idx).groupby(day).cumsum().to_numpy() - 1
    n = _aligned(nifty_close, idx).to_numpy(dtype=float)
    y = np.full(o.shape, np.nan)
    for i in range(len(idx) - HOLD_BARS - 1):
        if not (FIRST_K <= k[i] <= LAST_K) or day[i + HOLD_BARS + 1] != day[i]:
            continue
        y[i] = o[i + HOLD_BARS + 1] / o[i + 1] - 1 - (n[i + HOLD_BARS] / n[i] - 1)
    return np.clip(y, -LABEL_CLIP, LABEL_CLIP)


def build_training_set(data: MarketData, nifty_close: pd.Series, nifty_open: pd.Series | None = None,
                       start: date | None = None, end: date | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Rows from decision bars dated start <= day < end. Features may use earlier days
    (volume ratios need history); labels never reach past their own day."""
    feats = model_features(data, nifty_close, nifty_open)
    y = forward_labels(data, nifty_close)
    days = feats["day"]
    keep = np.ones(len(days), dtype=bool)
    if start is not None:
        keep &= days >= start
    if end is not None:
        keep &= days < end
    X = feats["X"][keep].reshape(-1, len(FEATURES))
    y = y[keep].reshape(-1)
    ok = np.isfinite(y) & np.isfinite(X[:, FEATURES.index("vwap_dev")])
    return X[ok], y[ok]


def train_model(X: np.ndarray, y: np.ndarray):
    from lightgbm import LGBMRegressor

    if len(X) == 0:
        raise ValueError("no training rows")
    return LGBMRegressor(**LGBM_PARAMS).fit(X, y)
