"""Sandbox strategies (docs/PLAN.md, pre-registered): model trader (accounts A and B),
random trader (C) and cash (D). All trade the 50 liquid stocks long and short, hold 6 bars,
enter only on bars starting 09:45 to 14:25, and are flattened by the engine at 15:10.

A and B are the SAME class with the same parameters; they differ only in which data the
model was trained on (and B's fixed warm-up before it may trade).
"""
from __future__ import annotations

import sys
from datetime import date, time as dtime

import numpy as np
import pandas as pd

from firm.backtest.data import DataView
from firm.features.intraday_model import (
    FEATURES, FIRST_K, HOLD_BARS, LAST_K, model_features,
)
from firm.strategies.intraday import IntradayBase

SLOT_VALUE = 24_000.0   # one slot (24% of Rs 1L). A stock costing more than this per share cannot be bought
                        # at all (the order would round to zero shares), so it is never picked.
THRESHOLD = 0.0025      # default entry bar when a model carries none (live runs use each model's own bar)
MAX_NEW_PER_BAR = 2
RANDOM_ENTRY_PROB = 0.02

LABELS = {
    "ret_1": "the last 5 minutes vs Nifty", "ret_3": "the last 15 minutes vs Nifty",
    "ret_6": "the last 30 minutes vs Nifty", "ret_12": "the last hour vs Nifty",
    "vwap_dev": "distance from VWAP", "vwap_z": "stretch from VWAP", "day_ret": "the day so far vs Nifty",
    "gap_rel": "the opening gap vs Nifty", "or_pos": "position in the opening range",
    "volr": "volume vs normal", "sd": "recent volatility", "tod": "time of day",
    "nifty_day": "Nifty's day", "nifty_ret_6": "Nifty's last 30 minutes",
}


class _Intraday(IntradayBase):
    """Shared plumbing: per-day reset of entry bookkeeping and the 09:45-14:25 window."""

    def __init__(self, nifty=None, nifty_open=None, **kw) -> None:
        super().__init__(nifty, max_pos=4, slot=0.24, **kw)
        self.nifty_open = nifty_open
        self._entry_i: dict[str, int] = {}
        self._ei_day = None

    def _new_day(self, d) -> None:
        if self._ei_day != d:
            self._ei_day, self._entry_i = d, {}

    def _open_window(self, f, i, tod) -> bool:
        return FIRST_K <= f["k"][i] <= LAST_K and dtime(9, 45) <= tod <= dtime(14, 25)

    def _due_exits(self, i: int) -> bool:
        changed = False
        for s in list(self._pos):
            if i - self._entry_i.get(s, i) >= HOLD_BARS:
                self._exit(s, f"held {HOLD_BARS} bars (30 minutes), time to leave")
                changed = True
        return changed

    def _take(self, cands: list[tuple[str, int, str]], i: int, tod, limit: int) -> bool:
        fresh = [c for c in cands if c[0] not in self._entered and c[0] not in self._pos][:limit]
        changed = self._enter(fresh, tod)
        for s, _, _ in fresh:
            if s in self._pos:
                self._entry_i[s] = i
        return changed


class ModelStrategy(_Intraday):
    name = "intraday_model"

    def __init__(self, nifty=None, nifty_open=None, model=None, threshold: float = THRESHOLD,
                 trade_from: date | None = None, label: str = "A", **kw) -> None:
        super().__init__(nifty, nifty_open, label=label, threshold=threshold, hold_bars=HOLD_BARS,
                         max_new_per_bar=MAX_NEW_PER_BAR, **kw)
        self.model, self.threshold, self.trade_from = model, threshold, trade_from
        self.last_scores: dict[str, float] = {}

    def _model_features(self, view: DataView):
        return view.cached("sandbox_model_features",
                           lambda d: model_features(d, self.nifty, self.nifty_open))

    def _reasons(self, X: np.ndarray, rows: list[int]) -> dict[int, str]:
        try:
            contrib = self.model.predict(X[rows], pred_contrib=True)[:, :-1]
        except Exception:
            return {}
        out = {}
        for n, r in enumerate(rows):
            top = np.argsort(-np.abs(contrib[n]))[:2]
            out[r] = "; ".join(f"{LABELS[FEATURES[j]]} {100 * contrib[n][j]:+.2f}%" for j in top)
        return out

    def update(self, view: DataView, f, i: int, tod) -> bool:
        self._new_day(view.now.date())
        changed = self._due_exits(i)
        if self.model is None or not self._open_window(f, i, tod):
            return changed
        if self.trade_from is not None and view.now.date() < self.trade_from:
            return changed
        feats = self._model_features(view)
        X = feats["X"][i]
        close = f["close"].iloc[i]
        ok = (np.isfinite(X[:, FEATURES.index("vwap_dev")]) & close.notna().to_numpy()
              & (close.to_numpy(dtype=float) <= SLOT_VALUE))      # only stocks we can actually buy
        rows = np.flatnonzero(ok)
        if not len(rows):
            return changed
        pred = np.asarray(self.model.predict(X[rows]), dtype=float)
        symbols = feats["symbols"]
        self.last_scores = {symbols[r]: float(p) for r, p in zip(rows, pred)}
        order = np.argsort(-np.abs(pred))
        picks = [(int(rows[j]), float(pred[j])) for j in order if abs(pred[j]) >= self.threshold]
        if not picks:
            return changed
        reasons = self._reasons(X, [r for r, _ in picks[:MAX_NEW_PER_BAR + 2]])
        cands = [(symbols[r], 1 if p > 0 else -1,
                  f"model expects {p:+.2%} vs Nifty over the next 30 min"
                  + (f" (mainly {reasons[r]})" if r in reasons else ""))
                 for r, p in picks]
        return self._take(cands, i, tod, MAX_NEW_PER_BAR) or changed


class RandomTrader(_Intraday):
    """Account C: the cost floor. Entry probability fixed in advance; seeded by date and bar."""
    name = "intraday_random"

    def __init__(self, nifty=None, nifty_open=None, prob: float = RANDOM_ENTRY_PROB, **kw) -> None:
        super().__init__(nifty, nifty_open, label="C", prob=prob, hold_bars=HOLD_BARS, **kw)
        self.prob = prob

    def update(self, view: DataView, f, i, tod) -> bool:
        day = view.now.date()
        self._new_day(day)
        changed = self._due_exits(i)
        if not self._open_window(f, i, tod):
            return changed
        rng = np.random.default_rng([int(day.strftime("%Y%m%d")), int(f["k"][i])])
        if rng.random() >= self.prob:
            return changed
        close = f["close"].iloc[i]
        names = [s for s in close.index[close.notna() & (close <= SLOT_VALUE)]
                 if s not in self._entered and s not in self._pos]
        if not names:
            return changed
        sym = names[int(rng.integers(len(names)))]
        side = 1 if rng.random() < 0.5 else -1
        return self._take([(sym, side, "random pick (the cost-floor control)")], i, tod, 1) or changed


class CashOnly:
    """Account D: holds nothing; earns the idle-cash rate like the others."""
    name = "intraday_cash"
    params = {"label": "D"}
    last_reason: list[str] = []

    def target_weights(self, view: DataView) -> dict[str, float] | None:
        return None


IntradayBase.depends_on = (sys.modules["firm.strategies.intraday"], sys.modules[__name__],
                           sys.modules["firm.features.intraday_model"])
