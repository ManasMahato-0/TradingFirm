"""MomentumPlus: momentum shortlists, an AI picker chooses, a dial sizes.

Monthly, at the last trading day:
  1. Shortlist: top `candidates` stocks by 12-1 momentum (positive momentum only).
  2. Picker scores the shortlist and we hold the best `top_n`, always fully
     invested by rank (no confidence threshold). A held stock is kept while it
     stays in the top `keep_rank` of the shortlist, to limit churn.
       picker="none": momentum order (the baseline inside this framework)
       picker="lgbm": LightGBM predicting next month's return vs the average stock
       picker="ppo":  PPO with symmetric reward (holding winners and skipping
                      laggards both pay), score = P(hold)
  3. Dial sets exposure:
       dial="none":   100%
       dial="rule":   100% if the broad market (equal-weight index) is above its
                      200-day average, else 50%
       dial="ddrule": rule, halved again if our own portfolio is >15% off its peak
       dial="ppo":    PPO over market features + our own drawdown
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

import firm.features.results
import firm.features.swing
import firm.strategies.swing_rl
from firm.backtest.data import DataView, MarketData
from firm.features.swing import (
    MARKET_FEATURES, all_stock_features, build_weekly_dataset, feature_names, market_features,
)
from firm.strategies.swing_rl import (
    _MODEL_CACHE, DIAL_LEVELS, DialEnv, PickerEnv, _ppo, _probs, train_lgbm_picker,
)

CASH_BUFFER = 0.98
DD_LIMIT = 0.15


class MomentumPlus:
    depends_on = (firm.features.swing, firm.features.results, firm.strategies.swing_rl)

    def __init__(self, sectors: dict[str, str], nifty: pd.Series, vix: pd.Series | None = None,
                 picker: str = "lgbm", dial: str = "none", candidates: int = 20, top_n: int = 10,
                 keep_rank: int = 15, seeds: tuple[int, ...] = (0, 1, 2),
                 picker_steps: int = 100_000, dial_steps: int = 60_000, cost: float = 0.15,
                 risk_pen: float = 0.002, dd_pen: float = 1.0,
                 meetings: pd.DataFrame | None = None) -> None:
        self.sectors, self.nifty, self.vix, self.meetings = sectors, nifty, vix, meetings
        self.picker, self.dial = picker, dial
        self.features = feature_names(meetings is not None)
        self.name = f"momplus_{picker}_{dial}" + ("+res" if meetings is not None else "")
        self.params = {"picker": picker, "dial": dial, "candidates": candidates, "top_n": top_n,
                       "keep_rank": keep_rank, "seeds": list(seeds), "picker_steps": picker_steps,
                       "dial_steps": dial_steps, "cost": cost, "risk_pen": risk_pen,
                       "dd_pen": dd_pen, "results_features": meetings is not None,
                       "rebalance": "monthly"}
        self._seeds = tuple(seeds)
        self._picker_model = self._dial_model = None
        self.last_decision: dict | None = None  # why it chose what it chose (for the dashboard)
        self._reset_state()

    def _reset_state(self) -> None:
        self._held: list[str] = []
        self._weights: dict[str, float] = {}
        self._prev_close: pd.Series | None = None
        self._nav = self._peak = 1.0
        self._exposure = 1.0

    # ------------------------------------------------------------ training

    def fit(self, data: MarketData, train_start: date, train_end: date, reset: bool = True) -> None:
        """Train on bars before train_end. reset=False keeps holdings memory and the
        drawdown tracker (live yearly retrain); walk-forward folds start fresh."""
        p = self.params
        cut = lambda s: None if s is None else s[s.index.date < train_end]  # noqa: E731
        nifty, vix = cut(self.nifty), cut(self.vix)
        if self.picker != "none":
            key = ("mp_picker", self.picker, self._seeds, p["picker_steps"], p["cost"],
                   p["risk_pen"], p["results_features"], train_end)
            if key not in _MODEL_CACHE:
                ds = build_weekly_dataset(data, self.sectors, nifty, vix, meetings=self.meetings, freq="M")
                if self.picker == "lgbm":
                    _MODEL_CACHE[key] = train_lgbm_picker(ds, self._seeds[0])
                else:
                    _MODEL_CACHE[key] = [
                        _ppo(lambda: PickerEnv(ds, p["cost"], p["risk_pen"], ep_len=12,
                                               symmetric=True, clip=40.0), s, p["picker_steps"], gamma=0.5)
                        for s in self._seeds
                    ]
            self._picker_model = _MODEL_CACHE[key]
        if self.dial == "ppo":
            key = ("mp_dial", self._seeds, p["dial_steps"], p["cost"], p["dd_pen"], train_end)
            if key not in _MODEL_CACHE:
                weekly = build_weekly_dataset(data, self.sectors, nifty, vix, freq="W")
                _MODEL_CACHE[key] = [
                    _ppo(lambda: DialEnv(weekly, p["cost"], p["dd_pen"]), s, p["dial_steps"],
                         gamma=0.9, n_envs=4)
                    for s in self._seeds
                ]
            self._dial_model = _MODEL_CACHE[key]
        if reset:
            self._reset_state()

    # ------------------------------------------------------------ live decisions

    def _features(self, view: DataView):
        def build(d: MarketData):
            feats = all_stock_features(d, self.sectors, self.meetings)
            return feats, market_features(d.index, d.panels["close"], self.nifty, self.vix)
        return view.cached(f"mp_features_{self.name}", build)

    def _track_nav(self, close: pd.Series) -> None:
        """Approximate our own portfolio value from prices, for the drawdown-aware dial."""
        if self._prev_close is not None and self._weights:
            r = sum(w * (close[s] / self._prev_close[s] - 1) for s, w in self._weights.items()
                    if pd.notna(close.get(s)) and pd.notna(self._prev_close.get(s)))
            self._nav *= 1 + r
            self._peak = max(self._peak, self._nav)
        self._prev_close = close

    @property
    def own_drawdown(self) -> float:
        return 1 - self._nav / self._peak

    def exposure(self, m: np.ndarray) -> float:
        broad_up = m[MARKET_FEATURES.index("ew_dist_sma200")] > 0
        if self.dial == "rule":
            return 1.0 if broad_up else 0.5
        if self.dial == "ddrule":
            return (1.0 if broad_up else 0.5) * (0.5 if self.own_drawdown > DD_LIMIT else 1.0)
        if self.dial == "ppo":
            obs = np.append(m, [self._exposure, self.own_drawdown]).astype(np.float32)[None, :]
            return float(_probs(self._dial_model, obs)[0] @ DIAL_LEVELS)
        return 1.0

    def target_weights(self, view: DataView) -> dict[str, float] | None:
        close = view.current("close")
        self._track_nav(close)
        if not view.is_period_end("M"):
            return None

        p = self.params
        mom = view.pct_change(252, 21)
        mom = mom[close.reindex(mom.index).notna() & (mom > 0)].dropna()
        shortlist = list(mom.nlargest(p["candidates"]).index)
        if not shortlist:
            self._weights, self._held = {}, []
            return {}

        feats, mkt = self._features(view)
        i = view.bar_index
        m = mkt[MARKET_FEATURES].iloc[i].to_numpy(dtype=float)
        if np.isnan(m).any():
            return None
        contrib = None
        if self.picker == "none":
            ranked = shortlist
        else:
            stock = np.stack([feats[k].iloc[i].reindex(shortlist).to_numpy() for k in self.features], axis=-1)
            X = np.concatenate([np.nan_to_num(stock), np.repeat(m[None, :], len(shortlist), axis=0)], axis=1)
            if self.picker == "lgbm":
                scores = self._picker_model.predict(X)
                contrib = self._picker_model.predict(X, pred_contrib=True)[:, :-1]
            else:
                held = np.array([1.0 if s in self._held else 0.0 for s in shortlist])
                scores = _probs(self._picker_model, np.concatenate([X, held[:, None]], axis=1))[:, 1]
            ranked = [shortlist[j] for j in np.argsort(-scores)]

        keep = [s for s in ranked[: p["keep_rank"]] if s in self._held][: p["top_n"]]
        chosen = keep + [s for s in ranked if s not in keep][: p["top_n"] - len(keep)]
        self._exposure = self.exposure(m)
        w = self._exposure * CASH_BUFFER / p["top_n"]
        names = self.features + MARKET_FEATURES
        self.last_decision = {
            "shortlist": shortlist, "ranked": ranked, "chosen": chosen, "kept": keep,
            "dropped": [s for s in self._held if s not in chosen],
            "exposure": self._exposure, "own_drawdown": self.own_drawdown,
            "broad_market_up": bool(m[MARKET_FEATURES.index("ew_dist_sma200")] > 0),
            "momentum": {s: float(mom[s]) for s in shortlist},
            "scores": None if self.picker == "none" else
            {s: float(scores[j]) for j, s in enumerate(shortlist)},
            # Per-stock reasons: each input's push on the predicted return (percentage points).
            "reasons": None if contrib is None else {
                s: sorted(((names[k], float(contrib[j, k])) for k in range(len(names))),
                          key=lambda kv: -abs(kv[1]))[:4]
                for j, s in enumerate(shortlist)},
        }
        self._held = chosen
        self._weights = {s: w for s in chosen} if w > 0 else {}
        return dict(self._weights)
