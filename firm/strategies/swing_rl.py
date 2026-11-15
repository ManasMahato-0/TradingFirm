"""Team D (swing stocks): RL picker + risk dial, with LightGBM and rule-based rivals.

Picker: one PPO policy shared by all stocks. Per stock-week it sees ranked features,
market context and whether we already hold the stock; it chooses hold (1) or not (0).
Reward: next week's return above the equal-weight average, minus trading costs,
minus a variance penalty. Score = probability of "hold", averaged over seeds.

Risk dial: PPO over market features choosing exposure 0/25/50/75/100%.
Reward: exposure * market (equal-weight) weekly return, minus switching costs,
minus a penalty on increases in drawdown. Trained on the market, not on the picker,
so its training data is independent of the picker's in-sample results.
"""
from __future__ import annotations

from datetime import date

import gymnasium as gym
import numpy as np
import pandas as pd

from firm.backtest.data import DataView, MarketData
from firm.features.swing import (
    MARKET_FEATURES, WeeklyDataset, all_stock_features, build_weekly_dataset, feature_names,
    market_features,
)

CASH_BUFFER = 0.98
DIAL_LEVELS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
_MODEL_CACHE: dict[tuple, object] = {}  # reuse trained models across configs in one run


# ---------------------------------------------------------------- environments

class PickerEnv(gym.Env):
    """symmetric=False: reward = a * excess (holding pays, skipping pays nothing).
    symmetric=True:  reward = (2a - 1) * excess (skipping a laggard pays too), so
    "never hold" is no longer a safe default and the policy must discriminate."""

    def __init__(self, ds: WeeklyDataset, cost: float = 0.15, risk_pen: float = 0.01,
                 ep_len: int = 26, symmetric: bool = False, clip: float = 30.0) -> None:
        self.X, self.valid = ds.X, ds.valid
        self.R = np.clip(ds.excess * 100, -clip, clip).astype(np.float32)  # percent
        self.cost, self.risk_pen, self.ep_len, self.symmetric = cost, risk_pen, ep_len, symmetric
        k = self.X.shape[-1] + 1
        self.observation_space = gym.spaces.Box(-5, 5, (k,), np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.starts = np.argwhere(self.valid[:-1])
        if len(self.starts) == 0:
            raise ValueError("no valid training samples")
        self.t = self.s = self.n = 0
        self.h = 0.0

    def _obs(self) -> np.ndarray:
        x = self.X[self.t, self.s] if self.valid[self.t, self.s] else np.zeros(self.X.shape[-1], np.float32)
        return np.append(x, np.float32(self.h)).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t, self.s = self.starts[self.np_random.integers(len(self.starts))]
        self.h, self.n = 0.0, 0
        return self._obs(), {}

    def step(self, action):
        a = float(action)
        if self.valid[self.t, self.s]:
            r = self.R[self.t, self.s]
            gain = (2 * a - 1) * r if self.symmetric else a * r
            reward = gain - self.cost * abs(a - self.h) - self.risk_pen * a * r * r
        else:
            a, reward = 0.0, 0.0
        self.h = a
        self.t += 1
        self.n += 1
        truncated = self.n >= self.ep_len or self.t >= len(self.X) - 1
        return self._obs(), float(reward), False, truncated, {}


class DialEnv(gym.Env):
    def __init__(self, ds: WeeklyDataset, cost: float = 0.15, dd_pen: float = 1.0,
                 ep_len: int = 52) -> None:
        weeks = ds.valid.any(axis=1)
        self.M = ds.market[weeks]
        self.R = np.clip(ds.ew_fwd[weeks], -0.3, 0.3)
        if len(self.M) < 10:
            raise ValueError("not enough weeks to train the dial")
        self.cost, self.dd_pen, self.ep_len = cost, dd_pen, ep_len
        self.observation_space = gym.spaces.Box(-5, 5, (self.M.shape[1] + 2,), np.float32)
        self.action_space = gym.spaces.Discrete(len(DIAL_LEVELS))
        self.t = self.n = 0
        self.e = self.eq = self.peak = 1.0

    def _obs(self) -> np.ndarray:
        own_dd = 1 - self.eq / self.peak  # our own portfolio's drawdown
        return np.append(self.M[self.t], [self.e, own_dd]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = int(self.np_random.integers(max(1, len(self.M) - self.ep_len)))
        self.e, self.eq, self.peak, self.n = 1.0, 1.0, 1.0, 0
        return self._obs(), {}

    def step(self, action):
        e = float(DIAL_LEVELS[int(action)])
        ret = e * self.R[self.t]
        dd_old = 1 - self.eq / self.peak
        self.eq *= 1 + ret
        self.peak = max(self.peak, self.eq)
        dd_new = 1 - self.eq / self.peak
        reward = 100 * np.log1p(ret) - self.cost * abs(e - self.e) - self.dd_pen * 100 * max(0.0, dd_new - dd_old)
        self.e = e
        self.t += 1
        self.n += 1
        truncated = self.n >= self.ep_len or self.t >= len(self.M) - 1
        return self._obs(), float(reward), False, truncated, {}


# ---------------------------------------------------------------- training

def _ppo(env_fn, seed: int, steps: int, gamma: float, n_envs: int = 8):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    env = make_vec_env(env_fn, n_envs=n_envs, seed=seed)
    model = PPO("MlpPolicy", env, n_steps=256, batch_size=512, n_epochs=5, gamma=gamma,
                learning_rate=3e-4, ent_coef=0.01, policy_kwargs={"net_arch": [64, 64]},
                seed=seed, device="cpu", verbose=0)
    model.learn(total_timesteps=steps)
    return model


def _probs(models, obs: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        t = torch.as_tensor(obs, dtype=torch.float32)
        return np.mean([m.policy.get_distribution(t).distribution.probs.numpy() for m in models], axis=0)


def train_ppo_picker(ds, seeds, steps, cost, risk_pen):
    return [_ppo(lambda: PickerEnv(ds, cost, risk_pen), s, steps, gamma=0.8) for s in seeds]


def train_ppo_dial(ds, seeds, steps, cost, dd_pen):
    return [_ppo(lambda: DialEnv(ds, cost, dd_pen), s, steps, gamma=0.9, n_envs=4) for s in seeds]


def train_lgbm_picker(ds, seed: int = 0):
    from lightgbm import LGBMRegressor

    X = ds.X[ds.valid]
    y = np.clip(ds.excess[ds.valid] * 100, -30, 30)
    model = LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                          min_child_samples=200, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, random_state=seed, verbose=-1)
    return model.fit(X, y)


# ---------------------------------------------------------------- strategy

class SwingStrategy:
    """picker: 'ppo' | 'lgbm';  dial: 'none' | 'rule' | 'ppo'."""

    def __init__(self, sectors: dict[str, str], nifty: pd.Series, vix: pd.Series | None = None,
                 picker: str = "ppo", dial: str = "none", top_n: int = 10,
                 seeds: tuple[int, ...] = (0, 1, 2), picker_steps: int = 200_000,
                 dial_steps: int = 60_000, cost: float = 0.15, risk_pen: float = 0.01,
                 dd_pen: float = 1.0, keep_rank: int | None = None,
                 meetings: pd.DataFrame | None = None) -> None:
        self.sectors, self.nifty, self.vix, self.meetings = sectors, nifty, vix, meetings
        self.picker, self.dial, self.top_n = picker, dial, top_n
        self.features = feature_names(meetings is not None)
        self.name = f"swing_{picker}_{dial}" + ("+res" if meetings is not None else "")
        self.params = {"picker": picker, "dial": dial, "top_n": top_n, "seeds": list(seeds),
                       "results_features": meetings is not None,
                       "picker_steps": picker_steps, "dial_steps": dial_steps, "cost": cost,
                       "risk_pen": risk_pen, "dd_pen": dd_pen, "rebalance": "weekly",
                       "keep_rank": keep_rank or 2 * top_n}
        self._seeds, self._held = tuple(seeds), set()
        self._picker_model = self._dial_model = None

    def fit(self, data: MarketData, train_start: date, train_end: date) -> None:
        p = self.params
        cut = lambda s: None if s is None else s[s.index.date < train_end]  # noqa: E731
        ds = build_weekly_dataset(data, self.sectors, cut(self.nifty), cut(self.vix),
                                  meetings=self.meetings)
        pkey = ("picker", self.picker, self._seeds, p["picker_steps"], p["cost"], p["risk_pen"],
                p["results_features"], train_end)
        if pkey not in _MODEL_CACHE:
            _MODEL_CACHE[pkey] = (
                train_ppo_picker(ds, self._seeds, p["picker_steps"], p["cost"], p["risk_pen"])
                if self.picker == "ppo" else train_lgbm_picker(ds, self._seeds[0])
            )
        self._picker_model = _MODEL_CACHE[pkey]
        if self.dial == "ppo":
            dkey = ("dial", self._seeds, p["dial_steps"], p["cost"], p["dd_pen"], train_end)
            if dkey not in _MODEL_CACHE:
                _MODEL_CACHE[dkey] = train_ppo_dial(ds, self._seeds, p["dial_steps"], p["cost"], p["dd_pen"])
            self._dial_model = _MODEL_CACHE[dkey]
        self._held = set()

    def _features(self, view: DataView):
        def build(d: MarketData):
            feats = all_stock_features(d, self.sectors, self.meetings)
            return feats, market_features(d.index, d.panels["close"], self.nifty, self.vix)
        return view.cached(f"swing_features_{self.name}", build)

    def exposure(self, m: np.ndarray) -> float:
        if self.dial == "rule":
            return 1.0 if m[MARKET_FEATURES.index("mkt_dist_sma200")] > 0 else 0.0
        if self.dial == "ppo":
            probs = _probs(self._dial_model, np.append(m, [1.0, 0.0]).astype(np.float32)[None, :])[0]
            return float(probs @ DIAL_LEVELS)
        return 1.0

    def target_weights(self, view: DataView) -> dict[str, float] | None:
        if not view.is_period_end("W"):
            return None
        feats, mkt = self._features(view)
        i = view.bar_index
        stock = np.stack([feats[k].iloc[i].to_numpy() for k in self.features], axis=-1)
        m = mkt[MARKET_FEATURES].iloc[i].to_numpy(dtype=float)
        if np.isnan(m).any():
            return {}
        symbols = view.symbols
        has = ~np.isnan(stock).any(axis=-1) & view.current("close").reindex(symbols).notna().to_numpy()
        X = np.concatenate([np.nan_to_num(stock), np.repeat(m[None, :], len(symbols), axis=0)], axis=1)

        if self.picker == "ppo":
            held = np.array([1.0 if s in self._held else 0.0 for s in symbols])
            scores = _probs(self._picker_model, np.concatenate([X, held[:, None]], axis=1))[:, 1]
            eligible = has & (scores > 0.5)
        else:
            scores = self._picker_model.predict(X)
            eligible = has & (scores > 0)
        ranked = [symbols[j] for j in np.argsort(-scores) if eligible[j]]
        # Holding buffer: keep a held stock while it ranks within keep_rank,
        # so the portfolio isn't reshuffled (and charged costs) every week.
        keep = [s for s in ranked[: self.params["keep_rank"]] if s in self._held][: self.top_n]
        chosen = keep + [s for s in ranked if s not in keep][: self.top_n - len(keep)]
        self._held = set(chosen)
        w = self.exposure(m) * CASH_BUFFER / self.top_n
        return {s: w for s in chosen} if w > 0 else {}
