"""Intraday baselines on 5-minute bars. Long and short, flat by 15:15 (the engine forces it).

Every strategy decides at a bar's close and fills at the next bar's open. Each holds at
most `max_pos` positions of `slot` of equity each, enters a stock at most once a day,
and enters nothing after `last_entry`.

  ORB            opening range breakout (first 30 minutes), exit back through the range middle
  NiftyMomentum  first half hour of Nifty predicts the last half hour; trade the most liquid stocks
  Gap            overnight gap vs Nifty's gap: 'fade' (bet it fills) or 'follow'
  VWAPRevert     stock stretched far from its VWAP: bet it comes back; exit at VWAP
  FilingReaction stock moves sharply right after an NSE filing: follow the move
"""
from __future__ import annotations

import sys
from datetime import time as dtime

import numpy as np
import pandas as pd

from firm.backtest.data import DataView, MarketData

SLOT = 0.24
MAX_POS = 4
OR_BARS = 6                 # 09:15-09:40 bars = first 30 minutes
LAST_ENTRY = dtime(14, 30)


# ---------------------------------------------------------------- features

def intraday_features(d: MarketData, nifty: pd.Series | None = None) -> dict[str, pd.DataFrame | np.ndarray]:
    """Causal per-bar features. Opening-range values exist only once the range is complete."""
    o, h, l, c, v = (d.panels[k] for k in ("open", "high", "low", "close", "volume"))
    day = np.asarray(d.index.date)
    k = pd.Series(1, index=d.index).groupby(day).cumsum().to_numpy() - 1
    tod = np.asarray(d.index.time)
    first = pd.DataFrame(np.repeat((k == 0)[:, None], c.shape[1], axis=1), index=c.index, columns=c.columns)

    daily_close = c.groupby(day).last()
    prev_close = daily_close.shift(1).reindex(day)
    prev_close.index = c.index
    day_open = o.groupby(day).transform("first")
    in_or = pd.DataFrame(np.repeat((k < OR_BARS)[:, None], c.shape[1], axis=1), index=c.index, columns=c.columns)
    or_high = h.where(in_or).groupby(day).transform("max").where(~in_or)
    or_low = l.where(in_or).groupby(day).transform("min").where(~in_or)
    tp = (h + l + c) / 3
    vwap = (tp * v).groupby(day).cumsum() / v.groupby(day).cumsum().replace(0, np.nan)
    vol_ratio = v / v.groupby(tod).transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())
    r = c.pct_change(fill_method=None).mask(first)  # no overnight jumps
    sd = r.rolling(60, min_periods=30).std()
    out = {"day": day, "k": k, "tod": tod, "close": c, "prev_close": prev_close, "day_open": day_open,
           "or_high": or_high, "or_low": or_low, "vwap": vwap, "vol_ratio": vol_ratio, "sd": sd}
    if nifty is not None:
        n = nifty.reindex(d.index).ffill()
        nd = n.groupby(day).last().shift(1).reindex(day)
        out["nifty"] = n.to_numpy()
        out["nifty_prev_close"] = nd.to_numpy()
        out["nifty_open"] = n.groupby(day).transform("first").to_numpy()
    return out


# ---------------------------------------------------------------- base

class IntradayBase:
    name = "intraday_base"

    def __init__(self, nifty: pd.Series | None = None, max_pos: int = MAX_POS, slot: float = SLOT,
                 max_entries_per_day: int | None = None, **params) -> None:
        self.nifty = nifty
        self.max_pos, self.slot = max_pos, slot
        self.max_entries_per_day = max_entries_per_day  # None = no cap (the baseline behaviour)
        self.params = {"max_pos": max_pos, "slot": slot, "last_entry": str(LAST_ENTRY), **params}
        if max_entries_per_day is not None:  # only recorded when set, so baseline params are unchanged
            self.params["max_entries_per_day"] = max_entries_per_day
        self._pos: dict[str, int] = {}
        self._entered: set[str] = set()
        self._day = None
        self.last_reason: list[str] = []  # plain-English reasons for the latest change

    def _feats(self, view: DataView):
        return view.cached(f"intraday_features_{id(self.nifty)}", lambda d: intraday_features(d, self.nifty))

    def target_weights(self, view: DataView) -> dict[str, float] | None:
        t = view.now
        if t.date() != self._day:
            self._day, self._pos, self._entered = t.date(), {}, set()
        f = self._feats(view)
        self.last_reason = []
        if not self.update(view, f, view.bar_index, t.time()):
            return None
        return {s: side * self.slot for s, side in self._pos.items()}

    def update(self, view, f, i, tod) -> bool:  # subclasses: change self._pos, return True if changed
        raise NotImplementedError

    def _enter(self, ranked: list[tuple[str, int, str]], tod) -> bool:
        """ranked: (symbol, side, reason), best first."""
        if tod >= LAST_ENTRY:
            return False
        changed = False
        for sym, side, why in ranked:
            if len(self._pos) >= self.max_pos:
                break
            # Daily cap on NEW entries across all stocks (exits and re-entries of nothing else count:
            # each stock can enter at most once a day, so len(_entered) is the day's entry count).
            if self.max_entries_per_day is not None and len(self._entered) >= self.max_entries_per_day:
                break
            if sym in self._entered or sym in self._pos:
                continue
            self._pos[sym] = side
            self._entered.add(sym)
            self.last_reason.append(f"{'LONG' if side > 0 else 'SHORT'} {sym}: {why}")
            changed = True
        return changed

    def _exit(self, sym: str, why: str) -> None:
        side = self._pos.pop(sym)
        self.last_reason.append(f"EXIT {'long' if side > 0 else 'short'} {sym}: {why}")


def _row(f, key, i) -> pd.Series:
    return f[key].iloc[i]


# ---------------------------------------------------------------- strategies

class ORB(IntradayBase):
    name = "orb"

    def __init__(self, nifty=None, vol_mult: float = 1.5, market_filter: bool = False, **kw) -> None:
        super().__init__(nifty, vol_mult=vol_mult, **kw)
        self.vol_mult, self.market_filter = vol_mult, market_filter
        if market_filter:  # only recorded when on, so the baseline's params are unchanged
            self.params["market_filter"] = True

    def update(self, view, f, i, tod) -> bool:
        if f["k"][i] < OR_BARS:
            return False
        c, hi, lo = _row(f, "close", i), _row(f, "or_high", i), _row(f, "or_low", i)
        mid, width = (hi + lo) / 2, (hi - lo).replace(0, np.nan)
        changed = False
        for s, side in list(self._pos.items()):
            if (side > 0 and c[s] < mid[s]) or (side < 0 and c[s] > mid[s]):
                self._exit(s, "back inside the opening range")
                changed = True
        vr = _row(f, "vol_ratio", i)
        up = ((c - hi) / width)[(c > hi) & (vr > self.vol_mult)].dropna()
        dn = ((lo - c) / width)[(c < lo) & (vr > self.vol_mult)].dropna()
        if self.market_filter:
            # Longs only while Nifty is above its previous close, shorts only while below it.
            nifty_now, nifty_prev = f["nifty"][i], f["nifty_prev_close"][i]
            if not (np.isfinite(nifty_now) and np.isfinite(nifty_prev)):
                up, dn = up.iloc[0:0], dn.iloc[0:0]
            elif nifty_now > nifty_prev:
                dn = dn.iloc[0:0]
            elif nifty_now < nifty_prev:
                up = up.iloc[0:0]
            else:
                up, dn = up.iloc[0:0], dn.iloc[0:0]
        cands = [(s, 1, f"broke above its opening range on {vr[s]:.1f}x normal volume") for s in up.index] + \
                [(s, -1, f"broke below its opening range on {vr[s]:.1f}x normal volume") for s in dn.index]
        score = {**up.to_dict(), **dn.to_dict()}
        cands.sort(key=lambda x: -score[x[0]])
        return self._enter(cands, tod) or changed


class NiftyMomentum(IntradayBase):
    """Gao et al.: the market's first half hour predicts its last half hour."""
    name = "nifty_momentum"

    def __init__(self, nifty=None, basket: list[str] | None = None, threshold: float = 0.004,
                 entry: dtime = dtime(14, 40), **kw) -> None:
        super().__init__(nifty, basket=list(basket or []), threshold=threshold, entry=str(entry), **kw)
        self.basket, self.threshold, self.entry = list(basket or []), threshold, entry

    def update(self, view, f, i, tod) -> bool:
        if tod != self.entry:
            return False
        day = f["day"][i]
        first = np.flatnonzero((f["day"] == day) & (f["k"] == OR_BARS - 1))
        if not len(first):
            return False
        r = f["nifty"][first[0]] / f["nifty_prev_close"][i] - 1  # previous close to 09:45
        if not np.isfinite(r) or abs(r) < self.threshold:
            return False
        side = 1 if r > 0 else -1
        c = _row(f, "close", i)
        cands = [(s, side, f"Nifty moved {r:+.2%} in the first half hour; riding the last half hour")
                 for s in self.basket if pd.notna(c.get(s))]
        return self._enter(cands, dtime(0, 0))  # entry time is fixed by design, after LAST_ENTRY


class Gap(IntradayBase):
    name = "gap"

    def __init__(self, nifty=None, mode: str = "fade", min_gap: float = 0.015, **kw) -> None:
        super().__init__(nifty, mode=mode, min_gap=min_gap, **kw)
        self.mode, self.min_gap = mode, min_gap
        self.name = f"gap_{mode}"

    def update(self, view, f, i, tod) -> bool:
        c, pc = _row(f, "close", i), _row(f, "prev_close", i)
        changed = False
        if self.mode == "fade":
            for s, side in list(self._pos.items()):
                if (side > 0 and c[s] >= pc[s]) or (side < 0 and c[s] <= pc[s]):
                    self._exit(s, "gap filled (back to yesterday's close)")
                    changed = True
        if f["k"][i] != 0:
            return changed
        n_gap = f["nifty_open"][i] / f["nifty_prev_close"][i] - 1 if "nifty_open" in f else 0.0
        gap = (_row(f, "day_open", i) / pc - 1 - (n_gap if np.isfinite(n_gap) else 0)).dropna()
        gap = gap[gap.abs() > self.min_gap]
        sign = -1 if self.mode == "fade" else 1
        cands = [(s, int(sign * np.sign(g)), f"opened {g:+.1%} vs Nifty's gap; {'betting it fills' if sign < 0 else 'betting it continues'}")
                 for s, g in gap.abs().sort_values(ascending=False).items() for g in [gap[s]]]
        return self._enter(cands, tod) or changed


class VWAPRevert(IntradayBase):
    name = "vwap_revert"

    def __init__(self, nifty=None, z: float = 4.0, start: dtime = dtime(10, 0), **kw) -> None:
        super().__init__(nifty, z=z, start=str(start), **kw)
        self.z, self.start = z, start

    def update(self, view, f, i, tod) -> bool:
        c, vw = _row(f, "close", i), _row(f, "vwap", i)
        changed = False
        for s, side in list(self._pos.items()):
            if (side > 0 and c[s] >= vw[s]) or (side < 0 and c[s] <= vw[s]):
                self._exit(s, "back to VWAP")
                changed = True
        if tod < self.start:
            return changed
        z = ((c / vw - 1) / _row(f, "sd", i)).dropna()
        z = z[z.abs() > self.z]
        cands = [(s, -int(np.sign(z[s])), f"{abs(z[s]):.1f} typical moves {'above' if z[s] > 0 else 'below'} VWAP")
                 for s in z.abs().sort_values(ascending=False).index]
        return self._enter(cands, tod) or changed


class FilingReaction(IntradayBase):
    """Follow a sharp move (vs Nifty) in the 10 minutes after an NSE filing."""
    name = "filing_reaction"

    def __init__(self, nifty=None, events: pd.DataFrame | None = None, min_move: float = 0.01, **kw) -> None:
        super().__init__(nifty, min_move=min_move, **kw)
        self.events, self.min_move = events, min_move
        self._ref: dict[str, float] = {}

    def _triggers(self, view: DataView):
        def build(d: MarketData):
            f = self._feats(view)
            idx, c, n = d.index, d.panels["close"], f["nifty"]
            trig: dict[int, list[tuple[str, float, str, float]]] = {}
            if self.events is None:
                return trig
            for e in self.events.itertuples():
                if e.symbol not in c:
                    continue
                b0 = idx.searchsorted(e.ts, side="right") - 1  # bar containing the filing
                b1 = b0 + 1
                if b0 < 1 or b1 >= len(idx) or idx[b1].date() != e.ts.date() or idx[b0 - 1].date() != e.ts.date():
                    continue
                before = c[e.symbol].iloc[b0 - 1]
                move = c[e.symbol].iloc[b1] / before - 1 - (n[b1] / n[b0 - 1] - 1)
                if np.isfinite(move) and abs(move) >= self.min_move:
                    trig.setdefault(b1, []).append((e.symbol, float(move), e.category, float(before)))
            return trig
        return view.cached(f"filing_triggers_{id(self.events)}", build)

    def update(self, view, f, i, tod) -> bool:
        trig = self._triggers(view)
        c = _row(f, "close", i)
        changed = False
        for s, side in list(self._pos.items()):
            ref = self._ref.get(s)
            if ref and ((side > 0 and c[s] < ref) or (side < 0 and c[s] > ref)):
                self._exit(s, "gave back the whole reaction")
                changed = True
        cands = []
        for sym, move, cat, before in sorted(trig.get(i, []), key=lambda x: -abs(x[1])):
            cands.append((sym, 1 if move > 0 else -1, f"moved {move:+.1%} vs Nifty in 10 min after a '{cat}' filing"))
            self._ref[sym] = before
        return self._enter(cands, tod) or changed

    def target_weights(self, view):
        if view.now.date() != self._day:
            self._ref = {}
        return super().target_weights(view)


IntradayBase.depends_on = (sys.modules[__name__],)
