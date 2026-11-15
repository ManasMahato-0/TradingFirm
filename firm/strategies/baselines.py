"""Simple, well-known strategies. Anything smarter must beat these."""
from __future__ import annotations

from firm.backtest.data import DataView

CASH_BUFFER = 0.98  # leave room for costs and slippage


def _tradable(view: DataView, scores):
    """Drop symbols with no price today."""
    today = view.current("close")
    return scores[today.reindex(scores.index).notna()]


class Momentum:
    """Buy last year's winners (12-month return, skipping the latest month), monthly."""

    name = "momentum_12_1"

    def __init__(self, top_n: int = 10, lookback: int = 252, skip: int = 21) -> None:
        self.top_n, self.lookback, self.skip = top_n, lookback, skip
        self.params = {"top_n": top_n, "lookback": lookback, "skip": skip, "rebalance": "monthly"}
        self._started = False

    def target_weights(self, view: DataView) -> dict[str, float] | None:
        if self._started and not view.is_new_period("M"):
            return None
        self._started = True
        r = _tradable(view, view.pct_change(self.lookback, self.skip).dropna())
        top = r[r > 0].nlargest(self.top_n)  # only positive momentum; otherwise cash
        return {s: CASH_BUFFER / self.top_n for s in top.index}


class TrendFollow:
    """Hold stocks in an uptrend (price > 200-day average, 50-day > 200-day),
    strongest 6-month performers first, weekly."""

    name = "trend_sma_50_200"

    def __init__(self, top_n: int = 10, fast: int = 50, slow: int = 200, rank_lookback: int = 126) -> None:
        self.top_n, self.fast, self.slow, self.rank_lookback = top_n, fast, slow, rank_lookback
        self.params = {"top_n": top_n, "fast": fast, "slow": slow,
                       "rank_lookback": rank_lookback, "rebalance": "weekly"}
        self._started = False

    def target_weights(self, view: DataView) -> dict[str, float] | None:
        if self._started and not view.is_new_period("W"):
            return None
        self._started = True
        close, fast, slow = view.current("close"), view.sma(self.fast), view.sma(self.slow)
        uptrend = (close > slow) & (fast > slow)
        scores = _tradable(view, view.pct_change(self.rank_lookback)[uptrend].dropna())
        top = scores.nlargest(self.top_n)
        return {s: CASH_BUFFER / self.top_n for s in top.index}
