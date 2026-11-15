"""The research lab: runs backtests and walk-forward tests, guards the locked
test period, and logs every trial."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable

import pandas as pd

from firm.backtest.data import MarketData
from firm.backtest.engine import BacktestConfig, BacktestResult, Strategy, run_backtest
from firm.backtest.metrics import compute_metrics, daily_returns
from firm.backtest.registry import TrialRegistry, strategy_version
from firm.config import DATA_DIR
from firm.risk import RiskLimits

# Research may only use data before these dates. Fixed, so the lock never moves.
LOCK_START = {
    "swing": date(2026, 5, 14),     # last 6 months locked
    "intraday": date(2026, 8, 14),  # last 3 months locked
}
RESEARCH_DB = DATA_DIR / "research.db"


class LockedPeriodError(RuntimeError):
    pass


@dataclass
class WalkForwardResult:
    strategy: str
    params: dict
    folds: list[tuple[date, date, date, date]]
    fold_results: list[BacktestResult]
    returns: pd.Series
    metrics: dict


def make_folds(start: date, end: date, train_years: int = 3, test_months: int = 6,
               train_months: int | None = None) -> list[tuple[date, date, date, date]]:
    """(train_start, train_end, test_start, test_end) windows rolling forward.
    train_months, if given, replaces train_years (intraday has only ~1 year of data)."""
    folds = []
    train = pd.DateOffset(months=train_months) if train_months else pd.DateOffset(years=train_years)
    test_start = (pd.Timestamp(start) + train).date()
    while test_start < end:
        test_end = min((pd.Timestamp(test_start) + pd.DateOffset(months=test_months)).date(), end)
        train_start = (pd.Timestamp(test_start) - train).date()
        folds.append((train_start, test_start, test_start, test_end))
        test_start = test_end
    return folds


class Lab:
    def __init__(self, data: MarketData, track: str, limits: RiskLimits,
                 config: BacktestConfig | None = None, benchmark: pd.Series | None = None,
                 registry: TrialRegistry | None = None, lock_start: date | None = None,
                 broker=None) -> None:
        self.data = data
        self.broker = broker  # None = default paper broker with real costs
        self.track = track
        self.limits = limits
        self.config = config or BacktestConfig()
        self.registry = registry or TrialRegistry(RESEARCH_DB)
        self.lock_start = lock_start or LOCK_START[track]
        self.bench_rets = None
        if benchmark is not None:
            b = benchmark.groupby(benchmark.index.date).last()
            b.index = pd.to_datetime(b.index)
            self.bench_rets = b.pct_change()

    def _metrics(self, rets: pd.Series, fills, exposure, version: str,
                 equity_by_run: list[pd.Series]) -> dict:
        n, var = self.registry.trial_stats(self.track)
        return compute_metrics(rets, fills, exposure, self.bench_rets, n_trials=max(n, 1),
                               trial_sr_var=var, initial_capital=self.config.initial_capital,
                               equity_by_run=equity_by_run)

    def _check_lock(self, end: date) -> None:
        if end > self.lock_start:
            raise LockedPeriodError(
                f"research must end by {self.lock_start}; the period after it is locked for final exams")

    def backtest(self, strategy: Strategy, start: date, end: date | None = None) -> BacktestResult:
        end = end or self.lock_start
        self._check_lock(end)
        res = run_backtest(strategy, self.data, start, end, self.limits, self.config)
        version = strategy_version(strategy)
        res.metrics = self._metrics(daily_returns(res.equity, res.initial_capital),
                                    res.fills, res.exposure, version, [res.equity])
        self.registry.log(self.track, strategy.name, version, strategy.params, "research",
                          start, end, res.metrics)
        return res

    def walk_forward(self, factory: Callable[[], Strategy], start: date, end: date | None = None,
                     train_years: int = 3, test_months: int = 6,
                     train_months: int | None = None) -> WalkForwardResult:
        end = end or self.lock_start
        self._check_lock(end)
        folds = make_folds(start, end, train_years, test_months, train_months)
        results, rets, exposures, fills = [], [], [], []
        strategy = None
        for train_start, train_end, test_start, test_end in folds:
            strategy = factory()
            if hasattr(strategy, "fit"):
                strategy.fit(self.data.truncate(train_end), train_start, train_end)
            res = run_backtest(strategy, self.data, test_start, test_end, self.limits, self.config,
                               self.broker)
            if res.equity.empty:
                continue
            results.append(res)
            rets.append(daily_returns(res.equity, res.initial_capital))
            exposures.append(res.exposure)
            fills.extend(res.fills)
        if strategy is None or not rets:
            raise ValueError("no walk-forward folds with data")
        all_rets = pd.concat(rets)
        version = strategy_version(strategy)
        metrics = self._metrics(all_rets, fills, pd.concat(exposures), version,
                                [r.equity for r in results])
        metrics["folds"] = len(results)
        self.registry.log(self.track, strategy.name, version, strategy.params, "walkforward",
                          start, end, metrics)
        return WalkForwardResult(strategy.name, dict(strategy.params), folds, results, all_rets, metrics)

    def final_exam(self, strategy: Strategy) -> BacktestResult:
        """One shot on the locked period per strategy version."""
        version = strategy_version(strategy)
        if self.registry.has_final_exam(version):
            raise LockedPeriodError(f"{strategy.name} version {version} already took its final exam")
        end = self.data.index[-1].date() + timedelta(days=1)
        res = run_backtest(strategy, self.data, self.lock_start, end, self.limits, self.config)
        res.metrics = self._metrics(daily_returns(res.equity, res.initial_capital),
                                    res.fills, res.exposure, version, [res.equity])
        self.registry.log(self.track, strategy.name, version, strategy.params, "final_exam",
                          self.lock_start, end, res.metrics)
        return res
