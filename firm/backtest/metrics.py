"""Performance metrics, with plain-English descriptions for the dashboard."""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from firm.broker import Fill

PERIODS_PER_YEAR = 252
RISK_FREE = 0.065  # roughly what a fixed deposit pays in India

DESCRIPTIONS = {
    "total_return": "How much the money grew over the whole period.",
    "cagr": "Average yearly growth rate (compounded).",
    "volatility": "How much returns swing around, per year. Lower is calmer.",
    "sharpe": "Return above a fixed deposit, per unit of risk. Above 1 is good, above 2 is excellent.",
    "sortino": "Like Sharpe but only counts downside swings as risk.",
    "max_drawdown": "Worst fall from a peak. -30% means at one point you were down 30% from the high.",
    "calmar": "Yearly growth divided by worst drawdown. Higher is better.",
    "trades": "Number of positions closed (fully or partly).",
    "win_rate": "Share of closed trades that made money.",
    "avg_win": "Average profit on a winning trade (Rs).",
    "avg_loss": "Average loss on a losing trade (Rs).",
    "profit_factor": "Total profits divided by total losses. Above 1 means it makes money.",
    "costs_paid": "Brokerage, taxes and fees paid (Rs).",
    "annual_turnover": "How many times per year the whole capital gets traded.",
    "exposure": "Share of days with money invested.",
    "benchmark_cagr": "Yearly growth of the benchmark over the same period.",
    "excess_cagr": "Strategy yearly growth minus benchmark. Positive means it beat the benchmark.",
    "deflated_sharpe": "Probability the Sharpe is real skill and not luck, after counting every strategy we tried. Above 0.95 is convincing.",
}


def daily(series: pd.Series) -> pd.Series:
    """Last value per calendar day (intraday equity -> daily)."""
    s = series.groupby(series.index.date).last()
    s.index = pd.to_datetime(s.index)
    return s


def daily_returns(equity: pd.Series, initial_capital: float) -> pd.Series:
    e = daily(equity)
    prev = e.shift(1)
    prev.iloc[0] = initial_capital
    return e / prev - 1


def deflated_sharpe(sr: float, n_obs: int, skew: float, kurt: float,
                    n_trials: int = 1, trial_sr_var: float = 0.0) -> float:
    """Bailey & Lopez de Prado deflated Sharpe ratio. `sr` is per-period (not annualised),
    `kurt` is ordinary (not excess) kurtosis."""
    if n_obs < 3 or not math.isfinite(sr):
        return float("nan")
    nd = NormalDist()
    sr0 = 0.0
    if n_trials > 1 and trial_sr_var > 0:
        g = 0.5772156649  # Euler-Mascheroni
        sr0 = math.sqrt(trial_sr_var) * (
            (1 - g) * nd.inv_cdf(1 - 1 / n_trials) + g * nd.inv_cdf(1 - 1 / (n_trials * math.e))
        )
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if denom <= 0:
        return float("nan")
    return nd.cdf((sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(denom))


def _turnover(fills: list[Fill], years: float, single_run_equity: pd.Series,
              equity_by_run: list[pd.Series] | None) -> float:
    """Value traded per year as a multiple of the capital it was traded from."""
    if years <= 0 or not fills:
        return 0.0 if years > 0 else float("nan")
    runs = equity_by_run or [single_run_equity]
    total = 0.0
    for eq in runs:
        if eq.empty:
            continue
        lo, hi = eq.index[0].date(), eq.index[-1].date()
        traded = sum(f.price * f.qty for f in fills if lo <= f.ts.date() <= hi)
        total += traded / float(eq.mean())
    return total / years


def compute_metrics(rets: pd.Series, fills: list[Fill] | None = None,
                    exposure: pd.Series | None = None, benchmark_rets: pd.Series | None = None,
                    n_trials: int = 1, trial_sr_var: float = 0.0,
                    initial_capital: float = 100_000.0,
                    equity_by_run: list[pd.Series] | None = None) -> dict:
    """`equity_by_run`: rupee equity of each separate run (e.g. walk-forward folds,
    each starting from fresh capital). Used for turnover; defaults to one run
    starting at initial_capital."""
    rets = rets.dropna()
    fills = fills or []
    n = len(rets)
    if n == 0:
        return {}
    curve = (1 + rets).cumprod()
    years = n / PERIODS_PER_YEAR
    total = float(curve.iloc[-1] - 1)
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else float("nan")
    std = float(rets.std())
    excess = rets - RISK_FREE / PERIODS_PER_YEAR
    sr_period = float(excess.mean() / std) if std > 0 else float("nan")
    downside = math.sqrt(float((np.minimum(rets, 0) ** 2).mean()))
    dd = float((curve / curve.cummax() - 1).min())

    closed = [f.realized_pnl for f in fills if f.realized_pnl != 0]
    wins = [p for p in closed if p > 0]
    losses = [p for p in closed if p < 0]

    m = {
        "total_return": total,
        "cagr": cagr,
        "volatility": std * math.sqrt(PERIODS_PER_YEAR),
        "sharpe": sr_period * math.sqrt(PERIODS_PER_YEAR),
        "sortino": float(excess.mean() / downside * math.sqrt(PERIODS_PER_YEAR)) if downside > 0 else float("nan"),
        "max_drawdown": dd,
        "calmar": cagr / abs(dd) if dd < 0 else float("nan"),
        "trades": len(closed),
        "win_rate": len(wins) / len(closed) if closed else float("nan"),
        "avg_win": float(np.mean(wins)) if wins else float("nan"),
        "avg_loss": float(np.mean(losses)) if losses else float("nan"),
        "profit_factor": sum(wins) / abs(sum(losses)) if losses else float("nan"),
        "costs_paid": sum(f.costs for f in fills),
        "annual_turnover": _turnover(fills, years, initial_capital * curve, equity_by_run),
        "exposure": float((exposure > 0.01).mean()) if exposure is not None and len(exposure) else float("nan"),
        "sharpe_per_period": sr_period,
        "n_days": n,
        "deflated_sharpe": deflated_sharpe(
            sr_period, n, float(rets.skew()), float(rets.kurt()) + 3, n_trials, trial_sr_var),
    }
    if benchmark_rets is not None:
        b = benchmark_rets.reindex(rets.index).fillna(0.0)
        b_total = float((1 + b).prod() - 1)
        b_cagr = (1 + b_total) ** (1 / years) - 1 if years > 0 and b_total > -1 else float("nan")
        m["benchmark_cagr"] = b_cagr
        m["excess_cagr"] = cagr - b_cagr
    return m
