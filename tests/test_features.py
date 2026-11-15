import numpy as np
import pandas as pd

from firm.backtest.data import MarketData
from firm.features.swing import (
    build_weekly_dataset, decision_rows, forward_returns, market_features, stock_features,
)


def random_data(n=400, k=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n, tz="Asia/Kolkata", name="ts")
    frames = {}
    for j in range(k):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
        o = c * (1 + rng.normal(0, 0.003, n))
        frames[f"S{j}"] = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.01,
                                        "low": np.minimum(o, c) * 0.99, "close": c,
                                        "volume": rng.integers(1000, 5000, n)}, index=idx)
    nifty = pd.Series(10_000 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    return MarketData.from_frames(frames), nifty


def test_features_are_causal():
    """Values at row t must not change when data after t is removed."""
    data, nifty = random_data()
    sectors = {"S0": "A", "S1": "A", "S2": "B"}
    full = stock_features(data.panels["close"], data.panels["volume"], sectors)
    mkt_full = market_features(data.index, data.panels["close"], nifty)
    cut = 300
    short = MarketData({f: p.iloc[:cut] for f, p in data.panels.items()})
    part = stock_features(short.panels["close"], short.panels["volume"], sectors)
    mkt_part = market_features(short.index, short.panels["close"], nifty.iloc[:cut])
    for k in full:
        pd.testing.assert_frame_equal(full[k].iloc[:cut], part[k])
    pd.testing.assert_frame_equal(mkt_full.iloc[:cut], mkt_part)


def test_results_features_causal_and_announcement_gated():
    from datetime import date

    from firm.features.results import results_features

    data, _ = random_data(n=120, k=2)
    days = list(data.index.date)
    meetings = pd.DataFrame({
        "symbol": ["S0", "S0"],
        "meeting_date": [days[40], days[90]],
        "intimated_date": [days[30], days[80]],
    })
    full = results_features(data.index, data.panels["close"], meetings)
    to_res = full["days_to_results"]["S0"]
    assert to_res.iloc[29] == 0.5            # not announced yet: "far away"
    assert to_res.iloc[30] < 0.5             # announced: countdown visible
    assert to_res.iloc[40] == -0.5           # meeting day
    since = full["days_since_results"]["S0"]
    assert since.iloc[39] == 0.5 and since.iloc[41] < 0.5
    cut = 60
    part = results_features(data.index[:cut], data.panels["close"].iloc[:cut], meetings)
    for k in full:
        pd.testing.assert_frame_equal(full[k].iloc[:cut], part[k])


def test_decision_rows_are_week_ends():
    idx = pd.bdate_range("2026-11-02", periods=10, tz="Asia/Kolkata")  # Mon..Fri x2
    rows = decision_rows(idx)
    assert [idx[r].day_name() for r in rows] == ["Friday", "Friday"]


def test_monthly_decision_rows_are_month_ends():
    idx = pd.bdate_range("2026-09-01", "2026-11-30", tz="Asia/Kolkata")
    rows = decision_rows(idx, "M")
    assert [str(idx[r].date()) for r in rows] == ["2026-09-30", "2026-10-30", "2026-11-30"]


def test_forward_returns_open_to_open_and_no_leak():
    data, _ = random_data(n=30, k=1)
    rows = decision_rows(data.index)
    fwd = forward_returns(data.panels["open"], rows)
    o = data.panels["open"]["S0"].to_numpy()
    assert fwd.iloc[0, 0] == o[rows[1] + 1] / o[rows[0] + 1] - 1
    assert np.isnan(fwd.iloc[-1, 0])  # exit beyond the data: unknown


def test_weekly_dataset_shapes_and_excess_is_relative():
    data, nifty = random_data()
    ds = build_weekly_dataset(data, {}, nifty)
    w, s = ds.valid.shape
    assert ds.X.shape[:2] == (w, s) and ds.fwd.shape == (w, s)
    rows_ok = ds.valid.sum(axis=1) == s
    if rows_ok.any():
        assert np.allclose(ds.excess[rows_ok].mean(axis=1), 0, atol=1e-9)
