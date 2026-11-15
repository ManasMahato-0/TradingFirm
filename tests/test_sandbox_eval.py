from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from firm.config import IST
from firm.live.sandbox_eval import BOOKS, evaluate
from firm.storage import Storage


def fill_db(path, daily_pnl, n_fills_a=70, quote_gap=0.0):
    """Four accounts over len(daily_pnl) days; D earns cash, A earns D + daily_pnl, C loses a little."""
    st = Storage(path)
    days = [d.to_pydatetime().replace(hour=15, minute=30, tzinfo=IST)
            for d in pd.bdate_range("2027-01-04", periods=len(daily_pnl))]
    eqs = {b: 100_000.0 for b in BOOKS}
    for d, pnl in zip(days, daily_pnl):
        cash = 100_000 * 0.065 / 252
        eqs["sandbox_D"] += cash
        eqs["sandbox_A"] += cash + pnl
        eqs["sandbox_B"] += cash
        eqs["sandbox_C"] += cash - 15
        for b in BOOKS:
            st.record_equity(d, b, eqs[b], eqs[b])
    from firm.broker import Fill
    from firm.costs import Segment, Side
    for k in range(n_fills_a):
        side = Side.BUY if k % 2 == 0 else Side.SELL
        ts = datetime(2027, 1, 5, 10, 0, tzinfo=IST).replace(minute=(k * 5) % 60, hour=10 + k // 12 % 5)
        st.record_fill(Fill(ts, "sandbox_A", "S0", Segment.EQ_INTRADAY, side, 10, 100.0, 1.0, 0.0, "t"))
        st.record_quotes([("sandbox_A", (ts - np.timedelta64(5, "m")).isoformat(), "S0", side.value,
                           100.0 + (quote_gap if side is Side.BUY else -quote_gap))])
    return st


def test_account_that_earns_steady_excess_works(tmp_path):
    rng = np.random.default_rng(0)
    fill_db(tmp_path / "f.db", 60 + rng.normal(0, 20, 60))
    r = evaluate(tmp_path / "f.db")
    a = r["accounts"]["sandbox_A"]
    assert r["days"] == 60 and r["checkpoint_reached"] == 60
    assert a["works"] and all(a["checks"].values())
    assert not r["accounts"]["sandbox_C"]["works"] and not r["accounts"]["sandbox_B"]["works"]


def test_too_few_trades_or_one_good_half_fails(tmp_path):
    rng = np.random.default_rng(1)
    fill_db(tmp_path / "f.db", 60 + rng.normal(0, 20, 60), n_fills_a=20)
    assert not evaluate(tmp_path / "f.db")["accounts"]["sandbox_A"]["checks"]["at_least_30_round_trips"]
    pnl = np.r_[np.full(30, 150.0), np.full(30, -40.0)] + np.random.default_rng(2).normal(0, 20, 60)
    fill_db(tmp_path / "g.db", pnl)
    a = evaluate(tmp_path / "g.db")["accounts"]["sandbox_A"]
    assert not a["checks"]["both_halves_positive"] and not a["works"]


def test_latency_rule_can_remove_an_edge(tmp_path):
    rng = np.random.default_rng(3)
    # The quotes are 0.5% worse than the ledger prices: a thin edge disappears once re-priced.
    fill_db(tmp_path / "f.db", 4 + rng.normal(0, 10, 60), quote_gap=0.5)
    a = evaluate(tmp_path / "f.db")["accounts"]["sandbox_A"]
    assert a["checks"]["beats_D"]                                   # it wins on ledger prices ...
    assert a["latency_adjustment"] < 0 and not a["checks"]["beats_D_after_latency"] and not a["works"]  # ... not re-priced


def test_no_data_yet(tmp_path):
    Storage(tmp_path / "f.db")
    assert evaluate(tmp_path / "f.db")["days"] == 0


def test_missing_decision_quotes_make_the_latency_check_unverifiable(tmp_path):
    rng = np.random.default_rng(0)
    st = fill_db(tmp_path / "f.db", 60 + rng.normal(0, 20, 60))
    st.conn.execute("DELETE FROM sandbox_quotes WHERE rowid % 3 = 0")      # a third of the quotes are gone
    st.conn.commit()
    a = evaluate(tmp_path / "f.db")["accounts"]["sandbox_A"]
    assert a["quote_coverage"] < 0.9 and not a["checks"]["decision_quotes_logged"] and not a["works"]
