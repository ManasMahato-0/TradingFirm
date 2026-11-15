"""The Head (shadow): proposals follow the pre-registered rules, text carries only real numbers, and it cannot move money."""
from __future__ import annotations

import ast
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

from firm.agents import head
from firm.agents.head import TeamView, allocate, sandbox_view, swing_view, write_report
from firm.storage import SCHEMA


def eq(values, start="2026-11-30"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


def test_swing_not_started_in_test_passed_failed():
    assert swing_view(eq([]), None, 0, date(2026, 11, 20)).state == "not_started"
    v = swing_view(eq([100, 101, 102]), 0.02, 0, date(2026, 12, 5))
    assert v.state == "in_test" and not v.eligible
    done = date(2027, 6, 1)
    assert swing_view(eq([100, 110, 112]), 0.08, 0, done).eligible
    assert swing_view(eq([100, 110, 112]), 0.08, 1, done).state == "failed"           # a risk block
    assert swing_view(eq([100, 110, 112]), None, 0, done).state == "failed"           # no benchmark: cannot promote
    assert swing_view(eq([100, 70, 75]), 0.0, 0, date(2027, 1, 5)).state == "failed"  # drawdown cut rule
    assert swing_view(eq([100, 95, 96]), 0.20, 0, date(2027, 1, 5)).state == "failed" # trails benchmark by > 10pts


def test_sandbox_states():
    assert sandbox_view({"days": 0, "accounts": {}}).state == "not_started"
    acc = lambda works: {b: {"equity": 100_000.0, "return": 0.0, "works": works.get(b, False)}
                         for b in ("sandbox_A", "sandbox_B", "sandbox_C", "sandbox_D")}
    mid = sandbox_view({"days": 30, "next_checkpoint": 60, "checkpoint_reached": None, "accounts": acc({})})
    assert mid.state == "in_test" and "day 60" in mid.detail
    ok = sandbox_view({"days": 61, "next_checkpoint": 120, "checkpoint_reached": 60, "accounts": acc({"sandbox_A": True})})
    assert ok.eligible and "A" in ok.detail
    no = sandbox_view({"days": 121, "next_checkpoint": None, "checkpoint_reached": 120, "accounts": acc({})})
    assert no.state == "failed"


def test_allocation_rules():
    t = lambda name, el: TeamView(name, name, "passed" if el else "in_test", el, "")
    assert allocate([t("a", False), t("b", False)])["action"] == "hold"
    one = allocate([t("a", True), t("b", False)])
    assert one["weights"] == {"a": 0.5} and abs(one["cash"] - 0.5) < 1e-9          # capped at 50%
    two = allocate([t("a", True), t("b", True)])
    assert two["weights"] == {"a": 0.5, "b": 0.5} and abs(two["cash"]) < 1e-9


def test_report_is_stored_and_says_hold_today(tmp_path):
    db = tmp_path / "firm.db"
    with sqlite3.connect(db) as c:
        c.executescript(SCHEMA)
    out = write_report(db, date(2026, 11, 16), ew_ret=None, halted=True)
    assert out["plan"]["action"] == "hold"
    text = head.latest(db)[1]
    assert "HOLD" in text and "on hold by the owner" in text and "Overnight options: parked" in text
    write_report(db, date(2026, 11, 16))                                              # same day: replaced, not duplicated
    with sqlite3.connect(db) as c:
        assert c.execute("SELECT COUNT(*) FROM head_reports").fetchone()[0] == 1


def test_head_cannot_move_money():
    tree = ast.parse(Path(head.__file__).read_text())
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    forbidden = ("firm.broker", "firm.risk", "firm.live.runner", "firm.backtest.engine", "firm.control")
    assert not [m for m in imported if m.startswith(forbidden)], imported
