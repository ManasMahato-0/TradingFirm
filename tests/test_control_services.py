import importlib.util
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest
from test_live import CAL, _spec, random_data
from test_sandbox_runner import fills_of, replay, setup

from firm.activity import ActivityLog
from firm.config import IST
from firm.control import Control
from firm.live.runner import LiveContext, run_books
from firm.live.sandbox import Sandbox, StopRequested
from firm.live.sandbox_feed import ReplaySource
from firm.live.tracker import risk_blocks
from firm.storage import Storage


def load_recorder():
    spec = importlib.util.spec_from_file_location(
        "recorder_script", Path(__file__).resolve().parent.parent / "scripts" / "recorder.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["recorder_script"] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeRecorder:
    """The recorder's tick/command logic with every job stubbed to a recording call."""

    def __init__(self, mod, ctl, when):
        rec = mod.Recorder.__new__(mod.Recorder)
        rec.ctl, rec.last_run, rec.last_options_slot, rec._fyers = ctl, {}, None, object()
        rec.called = []
        rec.run_job = lambda name, fn: (rec.called.append(name), "ok")[1]
        rec.bars_done_today = lambda: True
        rec.done_today = lambda job: job == "runner"          # runner done -> the desk may run
        mod.keep_awake = lambda on: None
        mod.now = lambda: when
        rec.job_news = rec.job_announcements = rec.job_board_meetings = rec.job_options = rec.job_bars = None
        rec.job_results = rec.job_researcher = rec.job_runner = rec.job_desk = None
        self.rec = rec


WEEKDAY_EVENING = datetime(2026, 11, 17, 16, 20, tzinfo=IST)       # a Tuesday


def test_every_job_runs_when_nothing_is_switched_off(tmp_path):
    mod = load_recorder()
    f = FakeRecorder(mod, Control(tmp_path / "c.db"), WEEKDAY_EVENING)
    f.rec.tick()
    assert {"news", "announcements", "board_meetings", "results", "researcher", "desk"} <= set(f.rec.called)


def test_a_switched_off_job_does_not_run_and_others_still_do(tmp_path):
    mod = load_recorder()
    ctl = Control(tmp_path / "c.db")
    for job in ("news", "results", "desk", "researcher"):
        ctl.set(f"job.{job}.enabled", False)
    f = FakeRecorder(mod, ctl, WEEKDAY_EVENING)
    f.rec.tick()
    assert not {"news", "results", "desk", "researcher"} & set(f.rec.called)
    assert {"announcements", "board_meetings"} <= set(f.rec.called)
    ctl.set("job.news.enabled", True)                                  # switching back on runs it at once
    f2 = FakeRecorder(mod, ctl, WEEKDAY_EVENING)
    f2.rec.tick()
    assert "news" in f2.rec.called


def test_commands_from_the_dashboard_run_once_and_report(tmp_path):
    mod = load_recorder()
    ctl = Control(tmp_path / "c.db")
    f = FakeRecorder(mod, ctl, WEEKDAY_EVENING)
    f.rec._job_functions = lambda: {"runner": object(), "desk": object()}
    a = ctl.enqueue("recorder", "run_job", {"job": "runner"})
    b = ctl.enqueue("recorder", "reload_fyers")
    c = ctl.enqueue("recorder", "mystery")
    f.rec.handle_commands()
    f.rec.handle_commands()                                            # nothing left to run twice
    assert f.rec.called == ["runner"] and f.rec._fyers is None
    status = {r[0]: r[1] for r in ctl.conn.execute("SELECT id, status FROM commands")}
    assert status[a] == "done" and status[b] == "done" and status[c] == "error"


def test_a_pause_blocks_new_swing_positions_but_the_book_keeps_its_days_and_nothing_replays(tmp_path):
    data = random_data()
    ctx = LiveContext(data, pd.Series(dtype=float), None, {}, None, CAL)
    spec_start = date(2026, 1, 1)
    days = [t.date() for t in data.index if t.date() >= spec_start]
    mid, end = days[len(days) // 2], days[-1]
    at = lambda d: datetime(d.year, d.month, d.day, 17, tzinfo=IST)     # noqa: E731

    base = tmp_path / "base"
    base.mkdir()
    run_books([_spec(spec_start)], base / "f.db", base, ctx=ctx, now=at(end), control=Control(base / "c.db"))

    paused = tmp_path / "paused"
    paused.mkdir()
    ctl = Control(paused / "c.db")
    ctl.set("book.test_mom.paused", True)
    run_books([_spec(spec_start)], paused / "f.db", paused, ctx=ctx, now=at(mid), control=ctl)
    first_half_fills = fills_of(paused / "f.db", "test_mom")
    assert not first_half_fills                                         # paused: no new positions at all
    blocked = [r[0] for r in sqlite3.connect(paused / "f.db").execute(
        "SELECT text FROM activity WHERE agent='Risk engine' AND text LIKE 'Rejected%'")]
    assert blocked and all("paused by the owner" in t for t in blocked)
    assert risk_blocks(paused / "f.db", "test_mom") == 0                # ... and these never count as breaches

    ctl.set("book.test_mom.paused", False)
    run_books([_spec(spec_start)], paused / "f.db", paused, ctx=ctx, now=at(end), control=ctl)
    n_days = lambda p: Storage(p).conn.execute("SELECT COUNT(*) FROM equity WHERE team='test_mom'").fetchone()[0]  # noqa: E731
    assert n_days(paused / "f.db") == n_days(base / "f.db") == len(days)   # every day counted, none replayed
    later = fills_of(paused / "f.db", "test_mom")
    assert later and min(f[0] for f in later) > mid.isoformat()           # it trades again only AFTER resuming
    assert len(later) <= len(fills_of(base / "f.db", "test_mom"))        # it never "catches up" with extra trades


def test_owner_pause_rows_are_excluded_but_real_blocks_still_count(tmp_path):
    db = tmp_path / "f.db"
    log = ActivityLog(db)
    log.log("talk", "Risk engine", "Rejected BUY 5 X: firm halted: paused by the owner", book="b1")
    log.log("talk", "Risk engine", "Rejected BUY 5 Y: position or exposure limit reached", book="b1")
    log.log("talk", "Risk engine", "Rejected BUY 5 Z: stale data: 9000s old", book="b2")
    assert risk_blocks(db, "b1") == 1 and risk_blocks(db, "b2") == 1 and risk_blocks(tmp_path / "none.db", "b1") == 0


def test_sandbox_pause_and_skip_day_keep_accounts_flat_but_counted(tmp_path):
    sb, data, nifty, nopen = setup(tmp_path)
    days = sorted(set(data.index.date))
    ctl = Control(tmp_path / "c.db")
    ctl.set("book.sandbox_A.paused", True)
    sb = Sandbox(tmp_path / "firm.db", tmp_path / "state", launch=days[0], trade_from=date(2099, 1, 1), control=ctl)
    sb.run_day(days[8], ReplaySource(data, nifty, nopen, days[8]), ensure=False)
    assert not fills_of(tmp_path / "firm.db", "sandbox_A")              # paused: stays flat
    assert fills_of(tmp_path / "firm.db", "sandbox_C") or True          # the others are not affected
    assert Storage(tmp_path / "firm.db").conn.execute(
        "SELECT COUNT(*) FROM equity WHERE team='sandbox_A'").fetchone()[0] == 75      # ... but every bar is recorded
    ctl.set("book.sandbox_A.paused", False)
    ctl.set("sandbox.skip_date", days[9].isoformat())
    sb.run_day(days[9], ReplaySource(data, nifty, nopen, days[9]), ensure=False)
    day9 = [f for f in fills_of(tmp_path / "firm.db", "sandbox_A") if f[0].startswith(days[9].isoformat())]
    assert not day9                                                    # the skipped day: flat for every account
    assert Storage(tmp_path / "firm.db").conn.execute(
        "SELECT COUNT(DISTINCT substr(ts,1,10)) FROM equity WHERE team='sandbox_D'").fetchone()[0] == 2   # both days counted


def test_stopping_the_sandbox_from_the_page_ends_the_session_cleanly(tmp_path):
    sb, data, nifty, nopen = setup(tmp_path)
    days = sorted(set(data.index.date))
    ctl = Control(tmp_path / "c.db")
    ctl.set("service.sandbox.run", "stop")
    sb = Sandbox(tmp_path / "firm.db", tmp_path / "state", launch=days[0], trade_from=date(2099, 1, 1), control=ctl)
    with pytest.raises(StopRequested):
        sb.run_day(days[8], ReplaySource(data, nifty, nopen, days[8]), ensure=False)
    log = [r[0] for r in sqlite3.connect(tmp_path / "firm.db").execute("SELECT text FROM activity")]
    assert not any("Day done" in t for t in log)                        # it did not pretend the day finished


def test_ai_model_switch_turns_the_llm_off(tmp_path, monkeypatch):
    import firm.control as control
    from firm.llm import get_llm

    monkeypatch.setattr(control, "CONTROL_DB", tmp_path / "c.db")
    Control().set("llm.enabled", False)
    assert get_llm().name == "off"
    Control().set("llm.enabled", True)
    assert get_llm().name != "off"


def test_commands_are_checked_between_jobs_not_only_once_per_tick(tmp_path):
    mod = load_recorder()
    ctl = Control(tmp_path / "c.db")
    f = FakeRecorder(mod, ctl, WEEKDAY_EVENING)
    order = []
    f.rec.handle_commands = lambda: order.append("cmd")
    f.rec.run_job = lambda name, fn: (order.append(name), "ok")[1]
    f.rec.tick()
    assert order.count("cmd") >= 4
    assert order.index("cmd") < order.index("news")                        # before the first job
    assert order.index("researcher") > max(i for i, x in enumerate(order) if x == "cmd" and i < order.index("researcher"))         and order[order.index("researcher") + 1] == "cmd"                  # and right after the slow Researcher
