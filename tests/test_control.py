import threading

import pytest

from firm.control import BOOKS, CONTROL_DB, JOBS, LOCKED, OWNER_HALT_REASON, SWITCHES, Control


@pytest.fixture
def ctl(tmp_path):
    return Control(tmp_path / "control.db")


def test_empty_board_changes_nothing_every_switch_defaults_to_todays_behaviour(ctl):
    assert not ctl.get_bool("halt.firm")
    assert all(ctl.job_enabled(j) for j in JOBS)                         # every job runs
    assert not any(ctl.halted(b, "2026-11-23") for b in BOOKS)            # nothing paused
    assert ctl.get("agent.guard") == "shadow" and ctl.get_bool("llm.enabled") and ctl.get_bool("agent.manager")
    assert ctl.service_should_run("recorder") and ctl.service_should_run("sandbox")
    assert ctl.get("sandbox.skip_date") == ""


def test_set_logs_changes_once_and_returns_whether_it_changed(ctl):
    assert ctl.set("job.news.enabled", False) is True
    assert ctl.set("job.news.enabled", False) is False                    # same value: no new audit line
    assert not ctl.job_enabled("news") and ctl.job_enabled("results")
    rows = ctl.audit_tail()
    assert len(rows) == 1 and rows[0][3] == "job.news.enabled" and rows[0][4:6] == ("true", "false")


def test_halt_and_pause_meaning_and_windows(ctl):
    ctl.set("book.sandbox_A.paused", True)
    assert ctl.halted("sandbox_A") and not ctl.halted("sandbox_B")
    ctl.set("halt.firm", True)
    assert all(ctl.halted(b) for b in BOOKS)
    ctl.set("halt.firm", False)
    ctl.set("book.sandbox_A.paused", False)
    wins = ctl.windows()
    assert {w[0] for w in wins} == {"halt.firm", "book.sandbox_A.paused"} and all(w[2] for w in wins)   # both closed
    ctl.set("halt.firm", True)
    assert ctl.windows("halt.firm")[-1][2] is None                        # an open window while it is on


def test_skip_date_only_affects_sandbox_accounts_on_that_day(ctl):
    ctl.set("sandbox.skip_date", "2026-11-24")
    assert ctl.halted("sandbox_C", "2026-11-24") and not ctl.halted("sandbox_C", "2026-11-25")
    assert not ctl.halted("swing_probation", "2026-11-24")


def test_commands_are_deduplicated_and_claimed_exactly_once(ctl):
    first = ctl.enqueue("recorder", "run_job", {"job": "runner"})
    assert first is not None
    assert ctl.enqueue("recorder", "run_job", {"job": "runner"}) is None   # a double click is ignored
    assert ctl.enqueue("recorder", "run_job", {"job": "desk"}) is not None
    got = []

    def worker():
        while (c := ctl.claim("recorder")) is not None:
            got.append(c["id"])

    other = Control(ctl.path)                                              # a second connection, like a second process
    t = threading.Thread(target=lambda: [got.append(c) for c in iter(lambda: other.claim("recorder"), None)])
    t.start()
    worker()
    t.join()
    flat = [g["id"] if isinstance(g, dict) else g for g in got]
    assert sorted(flat) == sorted(set(flat)) and len(flat) == 2             # nobody ran a command twice
    assert ctl.enqueue("recorder", "run_job", {"job": "runner"}) is None   # still 'running', not finished
    ctl.finish(first, "done", "ok")
    assert ctl.enqueue("recorder", "run_job", {"job": "runner"}) is not None   # finished: a new one is allowed


def test_commands_go_only_to_their_target(ctl):
    ctl.enqueue("sandbox", "skip_today")
    assert ctl.claim("recorder") is None and ctl.claim("sandbox")["name"] == "skip_today"


def test_heartbeat_roundtrip(ctl):
    assert ctl.last_beat("recorder") is None
    ctl.beat("recorder")
    assert ctl.last_beat("recorder") is not None


def test_locked_items_are_not_switches_and_the_real_db_is_not_the_data_mode_db():
    names = " ".join(t for t, _ in LOCKED).lower()
    assert "paper mode" in names and "threshold" in names and "risk limits" in names
    assert not any("threshold" in k or "limit" in k for k in SWITCHES)       # nothing pre-registered is a switch
    assert CONTROL_DB.name == "control.db" and "live" not in str(CONTROL_DB)
    assert OWNER_HALT_REASON == "paused by the owner"
