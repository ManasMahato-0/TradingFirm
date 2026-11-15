"""Morning reminders: when they go out, that each goes once, and that they stay quiet when they should."""
from __future__ import annotations

from datetime import date, datetime

from firm.alerts import Watch, notify
from firm.config import IST

MON = date(2026, 11, 16)          # a Monday


def at(h, m, day=MON):
    return datetime(day.year, day.month, day.day, h, m, tzinfo=IST)


def make(logged_in=False, sandbox=True, trading=True, enabled=True):
    sent, probes = [], []
    state = {"in": logged_in}
    w = Watch(probe=lambda: (probes.append(1), (state["in"], "x"))[1], notifier=lambda t, m: sent.append((t, m)),
              sandbox_running=lambda: sandbox, is_trading_day=lambda d: trading, enabled=lambda: enabled)
    return w, sent, probes, state


def test_no_login_gets_three_reminders_then_stops():
    w, sent, _, _ = make()
    assert w.check(at(8, 30)) == []
    assert w.check(at(8, 45)) == ["login-0845"]
    assert w.check(at(8, 50)) == []                        # same slot: once
    assert w.check(at(9, 0)) == ["login-0900"]
    assert w.check(at(9, 10)) == ["login-0910"]
    assert w.check(at(9, 20)) == [] and w.check(at(11, 0)) == []
    assert len(sent) == 3 and "Last reminder" in sent[-1][1]


def test_once_logged_in_no_more_reminders_that_day():
    w, sent, probes, state = make()
    w.check(at(8, 45))
    state["in"] = True
    assert w.check(at(9, 0)) == []
    n = len(probes)
    w.check(at(9, 10))
    assert len(probes) == n and len(sent) == 1            # not even probing any more


def test_already_logged_in_at_first_slot_says_nothing():
    w, sent, _, _ = make(logged_in=True)
    assert w.check(at(8, 45)) == [] and sent == []


def test_quiet_on_holidays_weekends_and_when_switched_off():
    for kw in ({"trading": False}, {"enabled": False}):
        w, sent, probes, _ = make(**kw)
        assert w.check(at(8, 45)) == [] and w.check(at(9, 5)) == [] and sent == [] and probes == []


def test_a_slot_slept_through_is_skipped_not_sent_late():
    w, sent, _, _ = make()
    assert w.check(at(13, 0)) == [] and sent == []         # laptop woke up at 13:00


def test_sandbox_down_is_reported_once_in_the_morning_window():
    w, sent, _, _ = make(logged_in=True, sandbox=False)
    assert w.check(at(9, 0)) == []
    assert w.check(at(9, 6)) == ["sandbox-down"]
    assert w.check(at(9, 7)) == []
    assert w.check(at(9, 40)) == []


def test_new_day_resets():
    w, sent, _, _ = make()
    w.check(at(8, 45))
    tue = date(2026, 11, 17)
    assert w.check(at(8, 45, tue)) == ["login-0845"]


def test_notify_sends_windows_and_optional_phone_and_never_raises():
    calls, posts = [], []
    ch = notify("T", "M", run=lambda *a, **k: calls.append(k["env"]["TF_MSG"]), post=lambda *a, **k: posts.append(a[0]),
                env={"NTFY_TOPIC": "abc123"})
    assert ch == ["windows", "phone"] and calls == ["M"] and posts == ["https://ntfy.sh/abc123"]
    assert notify("T", "M", run=lambda *a, **k: None, env={}) == ["windows"]

    def boom(*a, **k):
        raise OSError("nope")
    assert notify("T", "M", run=boom, post=boom, env={"NTFY_TOPIC": "x"}) == []
