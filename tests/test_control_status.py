from datetime import datetime

from firm.config import IST
from firm.control import Control
from firm.control_status import attention_items, last_job_runs

TUE_10AM = datetime(2026, 11, 17, 10, 0, tzinfo=IST)
TUE_NIGHT = datetime(2026, 11, 17, 22, 0, tzinfo=IST)
SAT = datetime(2026, 11, 21, 10, 0, tzinfo=IST)
OK = {"recorder": True, "sandbox": True}


def keys(items):
    return [i.key for i in items]


def test_all_normal_means_nothing_to_do(tmp_path):
    assert attention_items(Control(tmp_path / "c.db"), OK, True, True, TUE_10AM) == []


def test_each_problem_has_a_plain_message_and_a_one_click_fix(tmp_path):
    ctl = Control(tmp_path / "c.db")
    ctl.set("halt.firm", True)
    items = attention_items(ctl, {"recorder": False, "sandbox": True}, False, False, TUE_10AM)
    assert keys(items) == ["resume_halt", "start_recorder", "start_ollama", "login"]
    assert all(i.text and i.fix for i in items)


def test_a_service_the_owner_switched_off_says_so(tmp_path):
    ctl = Control(tmp_path / "c.db")
    ctl.set("service.sandbox.run", "stop")
    (item,) = attention_items(ctl, {"recorder": True, "sandbox": False}, True, True, TUE_10AM)
    assert "switched off" in item.text and item.key == "start_sandbox"


def test_fyers_login_is_only_nagged_during_trading_hours(tmp_path):
    ctl = Control(tmp_path / "c.db")
    assert "login" in keys(attention_items(ctl, OK, True, False, TUE_10AM))
    assert "login" not in keys(attention_items(ctl, OK, True, False, TUE_NIGHT))      # an expired token at night is normal
    assert "login" not in keys(attention_items(ctl, OK, True, False, SAT))
    assert "login" not in keys(attention_items(ctl, OK, True, None, TUE_10AM))        # not checked yet


def test_ai_model_offline_is_not_a_problem_if_the_owner_turned_it_off(tmp_path):
    ctl = Control(tmp_path / "c.db")
    ctl.set("llm.enabled", False)
    assert "start_ollama" not in keys(attention_items(ctl, OK, False, True, TUE_10AM))


def test_paused_books_and_switched_off_jobs_are_shown_as_information(tmp_path):
    ctl = Control(tmp_path / "c.db")
    ctl.set("book.sandbox_A.paused", True)
    ctl.set("job.news.enabled", False)
    items = attention_items(ctl, OK, True, True, TUE_10AM)
    assert keys(items) == ["show_books", "show_jobs"] and all(i.level == "info" for i in items)
    assert "Practice A" in items[0].text and "Headlines" in items[1].text


def test_last_job_runs_reads_the_recorder_log(tmp_path):
    from firm.recorders.db import RecorderDB
    db = RecorderDB(tmp_path / "r.db")
    db.log_run("news", "2026-11-17T10:00:00+05:30", "2026-11-17T10:00:05+05:30", "ok", 3, "")
    db.log_run("news", "2026-11-17T10:05:00+05:30", "2026-11-17T10:05:05+05:30", "partial", 1, "")
    assert last_job_runs(tmp_path / "r.db") == {"news": ("2026-11-17T10:05:00+05:30", "partial")}
    assert last_job_runs(tmp_path / "missing.db") == {}


def test_a_job_that_failed_for_lack_of_a_login_points_to_the_login_on_a_trading_day(tmp_path):
    ctl = Control(tmp_path / "c.db")
    runs = {"bars": ("2026-11-17T09:00:00+05:30", "auth_failed")}
    items = attention_items(ctl, OK, True, None, TUE_10AM, runs)
    assert keys(items) == ["login"] and "price bars" in items[0].text
    assert "login" not in keys(attention_items(ctl, OK, True, None, TUE_NIGHT, runs))   # nothing to do at 10pm
    assert "login" not in keys(attention_items(ctl, OK, True, None, SAT, runs))          # nor over a weekend
    ctl.set("job.bars.enabled", False)                                                   # switched off on purpose
    assert "login" not in keys(attention_items(ctl, OK, True, None, TUE_10AM, runs))
    old = {"bars": ("2026-11-10T21:00:00+05:30", "auth_failed")}                         # a week old: not current
    ctl.set("job.bars.enabled", True)
    assert attention_items(ctl, OK, True, None, TUE_10AM, old) == []


def test_a_failed_run_is_information_not_an_alarm(tmp_path):
    ctl = Control(tmp_path / "c.db")
    runs = {"announcements": ("2026-11-17T09:50:00+05:30", "error"), "news": ("2026-11-17T09:55:00+05:30", "ok")}
    (item,) = attention_items(ctl, OK, True, True, TUE_10AM, runs)
    assert item.level == "info" and "Company filings" in item.text and "News" not in item.text


def test_every_notice_has_a_unique_key_so_the_page_never_draws_two_buttons_with_one_key(tmp_path):
    ctl = Control(tmp_path / "c.db")
    ctl.set("halt.firm", True)
    ctl.set("book.sandbox_A.paused", True)
    ctl.set("job.news.enabled", False)
    runs = {"announcements": ("2026-11-17T09:50:00+05:30", "error"), "bars": ("2026-11-17T09:00:00+05:30", "auth_failed")}
    items = attention_items(ctl, {"recorder": False, "sandbox": False}, False, False, TUE_10AM, runs)
    ks = keys(items)
    assert len(ks) == len(set(ks)) and len(ks) >= 7
