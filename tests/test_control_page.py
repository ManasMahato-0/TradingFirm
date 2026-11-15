from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import firm.control as control
import firm.fyers_auth as fyers_auth
import firm.ops as ops
from firm.control import LOCKED, Control

APP = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
SECTIONS = ["Switches", "Programs", "Fyers login", "More"]


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "CONTROL_DB", tmp_path / "c.db")
    state = {"recorder": True, "sandbox": True, "ollama": True, "started": []}
    monkeypatch.setattr(ops, "is_running", lambda n, defs=None: state[n])
    monkeypatch.setattr(ops, "ollama_running", lambda: state["ollama"])
    monkeypatch.setattr(ops, "start_service", lambda n, ctl, *a, **k: (state["started"].append(n), "started")[1])
    monkeypatch.setattr(ops, "job_running", lambda n: False)
    monkeypatch.setattr(fyers_auth, "token_status", lambda: (state.get("fyers", True), "Fyers is reachable"))
    monkeypatch.setattr(fyers_auth, "begin", lambda: ("https://fyers.example/login", "STATE1"))
    from dashboard import control_page
    monkeypatch.setattr(control_page, "RECORDER_DB_PATH", tmp_path / "no_recorder.db")     # never read the real run log
    control_page.live_state.clear()
    control_page.fyers_state.clear()
    ctl = Control(tmp_path / "c.db")

    def make(section="Services", mode="Live"):
        control_page.live_state.clear()
        control_page.fyers_state.clear()
        at = AppTest.from_file(APP, default_timeout=120)
        at.session_state["_test_page"] = "control"
        at.session_state["ctl_section"] = section
        at.session_state["mode"] = mode
        at.run()
        return at

    return make, ctl, state


def text_of(at) -> str:
    parts = [m.value for m in at.markdown] + [m.value for m in at.error] + [m.value for m in at.success] \
        + [m.value for m in at.warning] + [m.value for m in at.caption]
    return "\n".join(parts)


@pytest.mark.parametrize("mode", ["Live", "Sandbox replay"])
def test_every_section_renders_without_errors_in_every_data_mode(app, mode):
    make, _, _ = app
    for section in SECTIONS:
        at = make(section, mode)
        assert not at.exception, (section, [e.value for e in at.exception])


def test_flipping_a_switch_reaches_the_control_board_and_is_logged(app):
    make, ctl, _ = app
    at = make("Switches")
    assert ctl.job_enabled("news")
    at.toggle(key="w_job.news.enabled").set_value(False).run()
    assert not ctl.job_enabled("news") and ctl.job_enabled("announcements")      # only that one
    assert any(r[3] == "job.news.enabled" for r in ctl.audit_tail())
    assert at.toggle(key="w_job.news.enabled").value is False                    # the page shows what is saved
    at.toggle(key="w_job.news.enabled").set_value(True).run()
    assert ctl.job_enabled("news")


def test_the_control_page_writes_to_the_live_board_even_in_another_data_mode(app):
    make, ctl, _ = app
    at = make("Switches", mode="Sandbox replay")
    at.toggle(key="w_book.sandbox_A.paused").set_value(True).run()
    assert ctl.get_bool("book.sandbox_A.paused")
    assert not ctl.get_bool("book.sandbox_B.paused")


def test_halt_is_one_click_and_so_is_resume(app):
    make, ctl, _ = app
    at = make("Programs")
    at.button(key="halt").click().run()
    assert ctl.get_bool("halt.firm")
    assert "MASTER HALT IS ON" in text_of(at)
    at.button(key="resume").click().run()
    assert not ctl.get_bool("halt.firm") and "MASTER HALT IS ON" not in text_of(at)
    assert [w[2] is not None for w in ctl.windows("halt.firm")] == [True]         # the halt window was recorded and closed


def test_the_guard_can_only_be_shadow_or_off(app):
    make, ctl, _ = app
    at = make("Switches")
    options = at.selectbox[0].options
    assert len(options) == 2 and all("act" not in o.lower().split(":")[0] for o in options)
    at.selectbox[0].select(options[1]).run()
    assert ctl.get("agent.guard") == "off"


def test_a_stopped_service_shows_a_plain_message_with_a_start_button_that_works(app):
    make, ctl, state = app
    state["recorder"] = False
    at = make("Programs")
    assert "Data collector is not running" in text_of(at)
    at.button(key="fix_start_recorder").click().run()
    assert state["started"] == ["recorder"]


def test_locked_rules_are_listed_and_have_no_controls(app):
    make, _, _ = app
    at = make("More")
    body = text_of(at)
    for title, _ in LOCKED:
        assert title in body
    assert len(at.toggle) == 0 and len(at.number_input) == 0                              # nothing here can be switched


def test_fyers_panel_shows_status_and_never_the_token(app, monkeypatch):
    make, ctl, state = app
    state["fyers"] = False
    at = make("Fyers login")
    assert "login needed" in text_of(at)
    monkeypatch.setattr(fyers_auth, "finish", lambda url, st_: (False, "That address is not from this login attempt."))
    at.text_input(key="fyers_paste").set_value("http://x/?auth_code=ZZZ&state=NOPE").run()
    at.button[[b.label for b in at.button].index("Finish login")].click().run()
    assert "not from this login attempt" in text_of(at)
    monkeypatch.setattr(fyers_auth, "finish", lambda url, st_: (True, "Logged in. The token is saved (it is not shown anywhere)."))
    at.text_input(key="fyers_paste").set_value("http://x/?auth_code=GOOD&state=STATE1").run()
    at.button[[b.label for b in at.button].index("Finish login")].click().run()
    body = text_of(at)
    assert "Logged in" in body and "GOOD" not in body                                # the one-time code is never echoed
    assert any(r[2] == "reload_fyers" for r in ctl.recent_commands())                # the recorder is told to reload


def test_run_now_queues_a_command_once(app):
    make, ctl, _ = app
    at = make("More")
    at.button(key="q_runner").click().run()
    at.button(key="q_runner").click().run()                                          # a double click
    runs = [r for r in ctl.recent_commands() if r[2] == "run_job"]
    assert len(runs) == 1 and '"runner"' in runs[0][3]


def test_the_page_survives_every_notice_at_once(app):
    make, ctl, state = app
    state["recorder"], state["sandbox"], state["ollama"], state["fyers"] = False, False, False, False
    ctl.set("halt.firm", True)
    ctl.set("book.sandbox_A.paused", True)
    ctl.set("job.news.enabled", False)
    for section in ("Programs", "Switches"):
        at = make(section)
        assert not at.exception, [e.value for e in at.exception]
    assert "MASTER HALT IS ON" in text_of(at)
