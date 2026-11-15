from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import firm.control as control
import firm.fyers_auth as fyers_auth
import firm.ops as ops
from firm.control import Control
from firm.todo import Task

APP = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "CONTROL_DB", tmp_path / "c.db")
    state = {"recorder": True, "sandbox": True, "ollama": True, "fyers": True}
    monkeypatch.setattr(ops, "is_running", lambda n, defs=None: state[n])
    monkeypatch.setattr(ops, "ollama_running", lambda: state["ollama"])
    monkeypatch.setattr(ops, "job_running", lambda n: False)
    monkeypatch.setattr(fyers_auth, "token_status", lambda: (state["fyers"], "Fyers is reachable"))
    monkeypatch.setattr(fyers_auth, "begin", lambda: ("https://fyers.example/login", "STATE1"))
    from dashboard import control_page, home_page
    monkeypatch.setattr(control_page, "RECORDER_DB_PATH", tmp_path / "none.db")
    monkeypatch.setattr(home_page, "RECORDER_DB_PATH", tmp_path / "none.db")
    monkeypatch.setattr(home_page, "LIVE_DIR", tmp_path / "live")
    ctl = Control(tmp_path / "c.db")

    def make(mode="Live", tasks=None):
        control_page.live_state.clear()
        control_page.fyers_state.clear()
        if tasks is not None:
            monkeypatch.setattr(home_page, "todays_tasks", lambda *a, **k: tasks)
        at = AppTest.from_file(APP, default_timeout=120)
        at.session_state["_test_page"] = "home"
        at.session_state["mode"] = mode
        at.run()
        return at

    return make, ctl, state


def body(at) -> str:
    return "\n".join([m.value for m in at.markdown] + [m.value for m in at.caption] + [m.value for m in at.success]
                     + [m.value for m in at.warning] + [m.value for m in at.error] + [t.value for t in at.title]
                     + [h.value for h in at.subheader])


@pytest.mark.parametrize("mode", ["Live", "Sandbox replay"])
def test_home_renders_and_explains_where_everything_is(home, mode):
    make, _, _ = home
    at = make(mode)
    assert not at.exception, [e.value for e in at.exception]
    text = body(at)
    for must in ("What you need to do today", "How the money is doing", "Where is everything?",
                 "Live feed", "Money", "Settings", "Paper money", "Practice accounts", "Swing experiment"):
        assert must in text, must


def test_when_all_is_well_it_says_so_in_one_line(home):
    make, _, _ = home
    at = make()
    assert any("Everything is running normally" in s.value for s in at.success)


def test_a_problem_shows_one_plain_sentence_and_a_button_that_fixes_it(home):
    make, ctl, state = home
    state["recorder"] = False
    at = make()
    assert any("need" in w.value for w in at.warning)
    assert "Data collector is not running" in body(at)
    assert any(b.key == "home_fix_start_recorder" for b in at.button)


def test_the_fyers_login_appears_right_on_home_when_it_is_the_thing_to_do(home):
    make, _, state = home
    state["fyers"] = False
    todo = [Task("Log in to Fyers", "Needed once every trading morning, before 9:15.", False, action="login")]
    at = make(tasks=todo)
    text = body(at)
    assert "Log in to Fyers" in text and "Step 1." in text and "Step 3." in text
    assert any(b.label == "Finish login" for b in at.button)


def test_nothing_to_do_says_nothing_to_do(home):
    make, _, _ = home
    at = make(tasks=[Task("Nothing to do today", "The system runs by itself.", True, info=True)])
    assert "Nothing to do today" in body(at) and "Step 1." not in body(at)


def test_the_emergency_stop_is_on_home_and_resumes_in_one_click(home):
    make, ctl, _ = home
    at = make()
    at.button(key="home_halt").click().run()
    assert ctl.get_bool("halt.firm") and "on HOLD" in body(at)
    at.button(key="home_resume").click().run()
    assert not ctl.get_bool("halt.firm")


def test_the_sidebar_is_grouped_and_uses_plain_names():
    src = Path(APP).read_text(encoding="utf-8")
    for name in ("Home", "Live feed", "Money", "Notes", "Settings", "Research", "System"):
        assert f'title="{name}"' in src
    assert '"Watch"' in src and '"Control"' in src and '"Advanced"' in src
