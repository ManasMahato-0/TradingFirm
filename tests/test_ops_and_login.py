import socket
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

import firm.ops as ops
from firm import fyers_auth
from firm.control import Control

# ---------------------------------------------------------------- Fyers login panel

ENV = "FYERS_CLIENT_ID=ABC-100\nFYERS_SECRET_KEY=shh\nFYERS_REDIRECT_URI=http://127.0.0.1/cb\nFYERS_ACCESS_TOKEN=OLDTOKEN\n"


class FakeSession:
    last = None

    def __init__(self, client_id, secret_key, redirect_uri, response_type, grant_type, state):
        self.state, self.code = state, None
        FakeSession.last = self

    def generate_authcode(self):
        return f"https://fyers.example/auth?state={self.state}&client_id=ABC-100"

    def set_token(self, code):
        self.code = code

    def generate_token(self):
        return {"s": "ok", "access_token": "SECRET-TOKEN-123"} if self.code == "goodcode" else {"s": "error", "message": "bad code"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text(ENV)
    import fyers_apiv3.fyersModel as fm
    monkeypatch.setattr(fm, "SessionModel", FakeSession)
    return p


def test_login_round_trip_saves_the_token_and_never_returns_it(env):
    url, state = fyers_auth.begin(env)
    assert parse_qs(urlparse(url).query)["state"] == [state]
    ok, msg = fyers_auth.finish(f"http://127.0.0.1/cb?auth_code=goodcode&state={state}", state, env)
    assert ok and "SECRET-TOKEN-123" not in msg and "goodcode" not in msg
    text = env.read_text()
    assert "FYERS_ACCESS_TOKEN='SECRET-TOKEN-123'" in text or "FYERS_ACCESS_TOKEN=SECRET-TOKEN-123" in text
    assert "OLDTOKEN" not in text


def test_login_rejects_a_foreign_address_a_missing_code_and_a_bad_code(env):
    _, state = fyers_auth.begin(env)
    ok, msg = fyers_auth.finish("http://127.0.0.1/cb?auth_code=goodcode&state=SOMEONE-ELSES", state, env)
    assert not ok and "not from this login attempt" in msg
    assert not fyers_auth.finish("http://127.0.0.1/cb?state=" + state, state, env)[0]
    ok, msg = fyers_auth.finish(f"http://127.0.0.1/cb?auth_code=wrong&state={state}", state, env)
    assert not ok and "bad code" in msg
    assert not fyers_auth.finish("anything", None, env)[0]                    # no login started
    assert "OLDTOKEN" in env.read_text() and "SECRET-TOKEN" not in env.read_text()   # nothing was overwritten


def test_login_needs_the_app_keys_in_env(tmp_path):
    p = tmp_path / ".env"
    p.write_text("FYERS_CLIENT_ID=x\n")
    with pytest.raises(ValueError, match="Missing"):
        fyers_auth.begin(p)


# ---------------------------------------------------------------- services

def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


CHILD = """
import sys, time, socket
from pathlib import Path
port, control_db, pid_file, obey = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4] == "obey"
sys.path.insert(0, sys.argv[5])
from firm.control import Control
lock = socket.socket(); lock.bind(("127.0.0.1", port))
Path(pid_file).write_text(str(__import__("os").getpid()))
ctl = Control(control_db)
while True:
    if obey and not ctl.service_should_run("recorder"):
        break
    time.sleep(0.3)
"""


@pytest.fixture
def child(tmp_path, monkeypatch):
    monkeypatch.setattr(ops, "PID_DIR", tmp_path / "pids")
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    ctl = Control(tmp_path / "c.db")
    root = str(Path(__file__).resolve().parent.parent)

    def make(obey: bool):
        port = free_port()
        pid_file = tmp_path / "pids" / "recorder.pid"
        pid_file.parent.mkdir(exist_ok=True)
        defs = {"recorder": ops.ServiceDef(port, (str(script), str(port), str(tmp_path / "c.db"), str(pid_file),
                                                  "obey" if obey else "ignore", root))}
        return ctl, defs, pid_file

    return make


def wait_for(cond, timeout=15):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.3)
    return False


def test_port_detection():
    port = free_port()
    assert not ops.port_in_use(port)
    s = socket.socket()
    s.bind(("127.0.0.1", port))
    try:
        assert ops.port_in_use(port)
    finally:
        s.close()


def test_start_is_idempotent_and_stop_is_cooperative(child):
    ctl, defs, _ = child(obey=True)
    assert ops.start_service("recorder", ctl, defs, interpreter=Path(sys.executable)) == "started"
    assert wait_for(lambda: ops.is_running("recorder", defs))
    assert ops.start_service("recorder", ctl, defs, interpreter=Path(sys.executable)) == "already running"
    assert ctl.service_should_run("recorder")
    assert ops.stop_service("recorder", ctl, defs, wait_s=15) == "stopped"                # it left on its own
    assert not ctl.service_should_run("recorder")                                         # ... and stays switched off


def test_a_service_that_ignores_the_stop_flag_is_forced_down_by_pid(child):
    ctl, defs, pid_file = child(obey=False)
    ops.start_service("recorder", ctl, defs, interpreter=Path(sys.executable))
    assert wait_for(lambda: ops.is_running("recorder", defs) and pid_file.exists())
    assert ops.stop_service("recorder", ctl, defs, wait_s=3) == "stopped (forced)"
    assert not ops.is_running("recorder", defs)


def test_long_job_launch_is_guarded_against_a_double_start(tmp_path, monkeypatch):
    monkeypatch.setattr(ops, "PID_DIR", tmp_path / "pids")
    monkeypatch.setattr(ops, "LOG_DIR", tmp_path / "logs")
    argv = ("-c", "import time; time.sleep(6)")
    assert ops.launch_job("demo", argv, interpreter=Path(sys.executable)) == "started"
    assert ops.job_running("demo")
    assert ops.launch_job("demo", argv, interpreter=Path(sys.executable)) == "already running"
    assert wait_for(lambda: not ops.job_running("demo"), timeout=20)                     # finishes, can run again
    assert ops.launch_job("demo", ("-c", "pass"), interpreter=Path(sys.executable)) == "started"


def test_job_catalogue_describes_each_job_in_plain_words():
    from datetime import date
    jobs = ops.job_defs(date(2026, 11, 12), 2)
    assert {"backfill_results", "refresh_filings", "backfill_options_data", "sandbox_replay"} <= set(jobs)
    assert all(desc and argv for desc, argv in jobs.values())
