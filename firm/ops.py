"""Start, stop and check the firm's background programs, and launch long jobs, for the Control page.

Is it running? Each service holds a lock port (its single-instance guard) and writes a PID file.
Stopping is cooperative first (the control board's stop flag; the service exits at its next tick) and
only then forced by PID. Long jobs (backfills, replays) run as their own detached processes so they
never freeze the recorder during market hours.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from firm.config import DATA_DIR, LOG_DIR, ROOT

PID_DIR = DATA_DIR / "pids"
PYTHONW = ROOT / ".venv" / "Scripts" / "pythonw.exe"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
OLLAMA = Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
DETACHED = (getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0))


@dataclass(frozen=True)
class ServiceDef:
    port: int
    argv: tuple[str, ...]       # after the interpreter


SERVICE_DEFS = {
    "recorder": ServiceDef(47653, (str(ROOT / "scripts" / "recorder.py"), "run")),
    "sandbox": ServiceDef(47655, (str(ROOT / "scripts" / "sandbox.py"), "live")),
}


# ---------------------------------------------------------------- pids and ports

def write_pid(name: str, pid: int | None = None) -> None:
    PID_DIR.mkdir(parents=True, exist_ok=True)
    (PID_DIR / f"{name}.pid").write_text(str(pid or os.getpid()))


def read_pid(name: str) -> int | None:
    try:
        return int((PID_DIR / f"{name}.pid").read_text().strip())
    except (OSError, ValueError):
        return None


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True, text=True,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    return f'"{pid}"' in out


def port_in_use(port: int) -> bool:
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port))
        return False
    except OSError:
        return True
    finally:
        s.close()


def is_running(name: str, defs: dict[str, ServiceDef] = SERVICE_DEFS) -> bool:
    return port_in_use(defs[name].port)


def kill_tree(pid: int) -> None:
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


# ---------------------------------------------------------------- services

def start_service(name: str, ctl, defs: dict[str, ServiceDef] = SERVICE_DEFS, interpreter: Path = PYTHONW) -> str:
    """Switch it on and start it if it is not already running."""
    ctl.set(f"service.{name}.run", "run")
    if is_running(name, defs):
        return "already running"
    subprocess.Popen([str(interpreter), *defs[name].argv], cwd=str(ROOT), creationflags=DETACHED,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
    return "started"


def stop_service(name: str, ctl, defs: dict[str, ServiceDef] = SERVICE_DEFS, wait_s: float = 90.0,
                 poll_s: float = 1.0) -> str:
    """Ask it to stop (the board flag), wait for it to leave, force it only as a last resort."""
    ctl.set(f"service.{name}.run", "stop")
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if not is_running(name, defs):
            return "stopped"
        time.sleep(poll_s)
    pid = read_pid(name)
    if pid and pid_alive(pid):
        kill_tree(pid)
        time.sleep(2)
    return "stopped (forced)" if not is_running(name, defs) else "could not stop"


# ---------------------------------------------------------------- Ollama (the local AI model)

def ollama_running() -> bool:
    import requests

    try:
        return requests.get("http://127.0.0.1:11434/api/version", timeout=2).status_code == 200
    except Exception:
        return False


def start_ollama() -> str:
    if ollama_running():
        return "already running"
    if not OLLAMA.exists():
        return "Ollama is not installed"
    subprocess.Popen([str(OLLAMA), "serve"], creationflags=DETACHED, stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
    return "started"


def stop_ollama() -> str:
    for image in ("ollama.exe", "ollama app.exe"):
        subprocess.run(["taskkill", "/IM", image, "/F"], capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    time.sleep(1.5)
    return "stopped" if not ollama_running() else "could not stop"


# ---------------------------------------------------------------- long jobs

def job_defs(day: date | None = None, ndays: int = 1) -> dict[str, tuple[str, tuple[str, ...]]]:
    """name -> (what it does in plain words, argv after the interpreter)."""
    today = date.today()
    out = {
        "backfill_results": ("Download all quarterly results history (about 20-40 minutes)",
                             (str(ROOT / "scripts" / "recorder.py"), "backfill-results")),
        "refresh_filings": ("Re-fetch the last 30 days of company filings",
                            (str(ROOT / "scripts" / "recorder.py"), "backfill-announcements",
                             "--start", (today - timedelta(days=30)).isoformat(), "--end", today.isoformat())),
        "backfill_options_data": ("Download NSE option prices since 2019 (about 40 minutes)",
                                  (str(ROOT / "scripts" / "fo_backfill.py"), "--start", "2019-01-01")),
    }
    out["sandbox_shakedown"] = ("Start today's UNCOUNTED sandbox test session (separate data; needs the Fyers login)",
                                (str(ROOT / "scripts" / "sandbox.py"), "live", "--shakedown", "--once"))
    if day is not None:
        out["sandbox_replay"] = (f"Replay {ndays} recorded day(s) from {day} through the sandbox (a rehearsal)",
                                 (str(ROOT / "scripts" / "sandbox.py"), "replay", "--day", day.isoformat(),
                                  "--ndays", str(ndays)))
    return out


def job_running(name: str) -> bool:
    return pid_alive(read_pid(f"job_{name}"))


def launch_job(name: str, argv: tuple[str, ...], interpreter: Path = PYTHON) -> str:
    """Start a long job as its own detached process. A second launch while it runs is refused."""
    if job_running(name):
        return "already running"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_DIR / f"job_{name}.log", "ab")
    p = subprocess.Popen([str(interpreter), *argv], cwd=str(ROOT), creationflags=DETACHED, stdin=subprocess.DEVNULL,
                         stdout=log, stderr=subprocess.STDOUT, close_fds=True)
    write_pid(f"job_{name}", p.pid)
    return "started"
