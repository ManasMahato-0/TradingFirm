"""What needs the owner's attention? The Control page's top strip. Pure logic, no network and no
Streamlit, so it is tested on its own.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from datetime import time as dtime

from firm.control import BOOKS, JOBS, SERVICES, Control


@dataclass(frozen=True)
class Issue:
    key: str            # also names the one-click fix the page offers
    text: str           # plain words
    fix: str            # button label
    level: str = "warn"  # warn | info


def attention_items(ctl: Control, running: dict[str, bool], ollama_up: bool, fyers_ok: bool | None,
                    now: datetime, runs: dict | None = None) -> list[Issue]:
    """running: service name -> process is running. fyers_ok: None = not checked yet.
    runs: job -> (when it last ran, status), from the recorder's run log."""
    out: list[Issue] = []
    runs = runs or {}
    if ctl.get_bool("halt.firm"):
        out.append(Issue("resume_halt", "Master halt is ON: no new positions anywhere.", "Resume trading"))
    for name, (label, _) in SERVICES.items():
        if not running.get(name, False):
            wanted_off = not ctl.service_should_run(name)
            out.append(Issue(f"start_{name}", f"{label} is " + ("switched off." if wanted_off else "not running."),
                             "Start"))
    if ctl.get_bool("llm.enabled") and not ollama_up:
        out.append(Issue("start_ollama", "The local AI model is offline, so notes and summaries are paused.", "Start it"))
    market_hours = now.weekday() < 5 and dtime(8, 0) <= now.time() < dtime(15, 45)
    recently = {j for j, (when, st) in runs.items() if st == "auth_failed" and _within_a_day(when, now)}
    # Only on a trading day, in the window where a login matters. A price-saving job that failed over a weekend
    # for lack of a login is not something to do anything about until the next trading morning.
    if market_hours and (fyers_ok is False or (recently and ctl.job_enabled(next(iter(recently))))):
        out.append(Issue("login", "Fyers login needed: " + (", ".join(JOBS[j][0].lower() for j in sorted(recently) if j in JOBS)
                                                           + " could not run without it." if recently else
                                                           "data collection and the live sandbox depend on it today."),
                         "Log in"))
    failed = [JOBS[j][0] for j, (when, st) in sorted(runs.items())
              if j in JOBS and st == "error" and ctl.job_enabled(j) and _within_a_day(when, now)]
    if failed:
        out.append(Issue("show_failed_jobs", "Last run failed: " + ", ".join(failed) + ". Usually a brief network hiccup; "
                         "it retries by itself.", "Review", "info"))
    paused = [BOOKS[b] for b in BOOKS if ctl.get_bool(f"book.{b}.paused")]
    if paused:
        out.append(Issue("show_books", f"{len(paused)} book(s) paused: " + ", ".join(paused) + ".", "Review", "info"))
    off = [JOBS[j][0] for j in JOBS if not ctl.job_enabled(j)]
    if off:
        out.append(Issue("show_jobs", f"{len(off)} data job(s) switched off: " + ", ".join(off) + ".", "Review", "info"))
    return out


def _within_a_day(when: str, now: datetime) -> bool:
    try:
        return (now - datetime.fromisoformat(when)).total_seconds() < 86400
    except (TypeError, ValueError):
        return False


def last_job_runs(recorder_db) -> dict[str, tuple[str, str]]:
    """job -> (when it last ran, how it went), from the recorder's run log. Empty if unavailable."""
    import sqlite3
    from pathlib import Path

    if not Path(recorder_db).exists():
        return {}
    try:
        with sqlite3.connect(f"file:{recorder_db}?mode=ro", uri=True, timeout=10) as c:
            rows = c.execute("SELECT job, started, status FROM recorder_runs r WHERE id = "
                             "(SELECT MAX(id) FROM recorder_runs WHERE job=r.job)").fetchall()
    except sqlite3.OperationalError:
        return {}
    return {job: (started, status) for job, started, status in rows}
