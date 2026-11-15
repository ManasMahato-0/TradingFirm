"""Control board: the switches, the command queue and the change log the dashboard writes and the
services (recorder, runner, sandbox) read. One small SQLite file, always `data/control.db`, no
matter which data mode the dashboard is viewing.

Rules:
- Every switch defaults to TODAY'S behaviour, so an empty control board changes nothing.
- Pausing and halting only BLOCK NEW ENTRIES. Services keep marking to market and recording equity, so
  a paused day still counts and nothing is replayed on resume.
- Strategy settings, thresholds, risk limits and pre-registered rules are NOT switches. They are shown
  read-only (LOCKED) and change only by a dated amendment in docs/PLAN.md.
- Nothing here can place a real order. There is no such switch.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from firm.config import DATA_DIR, IST

CONTROL_DB = DATA_DIR / "control.db"
OWNER_HALT_REASON = "paused by the owner"      # shown in risk messages; probation checks ignore it

SCHEMA = """
CREATE TABLE IF NOT EXISTS switches (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated TEXT NOT NULL, by TEXT);
CREATE TABLE IF NOT EXISTS commands (
    id INTEGER PRIMARY KEY, ts TEXT NOT NULL, target TEXT NOT NULL, name TEXT NOT NULL, args TEXT NOT NULL,
    status TEXT NOT NULL, result TEXT, started TEXT, finished TEXT, by TEXT);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY, ts TEXT NOT NULL, who TEXT, action TEXT NOT NULL, key TEXT, old TEXT, new TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS windows (   -- when a pause or halt was on: lets trackers show what was paused
    id INTEGER PRIMARY KEY, scope TEXT NOT NULL, start TEXT NOT NULL, end TEXT);
CREATE TABLE IF NOT EXISTS heartbeats (service TEXT PRIMARY KEY, ts TEXT NOT NULL);
"""


@dataclass(frozen=True)
class Switch:
    key: str
    default: str
    label: str
    help: str
    group: str


JOBS = {
    "news": ("Headlines", "Save news headlines every 5 minutes."),
    "announcements": ("Company filings", "Save company filings from the stock exchange every 5 minutes."),
    "board_meetings": ("Board-meeting dates", "Save upcoming company board meetings and results dates every hour."),
    "results": ("Quarterly results", "Check every 30 minutes for companies' new results numbers."),
    "options": ("Option prices", "Save option prices every 5 minutes while the market is open."),
    "bars": ("Price bars", "Save the day's share prices after the market closes."),
    "researcher": ("Researcher", "An AI helper that reads new filings and headlines and writes short notes."),
    "runner": ("Evening paper-trading update", "After 16:00 each trading day: updates the swing experiment and decides at month-end."),
    "desk": ("Daily notes", "After the evening update: a plain-English note on how each book did."),
}
BOOKS = {
    "swing_probation": "Swing experiment (the main test)",
    "swing_shadow_momentum": "Swing yardstick (plain method, for comparison)",
    "sandbox_A": "Practice A: learned from history",
    "sandbox_B": "Practice B: learns as it goes",
    "sandbox_C": "Practice C: random trades (the control)",
    "sandbox_D": "Practice D: just holds cash",
}
SERVICES = {
    "recorder": ("Data collector", "Saves prices, news and filings in the background, and runs the evening update."),
    "sandbox": ("Practice trading", "Runs the four practice accounts on trading days."),
}


def _registry() -> dict[str, Switch]:
    r: dict[str, Switch] = {}
    r["halt.firm"] = Switch("halt.firm", "false", "Master halt",
                            "No new positions anywhere. Reducing and flattening still work.", "firm")
    for j, (label, help_) in JOBS.items():
        r[f"job.{j}.enabled"] = Switch(f"job.{j}.enabled", "true", label, help_, "jobs")
    for b, label in BOOKS.items():
        r[f"book.{b}.paused"] = Switch(f"book.{b}.paused", "false", label,
                                       "Paused: no new positions. It still marks to market and counts its days.", "books")
    r["agent.manager"] = Switch("agent.manager", "true", "Daily notes", "Write the daily note for each book.", "agents")
    r["agent.guard"] = Switch("agent.guard", "shadow", "Risk watcher",
                              "Watch only: notes what it would do, never acts. Off: does not run.", "agents")
    r["llm.enabled"] = Switch("llm.enabled", "true", "AI model", "Use the AI model on this laptop for notes and summaries.", "agents")
    r["sandbox.skip_date"] = Switch("sandbox.skip_date", "", "Skip a sandbox day",
                                    "The date (YYYY-MM-DD) the sandbox accounts sit out. They stay flat and the day still counts.", "books")
    for s in SERVICES:
        r[f"service.{s}.run"] = Switch(f"service.{s}.run", "run", SERVICES[s][0], SERVICES[s][1], "services")
    return r


SWITCHES = _registry()

# Things that are deliberately NOT switches: shown read-only with the reason.
LOCKED = [
    ("Paper mode", "Always on. There is no real-order code anywhere in this system, and no switch for one."),
    ("Entry thresholds and model settings", "Pre-registered. Changing them after seeing results would invalidate the test. "
     "Only by a dated amendment in docs/PLAN.md."),
    ("Risk limits (position size, daily loss, orders per day)", "Fixed per team in the firm config and enforced by code."),
    ("Probation and sandbox rules", "Pre-registered promote/cut rules and the day-60/120 checkpoints."),
    ("Final-exam windows", "Each strategy version gets exactly one try on its locked period."),
]


def _now() -> str:
    return datetime.now(IST).isoformat(timespec="seconds")


class Control:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else CONTROL_DB
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=30, isolation_level=None, check_same_thread=False)
        if str(self.path) != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)

    # ------------------------------------------------------------ switches

    def get(self, key: str, default: str | None = None) -> str:
        row = self.conn.execute("SELECT value FROM switches WHERE key=?", (key,)).fetchone()
        if row is not None:
            return row[0]
        if key in SWITCHES:
            return SWITCHES[key].default
        return "" if default is None else default

    def get_bool(self, key: str) -> bool:
        return self.get(key).lower() in ("1", "true", "yes", "on")

    def set(self, key: str, value, who: str = "dashboard", note: str = "") -> bool:
        """Returns True if the value changed. Every change is logged; pause/halt windows are tracked."""
        new = ("true" if value else "false") if isinstance(value, bool) else str(value)
        old = self.get(key)
        if old == new:
            return False
        self.conn.execute("INSERT INTO switches (key, value, updated, by) VALUES (?,?,?,?) "
                          "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated=excluded.updated, by=excluded.by",
                          (key, new, _now(), who))
        self.conn.execute("INSERT INTO audit (ts, who, action, key, old, new, note) VALUES (?,?,?,?,?,?,?)",
                          (_now(), who, "switch", key, old, new, note))
        if key == "halt.firm" or key.endswith(".paused"):
            if new == "true":
                self.conn.execute("INSERT INTO windows (scope, start) VALUES (?,?)", (key, _now()))
            else:
                self.conn.execute("UPDATE windows SET end=? WHERE scope=? AND end IS NULL", (_now(), key))
        return True

    # ------------------------------------------------------------ meaning of the switches

    def halted(self, book: str | None = None, today: str | None = None) -> bool:
        """Should NEW entries be blocked for this book right now?"""
        if self.get_bool("halt.firm"):
            return True
        if book is not None and self.get_bool(f"book.{book}.paused"):
            return True
        if book is not None and book.startswith("sandbox_") and today and self.get("sandbox.skip_date") == today:
            return True
        return False

    def job_enabled(self, job: str) -> bool:
        return self.get(f"job.{job}.enabled").lower() == "true"

    def service_should_run(self, service: str) -> bool:
        return self.get(f"service.{service}.run") != "stop"

    # ------------------------------------------------------------ heartbeats

    def beat(self, service: str) -> None:
        self.conn.execute("INSERT INTO heartbeats (service, ts) VALUES (?,?) ON CONFLICT(service) DO UPDATE SET ts=excluded.ts",
                          (service, _now()))

    def last_beat(self, service: str) -> datetime | None:
        row = self.conn.execute("SELECT ts FROM heartbeats WHERE service=?", (service,)).fetchone()
        return datetime.fromisoformat(row[0]) if row else None

    # ------------------------------------------------------------ commands

    def enqueue(self, target: str, name: str, args: dict | None = None, who: str = "dashboard") -> int | None:
        """Queue a command. An identical one already queued or running is not queued again
        (Streamlit reruns and double clicks would otherwise run it twice)."""
        a = json.dumps(args or {}, sort_keys=True)
        dup = self.conn.execute("SELECT id FROM commands WHERE target=? AND name=? AND args=? "
                                "AND status IN ('queued','running')", (target, name, a)).fetchone()
        if dup:
            return None
        cur = self.conn.execute("INSERT INTO commands (ts, target, name, args, status, by) VALUES (?,?,?,?,?,?)",
                                (_now(), target, name, a, "queued", who))
        self.conn.execute("INSERT INTO audit (ts, who, action, key, note) VALUES (?,?,?,?,?)",
                          (_now(), who, "command", f"{target}:{name}", a))
        return int(cur.lastrowid)

    def claim(self, target: str) -> dict | None:
        """The oldest queued command for `target`, claimed so that no other reader can run it too."""
        while True:
            row = self.conn.execute("SELECT id, name, args FROM commands WHERE target=? AND status='queued' "
                                    "ORDER BY id LIMIT 1", (target,)).fetchone()
            if row is None:
                return None
            cur = self.conn.execute("UPDATE commands SET status='running', started=? WHERE id=? AND status='queued'",
                                    (_now(), row[0]))
            if cur.rowcount == 1:
                return {"id": row[0], "name": row[1], "args": json.loads(row[2])}

    def finish(self, command_id: int, status: str, result: str = "") -> None:
        self.conn.execute("UPDATE commands SET status=?, result=?, finished=? WHERE id=?",
                          (status, result[:500], _now(), command_id))

    def recent_commands(self, limit: int = 15) -> list[tuple]:
        return self.conn.execute("SELECT ts, target, name, args, status, result FROM commands ORDER BY id DESC LIMIT ?",
                                 (limit,)).fetchall()

    # ------------------------------------------------------------ change log and windows

    def audit_tail(self, limit: int = 30) -> list[tuple]:
        return self.conn.execute("SELECT ts, who, action, key, old, new, note FROM audit ORDER BY id DESC LIMIT ?",
                                 (limit,)).fetchall()

    def log(self, action: str, note: str, who: str = "dashboard", key: str = "") -> None:
        self.conn.execute("INSERT INTO audit (ts, who, action, key, note) VALUES (?,?,?,?,?)",
                          (_now(), who, action, key, note))

    def windows(self, scope: str | None = None) -> list[tuple]:
        if scope:
            return self.conn.execute("SELECT scope, start, end FROM windows WHERE scope=? ORDER BY id", (scope,)).fetchall()
        return self.conn.execute("SELECT scope, start, end FROM windows ORDER BY id").fetchall()

    def close(self) -> None:
        self.conn.close()
