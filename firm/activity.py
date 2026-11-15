"""Activity log: what every agent thinks, says and does. The Trading Floor reads this.

kind:
  think  reasoning behind a decision ("ranked BEL first because ...")
  talk   a message to another agent (`to`), e.g. strategy -> risk engine
  act    something done: order filled, data fetched, model trained
  alert  something went wrong or needs a human

Agents also report a status (working / idle / error / off) with their current task.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from firm.config import IST

KINDS = ("think", "talk", "act", "alert")

SCHEMA = """
CREATE TABLE IF NOT EXISTS activity (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,          -- when it happened (market time for replayed bars)
    logged TEXT NOT NULL,      -- when it was written
    team TEXT,
    book TEXT,              -- which paper account, if any
    agent TEXT NOT NULL,
    kind TEXT NOT NULL,
    to_agent TEXT,
    text TEXT NOT NULL,
    details TEXT
);
CREATE INDEX IF NOT EXISTS activity_ts ON activity(ts);
CREATE TABLE IF NOT EXISTS agent_status (
    agent TEXT NOT NULL,
    team TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL,       -- working | idle | error | off
    task TEXT,
    updated TEXT NOT NULL,
    PRIMARY KEY (agent, team)
);
"""


def now_iso() -> str:
    return datetime.now(IST).isoformat(timespec="seconds")


def connect(path: Path | str) -> sqlite3.Connection:
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")  # dashboard reads while the runner writes
    return conn


class ActivityLog:
    def __init__(self, path: Path | str) -> None:
        self.conn = connect(path)
        self.conn.executescript(SCHEMA)

    def log(self, kind: str, agent: str, text: str, team: str | None = None, to: str | None = None,
            details: dict | None = None, ts: datetime | str | None = None, book: str | None = None) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}")
        ts = ts.isoformat(timespec="seconds") if isinstance(ts, datetime) else (ts or now_iso())
        self.conn.execute(
            "INSERT INTO activity (ts, logged, team, book, agent, kind, to_agent, text, details)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (ts, now_iso(), team, book, agent, kind, to, text,
             json.dumps(details, default=str) if details else None),
        )
        self.conn.commit()

    def status(self, agent: str, state: str, task: str = "", team: str | None = None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO agent_status (agent, team, state, task, updated) VALUES (?,?,?,?,?)",
            (agent, team or "", state, task, now_iso()),
        )
        self.conn.commit()

    def recent(self, limit: int = 200) -> list[tuple]:
        return self.conn.execute(
            "SELECT ts, team, agent, kind, to_agent, text FROM activity ORDER BY id DESC LIMIT ?",
            (limit,)).fetchall()

    def close(self) -> None:
        self.conn.close()
