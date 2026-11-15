"""SQLite storage for fills, equity snapshots and risk events."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from firm.broker import Fill

SCHEMA = """
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    team TEXT NOT NULL,
    symbol TEXT NOT NULL,
    segment TEXT NOT NULL,
    side TEXT NOT NULL,
    qty INTEGER NOT NULL,
    price REAL NOT NULL,
    costs REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    reason TEXT
);
CREATE TABLE IF NOT EXISTS equity (
    ts TEXT NOT NULL,
    team TEXT NOT NULL,
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    PRIMARY KEY (ts, team)
);
CREATE TABLE IF NOT EXISTS risk_events (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    team TEXT NOT NULL,
    symbol TEXT,
    approved INTEGER NOT NULL,
    requested_qty INTEGER,
    approved_qty INTEGER,
    reason TEXT
);
CREATE TABLE IF NOT EXISTS positions (  -- latest snapshot per book
    team TEXT NOT NULL,
    symbol TEXT NOT NULL,
    qty INTEGER NOT NULL,
    avg_price REAL NOT NULL,
    last_price REAL,
    ts TEXT NOT NULL,
    PRIMARY KEY (team, symbol)
);
CREATE TABLE IF NOT EXISTS sandbox_quotes (  -- quote at each decision, to measure feed latency
    book TEXT NOT NULL,
    decision_ts TEXT NOT NULL,   -- start of the bar whose close triggered the order
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quote REAL NOT NULL,
    PRIMARY KEY (book, decision_ts, symbol)
);
CREATE TABLE IF NOT EXISTS books (  -- live paper books (team accounts, shadows)
    name TEXT PRIMARY KEY,
    team TEXT NOT NULL,
    label TEXT NOT NULL,
    status TEXT NOT NULL,       -- probation | active | shadow | cut
    capital REAL NOT NULL,
    start TEXT NOT NULL,
    strategy TEXT,
    version TEXT,
    last_bar TEXT,
    last_run TEXT,
    note TEXT
);
"""


class Storage:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)

    def record_fill(self, f: Fill) -> None:
        self.conn.execute(
            "INSERT INTO fills (ts, team, symbol, segment, side, qty, price, costs, realized_pnl, reason)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (f.ts.isoformat(), f.team, f.symbol, f.segment.value, f.side.value,
             f.qty, f.price, f.costs, f.realized_pnl, f.reason),
        )
        self.conn.commit()

    def record_equity(self, ts: datetime, team: str, equity: float, cash: float) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO equity (ts, team, equity, cash) VALUES (?,?,?,?)",
            (ts.isoformat(), team, equity, cash),
        )
        self.conn.commit()

    def record_risk_event(
        self, ts: datetime, team: str, symbol: str, approved: bool,
        requested_qty: int, approved_qty: int, reason: str,
    ) -> None:
        self.conn.execute(
            "INSERT INTO risk_events (ts, team, symbol, approved, requested_qty, approved_qty, reason)"
            " VALUES (?,?,?,?,?,?,?)",
            (ts.isoformat(), team, symbol, int(approved), requested_qty, approved_qty, reason),
        )
        self.conn.commit()

    def record_positions(self, team: str, ts: datetime, rows: list[tuple[str, int, float, float | None]]) -> None:
        """Replace the book's position snapshot: rows of (symbol, qty, avg_price, last_price)."""
        self.conn.execute("DELETE FROM positions WHERE team=?", (team,))
        self.conn.executemany(
            "INSERT INTO positions (team, symbol, qty, avg_price, last_price, ts) VALUES (?,?,?,?,?,?)",
            [(team, s, q, a, p, ts.isoformat()) for s, q, a, p in rows])
        self.conn.commit()

    def record_quotes(self, rows: list[tuple]) -> None:
        self.conn.executemany("INSERT OR REPLACE INTO sandbox_quotes (book, decision_ts, symbol, side, quote)"
                              " VALUES (?,?,?,?,?)", rows)
        self.conn.commit()

    def upsert_book(self, **row) -> None:
        cols = ", ".join(row)
        self.conn.execute(
            f"INSERT INTO books ({cols}) VALUES ({', '.join('?' * len(row))})"
            f" ON CONFLICT(name) DO UPDATE SET " + ", ".join(f"{c}=excluded.{c}" for c in row if c != "name"),
            tuple(row.values()))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
