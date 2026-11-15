"""SQLite store for news, exchange announcements and recorder run history."""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS news (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT,
    link TEXT,
    published TEXT,           -- as claimed by the feed
    first_seen TEXT NOT NULL  -- when we first saw it: use this in backtests
);
CREATE INDEX IF NOT EXISTS news_first_seen ON news(first_seen);

CREATE TABLE IF NOT EXISTS announcements (
    id TEXT PRIMARY KEY,
    symbol TEXT,
    company TEXT,
    category TEXT,
    text TEXT,
    attachment TEXT,
    has_xbrl INTEGER,
    industry TEXT,
    exchange_time TEXT,       -- NSE dissemination time: public from this moment
    first_seen TEXT NOT NULL,
    raw TEXT
);
CREATE INDEX IF NOT EXISTS ann_symbol_time ON announcements(symbol, exchange_time);

CREATE TABLE IF NOT EXISTS board_meetings (
    id TEXT PRIMARY KEY,
    symbol TEXT,
    meeting_date TEXT,        -- when the board meets (results usually same day)
    purpose TEXT,
    description TEXT,
    is_results INTEGER,
    intimated_at TEXT,        -- when the meeting was announced: public from here
    first_seen TEXT NOT NULL,
    raw TEXT
);
CREATE INDEX IF NOT EXISTS bm_symbol_date ON board_meetings(symbol, meeting_date);

CREATE TABLE IF NOT EXISTS recorder_runs (
    id INTEGER PRIMARY KEY,
    job TEXT NOT NULL,
    started TEXT NOT NULL,
    finished TEXT NOT NULL,
    status TEXT NOT NULL,     -- ok | partial | error | auth_failed | skipped
    rows INTEGER,
    message TEXT
);
CREATE INDEX IF NOT EXISTS runs_job_time ON recorder_runs(job, started);

CREATE TABLE IF NOT EXISTS results (   -- quarterly results from NSE XBRL, amounts in Rs crore
    symbol TEXT NOT NULL,
    to_date TEXT NOT NULL,             -- quarter end
    consolidated INTEGER NOT NULL,     -- 1 consolidated, 0 standalone
    from_date TEXT,
    broadcast TEXT NOT NULL,           -- when the filing became public (first filing wins)
    audited TEXT,
    seq TEXT,
    income REAL,                       -- total income
    revenue REAL,                      -- revenue from operations (total income if not reported)
    pbt REAL,                          -- profit before tax
    pat REAL,                          -- net profit (owners' share where reported)
    eps REAL,
    xbrl_url TEXT,
    first_seen TEXT NOT NULL,
    PRIMARY KEY (symbol, to_date, consolidated)
);
CREATE INDEX IF NOT EXISTS results_broadcast ON results(broadcast);
"""

NEWS_COLS = ["id", "source", "title", "summary", "link", "published", "first_seen"]
ANN_COLS = ["id", "symbol", "company", "category", "text", "attachment", "has_xbrl",
            "industry", "exchange_time", "first_seen", "raw"]
RESULT_COLS = ["symbol", "to_date", "consolidated", "from_date", "broadcast", "audited", "seq", "income",
               "revenue", "pbt", "pat", "eps", "xbrl_url", "first_seen"]
BM_COLS = ["id", "symbol", "meeting_date", "purpose", "description", "is_results",
           "intimated_at", "first_seen", "raw"]


class RecorderDB:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), timeout=30)
        self.conn.executescript(SCHEMA)

    def _insert_new(self, table: str, cols: list[str], rows: list[dict]) -> int:
        if not rows:
            return 0
        before = self.conn.total_changes
        self.conn.executemany(
            f"INSERT OR IGNORE INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            [tuple(r.get(c) for c in cols) for r in rows],
        )
        self.conn.commit()
        return self.conn.total_changes - before

    def insert_news(self, rows: list[dict]) -> int:
        """Insert unseen items; returns how many were new."""
        return self._insert_new("news", NEWS_COLS, rows)

    def insert_announcements(self, rows: list[dict]) -> int:
        return self._insert_new("announcements", ANN_COLS, rows)

    def insert_board_meetings(self, rows: list[dict]) -> int:
        return self._insert_new("board_meetings", BM_COLS, rows)

    def insert_results(self, rows: list[dict]) -> int:
        """Earliest filing per (symbol, quarter, consolidated) wins; later revisions are ignored."""
        return self._insert_new("results", RESULT_COLS, rows)

    def known_results(self) -> set[tuple[str, str, int]]:
        return set(self.conn.execute("SELECT symbol, to_date, consolidated FROM results").fetchall())

    def log_run(self, job: str, started: str, finished: str, status: str,
                rows: int = 0, message: str = "") -> None:
        self.conn.execute(
            "INSERT INTO recorder_runs (job, started, finished, status, rows, message)"
            " VALUES (?,?,?,?,?,?)",
            (job, started, finished, status, rows, message[:2000]),
        )
        self.conn.commit()

    def last_runs(self) -> list[tuple]:
        return self.conn.execute(
            "SELECT job, started, status, rows, message FROM recorder_runs r"
            " WHERE id = (SELECT MAX(id) FROM recorder_runs WHERE job = r.job) ORDER BY job"
        ).fetchall()

    def close(self) -> None:
        self.conn.close()
