"""Tracker helpers shared by the dashboard and tests."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from firm.control import OWNER_HALT_REASON


def risk_blocks(db_path: Path | str, book: str) -> int:
    """How many orders the risk engine blocked for this book, NOT counting blocks caused by the owner's
    own pause or halt (a manual pause must not fail a probation on 'no risk-engine breaches')."""
    if not Path(db_path).exists():
        return 0
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10) as c:
        try:
            row = c.execute("SELECT COUNT(*) FROM activity WHERE book=? AND agent='Risk engine' "
                            "AND text LIKE 'Rejected%' AND text NOT LIKE ?", (book, f"%{OWNER_HALT_REASON}%")).fetchone()
        except sqlite3.OperationalError:
            return 0
    return int(row[0])
