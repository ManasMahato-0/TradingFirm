"""Entry points the recorder schedules: research (every few minutes), evening desk work."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from firm.agents.manager import Manager
from firm.agents.researcher import Researcher
from firm.agents.trader import TraderGuard
from firm.llm import LoggedLLM, get_llm


def run_research(db_path: Path | str) -> dict:
    """Fewer LLM calls per run while the market is open, so the recorder's option
    snapshots are not held up (each call takes a few seconds)."""
    from datetime import datetime
    from datetime import time as dtime

    from firm.config import IST

    t = datetime.now(IST)
    busy = t.weekday() < 5 and dtime(9, 0) <= t.time() <= dtime(15, 40)
    return Researcher(LoggedLLM(get_llm(), db_path), db_path, max_llm_calls=15 if busy else 60).run()


def run_evening(db_path: Path | str, control=None) -> dict:
    """Manager note and AI Trader guard for every open book, for its latest processed day."""
    from firm.activity import connect
    from firm.backtest.benchmarks import equal_weight
    from firm.backtest.data import MarketData
    from firm.universe import nifty100

    conn = connect(db_path)
    books = [dict(zip([c[0] for c in conn.execute("SELECT * FROM books").description], r))
             for r in conn.execute("SELECT * FROM books WHERE last_bar IS NOT NULL AND last_bar != 'None'")]
    conn.close()
    if not books:
        return {"notes": 0, "proposals": 0, "msg": "no open books yet"}
    llm = LoggedLLM(get_llm(), db_path)
    manager, guard = Manager(llm, db_path), TraderGuard(llm, db_path)
    first = min(date.fromisoformat(b["start"]) for b in books)
    data = MarketData.from_store("1d_yahoo", nifty100(), start=first - timedelta(days=30))
    closes = data.panels["close"]
    notes = proposals = 0
    for b in books:
        day = date.fromisoformat(b["last_bar"])
        ew = equal_weight(closes, date.fromisoformat(b["start"]) + timedelta(days=1), day + timedelta(days=1))
        if control is None or control.get_bool("agent.manager"):
            notes += manager.write(b, day, closes[closes.index.date <= day], ew) is not None
        if b["status"] == "probation" and (control is None or control.get("agent.guard") != "off"):
            proposals += len(guard.review(b, day, closes[closes.index.date <= day]))
    return {"notes": notes, "proposals": proposals, "msg": f"{notes} notes, {proposals} guard reviews"}
