"""AI Trader, daily guard: may only propose REDUCING risk (cut a holding, skip a planned buy).

Trigger: a Researcher note on a stock we hold (or plan to buy) that is negative
(direction <= -1) and notable (importance >= 2), from the last 3 days.

SHADOW MODE: proposals are logged, never applied, so the probation book stays identical
to its backtest. Each proposal is later scored: if the stock then lagged the market,
the guard was right. It goes live only after it has proven itself.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from firm.activity import ActivityLog, connect, now_iso
from firm.config import IST
from firm.agents.researcher import SCHEMA as RESEARCH_SCHEMA
from firm.llm import LLMUnavailable, LoggedLLM

AGENT = "AI Trader"

SCHEMA = """
CREATE TABLE IF NOT EXISTS guard_proposals (
    id INTEGER PRIMARY KEY,
    date TEXT NOT NULL,
    book TEXT NOT NULL,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,       -- cut | skip_buy | none
    reason TEXT NOT NULL,
    price REAL,                 -- close on the proposal date (for scoring)
    notes TEXT,
    shadow INTEGER NOT NULL DEFAULT 1,
    model TEXT,
    created TEXT NOT NULL,
    UNIQUE (date, book, symbol)
);
"""

DECISION = {
    "type": "object",
    "properties": {"action": {"type": "string", "enum": ["none", "cut", "skip_buy"]},
                   "reason": {"type": "string"}},
    "required": ["action", "reason"],
}

SYSTEM = (
    "You are a cautious portfolio risk officer for a systematic stock strategy. The strategy picks stocks "
    "by momentum; you may only propose reducing risk when news is serious enough that the systematic "
    "model could not have known it: fraud or governance problems, regulator action, large unexpected "
    "losses, promoter distress, major legal or operational damage. Ordinary bad days, routine filings and "
    "mild news are NOT reasons: answer 'none' then. 'cut' = sell a stock we hold; 'skip_buy' = do not buy "
    "a stock we planned to buy. reason: one sentence, facts from the notes only."
)


class TraderGuard:
    def __init__(self, llm: LoggedLLM, db_path: Path | str) -> None:
        self.llm, self.db_path = llm, db_path
        self.conn = connect(db_path)
        self.conn.executescript(SCHEMA)
        self.conn.executescript(RESEARCH_SCHEMA)  # reads research notes
        self.activity = ActivityLog(db_path)

    def _planned_buys(self, book: str) -> list[str]:
        row = self.conn.execute(
            "SELECT details FROM activity WHERE book=? AND kind='talk' AND to_agent='Risk engine'"
            " ORDER BY id DESC LIMIT 1", (book,)).fetchone()
        if not row or not row[0]:
            return []
        targets = json.loads(row[0]).get("targets", {})
        return [s for s, q in targets.items() if q > 0]

    def review(self, book: dict, day: date, closes: pd.DataFrame) -> list[dict]:
        held = [r[0] for r in self.conn.execute("SELECT symbol FROM positions WHERE team=?", (book["name"],))]
        planned = [s for s in self._planned_buys(book["name"]) if s not in held]
        watch = held + planned
        if not watch:
            return []
        since = (day - timedelta(days=3)).isoformat()
        notes = pd.read_sql_query(
            f"SELECT symbol, ts, summary, direction, importance, event_type FROM research_notes"
            f" WHERE ts >= ? AND direction <= -1 AND importance >= 2"
            f" AND symbol IN ({','.join('?' * len(watch))}) ORDER BY ts", self.conn, params=(since, *watch))
        out = []
        for sym, g in notes.groupby("symbol"):
            if self.conn.execute("SELECT 1 FROM guard_proposals WHERE date=? AND book=? AND symbol=?",
                                 (day.isoformat(), book["name"], sym)).fetchone():
                continue
            status = "we hold it" if sym in held else "we plan to buy it at the next open"
            lines = "\n".join(f"- {r.ts[:10]} [{r.event_type}, direction {r.direction:+d}, importance "
                              f"{r.importance}] {r.summary}" for r in g.itertuples())
            try:
                d = self.llm.json(AGENT, SYSTEM, f"Stock: {sym} ({status}).\nResearch notes:\n{lines}\n\nDecide.",
                                  DECISION, max_tokens=150)
            except LLMUnavailable:
                break
            action = d["action"] if d["action"] != ("skip_buy" if sym in held else "cut") else "none"
            price = float(closes[sym].dropna().iloc[-1]) if sym in closes and closes[sym].notna().any() else None
            self.conn.execute(
                "INSERT OR IGNORE INTO guard_proposals (date, book, symbol, action, reason, price, notes, shadow,"
                " model, created) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (day.isoformat(), book["name"], sym, action, d["reason"], price,
                 g.to_json(orient="records"), 1, self.llm.name, now_iso()))
            self.conn.commit()
            ts = datetime.combine(day, datetime.min.time(), IST).replace(hour=16, minute=45)
            if action == "none":
                text = f"Reviewed bad news on {sym} ({status}): no action. {d['reason']}"
            else:
                verb = "SELL" if action == "cut" else "NOT BUY"
                text = (f"Shadow mode: would {verb} {sym}. {d['reason']} Not applied: the probation book "
                        "must match its backtest. Scored later against what the stock did.")
            self.activity.log("think", AGENT, text, team=book["team"], book=book["name"], ts=ts,
                              details={"symbol": sym, "action": action})
            out.append({"symbol": sym, "action": action})
        return out
