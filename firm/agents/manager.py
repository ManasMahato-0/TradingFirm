"""Manager: a plain-English daily note per paper book, written after the evening run.

Code gathers the facts (money, benchmark, trades, movers, research on holdings, risk
messages). The LLM only turns them into prose. Every number in the note must appear in
the facts, otherwise the note is rejected and a template note is used instead.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from firm.activity import ActivityLog, connect, now_iso
from firm.config import IST
from firm.agents.researcher import SCHEMA as RESEARCH_SCHEMA
from firm.llm import LLMUnavailable, LoggedLLM

AGENT = "Manager"

SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_notes (
    date TEXT NOT NULL,
    book TEXT NOT NULL,
    headline TEXT NOT NULL,
    note TEXT NOT NULL,
    flags TEXT,
    facts TEXT NOT NULL,
    model TEXT,
    checked INTEGER NOT NULL,   -- 1 = LLM note passed the number check, 0 = template
    created TEXT NOT NULL,
    PRIMARY KEY (date, book)
);
"""

NOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "note": {"type": "string"},
        "flags": {"type": "array", "items": {"type": "object", "properties": {
            "level": {"type": "string", "enum": ["info", "watch", "act"]},
            "text": {"type": "string"}}, "required": ["level", "text"]}},
    },
    "required": ["headline", "note", "flags"],
}

SYSTEM = (
    "You are the manager of a paper-trading desk writing a short daily note for the firm's owner, "
    "a beginner. Use ONLY the facts given. Copy every number exactly as written in the facts; "
    "never compute new numbers, never round differently, never invent prices or targets. "
    "For comparisons with the market use the wording in today_vs_market and since_start_vs_market; "
    "never decide yourself whether the book beat or lagged. State numbers instead of judging them "
    "(say 'cash is 38%', not 'cash is dominant'). "
    "headline: under 12 words. note: 3 to 6 plain sentences: how the book did vs the equal-weight "
    "market, what traded and why if known, anything notable in research on holdings. "
    "flags: only real concerns (a holding with clearly negative news, drawdown near a limit, "
    "risk-engine blocks); empty list if none."
)

NUM = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").lstrip("+-") for n in NUM.findall(text)}


def numbers_ok(note: str, facts_text: str) -> bool:
    allowed = _numbers(facts_text) | {str(i) for i in range(0, 11)}  # small counts are fine
    return _numbers(note) <= allowed


def pct(x: float) -> str:
    return f"{x:+.2%}"


def rs(x: float) -> str:
    return f"Rs {x:,.0f}"


def gather_facts(db_path: Path | str, book: dict, day: date, closes: pd.DataFrame, ew: pd.Series) -> dict:
    conn = connect(db_path)
    name = book["name"]
    eq = pd.read_sql_query("SELECT ts, equity, cash FROM equity WHERE team=? ORDER BY ts", conn, params=(name,))
    eq["d"] = eq.ts.str[:10]
    eq = eq[eq.d <= day.isoformat()]
    e_now = float(eq.equity.iloc[-1])
    e_prev = float(eq.equity.iloc[-2]) if len(eq) > 1 else book["capital"]
    peak = max(float(eq.equity.max()), book["capital"])
    ew_day = float(ew.get(pd.Timestamp(day), 0.0))
    ew_since = float((1 + ew[ew.index <= pd.Timestamp(day)]).prod() - 1)

    fills = pd.read_sql_query("SELECT side, symbol, qty, price, realized_pnl FROM fills WHERE team=? AND ts LIKE ?",
                              conn, params=(name, f"{day.isoformat()}%"))
    pos = pd.read_sql_query("SELECT symbol, qty, avg_price FROM positions WHERE team=?", conn, params=(name,))
    movers = []
    if not pos.empty and len(closes) >= 2:
        c = closes[closes.index.date <= day]
        for s in pos.symbol:
            if s in c and c[s].notna().sum() >= 2:
                movers.append((s, float(c[s].iloc[-1] / c[s].dropna().iloc[-2] - 1)))
        movers.sort(key=lambda x: x[1])
    since = (day - timedelta(days=3)).isoformat()
    held = pos.symbol.tolist()
    notes = pd.DataFrame()
    if held:
        notes = pd.read_sql_query(
            f"SELECT symbol, summary, direction, importance FROM research_notes WHERE ts >= ? AND importance >= 1"
            f" AND symbol IN ({','.join('?' * len(held))}) ORDER BY importance DESC LIMIT 8",
            conn, params=(since, *held))
    risk = pd.read_sql_query("SELECT text FROM activity WHERE book=? AND agent='Risk engine' AND ts LIKE ?",
                             conn, params=(name, f"{day.isoformat()}%"))
    conn.close()

    def vs(a: float, b: float) -> str:  # comparisons are decided by code, not the LLM
        gap = (a - b) * 100
        return (f"beat the equal-weight market by {gap:.2f} percentage points" if gap > 0.005 else
                f"lagged the equal-weight market by {-gap:.2f} percentage points" if gap < -0.005 else
                "matched the equal-weight market")

    day_ret, since_ret = e_now / e_prev - 1, e_now / book["capital"] - 1
    facts = {
        "book": book["label"], "status": book["status"], "date": day.strftime("%d %b %Y"),
        "equity": rs(e_now), "day_change": f"{rs(e_now - e_prev)} ({pct(e_now / e_prev - 1)})",
        "since_start": pct(e_now / book["capital"] - 1),
        "equal_weight_today": pct(ew_day), "equal_weight_since_start": pct(ew_since),
        "today_vs_market": vs(day_ret, ew_day),
        "since_start_vs_market": vs(since_ret, ew_since),
        "drawdown_from_peak": pct(e_now / peak - 1),
        "holdings": len(held), "cash_share": f"{float(eq.cash.iloc[-1]) / e_now:.0%}",
        "trades_today": [f"{r.side} {r.qty} {r.symbol} at Rs {r.price:,.2f}" for r in fills.itertuples()],
        "best_holding_today": [f"{s} {pct(r)}" for s, r in movers[-2:][::-1]],
        "worst_holding_today": [f"{s} {pct(r)}" for s, r in movers[:2]],
        "research_on_holdings": [f"{r.symbol}: {r.summary} (direction {r.direction:+d}, importance {r.importance})"
                                 for r in notes.itertuples()],
        "risk_engine": risk.text.tolist(),
    }
    return facts


def template_note(f: dict) -> tuple[str, str]:
    head = f"{f['book']}: {f['day_change']} today"
    body = (f"Equity {f['equity']}, {f['since_start']} since start vs equal-weight {f['equal_weight_since_start']}. "
            f"Today {f['day_change']} vs equal-weight {f['equal_weight_today']}. "
            f"{len(f['trades_today'])} trades. Drawdown from peak {f['drawdown_from_peak']}.")
    return head, body


class Manager:
    def __init__(self, llm: LoggedLLM, db_path: Path | str) -> None:
        self.llm, self.db_path = llm, db_path
        self.conn = connect(db_path)
        self.conn.executescript(SCHEMA)
        self.conn.executescript(RESEARCH_SCHEMA)  # reads research notes
        self.activity = ActivityLog(db_path)

    def write(self, book: dict, day: date, closes: pd.DataFrame, ew: pd.Series) -> dict | None:
        if self.conn.execute("SELECT 1 FROM daily_notes WHERE date=? AND book=?",
                             (day.isoformat(), book["name"])).fetchone():
            return None
        facts = gather_facts(self.db_path, book, day, closes, ew)
        facts_text = json.dumps(facts, ensure_ascii=False, indent=1)
        head, note, flags, checked, model = None, None, [], 0, None
        if self.llm.available():
            for _ in range(2):
                try:
                    out = self.llm.json(AGENT, SYSTEM, f"Facts:\n{facts_text}\n\nWrite today's note.",
                                        NOTE_SCHEMA, max_tokens=500)
                except LLMUnavailable:
                    break
                text = out["headline"] + " " + out["note"] + " " + " ".join(f["text"] for f in out["flags"])
                if numbers_ok(text, facts_text):
                    head, note, flags, checked, model = out["headline"], out["note"], out["flags"], 1, self.llm.name
                    break
        if note is None:
            head, note = template_note(facts)
            if self.llm.available():
                self.activity.log("alert", AGENT, f"{book['label']}: LLM note used numbers not in the facts; "
                                  "posted the plain template instead.", team=book["team"], book=book["name"])
        self.conn.execute(
            "INSERT OR REPLACE INTO daily_notes (date, book, headline, note, flags, facts, model, checked, created)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (day.isoformat(), book["name"], head, note, json.dumps(flags), facts_text, model, checked, now_iso()))
        self.conn.commit()
        ts = datetime.combine(day, datetime.min.time(), IST).replace(hour=16, minute=30)
        self.activity.log("think", AGENT, f"Daily note, {book['label']}: {head}. {note}", team=book["team"],
                          book=book["name"], ts=ts)
        for fl in flags:
            if fl["level"] in ("watch", "act"):
                self.activity.log("alert" if fl["level"] == "act" else "think", AGENT,
                                  f"Flag ({fl['level']}): {fl['text']}", team=book["team"], book=book["name"], ts=ts)
        return {"book": book["name"], "checked": checked}
