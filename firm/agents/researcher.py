"""Researcher: reads NSE announcements and news about our stocks, writes structured notes.

Every run picks up items the recorder saved since the last run:
  announcements  routine categories (newspaper ads, con-call notices, ESOPs ...) are filed
                 without the LLM; the rest get an LLM note
  news           headlines matched to our companies by name; FinBERT scores the tone,
                 the LLM decides which company it is really about and writes the note
Notes: (symbol, event type, direction -2..+2, importance 0..3, one-line summary).

Notes are research only: nothing here changes a live book. They are scored forward
(did the stock beat the market over the next 5 trading days in the noted direction?)
because a local LLM may already "know" what happened in past years.
"""
from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from firm.activity import ActivityLog, connect, now_iso
from firm.config import IST, RECORDER_DB_PATH
from firm.llm import LLMUnavailable, LoggedLLM

AGENT = "Researcher"
TEAM = "swing_stocks"

SCHEMA = """
CREATE TABLE IF NOT EXISTS research_notes (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,           -- when the information became public / known to us
    symbol TEXT NOT NULL,
    source TEXT NOT NULL,       -- nse | news
    source_id TEXT NOT NULL,
    category TEXT,
    event_type TEXT NOT NULL,
    direction INTEGER NOT NULL, -- -2 very negative .. +2 very positive for the stock
    importance INTEGER NOT NULL,-- 0 routine .. 3 likely to move the stock
    summary TEXT NOT NULL,
    finbert REAL,               -- headline tone, -1..1 (news only)
    model TEXT,
    created TEXT NOT NULL,
    UNIQUE (source, source_id, symbol)
);
CREATE INDEX IF NOT EXISTS notes_symbol_ts ON research_notes(symbol, ts);
CREATE TABLE IF NOT EXISTS agent_state (key TEXT PRIMARY KEY, value TEXT);
"""

ROUTINE = {
    "Copy of Newspaper Publication", "Analysts/Institutional Investor Meet/Con. Call Updates",
    "Investor Presentation", "Shareholders meeting", "ESOP/ESOS/ESPS", "Monitoring Agency Report",
    "Allotment of Securities", "Trading Window", "Statement of deviation(s) or variation(s) under Reg. 32",
    "Loss of Share Certificates", "Duplicate Share Certificate", "Record Date", "Book Closure",
    "Certificate under SEBI (Depositories and Participants) Regulations, 2018", "Updates",
    "Notice Of Shareholders Meetings-XBRL", "Clarification - Financial Results",
}
EVENT_TYPES = ["results", "guidance", "order_win", "acquisition_merger", "capital_raise",
               "dividend_buyback", "management_change", "regulatory_legal", "rating",
               "promoter_pledge_stake", "operations", "macro_sector", "routine", "other"]

NOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "event_type": {"type": "string", "enum": EVENT_TYPES},
        "direction": {"type": "integer", "minimum": -2, "maximum": 2},
        "importance": {"type": "integer", "minimum": 0, "maximum": 3},
        "summary": {"type": "string"},
    },
    "required": ["event_type", "direction", "importance", "summary"],
}
NEWS_SCHEMA = {
    "type": "object",
    "properties": {"notes": {"type": "array", "items": {
        "type": "object",
        "properties": {"symbol": {"type": "string"}, **NOTE_SCHEMA["properties"]},
        "required": ["symbol", "event_type", "direction", "importance", "summary"],
    }}},
    "required": ["notes"],
}

SYSTEM = (
    "You are an equity research analyst covering large Indian listed companies (NSE). "
    "Judge only from the text given; do not use outside knowledge of later events. "
    "direction: effect on the company's share price, -2 clearly negative, -1 mildly negative, "
    "0 neutral or unclear, +1 mildly positive, +2 clearly positive. "
    "importance: 0 routine paperwork, 1 minor, 2 notable, 3 likely to move the stock several percent. "
    "Roundups and lists naming many stocks ('stocks to watch', 'top gainers') are importance 0. "
    "summary: one plain-English sentence, at most 25 words, facts from the text only."
)

STOP_FIRST = {"adani", "bajaj", "bank", "bharat", "hdfc", "hindustan", "indian", "power", "state",
              "tata", "siemens", "vedanta", "jio", "max", "solar", "sun", "union", "coal", "oil",
              "tech", "asian", "avenue", "punjab", "jindal", "jsw", "power", "the"}
EXTRA_ALIASES = {
    "BHARTIARTL": ["Airtel"], "LT": ["L&T"], "SBIN": ["SBI"], "ETERNAL": ["Zomato"],
    "POWERINDIA": ["Hitachi Energy"], "HINDUNILVR": ["HUL"], "M&M": ["M&M", "Mahindra & Mahindra"],
    "BAJFINANCE": ["Bajaj Finance"], "DMART": ["DMart", "D-Mart"], "MAZDOCK": ["Mazagon Dock"],
    "INDIGO": ["IndiGo"], "IDEA": ["Vodafone Idea", "Vi"], "TMPV": ["Tata Motors"], "TMCV": ["Tata Motors"],
    "COALINDIA": ["Coal India"], "ONGC": ["ONGC"], "SUNPHARMA": ["Sun Pharma"], "ICICIBANK": ["ICICI Bank"],
}
SUFFIX = re.compile(r"\b(Ltd\.?|Limited|Company|Co\.|Corporation|Enterprise)\s*$", re.I)


def aliases(names: dict[str, str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for sym, name in names.items():
        full = SUFFIX.sub("", SUFFIX.sub("", name.strip())).strip(" .,")
        al = {full}
        first = full.split()[0].strip(".,'")
        if len(first) >= 4 and first.lower() not in STOP_FIRST:
            al.add(first)
        al.update(EXTRA_ALIASES.get(sym, []))
        out[sym] = sorted(al, key=len, reverse=True)
    return out


def match_companies(text: str, alias_map: dict[str, list[str]]) -> list[str]:
    hits = []
    for sym, als in alias_map.items():
        for a in als:
            flags = 0 if a.isupper() or len(a) <= 4 else re.I  # short/acronym aliases: exact case
            if re.search(rf"(?<![\w&]){re.escape(a)}(?![\w&])", text, flags):
                hits.append(sym)
                break
    return hits


def _clean_html(s: str | None) -> str:
    return re.sub(r"<[^>]+>", " ", s or "").replace("&nbsp;", " ").strip()


class Researcher:
    def __init__(self, llm: LoggedLLM, db_path: Path | str, recorder_db: Path | str = RECORDER_DB_PATH,
                 universe: dict[str, str] | None = None, use_finbert: bool = True,
                 max_llm_calls: int = 60) -> None:
        from firm.universe import company_names

        self.llm, self.db_path = llm, db_path
        self.conn = connect(db_path)
        self.conn.executescript(SCHEMA)
        self.rec = sqlite3.connect(str(recorder_db), timeout=30)
        self.names = universe or company_names()
        self.alias_map = aliases(self.names)
        self.activity = ActivityLog(db_path)
        self.use_finbert, self.max_llm_calls = use_finbert, max_llm_calls
        self.calls = 0
        self._budget = max_llm_calls

    # ------------------------------------------------------------ state

    def _get(self, key: str, default: str) -> str:
        row = self.conn.execute("SELECT value FROM agent_state WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def _set(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO agent_state (key, value) VALUES (?,?)", (key, value))
        self.conn.commit()

    def _save(self, ts, symbol, source, source_id, category, note, finbert=None, model=None) -> bool:
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO research_notes (ts, symbol, source, source_id, category, event_type,"
            " direction, importance, summary, finbert, model, created) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (ts, symbol, source, source_id, category, note["event_type"], int(note["direction"]),
             int(note["importance"]), note["summary"].strip(), finbert, model, now_iso()))
        self.conn.commit()
        return cur.rowcount > 0

    def _post(self, symbol: str, note: dict, ts: str, source: str) -> None:
        if note["importance"] < 2:
            return
        tone = {-2: "clearly negative", -1: "mildly negative", 0: "neutral", 1: "mildly positive",
                2: "clearly positive"}[int(note["direction"])]
        self.activity.log("think", AGENT, f"{symbol}: {note['summary']} ({tone}, importance "
                          f"{note['importance']}/3, from {source}).", team=TEAM, ts=ts,
                          details={"symbol": symbol, **note})

    # ------------------------------------------------------------ announcements

    def announcements(self, since: str) -> tuple[int, str]:
        rows = self.rec.execute(
            "SELECT id, symbol, company, category, text, exchange_time, first_seen FROM announcements"
            " WHERE first_seen > ? ORDER BY first_seen", (since,)).fetchall()
        saved, last = 0, since
        for aid, sym, company, cat, text, ex_time, seen in rows:
            if sym not in self.names:
                last = seen
                continue
            ts = ex_time or seen
            if cat in ROUTINE:
                note = {"event_type": "routine", "direction": 0, "importance": 0,
                        "summary": f"{cat}: {(text or '')[:150]}"}
                saved += self._save(ts, sym, "nse", str(aid), cat, note)
                last = seen
                continue
            if self.calls >= self._budget:
                break  # continue next run from here
            prompt = (f"Company: {self.names[sym]} (NSE: {sym})\nFiling category: {cat}\n"
                      f"Filing text: {(text or '')[:1500]}\n\nWrite the research note.")
            self.calls += 1
            note = self.llm.json(AGENT, SYSTEM, prompt, NOTE_SCHEMA, max_tokens=200)
            if self._save(ts, sym, "nse", str(aid), cat, note, model=self.llm.name):
                saved += 1
                self._post(sym, note, ts, "NSE filing")
            last = seen
        return saved, last

    # ------------------------------------------------------------ news

    def news(self, since: str) -> tuple[int, str]:
        rows = self.rec.execute(
            "SELECT id, source, title, summary, first_seen FROM news WHERE first_seen > ? ORDER BY first_seen",
            (since,)).fetchall()
        items = []
        for nid, src, title, summary, seen in rows:
            text = f"{title}. {_clean_html(summary) if src != 'gnews_markets' else ''}".strip()
            syms = match_companies(title + " " + text, self.alias_map)
            items.append((nid, src, title, text, seen, syms))
        tones = {}
        relevant = [it for it in items if it[5]]
        if self.use_finbert and relevant:
            try:
                from firm.llm.finbert import sentiment

                tones = dict(zip([it[0] for it in relevant], sentiment([it[2] for it in relevant])))
            except Exception as e:
                self.activity.log("alert", AGENT, f"FinBERT unavailable: {type(e).__name__}: {e}", team=TEAM)
                self.use_finbert = False
        saved, last = 0, since
        for nid, src, title, text, seen, syms in items:
            if not syms:
                last = seen
                continue
            if self.calls >= self._budget:
                break
            cands = "\n".join(f"- {s}: {self.names[s]}" for s in syms)
            prompt = (f"Headline: {text[:800]}\n\nCompanies that may be mentioned:\n{cands}\n\n"
                      "Return a note only for companies this news is actually about (it may be none). "
                      "Use the NSE symbol exactly as listed.")
            self.calls += 1
            out = self.llm.json(AGENT, SYSTEM, prompt, NEWS_SCHEMA, max_tokens=350)
            for note in out.get("notes", []):
                sym = note.get("symbol", "").strip().upper()
                if sym not in syms:
                    continue  # the model may only pick from the candidates
                if self._save(seen, sym, "news", str(nid), src, note, tones.get(nid), self.llm.name):
                    saved += 1
                    self._post(sym, note, seen, "news")
            last = seen
        return saved, last

    # ------------------------------------------------------------ run

    def run(self, now: datetime | None = None) -> dict:
        now = now or datetime.now(IST)
        start = (now - timedelta(days=1)).isoformat(timespec="seconds")  # first run: last 24h only
        self.calls = 0
        if not self.llm.available():
            self.activity.status(AGENT, "idle", "LLM offline: start Ollama to resume reading", TEAM)
            return {"status": "skipped", "saved": 0, "msg": "LLM offline"}
        self.activity.status(AGENT, "working", "Reading new filings and headlines", TEAM)
        out = {}
        try:
            # Filings get 60% of the calls; news always gets the rest, so a filing
            # backlog cannot starve it.
            for key, fn, share in (("announcements", self.announcements, 0.6), ("news", self.news, 1.0)):
                self._budget = max(1, int(self.max_llm_calls * share))
                since = self._get(f"researcher.{key}", start)
                saved, last = fn(since)
                self._set(f"researcher.{key}", last)
                out[key] = saved
        except LLMUnavailable as e:
            self.activity.status(AGENT, "error", str(e)[:200], TEAM)
            return {"status": "error", "saved": sum(out.values()), "msg": str(e)}
        n = sum(out.values())
        self.activity.status(AGENT, "idle", f"Last read {now:%H:%M}: {out.get('announcements', 0)} filing notes, "
                             f"{out.get('news', 0)} news notes ({self.calls} LLM calls)", TEAM)
        return {"status": "ok", "saved": n, "msg": f"{out} LLM calls {self.calls}"}
