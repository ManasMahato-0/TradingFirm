import json
import sqlite3
from datetime import date, datetime

import numpy as np
import pandas as pd

from firm.activity import ActivityLog
from firm.agents.manager import Manager, numbers_ok
from firm.agents.researcher import Researcher, aliases, match_companies
from firm.agents.trader import TraderGuard
from firm.config import IST
from firm.llm import LoggedLLM
from firm.recorders.db import RecorderDB
from firm.storage import Storage

NAMES = {"TCS": "Tata Consultancy Services Ltd.", "SBIN": "State Bank of India",
         "INFY": "Infosys Ltd.", "BHARTIARTL": "Bharti Airtel Ltd.", "TMPV": "Tata Motors Passenger Vehicles Ltd."}


class FakeLLM:
    name = "fake"

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def available(self):
        return True

    def json(self, system, prompt, schema, max_tokens=400):
        self.prompts.append(prompt)
        r = self.replies.pop(0)
        return r(prompt) if callable(r) else r


def test_alias_matching():
    al = aliases(NAMES)
    assert match_companies("Infosys wins $1bn deal", al) == ["INFY"]
    assert match_companies("SBI hikes lending rates", al) == ["SBIN"]
    assert match_companies("Airtel tariffs up 10%", al) == ["BHARTIARTL"]
    assert "TMPV" in match_companies("Tata Motors sales jump", al)
    assert match_companies("Nifty ends flat; banks drag", al) == []
    assert match_companies("State of the market", al) == []   # 'State' alone is not SBI


def test_researcher_routes_filings_and_news(tmp_path):
    rec = RecorderDB(tmp_path / "rec.db")
    now = datetime(2026, 11, 20, 12, tzinfo=IST)
    seen = "2026-11-20T10:00:00+05:30"
    rec.conn.executemany(
        "INSERT INTO announcements (id, symbol, company, category, text, exchange_time, first_seen)"
        " VALUES (?,?,?,?,?,?,?)",
        [("a1", "TCS", "TCS", "Copy of Newspaper Publication", "ad", seen, seen),
         ("a2", "TCS", "TCS", "Outcome of Board Meeting", "Q2 results: revenue up 8%", seen, seen),
         ("a3", "XYZ", "Other", "Outcome of Board Meeting", "not our stock", seen, seen)])
    rec.conn.executemany("INSERT INTO news (id, source, title, summary, link, published, first_seen)"
                         " VALUES (?,?,?,?,?,?,?)",
                         [("n1", "et", "Infosys wins large deal", "", "", seen, seen),
                          ("n2", "et", "Rupee ends flat", "", "", seen, seen)])
    rec.conn.commit()
    llm = FakeLLM([
        {"event_type": "results", "direction": 1, "importance": 2, "summary": "Revenue grew 8%."},
        {"notes": [{"symbol": "INFY", "event_type": "order_win", "direction": 1, "importance": 2,
                    "summary": "Won a large deal."},
                   {"symbol": "WIPRO", "event_type": "other", "direction": 0, "importance": 1,
                    "summary": "not a candidate"}]},
    ])
    db = tmp_path / "firm.db"
    r = Researcher(LoggedLLM(llm, db), db, recorder_db=tmp_path / "rec.db", universe=NAMES, use_finbert=False)
    out = r.run(now=now)
    assert out["status"] == "ok" and len(llm.prompts) == 2       # routine filing skipped the LLM
    notes = sqlite3.connect(db).execute("SELECT symbol, event_type, importance FROM research_notes"
                                        " ORDER BY symbol, event_type").fetchall()
    assert notes == [("INFY", "order_win", 2), ("TCS", "results", 2), ("TCS", "routine", 0)]
    assert r.run(now=now)["saved"] == 0                           # watermark: nothing twice
    calls = sqlite3.connect(db).execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0]
    assert calls == 2


def test_number_check():
    facts = json.dumps({"equity": "Rs 101,234", "day": "+0.42%", "ew": "-0.10%"})
    assert numbers_ok("Equity rose to Rs 101,234, up +0.42% vs -0.10% for the market.", facts)
    assert not numbers_ok("Equity rose to Rs 101,300, up 0.5%.", facts)


def _book_db(tmp_path):
    db = tmp_path / "firm.db"
    st = Storage(db)
    for d, e in [("2026-12-01", 100_000.0), ("2026-12-02", 101_000.0)]:
        st.record_equity(datetime.fromisoformat(d + "T15:30:00+05:30"), "bk", e, 10_000.0)
    st.record_positions("bk", datetime(2026, 12, 2, 17, tzinfo=IST), [("TCS", 2, 3000.0, 3100.0)])
    book = {"name": "bk", "team": "swing_stocks", "label": "Test book", "status": "probation",
            "capital": 100_000.0, "start": "2026-11-30"}
    idx = pd.to_datetime(["2026-12-01", "2026-12-02"]).tz_localize("Asia/Kolkata")
    closes = pd.DataFrame({"TCS": [3000.0, 3100.0]}, index=idx)
    ew = pd.Series([0.001, 0.002], index=pd.to_datetime(["2026-12-01", "2026-12-02"]))
    return db, book, closes, ew


def test_manager_rejects_invented_numbers(tmp_path):
    db, book, closes, ew = _book_db(tmp_path)
    bad = {"headline": "Up 5% today", "note": "Equity hit Rs 999,999.", "flags": []}
    m = Manager(LoggedLLM(FakeLLM([bad, bad]), db), db)
    m.write(book, date(2026, 12, 2), closes, ew)
    row = sqlite3.connect(db).execute("SELECT checked, note FROM daily_notes").fetchone()
    assert row[0] == 0 and "Rs 101,000" in row[1]      # template used, real numbers


def test_manager_accepts_grounded_note(tmp_path):
    db, book, closes, ew = _book_db(tmp_path)

    def good(prompt):
        f = json.loads(prompt.split("Facts:\n", 1)[1].split("\n\nWrite", 1)[0])
        return {"headline": "A steady day", "note": f"Equity {f['equity']}, {f['since_start']} since start.",
                "flags": []}

    Manager(LoggedLLM(FakeLLM([good]), db), db).write(book, date(2026, 12, 2), closes, ew)
    assert sqlite3.connect(db).execute("SELECT checked FROM daily_notes").fetchone()[0] == 1


def test_guard_is_shadow_and_only_reduces(tmp_path):
    db, book, closes, ew = _book_db(tmp_path)
    g = TraderGuard(LoggedLLM(FakeLLM([{"action": "skip_buy", "reason": "held, so cannot skip"}]), db), db)
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO research_notes (ts, symbol, source, source_id, event_type, direction, importance,"
                 " summary, created) VALUES ('2026-12-02T10:00', 'TCS', 'nse', 'x', 'regulatory_legal', -2, 3,"
                 " 'SEBI order against the company.', 'now')")
    conn.commit()
    out = g.review(book, date(2026, 12, 2), closes)
    assert out == [{"symbol": "TCS", "action": "none"}]          # skip_buy on a holding is invalid
    row = conn.execute("SELECT shadow, action FROM guard_proposals").fetchone()
    assert row == (1, "none")
    kinds = {k for _, _, _, k, _, _ in ActivityLog(db).recent(50)}
    assert "think" in kinds


def test_scoreboard_uses_next_open_and_direction(tmp_path):
    from firm.agents.scoreboard import score_notes

    idx = pd.bdate_range("2026-12-01", periods=10, tz="Asia/Kolkata")
    opens = pd.DataFrame({"A": np.full(10, 100.0), "B": np.full(10, 100.0)}, index=idx)
    closes = opens.copy()
    closes["A"] = [100, 100, 100, 100, 100, 110, 110, 110, 110, 110.0]  # A jumps on day 5
    db = tmp_path / "firm.db"
    from firm.agents.researcher import SCHEMA
    conn = sqlite3.connect(db)
    conn.executescript(SCHEMA)
    conn.execute("INSERT INTO research_notes (ts, symbol, source, source_id, event_type, direction, importance,"
                 " summary, created) VALUES ('2026-12-02T11:00:00+05:30', 'A', 'nse', '1', 'other', 1, 2, 's', 'x')")
    conn.commit()
    s = score_notes(db, opens, closes)
    assert len(s) == 1 and s.right.iloc[0]        # entry 3 Dec open, exit 9 Dec close: A beat B
    assert s.excess_5d.iloc[0] > 0
