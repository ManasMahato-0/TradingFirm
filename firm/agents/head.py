"""The Head, SHADOW MODE. Reads every book, writes one plain firm report per day and an allocation PROPOSAL.

It moves no money and cannot: it imports no broker, risk engine or runner, only reads the databases and writes
its own `head_reports` table (a test enforces this). No language model is involved: the text is a template over
numbers read from the data, so nothing in it can be invented.

Allocation rules (pre-registered in docs/PLAN.md, "Head v0"):
  * A team gets capital only if it has PASSED its own pre-registered test. Today none has, so the proposal is
    "hold": every team stays on its own Rs 1 lakh of paper money and nothing is reallocated.
  * If teams have passed: equal split across them, at most 50% of firm capital each, the rest in cash.
  * Parked teams (research failed, or no data) always get 0.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from firm.live.sandbox_eval import daily_equity, evaluate

PROBATION_END = date(2027, 5, 31)
MAX_WEIGHT = 0.5
PROMOTE_DD, CUT_DD = -0.25, -0.25
PROMOTE_GAP, CUT_GAP = 0.05, 0.10          # vs equal-weight Nifty 100, from PLAN.md

SCHEMA = """
CREATE TABLE IF NOT EXISTS head_reports (
    day TEXT PRIMARY KEY, written TEXT NOT NULL, proposal TEXT NOT NULL, report TEXT NOT NULL
)"""

PARKED = [
    ("intraday_options", "Intraday options", "no option price history exists to test on; the recorder is collecting it"),
    ("overnight_options", "Overnight options", "every pre-registered baseline failed its test"),
]


@dataclass
class TeamView:
    team: str
    label: str
    state: str                   # not_started | in_test | passed | failed | parked
    eligible: bool
    detail: str
    equity: float | None = None
    ret: float | None = None


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def swing_view(eq: pd.Series, ew_ret: float | None, blocks: int, today: date) -> TeamView:
    label = "Swing stocks"
    if eq.empty:
        return TeamView("swing_stocks", label, "not_started", False, "its paper test has not started yet")
    ret = float(eq.iloc[-1] / eq.iloc[0] - 1)
    dd = float((eq / eq.cummax() - 1).min())
    base = dict(equity=float(eq.iloc[-1]), ret=ret)
    gap = None if ew_ret is None else ret - ew_ret
    if dd < CUT_DD or (gap is not None and gap < -CUT_GAP):
        why = f"drawdown {_pct(dd)}" if dd < CUT_DD else f"{_pct(gap)} behind the equal-weight benchmark"
        return TeamView("swing_stocks", label, "failed", False, f"cut rule hit: {why}", **base)
    if today < PROBATION_END:
        return TeamView("swing_stocks", label, "in_test",  False,
                        f"{len(eq)} days in, return {_pct(ret)}, worst drop {_pct(dd)}; the test runs to {PROBATION_END:%d %b %Y}",
                        **base)
    ok = gap is not None and gap >= -PROMOTE_GAP and dd > PROMOTE_DD and blocks == 0
    if ok:
        return TeamView("swing_stocks", label, "passed", True,
                        f"passed on numbers (return {_pct(ret)}, drop {_pct(dd)}); the owner's manual trades-match check is still required",
                        **base)
    return TeamView("swing_stocks", label, "failed", False, f"did not meet the promotion rules (return {_pct(ret)}, drop {_pct(dd)}, "
                    f"{blocks} risk blocks)", **base)


def sandbox_view(result: dict) -> TeamView:
    label = "Intraday stocks"
    days = result.get("days", 0)
    if not days:
        return TeamView("intraday_stocks", label, "not_started", False, "the practice accounts have not started yet")
    acc = result["accounts"]
    best = max(("sandbox_A", "sandbox_B"), key=lambda b: acc[b]["equity"])
    base = dict(equity=acc[best]["equity"], ret=acc[best]["return"])
    cp = result.get("checkpoint_reached")
    if cp and any(acc[b]["works"] for b in ("sandbox_A", "sandbox_B")):
        who = [b[-1] for b in ("sandbox_A", "sandbox_B") if acc[b]["works"]]
        return TeamView("intraday_stocks", label, "passed", True, f"practice account {' and '.join(who)} passed the day-{cp} checkpoint", **base)
    if cp and not result.get("next_checkpoint"):
        return TeamView("intraday_stocks", label, "failed", False, f"no practice account passed the day-{cp} checkpoint", **base)
    nxt = result.get("next_checkpoint")
    tail = f"; next checkpoint at day {nxt}" if nxt else ""
    return TeamView("intraday_stocks", label, "in_test", False, f"{days} recorded days so far{tail}", **base)


def allocate(views: list[TeamView]) -> dict:
    eligible = [v for v in views if v.eligible]
    if not eligible:
        return {"action": "hold", "weights": {},
                "why": "No team has passed its own test yet, so there is nothing to move. Every team stays on its own paper money."}
    w = min(MAX_WEIGHT, 1 / len(eligible))
    weights = {v.team: w for v in eligible}
    cash = 1 - sum(weights.values())
    names = " and ".join(v.label for v in eligible)
    return {"action": "reallocate", "weights": weights, "cash": cash,
            "why": f"{names} passed; split evenly, at most {MAX_WEIGHT:.0%} each, {cash:.0%} left in cash. Parked teams get nothing."}


def report_text(day: date, views: list[TeamView], plan: dict, halted: bool, blocks: int) -> str:
    lines = [f"Firm report for {day:%A %d %B %Y} (the Head only advises; it moves no money)."]
    lines.append("Trading is on hold by the owner." if halted else "Trading is not on hold.")
    for v in views:
        money = "" if v.equity is None else f" Paper equity Rs {v.equity:,.0f} ({_pct(v.ret)})."
        lines.append(f"- {v.label}: {v.state.replace('_', ' ')}. {v.detail}.{money}")
    for _, label, why in PARKED:
        lines.append(f"- {label}: parked: {why}.")
    lines.append(f"Risk engine blocks on the swing test so far: {blocks}.")
    lines.append(f"Proposal: {plan['action'].upper()}. {plan['why']}")
    return "\n".join(lines)


def _blocks(db: Path | str, book: str) -> int:
    from firm.live.tracker import risk_blocks

    return risk_blocks(db, book)


def write_report(db: Path | str, day: date, ew_ret: float | None = None, halted: bool = False) -> dict:
    """Gather, decide, store. Returns the stored row as a dict."""
    views = [swing_view(daily_equity(db, "swing_probation"), ew_ret, _blocks(db, "swing_probation"), day),
             sandbox_view(evaluate(db))]
    plan = allocate(views)
    text = report_text(day, views, plan, halted, _blocks(db, "swing_probation"))
    with sqlite3.connect(str(db), timeout=30) as c:
        c.execute(SCHEMA)
        c.execute("INSERT OR REPLACE INTO head_reports VALUES (?,?,?,?)",
                  (day.isoformat(), pd.Timestamp.now(tz="Asia/Kolkata").isoformat(timespec="seconds"),
                   json.dumps({"plan": plan, "teams": [asdict(v) for v in views]}), text))
    return {"day": day.isoformat(), "plan": plan, "teams": views, "report": text}


def latest(db: Path | str) -> tuple[str, str] | None:
    """(day, report text) of the newest report, or None."""
    if not Path(db).exists():
        return None
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10) as c:
            row = c.execute("SELECT day, report FROM head_reports ORDER BY day DESC LIMIT 1").fetchone()
    except sqlite3.OperationalError:
        return None
    return row
