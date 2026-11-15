"""Home: where to start. Says in plain words whether things are fine, what (if anything) you need to do
today, how the practice money is doing, and where everything else lives."""
from __future__ import annotations

import html
import sqlite3
from datetime import date, datetime
from pathlib import Path

import streamlit as st

from dashboard.control_page import chip, fix, fyers_state, fyers_section, live_state, ago, GOOD, MUTED
from firm.calendar import TradingCalendar
from firm.config import IST, RECORDER_DB_PATH
from firm.control import Control
from firm.control_status import attention_items, last_job_runs
from firm.live.books import PROBATION_START
from firm.live.runner import LIVE_DIR
from firm.live.sandbox import LAUNCH
from firm.todo import coming_up, days_until, greeting, todays_tasks

PRACTICE = {"sandbox_A": "Practice A: learned from history", "sandbox_B": "Practice B: learns as it goes",
            "sandbox_C": "Practice C: random trades (the control)", "sandbox_D": "Practice D: just holds cash"}

MAP = [
    ("🏠 Home", "This page. Start here."),
    ("📡 Live feed", "A running diary of what every program and AI helper is doing, in plain sentences."),
    ("💰 Money", "How the paper money is doing: the practice accounts and the main swing experiment."),
    ("📝 Notes", "The AI helpers' daily notes, what they read, and how right they turned out to be."),
    ("⚙️ Settings", "All the switches and buttons: log in to Fyers, start and stop programs, pause things."),
    ("🔬 Research · 🩺 System", "For later: the test results behind every decision, and the health of the data."),
]
WORDS = [
    ("Paper money", "Make-believe money. Nothing here ever places a real order."),
    ("Practice accounts", "Four make-believe ₹1 lakh accounts testing ideas for buying and selling within a day. "
                          "They compare an AI trained on history, an AI that learns as it goes, a random trader and plain cash."),
    ("Swing experiment", "The main test: a stock-picking method that holds shares for weeks. It starts on 30 Nov."),
    ("Data collector", "The program running in the background that saves prices, news, company filings and results."),
    ("AI helpers", "Programs that read news and write notes. They never place orders."),
]


def _equity(book: str) -> float | None:
    path = LIVE_DIR / "firm.db"
    if not Path(path).exists():
        return None
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10) as c:
            row = c.execute("SELECT equity FROM equity WHERE team=? ORDER BY ts DESC LIMIT 1", (book,)).fetchone()
    except sqlite3.OperationalError:
        return None
    return float(row[0]) if row else None


def _recent(n: int = 6) -> list[tuple[str, str, str]]:
    path = LIVE_DIR / "firm.db"
    if not Path(path).exists():
        return []
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10) as c:
            rows = c.execute("SELECT ts, agent, text FROM activity ORDER BY id DESC LIMIT ?", (n,)).fetchall()
    except sqlite3.OperationalError:
        return []
    return [(ts, agent, text) for ts, agent, text in rows]


def render() -> None:
    ctl = Control()
    now = datetime.now(IST)
    cal = TradingCalendar(refresh=False)
    state = live_state()

    st.title(f"{greeting(now)}")
    st.caption(f"{now:%A %d %B}, {now:%H:%M}. Everything here is paper money: make-believe, nothing real is ever traded.")

    # ---- is everything fine?
    trading = cal.is_trading_day(now.date())
    needs_check = trading and now.time() < datetime(2000, 1, 1, 15, 30).time()
    fyers_ok = fyers_state()[0] if needs_check else None
    st.session_state["fyers_checked"] = fyers_ok
    items = attention_items(ctl, state["running"], state["ollama"], fyers_ok, now, last_job_runs(RECORDER_DB_PATH))
    warn = [i for i in items if i.level == "warn"]
    if not warn:
        st.success("Everything is running normally.", icon="✅")
    else:
        st.warning("1 thing needs you." if len(warn) == 1 else f"{len(warn)} things need you.", icon="⚠️")
        for it in warn:
            c1, c2 = st.columns([5, 1])
            c1.markdown("• " + it.text)
            c2.button(it.fix, key=f"home_fix_{it.key}", on_click=fix, args=(it.key, ctl), use_container_width=True)

    # ---- what to do today
    st.subheader("What you need to do today")
    tasks = todays_tasks(now, trading, fyers_ok, LAUNCH, PROBATION_START)
    for t in tasks:
        mark = "✅" if t.done else "⬜"
        st.markdown(f"**{mark}  {t.title}**" + ("" if not t.info else ""))
        st.caption(t.detail)
        if t.action == "login" and not t.done:
            with st.expander("Log in to Fyers now", expanded=True):
                fyers_section(ctl, compact=True)
    up = coming_up(now.date(), LAUNCH, PROBATION_START)
    if up:
        st.markdown("**Coming up**")
        for c in up:
            st.markdown(f"- **{c.day:%a %d %b}** ({days_until(now.date(), c.day)}): {c.text}")

    # ---- the money
    st.subheader("How the money is doing")
    cols = st.columns(5)
    cards = [("Swing experiment", _equity("swing_probation"), f"Starts {PROBATION_START:%d %b}", "the main test")]
    for book, label in PRACTICE.items():
        cards.append((label.split(":")[0], _equity(book), f"Starts {LAUNCH:%d %b}", label.split(": ", 1)[1]))
    for col, (title, eq, starts, what) in zip(cols, cards):
        with col.container(border=True):
            st.markdown(f"**{title}**")
            if eq:
                st.markdown(f"<span style='font-size:1.5rem;font-weight:600'>₹{eq:,.0f}</span>", unsafe_allow_html=True)
                st.caption(what)
            else:
                st.markdown(f"<span style='color:{MUTED}'>Not started yet</span>", unsafe_allow_html=True)
                st.caption(starts)
    st.caption("Open **Money** in the sidebar for charts and details.")

    # ---- the Head
    from firm.agents.head import latest

    st.subheader("What the Head says")
    head = latest(LIVE_DIR / "firm.db")
    if head:
        st.caption(f"Report of {head[0]}. The Head only advises; it never moves money.")
        for line in head[1].splitlines()[1:]:
            st.markdown(line)
    else:
        st.caption("No report yet. The Head writes one each evening after the daily notes.")

    # ---- what is it doing right now
    st.subheader("What the system is doing right now")
    rows = _recent()
    if not rows:
        st.caption("Nothing yet.")
    for ts, agent, text in rows:
        st.markdown(f"<span style='color:{MUTED}'>{ts[11:16]}</span>  **{html.escape(agent)}**: {html.escape(text[:170])}",
                    unsafe_allow_html=True)
    st.caption("The **Live feed** page shows all of it, as it happens.")

    # ---- where is everything
    st.subheader("Where is everything?")
    with st.expander("A map of this app, and what the words mean", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Pages (left sidebar)**")
            for page, what in MAP:
                st.markdown(f"{page}: {what}")
        with c2:
            st.markdown("**Words used here**")
            for word, what in WORDS:
                st.markdown(f"**{word}**: {what}")

    # ---- emergency
    st.divider()
    halted = ctl.get_bool("halt.firm")
    if halted:
        st.error("Trading is on HOLD: no new positions anywhere.", icon="🛑")
        st.button("▶  Resume trading", key="home_resume", type="primary", on_click=fix, args=("resume_halt", ctl))
    else:
        st.button("⏹  Emergency stop: no new trading", key="home_halt",
                  on_click=lambda: (Control().set("halt.firm", True), st.toast("Trading on hold.")))
        st.caption("Blocks every new paper position everywhere. One click to resume. It is safe to press.")
