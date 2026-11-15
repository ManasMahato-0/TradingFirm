"""The Control page: every switch, in plain words, operated from here.

Design rules: one thing per row, a short sentence saying what it does, changes apply by themselves
(no Save button), the emergency halt is one click, anything disruptive asks first, and the things that
are locked by pre-registration are shown (not hidden) with the reason. Always writes to the live
control board, whatever data mode the sidebar shows.
"""
from __future__ import annotations

import html
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

from firm import fyers_auth, ops
from firm.config import IST, RECORDER_DB_PATH
from firm.control import BOOKS, JOBS, LOCKED, SERVICES, SWITCHES, Control
from firm.control_status import attention_items, last_job_runs

SECTIONS = ["Switches", "Programs", "Fyers login", "More"]
GOOD, WARN, BAD, MUTED = "#0ca30c", "#fab219", "#d03b3b", "#898781"


def chip(text: str, color: str) -> str:
    return (f'<span style="display:inline-block;font-size:.75rem;padding:1px 8px;border-radius:9px;'
            f'border:1px solid {color};color:{color};font-weight:600">{html.escape(text)}</span>')


def ago(iso: str | None) -> str:
    if not iso:
        return "never"
    try:
        s = (datetime.now(IST) - datetime.fromisoformat(iso)).total_seconds()
    except ValueError:
        return iso
    return "just now" if s < 90 else f"{s / 60:.0f} min ago" if s < 3600 else f"{s / 3600:.0f} h ago" if s < 172800 else f"{s / 86400:.0f} days ago"


def toast(msg: str) -> None:
    st.toast(msg)


# ---------------------------------------------------------------- cached facts

@st.cache_data(ttl=4, show_spinner=False)
def live_state() -> dict:
    return {"running": {n: ops.is_running(n) for n in SERVICES}, "ollama": ops.ollama_running()}


@st.cache_data(ttl=60, show_spinner=False)
def fyers_state() -> tuple[bool, str]:
    return fyers_auth.token_status()


def refresh_state() -> None:
    live_state.clear()


# ---------------------------------------------------------------- one-click fixes

def fix(key: str, ctl: Control) -> None:
    if key == "resume_halt":
        ctl.set("halt.firm", False)
        toast("Trading resumed.")
    elif key in ("start_recorder", "start_sandbox"):
        name = key.split("_", 1)[1]
        toast(f"{SERVICES[name][0]}: {ops.start_service(name, ctl)}")
    elif key == "start_ollama":
        toast("AI model: " + ops.start_ollama())
    elif key == "login":
        st.session_state["ctl_section"] = "Fyers login"
    elif key == "show_books":
        st.session_state["ctl_section"] = "Switches"
    elif key in ("show_jobs", "show_failed_jobs"):
        st.session_state["ctl_section"] = "Switches"
    refresh_state()


# ---------------------------------------------------------------- switches

def toggle(ctl: Control, key: str, label: str, help_: str = "", invert: bool = False, disabled: bool = False) -> None:
    """A switch that applies the moment you flip it. invert: the switch shows the opposite of the stored
    value (e.g. 'Paused' on = stored paused true is already direct; used for 'enabled' wording if needed)."""
    wk = f"w_{key}"
    st.session_state[wk] = ctl.get_bool(key) != invert          # the board is the truth

    def changed() -> None:
        on = st.session_state[wk]
        c = Control()
        c.set(key, (not on) if invert else on)
        st.toast(f"{label}: {'on' if on else 'off'}")

    st.toggle(label, key=wk, help=help_ or None, on_change=changed, disabled=disabled)


# ---------------------------------------------------------------- top strip and halt

@st.fragment(run_every=10)
def status_strip() -> None:
    ctl = Control()
    state = live_state()
    ok = st.session_state.get("fyers_checked")
    items = attention_items(ctl, state["running"], state["ollama"], ok, datetime.now(IST),
                            last_job_runs(RECORDER_DB_PATH))
    warn = [i for i in items if i.level == "warn"]
    info = [i for i in items if i.level == "info"]
    if not warn:
        st.success("All systems normal." + ("" if not info else " (See notes below.)"), icon="✅")
    else:
        st.warning(f"{len(warn)} thing(s) need your attention", icon="⚠️")
    for it in warn + info:
        c1, c2 = st.columns([5, 1])
        c1.markdown(("🟠 " if it.level == "warn" else "ℹ️ ") + it.text)
        c2.button(it.fix, key=f"fix_{it.key}", on_click=fix, args=(it.key, ctl), use_container_width=True)

    halted = ctl.get_bool("halt.firm")
    if halted:
        wins = ctl.windows("halt.firm")
        since = wins[-1][1] if wins else None
        st.error(f"MASTER HALT IS ON since {ago(since)}. No new positions are opened anywhere. "
                 "Existing positions can still be reduced.", icon="🛑")
        st.button("▶  Resume trading", key="resume", type="primary", on_click=fix, args=("resume_halt", ctl))
    else:
        c1, c2 = st.columns([1, 3])
        c1.button("⏹  Halt all new trading", key="halt", type="secondary",
                  on_click=lambda: (Control().set("halt.firm", True), st.toast("Master halt ON.")))
        c2.caption("Emergency stop: blocks every new position in every book at once. Reducing and the end-of-day "
                   "flatten still work. One click to stop, one click to resume.")


# ---------------------------------------------------------------- sections

@st.dialog("Are you sure?")
def confirm_stop(name: str, typed: bool) -> None:
    label = SERVICES[name][0] if name in SERVICES else "the local AI model"
    st.write(f"This stops **{label}**. " + {
        "recorder": "Nothing is collected (headlines, filings, prices, option chains) and the evening jobs do not run "
                    "until you start it again.",
        "sandbox": "The four intraday sandbox accounts stop trading. Their state is saved and a restart resumes. "
                   "A day with no session is skipped.",
        "ollama": "Notes and summaries pause until it is started again.",
    }[name])
    ok = True
    if typed:
        ok = st.text_input("This is a live session. Type STOP to confirm.").strip().upper() == "STOP"
    c1, c2 = st.columns(2)
    if c1.button(f"Stop {label}", type="primary", disabled=not ok):
        with st.spinner("Stopping..."):
            ctl = Control()
            res = ops.stop_ollama() if name == "ollama" else ops.stop_service(name, ctl, wait_s=60)
        refresh_state()
        st.toast(f"{label}: {res}")
        st.rerun()
    if c2.button("Cancel"):
        st.rerun()


def in_live_session(ctl: Control) -> bool:
    now = datetime.now(IST)
    beat = ctl.last_beat("sandbox")
    fresh = beat is not None and (now - beat).total_seconds() < 180
    return fresh and now.weekday() < 5 and (9, 15) <= (now.hour, now.minute) < (15, 30)


def services_section(ctl: Control) -> None:
    st.caption("The programs that run in the background. They start by themselves when you log in to Windows.")
    state = live_state()
    cols = st.columns(3)
    for col, name in zip(cols, list(SERVICES) + ["ollama"]):
        label, help_ = SERVICES.get(name, ("Local AI model", "Ollama: writes notes and summaries. It never trades."))
        up = state["ollama"] if name == "ollama" else state["running"][name]
        with col.container(border=True):
            st.markdown(f"**{label}**  " + chip("running" if up else "stopped", GOOD if up else BAD), unsafe_allow_html=True)
            st.caption(help_)
            if name != "ollama":
                beat = ctl.last_beat(name)
                st.caption(f"Last sign of life: {ago(beat.isoformat()) if beat else 'never'}")
            b1, b2, b3 = st.columns(3)
            if not up:
                b1.button("Start", key=f"start_{name}", on_click=fix, args=(f"start_{name}", ctl), use_container_width=True)
            else:
                if b1.button("Stop", key=f"stop_{name}", use_container_width=True):
                    confirm_stop(name, typed=(name == "sandbox" and in_live_session(ctl)))
                if b2.button("Restart", key=f"restart_{name}", use_container_width=True):
                    with st.spinner("Restarting..."):
                        if name == "ollama":
                            ops.stop_ollama()
                            res = ops.start_ollama()
                        else:
                            ops.stop_service(name, ctl, wait_s=60)
                            res = ops.start_service(name, ctl)
                    refresh_state()
                    st.toast(f"{label}: {res}")
                    st.rerun()
    st.caption("If you reboot the laptop, anything you switched off here stays off until you start it.")


def job_row(ctl: Control, job: str, runs: dict) -> None:
    label, help_ = JOBS[job]
    c1, c2 = st.columns([4, 2])
    with c1:
        toggle(ctl, f"job.{job}.enabled", label, help_)
    when, status = runs.get(job, (None, None))
    color = {"ok": GOOD, "skipped": MUTED, "partial": WARN}.get(status or "", BAD if status else MUTED)
    c2.markdown(chip(status or "not run yet", color) + f" <span style='color:{MUTED};font-size:.8rem'>{ago(when)}</span>",
                unsafe_allow_html=True)


def data_section(ctl: Control) -> None:
    runs = last_job_runs(RECORDER_DB_PATH)
    st.markdown("##### Saving data")
    for job in ("news", "announcements", "board_meetings", "results", "options", "bars"):
        job_row(ctl, job, runs)
    st.markdown("##### The evening update")
    for job in ("runner", "desk"):
        job_row(ctl, job, runs)
    st.caption("The evening update needs the day's share prices. The daily notes need the evening update.")
    st.markdown("##### AI helpers")
    st.caption("They read and write. They never place orders.")
    job_row(ctl, "researcher", runs)
    guard_now = ctl.get("agent.guard")
    options = {"Watch only: note what it would do, never act": "shadow", "Off: do not run": "off"}
    pick = st.selectbox("Risk watcher", list(options), index=0 if guard_now != "off" else 1,
                        help="It watches for bad news on shares we hold and writes down what it would do. It can never act, "
                             "so the main experiment stays exactly as tested.")
    if options[pick] != guard_now:
        ctl.set("agent.guard", options[pick])
        st.toast(f"Risk watcher: {options[pick]}")
    toggle(ctl, "llm.enabled", SWITCHES["llm.enabled"].label, SWITCHES["llm.enabled"].help + " Off means no notes or summaries.")


def books_section(ctl: Control) -> None:
    st.caption("Pausing a book blocks NEW positions only. It keeps marking to market and counting its days, so "
               "nothing is replayed when you resume. Your pauses are logged and shown on the trackers.")
    st.markdown("##### Pause the swing experiment")
    for b in ("swing_probation", "swing_shadow_momentum"):
        toggle(ctl, f"book.{b}.paused", f"Pause  ·  {BOOKS[b]}", SWITCHES[f'book.{b}.paused'].help)
    st.markdown("##### Pause the practice accounts")
    for b in ("sandbox_A", "sandbox_B", "sandbox_C", "sandbox_D"):
        toggle(ctl, f"book.{b}.paused", f"Pause  ·  {BOOKS[b]}", SWITCHES[f'book.{b}.paused'].help)
    skip = ctl.get("sandbox.skip_date")
    c1, c2, c3 = st.columns([2, 1, 1])
    when = c1.date_input("Skip a whole practice day", value=date.today(), min_value=date.today(),
                         help="All four accounts sit that day out. They stay flat and the day still counts.")
    if c2.button("Skip this day", use_container_width=True):
        ctl.set("sandbox.skip_date", when.isoformat())
        st.toast(f"Sandbox will sit out {when}.")
        st.rerun()
    if skip:
        c3.button("Clear skip", on_click=lambda: Control().set("sandbox.skip_date", ""), use_container_width=True)
        st.info(f"The sandbox is set to sit out {skip}.")
    st.markdown("##### Not running (and why)")
    st.caption("Option-trading ideas: tested and set aside for now.  ·  A \"head\" that moves money between teams: not "
               "built yet, nothing to move between.")


def fyers_section(ctl: Control, compact: bool = False) -> None:
    ok, msg = fyers_state()
    st.session_state["fyers_checked"] = ok
    if not compact:
        c1, c2 = st.columns([4, 1])
        c1.markdown("**Fyers:** " + (chip("logged in", GOOD) if ok else chip("login needed", BAD)) +
                    f" <span style='color:{MUTED}'>{html.escape(msg[:110])}</span>", unsafe_allow_html=True)
        c2.button("Check again", on_click=fyers_state.clear, use_container_width=True)
        st.caption("Fyers logins expire every day. Do this before 9:15 on trading days. It takes about a minute, and "
                   "the login is saved without ever being shown on screen.")
    if "fyers_login" not in st.session_state:
        try:
            st.session_state["fyers_login"] = fyers_auth.begin()
        except Exception as e:
            st.error(str(e))
            return
    url, state = st.session_state["fyers_login"]
    st.markdown("**Step 1.** Click the button, then sign in to Fyers as you normally do.")
    st.link_button("Open Fyers login", url)
    st.markdown("**Step 2.** After you sign in, the page that opens may look broken or empty. That's normal. Click the "
                "browser's address bar at the top, copy the **whole address**, and paste it in the box below.")
    pasted = st.text_input("Address you were sent to", key="fyers_paste", placeholder="http://127.0.0.1/...auth_code=...",
                           type="password", help="Hidden while you type: it contains a one-time login code.")
    st.markdown("**Step 3.** Press the button. That's it.")
    if st.button("Finish login", type="primary", disabled=not pasted.strip()):
        good, message = fyers_auth.finish(pasted, state)
        (st.success if good else st.error)(message)
        if good:
            ctl.enqueue("recorder", "reload_fyers")
            ctl.log("login", "Fyers login refreshed from the dashboard")
            fyers_state.clear()
            st.session_state.pop("fyers_login", None)
            st.session_state["fyers_checked"] = True
            st.toast("Fyers: logged in.")
    if st.button("Start over with a new link"):
        st.session_state.pop("fyers_login", None)
        st.rerun()


QUICK = [
    ("news", "Collect headlines now"), ("announcements", "Collect company filings now"),
    ("results", "Check for new results now"), ("runner", "Run the evening paper runner now"),
    ("desk", "Write today's manager notes now"), ("researcher", "Run the Researcher now"),
    ("bars", "Save today's price bars now"), ("options", "Save an option-chain snapshot now"),
]


def run_section(ctl: Control) -> None:
    st.caption("Do something right now instead of waiting for its schedule. The recorder picks it up between jobs, "
               "usually within a minute (longer while the Researcher is busy), and the result shows below.")
    if not live_state()["running"]["recorder"]:
        st.warning("The recorder is not running, so these will wait until you start it (Services).")
    cols = st.columns(2)
    for n, (job, label) in enumerate(QUICK):
        def go(job=job, label=label):
            cid = Control().enqueue("recorder", "run_job", {"job": job})
            st.toast(f"Queued: {label.lower()}." if cid else "Already queued or running.")
        cols[n % 2].button(label, key=f"q_{job}", on_click=go, use_container_width=True)
    st.markdown("##### Big jobs")
    st.caption("These run in their own window of time, separately, so they never slow the recorder down.")
    last_day = date.today() - timedelta(days=1)
    while last_day.weekday() >= 5:
        last_day -= timedelta(days=1)
    replay_day = st.date_input("Day to replay through the sandbox (a rehearsal)", value=last_day, max_value=last_day)
    ndays = st.selectbox("How many days", [1, 2, 3, 5], index=0)
    for name, (desc, argv) in ops.job_defs(replay_day, ndays).items():
        c1, c2 = st.columns([4, 1])
        running = ops.job_running(name)
        c1.markdown(f"{desc}  " + (chip("running", WARN) if running else ""), unsafe_allow_html=True)
        if c2.button("Start", key=f"job_{name}", disabled=running, use_container_width=True):
            st.toast(f"{desc.split('(')[0].strip()}: {ops.launch_job(name, argv)}")
            ctl.log("job", f"started {name}")
            st.rerun()
    st.markdown("##### Recent commands")
    rows = ctl.recent_commands(10)
    if rows:
        st.dataframe(pd.DataFrame(rows, columns=["When", "For", "What", "Details", "Status", "Result"]),
                     hide_index=True, use_container_width=True)
    else:
        st.caption("Nothing queued yet.")


def locked_section(ctl: Control) -> None:
    st.caption("These are deliberately not switches. They are what makes the test results trustworthy. "
               "They change only through a dated amendment in docs/PLAN.md.")
    for title, reason in LOCKED:
        with st.container(border=True):
            st.markdown(f"🔒 **{title}**")
            st.caption(reason)


def friendly(key: str) -> str:
    if key in SWITCHES:
        sw = SWITCHES[key]
        return {"jobs": "Data job", "books": "Book", "agents": "Agent", "services": "Service", "firm": "Firm"}.get(sw.group, "") \
            + (": " if sw.group in ("jobs", "books", "agents", "services", "firm") else "") + sw.label
    return key


def log_section(ctl: Control) -> None:
    rows = []
    for ts, who, action, key, old, new, note in ctl.audit_tail(60):
        if action == "switch":
            nice = {"true": "on", "false": "off"}
            what = f"{friendly(key)}: {nice.get(old, old or '-')} → {nice.get(new, new or '-')}"
        else:
            what = f"{action}: {key or ''} {note or ''}".strip()
        rows.append({"When": ts[:19].replace("T", " "), "What": what, "By": who})
    st.markdown("##### Every change you made")
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.caption("No changes yet.")
    wins = ctl.windows()
    st.markdown("##### Pauses and halts (so the trackers can be read fairly)")
    if wins:
        st.dataframe(pd.DataFrame([{"What": friendly(s), "From": a[:19].replace("T", " "),
                                    "Until": (b or "still on")[:19].replace("T", " ")} for s, a, b in wins[::-1]]),
                     hide_index=True, use_container_width=True)
    else:
        st.caption("Nothing has been paused or halted.")


def switches_section(ctl: Control) -> None:
    st.caption("Flip a switch and it applies on its own. Nothing to save.")
    data_section(ctl)
    st.markdown("---")
    books_section(ctl)


def more_section(ctl: Control) -> None:
    with st.expander("Do something right now (instead of waiting for its schedule)"):
        run_section(ctl)
    with st.expander("What can't be changed here, and why"):
        locked_section(ctl)
    with st.expander("A log of everything you changed"):
        log_section(ctl)


def render() -> None:
    ctl = Control()
    st.title("Settings")
    st.caption("Everything you can switch on or off. Changes apply by themselves, usually within a minute. This always "
               "controls the live system, whatever the sidebar says. Everything stays paper money.")
    st.markdown("<style>.st-key-halt button{background:#d03b3b;color:#fff;border-color:#d03b3b;font-weight:700}"
                ".st-key-halt button:hover{background:#b02f2f;color:#fff;border-color:#b02f2f}"
                ".st-key-resume button{background:#0ca30c;color:#fff;border-color:#0ca30c;font-weight:700}</style>",
                unsafe_allow_html=True)
    status_strip()
    st.divider()
    st.session_state.setdefault("ctl_section", SECTIONS[0])
    section = st.segmented_control("Section", SECTIONS, key="ctl_section", label_visibility="collapsed")
    {"Switches": switches_section, "Programs": services_section, "Fyers login": fyers_section,
     "More": more_section}.get(section or SECTIONS[0], switches_section)(ctl)
