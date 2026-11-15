"""Trading Firm dashboard.

    .venv\\Scripts\\streamlit run dashboard\\app.py

Trading Floor: agents, what they're doing, and the live feed of their thoughts and actions.
Analytics: money, performance vs benchmarks, holdings, trades, probation tracker.
Research: every strategy trial and walk-forward result.
System: recorder jobs, data freshness, disk.
"""
from __future__ import annotations

import html
import json
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from firm.backtest.metrics import DESCRIPTIONS  # noqa: E402
from firm.config import BARS_DIR, DATA_DIR, IST, OPTIONS_DIR, RECORDER_DB_PATH, TEAMS  # noqa: E402
from firm.live.runner import DRYRUN_DIR, LIVE_DIR  # noqa: E402

# ---------------------------------------------------------------- look

C = {  # validated dark-mode palette (categorical slots 1-3 pass all-pairs CVD checks)
    "page": "#0d0d0d", "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
    "grid": "#2c2c2a", "axis": "#383835", "s1": "#3987e5", "s2": "#d95926", "s3": "#199e70",
    "good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b",
}
KIND = {  # icon + label, never colour alone
    "think": ("💭", "thinks", C["s1"]), "talk": ("💬", "says", C["s3"]),
    "act": ("⚙️", "does", C["ink2"]), "alert": ("⚠️", "ALERT", C["critical"]),
}
STATE = {"working": ("●", "working", C["good"]), "idle": ("●", "idle", C["muted"]),
         "error": ("▲", "error", C["critical"]), "off": ("○", "not built yet", C["axis"])}

st.set_page_config(page_title="Trading Firm", page_icon="📈", layout="wide")
st.markdown(f"""
<style>
.block-container {{padding-top: 2.2rem;}}
.card {{background:{C['surface']}; border:1px solid rgba(255,255,255,.08); border-radius:10px;
        padding:10px 12px; margin-bottom:8px; min-height:92px;}}
.card .name {{font-weight:600; color:{C['ink']}; font-size:.95rem;}}
.card .role {{color:{C['muted']}; font-size:.75rem; margin-bottom:4px;}}
.card .task {{color:{C['ink2']}; font-size:.8rem;}}
.card.off {{opacity:.45;}}
.team-h {{color:{C['ink2']}; font-size:.8rem; letter-spacing:.06em; text-transform:uppercase;
          margin:6px 0 6px 2px;}}
.feed-row {{display:grid; grid-template-columns:92px 190px 1fr; gap:10px; padding:7px 4px;
            border-bottom:1px solid {C['grid']}; font-size:.86rem; line-height:1.35;}}
.feed-row .t {{color:{C['muted']}; font-variant-numeric:tabular-nums;}}
.feed-row .who {{color:{C['ink']}; font-weight:600;}}
.feed-row .who small {{display:block; font-weight:400; color:{C['muted']};}}
.feed-row .txt {{color:{C['ink2']};}}
.chip {{display:inline-block; font-size:.7rem; padding:1px 7px; border-radius:9px; margin-right:6px;
        border:1px solid; font-weight:600;}}
.day-h {{color:{C['muted']}; font-size:.75rem; padding:12px 4px 2px; text-transform:uppercase;
         letter-spacing:.06em;}}
.rule {{padding:6px 0; border-bottom:1px solid {C['grid']}; font-size:.9rem;}}
</style>""", unsafe_allow_html=True)


def chip(text: str, color: str) -> str:
    return f'<span class="chip" style="color:{color};border-color:{color}">{html.escape(text)}</span>'


# ---------------------------------------------------------------- data access

def q(db: Path, sql: str, params: tuple = ()) -> pd.DataFrame:
    if not db.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10) as conn:
            return pd.read_sql_query(sql, conn, params=params)
    except Exception:
        return pd.DataFrame()


def firm_db() -> Path:
    mode = st.session_state.get("mode", "Live")
    if mode.startswith("Sandbox replay"):
        return DATA_DIR / "sandbox_replay" / "firm.db"
    if mode.startswith("Sandbox shake"):
        return DATA_DIR / "sandbox_shakedown" / "firm.db"
    return (DRYRUN_DIR if mode.startswith("Dry") else LIVE_DIR) / "firm.db"


def ago(ts: str | None) -> str:
    if not ts:
        return "never"
    try:
        s = (datetime.now(IST) - datetime.fromisoformat(ts)).total_seconds()
    except ValueError:
        return ts
    if s < 90:
        return "just now"
    if s < 3600:
        return f"{s / 60:.0f} min ago"
    if s < 86400 * 2:
        return f"{s / 3600:.0f} h ago"
    return f"{s / 86400:.0f} days ago"


def rs(x: float) -> str:
    return f"₹{x:,.0f}"


@st.cache_data(ttl=900, show_spinner=False)
def benchmarks(start: date) -> tuple[pd.Series, pd.Series]:
    """Daily returns after `start`: equal-weight Nifty 100 and Nifty 50."""
    from firm.backtest.benchmarks import buy_and_hold, equal_weight
    from firm.backtest.data import MarketData
    from firm.universe import INDEXES, nifty100

    d = MarketData.from_store("1d_yahoo", nifty100() + INDEXES, start=start - timedelta(days=10))
    close = d.panels["close"]
    end = date.today() + timedelta(days=1)
    ew = equal_weight(close.drop(columns=[c for c in INDEXES if c in close]), start, end)
    nifty = buy_and_hold(close["NIFTY50"].dropna(), start, end) if "NIFTY50" in close else pd.Series(dtype=float)
    keep = lambda s: s[s.index.date > start]  # noqa: E731  (returns earned after the start close)
    return keep(ew), keep(nifty)


# ---------------------------------------------------------------- roster

FIRM_AGENTS = [
    ("Head", "LLM · moves capital between teams, writes the firm report", False),
    ("Risk engine", "Code · checks every order; no AI can override it", True),
    ("Broker", "Code · paper fills at real prices with Indian costs", True),
    ("Runner", "Code · each evening: fills, equity, month-end decisions", True),
    ("Recorder", "Code · saves prices, option chains, news, filings", True),
]
TEAM_AGENTS = {
    "intraday_stocks": [
        ("Sandbox A (history)", "ML · trained on history, trades long and short every 5 min", True),
        ("Sandbox B (blank)", "ML · learns only from live days; 10-day warm-up", True),
        ("Sandbox C (random)", "Control · random trades, the cost floor", True),
        ("Sandbox D (cash)", "Control · holds cash, the bar to beat", True),
    ],
    "swing_stocks": [
        ("Picker (LightGBM)", "ML · picks 10 of the top-20 momentum stocks monthly", True),
        ("Momentum (shadow)", "Rule · yardstick only, no firm capital", True),
        ("Researcher", "LLM + FinBERT · notes on filings and headlines, every 5 min", True),
        ("Manager", "LLM · plain-English daily note per book", True),
        ("AI Trader", "LLM · daily risk guard (shadow) + weekly review in Claude Code", True),
    ],
}


def agent_card(name: str, role: str, built: bool, status: dict) -> str:
    st_ = status.get(name)
    if not built:
        state, task, when = "off", "Coming in a later build step", ""
    elif st_ is None:
        state, task, when = "idle", "Waiting for its first run", ""
    else:
        state, task, when = st_["state"], st_["task"] or "", ago(st_["updated"])
    icon, label, color = STATE.get(state, STATE["idle"])
    return (f'<div class="card {"off" if state == "off" else ""}"><div class="name">{html.escape(name)}</div>'
            f'<div class="role">{html.escape(role)}</div>'
            f'<div><span style="color:{color}">{icon}</span> <b style="color:{color};font-size:.8rem">{label}</b>'
            f'<span style="color:{C["muted"]};font-size:.75rem"> {when}</span></div>'
            f'<div class="task">{html.escape(task)}</div></div>')


# ================================================================ pages

def page_floor() -> None:
    db = firm_db()
    st.title("Live feed")
    st.caption("Who's working, what they're thinking, what they did. Refreshes every 5 seconds.")

    @st.fragment(run_every="5s")
    def roster() -> None:
        s = q(db, "SELECT agent, team, state, task, updated FROM agent_status")
        status = {r.agent: r._asdict() for r in s.itertuples()} if not s.empty else {}
        now = datetime.now(IST)
        from firm.calendar import TradingCalendar
        cal = TradingCalendar(refresh=False)
        open_ = cal.is_trading_day(now.date()) and (9, 15) <= (now.hour, now.minute) < (15, 30)
        alerts = q(db, "SELECT COUNT(*) n FROM activity WHERE kind='alert' AND logged >= ?",
                   (now.date().isoformat(),))
        n_alerts = int(alerts.n.iloc[0]) if not alerts.empty else 0
        a, b, c = st.columns(3)
        a.markdown(f"**NSE** {chip('OPEN', C['good']) if open_ else chip('CLOSED', C['muted'])}"
                   f" <span style='color:{C['muted']}'>{now:%a %d %b, %H:%M} IST</span>", unsafe_allow_html=True)
        b.markdown(f"**Alerts today** {chip(str(n_alerts), C['critical'] if n_alerts else C['good'])}",
                   unsafe_allow_html=True)
        c.markdown(f"**Data** {chip(st.session_state.get('mode', 'Live'), C['s1'])}", unsafe_allow_html=True)

        st.markdown('<div class="team-h">Firm</div>', unsafe_allow_html=True)
        cols = st.columns(len(FIRM_AGENTS))
        for col, (n, role, built) in zip(cols, FIRM_AGENTS):
            col.markdown(agent_card(n, role, built, status), unsafe_allow_html=True)
        tcols = st.columns(len(TEAMS))
        for col, (key, team) in zip(tcols, TEAMS.items()):
            col.markdown(f'<div class="team-h">{team.display_name}</div>', unsafe_allow_html=True)
            agents = TEAM_AGENTS.get(key)
            if not agents:
                col.markdown(agent_card("Team not started", "Built in a later step", False, {}),
                             unsafe_allow_html=True)
                continue
            for n, role, built in agents:
                col.markdown(agent_card(n, role, built, status), unsafe_allow_html=True)

    roster()
    st.divider()

    st.subheader("Live feed")
    agents_df = q(db, "SELECT DISTINCT agent FROM activity ORDER BY agent")
    f1, f2, f3, f4 = st.columns([2, 3, 2, 2])
    kinds = f1.multiselect("Show", list(KIND), default=list(KIND),
                           format_func=lambda k: f"{KIND[k][0]} {KIND[k][1]}")
    who = f2.multiselect("Agents", agents_df.agent.tolist() if not agents_df.empty else [],
                         placeholder="All agents")
    text = f3.text_input("Search", placeholder="e.g. TCS")
    limit = f4.selectbox("How many", [100, 300, 1000], index=0)

    @st.fragment(run_every="5s")
    def feed() -> None:
        if not kinds:
            st.info("Pick at least one type above.")
            return
        sql = (f"SELECT ts, team, book, agent, kind, to_agent, text FROM activity WHERE kind IN "
               f"({','.join('?' * len(kinds))})")
        params: list = list(kinds)
        if who:
            sql += f" AND (agent IN ({','.join('?' * len(who))}) OR to_agent IN ({','.join('?' * len(who))}))"
            params += who + who
        if text:
            sql += " AND text LIKE ?"
            params.append(f"%{text}%")
        df = q(db, sql + " ORDER BY ts DESC, id DESC LIMIT ?", tuple(params + [limit]))
        if df.empty:
            st.info("Nothing yet. The runner posts here each evening after 16:00; the recorder posts "
                    "whenever it saves something new. Switch to 'Dry run' in the sidebar to see a rehearsal.")
            return
        df = df.fillna("")
        rows, day = [], None
        for r in df.itertuples():
            d = r.ts[:10]
            if d != day:
                day = d
                rows.append(f'<div class="day-h">{datetime.fromisoformat(r.ts).strftime("%A %d %B %Y")}</div>')
            icon, label, color = KIND[r.kind]
            to = f' → {html.escape(r.to_agent)}' if r.to_agent else ""
            book = f"<small>{html.escape(r.book)}</small>" if r.book else ""
            rows.append(f'<div class="feed-row"><div class="t">{r.ts[11:16]}</div>'
                        f'<div class="who">{html.escape(r.agent)}{to}{book}</div>'
                        f'<div class="txt">{chip(icon + " " + label, color)}{html.escape(r.text)}</div></div>')
        st.markdown("".join(rows), unsafe_allow_html=True)

    feed()


# ---------------------------------------------------------------- analytics

def equity_frame(db: Path, book: str) -> pd.Series:
    df = q(db, "SELECT ts, equity FROM equity WHERE team=? ORDER BY ts", (book,))
    if df.empty:
        return pd.Series(dtype=float)
    s = pd.Series(df.equity.values, index=pd.to_datetime(df.ts.str[:10]))
    return s[~s.index.duplicated(keep="last")]


def line_chart(series: dict[str, pd.Series], colors: list[str], yfmt: str, title: str, height=340) -> go.Figure:
    fig = go.Figure()
    for n, ((name, s), color) in enumerate(zip(series.items(), colors)):
        if s.empty:
            continue
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name, mode="lines",
                                 line={"color": color, "width": 2},
                                 hovertemplate=f"%{{y:{yfmt}}}<extra>{name}</extra>"))
        fig.add_annotation(x=s.index[-1], y=s.values[-1], text=name, showarrow=False, xanchor="left",
                           xshift=6, yshift=(n - (len(series) - 1) / 2) * 12, font={"color": C["ink2"], "size": 11})
    fig.update_layout(
        title={"text": title, "font": {"size": 14, "color": C["ink"]}}, height=height,
        paper_bgcolor=C["surface"], plot_bgcolor=C["surface"], font={"color": C["ink2"]},
        hovermode="x unified", margin={"l": 50, "r": 130, "t": 40, "b": 30},
        legend={"orientation": "h", "y": -0.15},
        xaxis={"gridcolor": C["grid"], "linecolor": C["axis"]},
        yaxis={"gridcolor": C["grid"], "linecolor": C["axis"], "tickformat": yfmt},
    )
    return fig


def sandbox_section(db: Path, sb: pd.DataFrame) -> None:
    """The four intraday sandbox accounts side by side, and the pre-registered day-60/120 tracker."""
    if sb.empty:
        return
    from firm.live.sandbox_eval import CHECKPOINTS, evaluate

    st.subheader("Intraday sandbox (live forward test)")
    st.caption("Four paper accounts of ₹1L, same rules and costs: A trained on history, B a blank learner, "
               "C random trades (the cost floor), D cash. Not a proven team. An account counts as working only "
               "if it beats cash (D) and the random trader (C) at day 60 and day 120. Rules: docs/PLAN.md.")
    series = {}
    colors = {"sandbox_A": C["s1"], "sandbox_B": C["s2"], "sandbox_C": C["s3"], "sandbox_D": C["muted"]}
    for b in sb.itertuples():
        eq = equity_frame(db, b.name)
        if not eq.empty:
            series[b.label.split(":")[0].replace("Sandbox ", "") + " " + b.label.split(":")[1].strip()] = 100 * eq / b.capital
    if series:

        st.plotly_chart(line_chart(series, [colors[n] for n in sb.name if n in colors][: len(series)],
                                   ".2f", "Growth of ₹100 (daily close)"), use_container_width=True)
    ev = evaluate(db)
    if not ev["accounts"]:
        st.info("No sandbox days recorded yet.")
        return
    days = ev["days"]
    nxt = ev.get("next_checkpoint")
    st.markdown(f"**Day {days} of the test.** " + (f"Next checkpoint: day {nxt}." if nxt else "All checkpoints reached."))
    rows = []
    for name, a in ev["accounts"].items():
        label = sb.set_index("name").label.get(name, name)
        rows.append({"Account": label, "Equity": a["equity"], "vs cash (D)": a["vs_D"], "vs random (C)": a["vs_C"],
                     "Round trips": a["round_trips"],
                     "Sharpe vs cash": a["sharpe_excess"] if np.isfinite(a["sharpe_excess"]) else None,
                     "Latency-adjusted vs D": a["latency_adjusted_vs_D"],
                     "Working?": ("yes" if a["works"] else "no") if ev.get("checkpoint_reached") else "too early"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, column_config={
        "Equity": st.column_config.NumberColumn(format="₹%.0f"),
        "vs cash (D)": st.column_config.NumberColumn(format="₹%.0f", help="Final equity minus the cash account's."),
        "vs random (C)": st.column_config.NumberColumn(format="₹%.0f", help="Final equity minus the random trader's."),
        "Sharpe vs cash": st.column_config.NumberColumn(format="%.2f", help="Needs to be above 0.5."),
        "Latency-adjusted vs D": st.column_config.NumberColumn(
            format="₹%.0f", help="Re-priced at the quote taken at each decision: live polling adds delay the "
                                 "ledger ignores. Must still be above zero.")})
    st.caption("Checkpoint tests: beats D, beats C, Sharpe above 0.5, positive in both halves, at least 30 round "
               "trips, and still ahead after latency. Zero trades is a valid result (it means not working).")
    st.divider()


def page_analytics() -> None:
    db = firm_db()
    st.title("Money")
    all_books = q(db, "SELECT * FROM books ORDER BY status")
    if all_books.empty:
        st.info("No paper books yet. The first live run is the evening of 30 Nov 2026. "
                "Switch to 'Dry run' in the sidebar to see a rehearsal.")
        return
    sandbox_section(db, all_books[all_books.status == "sandbox"])
    books = all_books[all_books.status != "sandbox"].reset_index(drop=True)
    if books.empty:
        return

    # Firm overview: a card per book.
    st.subheader("Firm overview")
    cols = st.columns(max(len(books), 1) + 1)
    total = 0.0
    for col, b in zip(cols, books.itertuples()):
        eq = equity_frame(db, b.name)
        if eq.empty:
            col.metric(b.label, rs(b.capital), help="No days processed yet.")
            continue
        start = date.fromisoformat(b.start)
        ew, _ = benchmarks(start)
        ew_ret = float((1 + ew[ew.index <= eq.index[-1]]).prod() - 1)
        ret = eq.iloc[-1] / b.capital - 1
        today = eq.iloc[-1] - (eq.iloc[-2] if len(eq) > 1 else b.capital)
        if b.status != "shadow":
            total += eq.iloc[-1]
        col.metric(f"{b.label} ({b.status})", rs(eq.iloc[-1]), f"{today:+,.0f} last day",
                   help="Paper equity: cash + holdings at the last close. Arrow = change on the last trading day.")
        col.caption(f"Since start {ret:+.2%} · equal-weight {ew_ret:+.2%} · "
                    f"{'ahead' if ret > ew_ret else 'behind'} by {abs(ret - ew_ret):.2%}")
    cols[-1].metric("Teams not started", "3", help="Intraday stocks, intraday options, overnight options.")

    st.divider()
    name = st.selectbox("Book", books.name.tolist(),
                        format_func=lambda n: books.set_index("name").label[n] + f" ({n})")
    b = books.set_index("name").loc[name]
    eq = equity_frame(db, name)
    if eq.empty:
        st.info("This book has no days processed yet.")
        return
    start = date.fromisoformat(b.start)
    ew, nifty = benchmarks(start)
    last = eq.index[-1]
    idx = lambda r: 100 * (1 + r[r.index <= last]).cumprod()  # noqa: E731
    shadow = q(db, "SELECT name FROM books WHERE status='shadow' AND team=? AND name != ?", (b.team, name))
    series = {b.label: 100 * eq / b.capital}
    if not shadow.empty:
        sh = equity_frame(db, shadow.name.iloc[0])
        if not sh.empty:
            series["Shadow momentum"] = 100 * sh / sh.iloc[0]
    series["Equal-weight Nifty 100"] = idx(ew)
    colors = [C["s1"], C["s3"], C["s2"]] if len(series) == 3 else [C["s1"], C["s2"]]

    m = st.columns(5)
    ret = eq.iloc[-1] / b.capital - 1
    ew_ret = float((1 + ew[ew.index <= last]).prod() - 1)
    dd = eq / eq.cummax() - 1
    fills = q(db, "SELECT * FROM fills WHERE team=? ORDER BY ts", (name,))
    costs = float(fills.costs.sum()) if not fills.empty else 0.0
    m[0].metric("Equity", rs(eq.iloc[-1]), f"{ret:+.2%} since start",
                help="Cash plus holdings valued at the last close.")
    m[1].metric("vs equal-weight", f"{ret - ew_ret:+.2%}", help="Our return minus the equal-weight Nifty 100 "
                "over the same days. This is the fair bar for a stock picker on this universe.")
    m[2].metric("Worst drawdown", f"{dd.min():.1%}", help=DESCRIPTIONS["max_drawdown"])
    m[3].metric("Costs paid", rs(costs), help=DESCRIPTIONS["costs_paid"])
    m[4].metric("Days live", f"{len(eq)}", help="Trading days processed since the book opened.")

    st.plotly_chart(line_chart(series, colors, ".1f", "Growth of ₹100"), use_container_width=True)
    st.plotly_chart(line_chart({"Drawdown": dd}, [C["s2"]], ".0%",
                               "Drawdown: how far below its previous peak", height=220),
                    use_container_width=True)

    if b.status == "probation":
        st.subheader("Probation tracker")
        st.caption(b.note)
        from firm.control import Control
        from firm.live.tracker import risk_blocks

        n_blocks = risk_blocks(db, name)                       # your own pauses are not risk-engine breaches
        wins = [w for w in Control().windows() if w[0] in ("halt.firm", f"book.{name}.paused")]
        if wins:
            st.caption("Paused by you: " + "; ".join(f"{a[:16].replace('T', ' ')} to {(b or 'now')[:16].replace('T', ' ')}"
                                                      for _, a, b in wins) + ". Read the results with these windows in mind.")
        end = date(2027, 5, 31)
        rules = [
            ("Return at least equal-weight minus 5 points", ret >= ew_ret - 0.05,
             f"{ret:+.2%} vs bar {ew_ret - 0.05:+.2%}"),
            ("Worst drawdown better than -25%", dd.min() > -0.25, f"{dd.min():.1%}"),
            ("No orders blocked by the risk engine", n_blocks == 0, f"{n_blocks} blocked"),
            ("Not trailing equal-weight by more than 10 points (cut rule)", ret > ew_ret - 0.10,
             f"gap {ret - ew_ret:+.2%}"),
        ]
        for text_, ok, val in rules:
            st.markdown(f'<div class="rule">{chip("✓ on track", C["good"]) if ok else chip("✗ failing", C["critical"])}'
                        f' {html.escape(text_)} <span style="color:{C["muted"]}">· {val}</span></div>',
                        unsafe_allow_html=True)
        st.caption(f"{max((end - last.date()).days, 0)} days left (ends {end:%d %b %Y}). "
                   "'Trades match backtest' is checked by re-running the backtest at the end.")

    st.subheader("Holdings")
    pos = q(db, "SELECT symbol, qty, avg_price, last_price FROM positions WHERE team=?", (name,))
    if pos.empty:
        st.caption("All cash.")
    else:
        pos["value"] = pos.qty * pos.last_price
        pos["weight"] = pos.value / eq.iloc[-1]
        pos["pnl"] = (pos.last_price - pos.avg_price) * pos.qty
        pos["pnl_pct"] = pos.last_price / pos.avg_price - 1
        st.dataframe(pos.sort_values("value", ascending=False), hide_index=True, use_container_width=True,
                     column_config={
                         "symbol": "Stock", "qty": "Shares",
                         "avg_price": st.column_config.NumberColumn("Bought at", format="₹%.2f"),
                         "last_price": st.column_config.NumberColumn("Last close", format="₹%.2f"),
                         "value": st.column_config.NumberColumn("Value", format="₹%.0f"),
                         "weight": st.column_config.NumberColumn("Share of book", format="percent"),
                         "pnl": st.column_config.NumberColumn("Profit/loss", format="₹%.0f"),
                         "pnl_pct": st.column_config.NumberColumn("P/L %", format="percent")})

    st.subheader("Trades")
    if fills.empty:
        st.caption("No trades yet.")
    else:
        thoughts = q(db, "SELECT ts, text FROM activity WHERE book=? AND kind='think' ORDER BY ts", (name,))

        def why(r) -> str:
            if thoughts.empty:
                return ""
            before = thoughts[thoughts.ts < r.ts[:10] + "T09:00"]
            key = "Adding" if r.side == "BUY" else "Dropping"
            hit = before[before.text.str.contains(rf"\b{r.symbol}\b", regex=True)
                         & before.text.str.startswith(key)]
            if hit.empty:
                hit = before[before.text.str.contains(rf"\b{r.symbol}\b", regex=True)]
            return hit.text.iloc[-1] if not hit.empty else ""

        fills["why"] = [why(r) for r in fills.itertuples()]
        fills["date"] = fills.ts.str[:10]
        st.dataframe(fills[["date", "side", "symbol", "qty", "price", "costs", "realized_pnl", "why"]]
                     .iloc[::-1], hide_index=True, use_container_width=True,
                     column_config={"date": "Date (at open)", "side": "Side", "symbol": "Stock", "qty": "Shares",
                                    "price": st.column_config.NumberColumn("Price", format="₹%.2f"),
                                    "costs": st.column_config.NumberColumn("Costs", format="₹%.2f"),
                                    "realized_pnl": st.column_config.NumberColumn("Booked P/L", format="₹%.0f"),
                                    "why": st.column_config.TextColumn("Why (the agent's reasoning)",
                                                                       width="large")})


# ---------------------------------------------------------------- desk (LLM agents' work)

@st.cache_data(ttl=900, show_spinner=False)
def price_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    from firm.backtest.data import MarketData
    from firm.universe import nifty100

    d = MarketData.from_store("1d_yahoo", nifty100(), start=date.today() - timedelta(days=200))
    return d.panels["open"], d.panels["close"]



@st.cache_data(ttl=300, show_spinner=False)
def latest_results() -> pd.DataFrame:
    """Latest filed quarter per stock with year-on-year growth (consolidated preferred)."""
    from firm.features.fundamentals import best_basis, load_results, quarter_label, quarter_measures

    try:
        res = load_results(RECORDER_DB_PATH)
    except Exception:
        return pd.DataFrame()
    if res.empty:
        return pd.DataFrame()
    m = best_basis(quarter_measures(res))
    key = res.assign(basis_n=res.consolidated)[["symbol", "to_date", "consolidated", "revenue", "pat", "eps"]]
    m = m.merge(key, on=["symbol", "to_date", "consolidated"], how="left")
    m = m.sort_values("to_date").groupby("symbol").tail(1)
    return pd.DataFrame({
        "symbol": m.symbol, "quarter": [quarter_label(d) for d in m.to_date],
        "filed": m.broadcast.dt.strftime("%Y-%m-%d %H:%M"), "revenue": m.revenue, "rev_yoy": m.rev_yoy,
        "pat": m.pat, "pat_yoy": m.pat_yoy, "eps": m.eps,
        "basis": np.where(m.consolidated == 1, "consolidated", "standalone")})


def page_desk() -> None:
    db = firm_db()
    st.title("Notes")
    st.caption("What the LLM agents wrote: daily notes, research notes, and how right they turned out to be. "
               "LLM output is research only; it never trades on its own.")
    t1, t2, t3, t4, t5, t6 = st.tabs(["Daily notes", "Research notes", "Researcher scoreboard",
                                      "AI Trader guard", "LLM health", "Results numbers"])

    with t1:
        notes = q(db, "SELECT date, book, headline, note, flags, checked FROM daily_notes "
                      "ORDER BY date DESC, book LIMIT 60")
        if notes.empty:
            st.info("The Manager writes a note each evening after the runner, once a book is open (from 30 Nov).")
        for r in notes.itertuples():
            badge = (chip("✓ numbers checked", C["good"]) if r.checked
                     else chip("template (LLM note failed check)", C["warning"]))
            st.markdown(f"**{r.date} · {html.escape(r.book)}**: {html.escape(r.headline)} {badge}",
                        unsafe_allow_html=True)
            st.write(r.note)
            for f in json.loads(r.flags or "[]"):
                color = {"act": C["critical"], "watch": C["warning"]}.get(f["level"], C["muted"])
                st.markdown(f"{chip(f['level'].upper(), color)} {html.escape(f['text'])}", unsafe_allow_html=True)
            st.divider()

    with t2:
        c1, c2, c3 = st.columns([2, 2, 3])
        imp = c1.slider("Minimum importance", 0, 3, 1, help="0 routine paperwork, 1 minor, 2 notable, "
                        "3 likely to move the stock several percent.")
        src = c2.multiselect("Source", ["nse", "news"], default=["nse", "news"])
        sym = c3.text_input("Stock", placeholder="e.g. TCS").strip().upper()
        sql = ("SELECT ts, symbol, source, event_type, direction, importance, summary, finbert "
               "FROM research_notes WHERE importance >= ?")
        params: list = [imp]
        if src:
            sql += f" AND source IN ({','.join('?' * len(src))})"
            params += src
        if sym:
            sql += " AND symbol = ?"
            params.append(sym)
        rn = q(db, sql + " ORDER BY ts DESC LIMIT 500", tuple(params))
        if rn.empty:
            st.info("No research notes yet. The Researcher reads new filings and headlines every 5 minutes "
                    "while Ollama is running.")
        else:
            st.dataframe(rn, hide_index=True, use_container_width=True, column_config={
                "ts": "When public", "symbol": "Stock", "source": "Source", "event_type": "Event",
                "direction": st.column_config.NumberColumn(
                    "Direction", help="-2 clearly negative .. +2 clearly positive"),
                "importance": st.column_config.NumberColumn(
                    "Importance", help="0 routine .. 3 likely to move the stock"),
                "summary": st.column_config.TextColumn("Note", width="large"),
                "finbert": st.column_config.NumberColumn(
                    "Headline tone", format="%.2f", help="FinBERT: -1 negative .. +1 positive (news only)")})

    with t3:
        from firm.agents.scoreboard import score_notes, summary
        opens, closes = price_panels()
        sc = score_notes(db, opens, closes)
        sm = summary(sc)
        st.caption("A note is right if, from the next open after it became public, the stock beat the "
                   "equal-weight average over 5 trading days in the noted direction. 50% = coin flip. "
                   "Needs 50+ scored notes before it means anything.")
        a, b, c, d = st.columns(4)
        a.metric("Scored notes", sm.get("scored", 0), help="Notes old enough to score (5 trading days).")
        hr = sm.get("hit_rate")
        b.metric("Hit rate", f"{hr:.0%}" if hr is not None else "-",
                 help="Share of notes whose direction was right.")
        pos, neg = sm.get("avg_excess_positive_notes"), sm.get("avg_excess_negative_notes")
        c.metric("After positive notes", f"{pos:+.2%}" if pos is not None else "-",
                 help="Average 5-day return vs equal-weight after a positive note. Want: above zero.")
        d.metric("After negative notes", f"{neg:+.2%}" if neg is not None else "-",
                 help="Average 5-day return vs equal-weight after a negative note. Want: below zero.")
        if not sc.empty:
            st.dataframe(sc[["ts", "symbol", "event_type", "direction", "importance", "excess_5d", "right",
                             "summary"]].sort_values("ts", ascending=False),
                         hide_index=True, use_container_width=True,
                         column_config={"excess_5d": st.column_config.NumberColumn("5-day vs EW", format="percent"),
                                        "right": "Right?"})

    with t4:
        st.caption("Shadow mode: proposals are logged, never applied, so the probation book stays identical to "
                   "its backtest. 'Since' = the stock's move after the proposal. For a cut, a fall means the "
                   "guard was right.")
        g = q(db, "SELECT date, book, symbol, action, reason, price FROM guard_proposals ORDER BY date DESC")
        if g.empty:
            st.info("No proposals yet. The guard only looks at clearly negative, notable news on stocks we "
                    "hold or plan to buy.")
        else:
            _, closes = price_panels()
            g["since"] = [float(closes[s].dropna().iloc[-1] / p - 1) if s in closes and p else None
                          for s, p in zip(g.symbol, g.price)]
            st.dataframe(g, hide_index=True, use_container_width=True, column_config={
                "price": st.column_config.NumberColumn("Price then", format="₹%.2f"),
                "since": st.column_config.NumberColumn("Since", format="percent"),
                "reason": st.column_config.TextColumn("Reason", width="large")})

    with t5:
        from firm.llm import get_llm
        llm = get_llm()
        up = llm.available()
        st.markdown(f"**Local model** {chip('online', C['good']) if up else chip('offline', C['critical'])} "
                    f"<span style='color:{C['muted']}'>{html.escape(llm.name)}</span>", unsafe_allow_html=True)
        if not up:
            st.caption("Start Ollama (Start menu > Ollama). The agents pause and resume by themselves.")
        h = q(db, "SELECT agent, COUNT(*) calls, SUM(1-ok) failed, ROUND(AVG(ms)) avg_ms, MAX(ts) last "
                  "FROM llm_calls GROUP BY agent")
        if not h.empty:
            st.dataframe(h, hide_index=True, use_container_width=True, column_config={
                "avg_ms": st.column_config.NumberColumn("Avg time (ms)"), "last": "Last call"})
            last = q(db, "SELECT ts, agent, ok, ms, prompt, response FROM llm_calls ORDER BY id DESC LIMIT 20")
            with st.expander("Last 20 calls (prompt and reply, for auditing)"):
                st.dataframe(last, hide_index=True, use_container_width=True)

    with t6:
        st.caption("Quarterly results straight from the filed XBRL numbers (code, no LLM). Growth is against the "
                   "same quarter a year earlier. Amounts in ₹ crore. Latest quarter per stock.")
        latest = latest_results()
        if latest.empty:
            st.info("No results stored yet. The recorder is downloading the history (a few hours), then checks "
                    "for new filings every 30 minutes.")
        else:
            c1, c2 = st.columns([3, 2])
            only = c2.text_input("Stock", placeholder="e.g. TCS", key="res_sym").strip().upper()
            if only:
                latest = latest[latest.symbol.str.contains(only)]
            c1.markdown(f"**{len(latest)} stocks** with a filed quarter on record.")
            st.dataframe(latest.sort_values("filed", ascending=False), hide_index=True, use_container_width=True,
                         column_config={
                             "symbol": "Stock", "quarter": "Quarter", "filed": "Filed (public)",
                             "revenue": st.column_config.NumberColumn("Revenue", format="%.0f"),
                             "rev_yoy": st.column_config.NumberColumn("Revenue vs year ago", format="percent"),
                             "pat": st.column_config.NumberColumn("Net profit", format="%.0f"),
                             "pat_yoy": st.column_config.NumberColumn("Profit vs year ago", format="percent"),
                             "eps": st.column_config.NumberColumn("EPS (₹)", format="%.2f"),
                             "basis": "Basis"})


# ---------------------------------------------------------------- research

def page_research() -> None:
    st.title("Research")
    st.caption("Every strategy we tested. More tries means a higher bar for luck (deflated Sharpe).")
    rows = {}
    for f in sorted((DATA_DIR / "research").glob("team_d_*.json")):
        if f.stem.endswith("_quick"):
            continue
        d = json.loads(f.read_text())
        for k, v in d["results"].items():
            rows[k] = {"Strategy": k, "CAGR": v.get("cagr"), "Sharpe": v.get("sharpe"),
                       "Worst drawdown": v.get("max_drawdown"), "Years beating EW": v.get("fold_win_rate"),
                       "Deflated Sharpe": v.get("deflated_sharpe"), "Turnover/yr": v.get("annual_turnover"),
                       "Run": f.stem[7:]}
    if rows:
        st.subheader("Swing walk-forward, 2011 to May 2026 (latest run per strategy)")
        st.dataframe(pd.DataFrame(rows.values()), hide_index=True, use_container_width=True,
                     column_config={
                         "CAGR": st.column_config.NumberColumn(format="percent", help=DESCRIPTIONS["cagr"]),
                         "Sharpe": st.column_config.NumberColumn(format="%.2f", help=DESCRIPTIONS["sharpe"]),
                         "Worst drawdown": st.column_config.NumberColumn(format="percent",
                                                                         help=DESCRIPTIONS["max_drawdown"]),
                         "Years beating EW": st.column_config.NumberColumn(format="percent",
                                                                           help="Share of test years beating equal-weight Nifty 100."),
                         "Deflated Sharpe": st.column_config.NumberColumn(format="%.2f",
                                                                          help=DESCRIPTIONS["deflated_sharpe"]),
                         "Turnover/yr": st.column_config.NumberColumn(format="%.1f",
                                                                      help=DESCRIPTIONS["annual_turnover"])})
    trials = q(DATA_DIR / "research.db", "SELECT ts, track, strategy, version, kind, start, end, metrics "
                                         "FROM trials ORDER BY id DESC")
    st.subheader(f"Trial log ({len(trials)} trials)")
    if not trials.empty:
        mt = trials.metrics.apply(lambda s: json.loads(s) if s else {})
        trials["Sharpe"] = mt.apply(lambda m: m.get("sharpe"))
        trials["Worst DD"] = mt.apply(lambda m: m.get("max_drawdown"))
        trials["CAGR"] = mt.apply(lambda m: m.get("cagr"))
        st.dataframe(trials.drop(columns="metrics"), hide_index=True, use_container_width=True,
                     column_config={"Sharpe": st.column_config.NumberColumn(format="%.2f"),
                                    "Worst DD": st.column_config.NumberColumn(format="percent"),
                                    "CAGR": st.column_config.NumberColumn(format="percent")})
        exams = trials[trials.kind == "final_exam"]
        if not exams.empty:
            st.subheader("Final exams (locked period, one try each)")
            st.dataframe(exams.drop(columns="metrics"), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------- system

def page_system() -> None:
    st.title("System health")
    runs = q(RECORDER_DB_PATH, "SELECT job, started, status, rows, message FROM recorder_runs r "
                               "WHERE id = (SELECT MAX(id) FROM recorder_runs WHERE job=r.job) ORDER BY job")
    color = {"ok": C["good"], "skipped": C["muted"], "partial": C["warning"], "error": C["critical"],
             "auth_failed": C["critical"]}
    st.subheader("Recorder and runner jobs (last run each)")
    if runs.empty:
        st.caption("No runs recorded.")
    for r in runs.itertuples():
        st.markdown(f'<div class="rule">{chip(r.status, color.get(r.status, C["muted"]))} <b>{r.job}</b> '
                    f'<span style="color:{C["muted"]}">· {ago(r.started)} · {r.rows} rows</span><br>'
                    f'<span style="color:{C["ink2"]};font-size:.8rem">{html.escape((r.message or "")[:240])}'
                    f'</span></div>', unsafe_allow_html=True)
    if not runs.empty and (runs.status == "auth_failed").any():
        st.error("Fyers login expired. Run `.venv\\Scripts\\python scripts\\fyers_login.py`.")

    st.subheader("Data freshness")
    from firm.backtest.data import MarketData
    try:
        d = MarketData.from_store("1d_yahoo", ["RELIANCE", "NIFTY50"], start=date.today() - timedelta(days=20))
        last_bar = d.index[-1].date()
    except Exception:
        last_bar = None
    news = q(RECORDER_DB_PATH, "SELECT MAX(first_seen) t, COUNT(*) n FROM news")
    ann = q(RECORDER_DB_PATH, "SELECT COUNT(*) n FROM announcements")
    a, b, c = st.columns(3)
    a.metric("Latest daily bar", str(last_bar), help="Last trading day with prices saved (Yahoo, adjusted).")
    b.metric("Newest headline", ago(news.t.iloc[0]) if not news.empty else "-",
             f"{int(news.n.iloc[0]):,} stored" if not news.empty else None, help="From ET, BS, Mint, Google News.")
    c.metric("NSE announcements", f"{int(ann.n.iloc[0]):,}" if not ann.empty else "-",
             help="Company filings since 2011 (backfill may still be running).")

    st.subheader("Disk")
    import shutil
    size = lambda p: sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9 if p.exists() else 0  # noqa: E731
    free = shutil.disk_usage(DATA_DIR).free / 1e9
    a, b, c = st.columns(3)
    a.metric("Free on drive", f"{free:.1f} GB", help="Below 5 GB, move data with FIRM_DATA_DIR.")
    b.metric("Price bars", f"{size(BARS_DIR):.2f} GB")
    c.metric("Option chains", f"{size(OPTIONS_DIR):.2f} GB")


# ================================================================ app

with st.sidebar:
    st.markdown("### 📈 Trading Firm")
    st.radio("Data", ["Live", "Dry run (rehearsal)", "Sandbox replay", "Sandbox shake-down"], key="mode",
             help="Live = the real paper books and the sandbox. Dry run = a swing rehearsal over past days. "
                  "Sandbox replay = recorded days replayed through the sandbox (scripts/sandbox.py replay). "
                  "Sandbox shake-down = an uncounted live test run.")
    st.caption("Paper trading only. No real orders exist anywhere in this system.")

def page_control() -> None:
    from dashboard.control_page import render

    render()


def page_home() -> None:
    from dashboard.home_page import render

    render()


PAGES = {"home": page_home, "floor": page_floor, "control": page_control, "desk": page_desk,
         "analytics": page_analytics, "research": page_research, "system": page_system}
if st.session_state.get("_test_page"):  # automated smoke tests render one page directly
    PAGES[st.session_state["_test_page"]]()
    st.stop()

pg = st.navigation({
    "": [st.Page(page_home, title="Home", icon="🏠", default=True)],
    "Watch": [st.Page(page_floor, title="Live feed", icon="📡"),
              st.Page(page_analytics, title="Money", icon="💰"),
              st.Page(page_desk, title="Notes", icon="📝")],
    "Control": [st.Page(page_control, title="Settings", icon="⚙️")],
    "Advanced": [st.Page(page_research, title="Research", icon="🔬"),
                 st.Page(page_system, title="System", icon="🩺")],
})
pg.run()
