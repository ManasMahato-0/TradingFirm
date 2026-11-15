r"""Intraday sandbox: four live paper accounts (rules in docs/PLAN.md).

    .venv\Scripts\python scripts\sandbox.py replay --day 2026-11-12 --ndays 2   # rehearsal on recorded days
    .venv\Scripts\python scripts\sandbox.py live                                # the service: every trading day
    .venv\Scripts\python scripts\sandbox.py live --shakedown                    # uncounted test run, separate data
    .venv\Scripts\python scripts\sandbox.py status
    .venv\Scripts\python scripts\sandbox.py install-startup | uninstall-startup

`replay` writes to data\sandbox_replay (wiped first), never to the live database.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import date, datetime
from datetime import time as dtime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from firm.config import DATA_DIR, IST  # noqa: E402

REPLAY_DIR = DATA_DIR / "sandbox_replay"


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def cmd_replay(day: date, ndays: int) -> int:
    from firm.live.sandbox import Sandbox, load_train_data
    from firm.live.sandbox_feed import ReplaySource

    if REPLAY_DIR.exists():
        shutil.rmtree(REPLAY_DIR)
    REPLAY_DIR.mkdir(parents=True)
    train, universe = load_train_data()
    all_days = sorted(set(train.stocks.index.date))
    days = [d for d in all_days if d >= day][:ndays]
    log(f"replaying {days[0]} to {days[-1]} ({len(days)} days), {len(universe)} stocks, launch = {days[0]}")
    sb = Sandbox(REPLAY_DIR / "firm.db", REPLAY_DIR / "state", launch=days[0])
    for d in days:
        t0 = time.time()
        # Models may only see data before the day being replayed.
        src = ReplaySource(train.stocks, train.nifty_close, train.nifty_open, d)
        res = sb.run_day(d, src, train=train)
        log(f"{d}: {res['bars']} bars in {time.time() - t0:.0f}s | " + " | ".join(
            f"{k[-1]} Rs {v['equity']:,.0f} ({v['entries']} entries)" for k, v in res.items() if k != "bars"))
    sb.close()
    return 0


# ---------------------------------------------------------------- live service

LOCK_PORT = 47655           # the real service; the uncounted shake-down uses the next port
STARTUP_FILE = Path(os.getenv("APPDATA", "")) / r"Microsoft\Windows\Start Menu\Programs\Startup\TradingFirmSandbox.cmd"
LOGIN_RETRY_UNTIL = dtime(9, 30)
START_FROM = dtime(8, 45)
DAY_END = dtime(15, 40)


def setup_logging() -> None:
    from logging.handlers import RotatingFileHandler

    from firm.config import LOG_DIR

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handlers = [RotatingFileHandler(LOG_DIR / "sandbox.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")]
    if sys.stdout is not None:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(level=logging.INFO, handlers=handlers, format="%(asctime)s %(levelname)s %(message)s")


def paths(shakedown: bool):
    from firm.live.runner import LIVE_DIR
    from firm.live.sandbox import SANDBOX_DIR

    if shakedown:
        base = DATA_DIR / "sandbox_shakedown"
        return base / "firm.db", base / "state"
    return LIVE_DIR / "firm.db", SANDBOX_DIR


def should_run(day: date, now: datetime, cal, done: set, shakedown: bool, launch: date) -> bool:
    """Run a session today? Trading day, within the morning-to-close window, not already run,
    and (unless it is the uncounted shake-down) on or after the pre-registered launch date."""
    return (day not in done and cal.is_trading_day(day) and START_FROM <= now.time() < DAY_END
            and (shakedown or day >= launch))


def run_live_day(day: date, shakedown: bool, tlog, now_fn=None, sleep_fn=None, make_fyers=None, renew=None) -> str:
    """One trading day. Returns 'done', 'skipped' (no Fyers login by 09:30) or raises.
    now_fn, sleep_fn and make_fyers exist so the whole day can be tested against a fake Fyers."""
    from firm.activity import ActivityLog
    from firm.data.fyers import FyersData
    from firm.control import Control
    from firm.live.sandbox import LAUNCH, Sandbox, load_train_data
    from firm.live.sandbox_live import FyersSource, probe_token

    now_fn = now_fn or (lambda: datetime.now(IST))
    sleep_fn = sleep_fn or time.sleep
    make_fyers = make_fyers or FyersData
    if renew is None and make_fyers is FyersData:      # fakes in tests must never touch the real .env
        from firm.fyers_auth import auto_renew as renew
    db, state = paths(shakedown)
    activity = ActivityLog(db)
    agents = ["Sandbox A (history)", "Sandbox B (blank)", "Sandbox C (random)", "Sandbox D (cash)"]
    uni_file = state / "universe.json"
    state.mkdir(parents=True, exist_ok=True)
    universe = json.loads(uni_file.read_text()) if uni_file.exists() else None
    tlog("loading training data ...")
    train, universe = load_train_data(universe)
    if not uni_file.exists():
        uni_file.write_text(json.dumps(universe))

    warned = False
    while True:
        ok, msg = probe_token(make_fyers, renew=renew)
        if ok:
            break
        for a in agents:
            activity.status(a, "error", "Waiting for the Fyers login", "intraday_stocks")
        if not warned:
            activity.log("alert", "Sandbox", "Fyers login needed: run scripts\\fyers_login.py. Retrying every 2 "
                         f"minutes until {LOGIN_RETRY_UNTIL:%H:%M}. ({msg})", team="intraday_stocks")
            warned = True
        tlog(f"no Fyers login yet: {msg}")
        if now_fn().time() >= LOGIN_RETRY_UNTIL:
            activity.log("alert", "Sandbox", f"No Fyers login by {LOGIN_RETRY_UNTIL:%H:%M}: {day} is skipped, "
                         "all four accounts stay flat.", team="intraday_stocks")
            for a in agents:
                activity.status(a, "idle", f"Skipped {day}: no Fyers login", "intraday_stocks")
            activity.close()
            return "skipped"
        sleep_fn(120)

    activity.log("act", "Sandbox", f"Fyers login OK. Starting the {day} session"
                 + (" (SHAKE-DOWN, not counted)" if shakedown else "") + ".", team="intraday_stocks")
    fyers = make_fyers()
    history = FyersSource.build_history(fyers, universe, day, log=tlog)
    source = FyersSource(fyers, universe, day, history, now=now_fn, sleep=sleep_fn, log=tlog)
    sb = Sandbox(db, state, launch=day if shakedown else LAUNCH, control=Control())
    try:
        res = sb.run_day(day, source, train=train, clock=now_fn)
        tlog(f"{day} done: " + ", ".join(f"{k} {v['equity']:,.0f}" for k, v in res.items() if k != "bars"))
    finally:
        sb.close()
        activity.close()
    return "done"


def cmd_live(shakedown: bool, once: bool = False) -> int:
    import socket

    from firm.calendar import TradingCalendar
    from firm.config import LOG_DIR
    from firm.live.sandbox import LAUNCH

    setup_logging()
    tlog = logging.getLogger("sandbox").info
    lock = socket.socket()
    try:
        lock.bind(("127.0.0.1", LOCK_PORT + (1 if shakedown else 0)))
    except OSError:
        tlog("sandbox already running; exiting")
        return 0
    from firm.control import Control
    from firm.live.sandbox import StopRequested

    ctl = Control()
    from firm.ops import write_pid

    write_pid("sandbox_shakedown" if shakedown else "sandbox")
    cal = TradingCalendar()
    done: set[date] = set()
    tlog("sandbox service started" + (" (shake-down)" if shakedown else f", launch {LAUNCH}"))
    while True:
        now = datetime.now(IST)
        day = now.date()
        ctl.beat("sandbox")
        if not ctl.service_should_run("sandbox"):
            tlog("sandbox stopped from the Control page; exiting")
            return 0
        try:
            if should_run(day, now, cal, done, shakedown, LAUNCH):
                status = run_live_day(day, shakedown, tlog)
                done.add(day)
                tlog(f"{day}: {status}")
                if once:
                    return 0
            elif once and now.time() >= DAY_END:
                tlog("--once: today's session window has passed; exiting")
                return 0
        except StopRequested:
            tlog("sandbox stopped from the Control page mid-session; state is saved, a restart resumes")
            return 0
        except Exception:
            logging.getLogger("sandbox").exception("sandbox day failed")
            try:
                from firm.activity import ActivityLog
                db, _ = paths(shakedown)
                a = ActivityLog(db)
                a.log("alert", "Sandbox", "The sandbox session crashed; retrying in 2 minutes (it resumes from "
                      "its saved state). Details in logs\\sandbox.log", team="intraday_stocks")
                a.close()
            except Exception:
                pass
            time.sleep(120)
            continue
        time.sleep(60)


def cmd_install_startup() -> int:
    pythonw = ROOT / ".venv" / "Scripts" / "pythonw.exe"
    STARTUP_FILE.write_text(f'@echo off\r\ncd /d "{ROOT}"\r\nstart "" "{pythonw}" "{ROOT / "scripts" / "sandbox.py"}" live\r\n',
                            encoding="utf-8")
    print(f"Installed: {STARTUP_FILE}\nThe sandbox service starts at every Windows login (no window).")
    return 0


def cmd_status() -> int:
    from firm.live.runner import LIVE_DIR
    from firm.storage import Storage

    st = Storage(LIVE_DIR / "firm.db")
    for row in st.conn.execute("SELECT name, status, last_bar, last_run FROM books WHERE status='sandbox'"):
        print("  ", *row)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("replay")
    rp.add_argument("--day", type=date.fromisoformat, required=True)
    rp.add_argument("--ndays", type=int, default=1)
    lv = sub.add_parser("live")
    lv.add_argument("--shakedown", action="store_true", help="uncounted test run with separate data")
    lv.add_argument("--once", action="store_true", help="run today's session, then exit (instead of waiting for tomorrow)")
    sub.add_parser("status")
    sub.add_parser("install-startup")
    sub.add_parser("uninstall-startup")
    args = p.parse_args()
    if args.cmd == "replay":
        return cmd_replay(args.day, args.ndays)
    if args.cmd == "status":
        return cmd_status()
    if args.cmd == "install-startup":
        return cmd_install_startup()
    if args.cmd == "uninstall-startup":
        STARTUP_FILE.unlink(missing_ok=True)
        print("Removed startup entry.")
        return 0
    return cmd_live(args.shakedown, args.once)


if __name__ == "__main__":
    raise SystemExit(main())
