"""Recorder service: saves news, filings, option chains and price bars every day.

Usage (from the TradindFirm folder):
    .venv\\Scripts\\python scripts\\recorder.py run               # keep running all day
    .venv\\Scripts\\python scripts\\recorder.py once news         # one job: news | announcements | options | bars
    .venv\\Scripts\\python scripts\\recorder.py backfill          # one-time grab of available history
    .venv\\Scripts\\python scripts\\recorder.py status            # health check
    .venv\\Scripts\\python scripts\\recorder.py install-startup   # start automatically at Windows login
    .venv\\Scripts\\python scripts\\recorder.py uninstall-startup

Schedule while running:
    news + NSE announcements   every 5 minutes, all day
    option chains              every 5 minutes while NSE is open (Fyers market status)
    price bars                 once a day after 15:45 IST, last 7 days (catches up missed days)
    researcher (LLM)           every 5 minutes: notes on new filings and headlines
    paper runner               once a day after bars are saved and it is past 16:00 IST
    manager + AI Trader guard  after the runner: daily note, shadow risk proposals
                               (fills, equity, month-end decisions; see scripts/runner.py)
Every job also reports to the Trading Floor feed (data/live/firm.db).
"""
from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import time
from datetime import date, datetime, timedelta
from datetime import time as dtime
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from firm.activity import ActivityLog  # noqa: E402
from firm.control import Control  # noqa: E402
from firm.config import BARS_DIR, IST, LOG_DIR, OPTIONS_DIR, RECORDER_DB_PATH  # noqa: E402
from firm.data.fyers import FyersAuthError  # noqa: E402
from firm.data.yahoo import YahooData  # noqa: E402
from firm.recorders.bars import record_bars  # noqa: E402
from firm.recorders.db import RecorderDB  # noqa: E402
from firm.recorders.news import (  # noqa: E402
    NSEAnnouncements, fetch_news, now_ist, parse_announcement, parse_board_meeting,
)
from firm.recorders.options import record_option_chains
from firm.recorders.results import NSEResults, collect  # noqa: E402
from firm.universe import INDEXES, nifty100  # noqa: E402

log = logging.getLogger("recorder")

MARKET_OPEN, MARKET_CLOSE = dtime(9, 15), dtime(15, 30)
BARS_AFTER = dtime(15, 45)
RUNNER_AFTER = dtime(16, 0)
NEWS_EVERY_S = 300
BOARD_EVERY_S = 3600
OPTIONS_SLOT_MIN = 5
BARS_RETRY_S = 1800
TICK_S = 20
LOCK_PORT = 47653
STARTUP_FILE = Path(os.getenv("APPDATA", "")) / r"Microsoft\Windows\Start Menu\Programs\Startup\TradingFirmRecorder.cmd"


def now() -> datetime:
    return datetime.now(IST)


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        RotatingFileHandler(LOG_DIR / "recorder.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    ]
    if sys.stdout is not None:  # pythonw has no console
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(message)s")


def keep_awake(on: bool) -> None:
    """Stop Windows sleeping on idle while on. Does not override closing the lid."""
    if sys.platform != "win32":
        return
    import ctypes

    es_continuous, es_system_required = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | (es_system_required if on else 0))


def summarize(results: dict[str, dict]) -> tuple[int, str, str]:
    rows = sum(r["rows"] for r in results.values())
    parts, n_err = [], 0
    for name, r in results.items():
        parts.append(f"{name} +{r['rows']} rows ({r['symbols_ok']} ok)")
        if r["errors"]:
            n_err += len(r["errors"])
            sample = "; ".join(f"{k}: {v}" for k, v in list(r["errors"].items())[:3])
            parts.append(f"{name} errors: {len(r['errors'])} [{sample}]")
    return rows, ("partial" if n_err else "ok"), " | ".join(parts)


class Recorder:
    def __init__(self) -> None:
        self.db = RecorderDB(RECORDER_DB_PATH)
        self.ctl = Control()      # the owner's switches and commands (dashboard Control page)
        from firm.live.runner import LIVE_DIR

        self.activity = ActivityLog(LIVE_DIR / "firm.db")
        self.nse = NSEAnnouncements()
        self._fyers = None
        self.watch = self._make_watch()
        self.last_run: dict[str, float] = {}
        self.last_options_slot: datetime | None = None

    def _make_watch(self):
        from firm import fyers_auth, ops
        from firm.alerts import Watch, notify
        from firm.calendar import TradingCalendar

        cal = TradingCalendar(refresh=False)
        return Watch(
            probe=fyers_auth.token_status,
            notifier=lambda title, msg: (notify(title, msg),
                                         self.activity.log("alert", "Recorder", f"{title}: {msg}")),
            # a sandbox the owner switched off on purpose is not "down"
            sandbox_running=lambda: ops.is_running("sandbox") or self.ctl.get("service.sandbox.run") != "run",
            is_trading_day=cal.is_trading_day,
            enabled=lambda: self.ctl.get_bool("alert.enabled"),
            log=log.info)

    def fyers(self):
        if self._fyers is None:
            from firm.data.fyers import FyersData

            self._fyers = FyersData()
        return self._fyers

    def run_job(self, job: str, fn) -> str:
        started = now()
        try:
            try:
                rows, status, msg = fn()
            except FyersAuthError:
                from firm import fyers_auth

                renewed, _ = fyers_auth.auto_renew()     # quiet daily renewal from the refresh token + PIN
                if not renewed:
                    raise
                self._fyers = None
                self.activity.log("act", "Recorder", "Fyers login renewed automatically.")
                rows, status, msg = fn()
        except FyersAuthError as e:
            self._fyers = None  # reload token from .env next time
            rows, status, msg = 0, "auth_failed", f"{e}. Run scripts/fyers_login.py"
        except Exception as e:
            log.exception("job %s crashed", job)
            rows, status, msg = 0, "error", f"{type(e).__name__}: {e}"
        self.db.log_run(job, started.isoformat(timespec="seconds"),
                        now().isoformat(timespec="seconds"), status, rows, msg)
        (log.info if status in ("ok", "skipped") else log.warning)(
            "%s: %s, %s rows. %s", job, status, rows, msg[:300])
        self.report(job, status, rows, msg)
        return status

    def report(self, job: str, status: str, rows: int, msg: str) -> None:
        """Post to the Trading Floor. Quiet when nothing new happened."""
        try:
            if status in ("ok", "skipped"):
                if rows and job not in ("runner", "researcher", "desk", "results"):
                    self.activity.log("act", "Recorder", f"{job}: saved {rows} new rows. {msg[:200]}")
                self.activity.status("Recorder", "idle", f"Last: {job} at {now():%H:%M} ({status})")
            else:
                self.activity.log("alert", "Recorder", f"{job} {status}: {msg[:300]}")
                self.activity.status("Recorder", "error" if status != "partial" else "idle",
                                     f"{job} {status} at {now():%H:%M}")
        except Exception:
            log.exception("activity report failed")

    # Jobs return (rows, status, message).

    def job_news(self):
        items, errors = fetch_news()
        new = self.db.insert_news(items)
        msg = f"{len(items)} fetched" + ("; " + "; ".join(f"{k}: {v}" for k, v in errors.items()) if errors else "")
        return new, ("partial" if errors else "ok"), msg

    def job_announcements(self, day: date | None = None):
        day = day or now().date()
        seen = now_ist()
        rows = [parse_announcement(a, seen) for a in self.nse.fetch(day)]
        return self.db.insert_announcements(rows), "ok", f"{len(rows)} listed for {day}"

    def job_board_meetings(self, start: date | None = None, end: date | None = None):
        start = start or now().date() - timedelta(days=7)
        end = end or now().date() + timedelta(days=60)
        seen = now_ist()
        rows = [parse_board_meeting(m, seen) for m in self.nse.fetch_board_meetings(start, end)]
        return self.db.insert_board_meetings(rows), "ok", f"{len(rows)} listed {start} to {end}"

    def backfill_board_meetings(self, start: date) -> None:
        def run():
            total, chunk = 0, start
            stop = now().date() + timedelta(days=60)
            while chunk <= stop:
                chunk_end = min(chunk + timedelta(days=30), stop)
                total += self.job_board_meetings(chunk, chunk_end)[0]
                chunk = chunk_end + timedelta(days=1)
                time.sleep(2)
            return total, "ok", f"{start} to {stop}"

        self.run_job("backfill_board_meetings", run)

    def job_results(self, days: int = 4):
        """New quarterly results (NSE integrated filings) for our stocks: numbers into the DB,
        a plain-English line onto the Trading Floor."""
        syms = set(nifty100())
        nse = NSEResults(pause=0.5)
        end = now().date()
        listed = [r for r in nse.list_integrated_range(end - timedelta(days=days), end) if r["symbol"] in syms]
        recs = collect(nse, listed, self.db.known_results(), log=log.warning)
        added = self.db.insert_results(recs)
        try:
            from firm.features.fundamentals import describe_result

            for rec in recs:
                self.activity.log("think", "Researcher", describe_result(self.db.conn, rec), team="swing_stocks",
                                  ts=rec["broadcast"], details={"symbol": rec["symbol"], "to_date": rec["to_date"]})
        except Exception:
            log.exception("could not post results to the feed")
        return added, "ok", f"{len(listed)} filings listed for our stocks, {added} new"

    def backfill_results(self) -> None:
        """All results history for our stocks: old feed (mid-2018 to Dec 2024) plus integrated
        filings (Apr 2025 on). Resumable: stored quarters are skipped."""
        def run():
            syms = nifty100()
            nse = NSEResults(pause=0.6)
            total = 0
            for i, sym in enumerate(syms):
                try:
                    rows = nse.list_symbol(sym) + nse.list_integrated_symbol(sym)
                    recs = collect(nse, rows, self.db.known_results(), log=log.warning)
                    total += self.db.insert_results(recs)
                except Exception as e:
                    log.warning("results backfill %s: %s: %s", sym, type(e).__name__, e)
                if i % 10 == 9:
                    log.info("results backfill: %d/%d stocks done, %d quarters stored", i + 1, len(syms), total)
            return total, "ok", f"{len(syms)} stocks, {total} quarters"

        self.run_job("backfill_results", run)

    def job_options(self):
        status = self.fyers().nse_cash_status()
        if status != "OPEN":
            return 0, "skipped", f"NSE status {status}"
        return record_option_chains(self.fyers(), now().replace(microsecond=0)), "ok", ""

    def job_bars(self, days: int = 7):
        end = now().date() + timedelta(days=1)
        symbols = nifty100() + INDEXES
        f = self.fyers()
        return summarize({
            "1m": record_bars(f, "1m", "1m", symbols, end - timedelta(days=days), end),
            "1d": record_bars(f, "1d", "1d", symbols, end - timedelta(days=max(days, 15)), end),
            "1d_yahoo": record_bars(YahooData(), "1d_yahoo", "1d", symbols,
                                    end - timedelta(days=max(days, 15)), end),
        })

    def bars_done_today(self) -> bool:
        row = self.db.conn.execute(
            "SELECT 1 FROM recorder_runs WHERE job='bars' AND status IN ('ok','partial')"
            " AND started >= ? LIMIT 1", (now().date().isoformat(),)
        ).fetchone()
        return row is not None

    def _due(self, job: str, every_s: float) -> bool:
        if time.monotonic() - self.last_run.get(job, -1e12) >= every_s:
            self.last_run[job] = time.monotonic()
            return True
        return False

    def _go(self, job: str, every_s: float) -> bool:
        """The job is switched on AND due. A switched-off job keeps no schedule state."""
        return self.ctl.job_enabled(job) and self._due(job, every_s)

    def _job_functions(self) -> dict:
        return {"news": self.job_news, "announcements": self.job_announcements,
                "board_meetings": self.job_board_meetings, "options": self.job_options, "bars": self.job_bars,
                "results": self.job_results, "researcher": self.job_researcher, "runner": self.job_runner,
                "desk": self.job_desk, "backup": self.job_backup,
                "head": self.job_head}

    def handle_commands(self) -> None:
        """Run what the dashboard queued for the recorder ("run now", reload the Fyers token)."""
        while (cmd := self.ctl.claim("recorder")) is not None:
            try:
                if cmd["name"] == "run_job":
                    job = cmd["args"]["job"]
                    status = self.run_job(job, self._job_functions()[job])
                    self.ctl.finish(cmd["id"], "done" if status in ("ok", "partial", "skipped") else "error", status)
                elif cmd["name"] == "reload_fyers":
                    self._fyers = None          # rebuild the client from the (new) token in .env
                    self.ctl.finish(cmd["id"], "done", "Fyers token reloaded")
                else:
                    self.ctl.finish(cmd["id"], "error", f"unknown command {cmd['name']}")
            except Exception as e:
                log.exception("command %s failed", cmd["name"])
                self.ctl.finish(cmd["id"], "error", f"{type(e).__name__}: {e}")

    def tick(self) -> None:
        t = now()
        weekday = t.weekday() < 5
        keep_awake(weekday and dtime(9, 0) <= t.time() <= dtime(16, 30))

        # The owner's queued commands are checked between jobs, so a long job (the Researcher can run
        # for minutes) never leaves a "run now" waiting for the whole tick.
        self.handle_commands()
        try:
            self.watch.check(t)
        except Exception:
            log.exception("reminder check failed")
        if self._go("news", NEWS_EVERY_S):
            self.run_job("news", self.job_news)
        if self._go("announcements", NEWS_EVERY_S):
            self.run_job("announcements", self.job_announcements)
        if self._go("board_meetings", BOARD_EVERY_S):
            self.run_job("board_meetings", self.job_board_meetings)
        self.handle_commands()

        if weekday and MARKET_OPEN <= t.time() <= MARKET_CLOSE and self.ctl.job_enabled("options"):
            slot = t.replace(minute=t.minute - t.minute % OPTIONS_SLOT_MIN, second=0, microsecond=0)
            if slot != self.last_options_slot:
                self.last_options_slot = slot
                self.run_job("options", self.job_options)

        if (t.time() >= BARS_AFTER and self.ctl.job_enabled("bars") and not self.bars_done_today()
                and self._due("bars", BARS_RETRY_S)):
            self.run_job("bars", self.job_bars)

        if (t.time() >= RUNNER_AFTER and self.ctl.job_enabled("runner") and self.bars_done_today()
                and not self.done_today("runner") and self._due("runner", BARS_RETRY_S)):
            self.run_job("runner", self.job_runner)

        if (self.ctl.job_enabled("head") and self.done_today("desk") and not self.done_today("head")
                and self._due("head", BARS_RETRY_S)):
            self.run_job("head", self.job_head)
        if (t.time() >= dtime(18, 0) and self.ctl.job_enabled("backup") and not self.done_today("backup")
                and self._due("backup", BARS_RETRY_S)):
            self.run_job("backup", self.job_backup)
        if self._due("holidays", 86400):
            self.run_job("holidays", self.job_holidays)

        self.handle_commands()
        if self._go("results", 1800):
            self.run_job("results", self.job_results)
        self.handle_commands()
        if self._go("researcher", NEWS_EVERY_S):
            self.run_job("researcher", self.job_researcher)
        self.handle_commands()
        if (self.ctl.job_enabled("desk") and self.done_today("runner") and not self.done_today("desk")
                and self._due("desk", BARS_RETRY_S)):
            self.run_job("desk", self.job_desk)

    def job_backup(self):
        from firm.backup import default_destination, run_backup
        from firm.config import DATA_DIR

        dest = Path(self.ctl.get("backup.dir").strip() or default_destination())
        r = run_backup(DATA_DIR, dest, now().date())
        return len(r.databases) + r.files_copied, ("ok" if r.status == "ok" else "error"), r.message

    def job_holidays(self):
        from firm.calendar import refresh_holiday_cache

        added = refresh_holiday_cache()
        return len(added), "ok", ("new years cached: " + ", ".join(map(str, added))) if added else "nothing new"

    def job_researcher(self):
        from firm.agents.desk import run_research
        from firm.live.runner import LIVE_DIR

        r = run_research(LIVE_DIR / "firm.db")
        return r["saved"], r["status"], r["msg"]

    def job_desk(self):
        from firm.agents.desk import run_evening
        from firm.live.runner import LIVE_DIR

        r = run_evening(LIVE_DIR / "firm.db", control=self.ctl)
        return r["notes"] + r["proposals"], "ok", r["msg"]

    def job_head(self):
        """The Head (shadow): one firm report and an allocation proposal. Moves no money."""
        from firm.agents.head import write_report
        from firm.backtest.benchmarks import equal_weight
        from firm.backtest.data import MarketData
        from firm.live.books import PROBATION_START
        from firm.live.runner import LIVE_DIR
        from firm.universe import nifty100

        today = now().date()
        ew_ret = None
        try:
            closes = MarketData.from_store("1d_yahoo", nifty100(), start=PROBATION_START - timedelta(days=30)).panels["close"]
            ew = equal_weight(closes, PROBATION_START + timedelta(days=1), today + timedelta(days=1))
            ew_ret = float((1 + ew).prod() - 1) if len(ew) else None
        except Exception:
            log.warning("head: equal-weight benchmark unavailable", exc_info=True)
        r = write_report(LIVE_DIR / "firm.db", today, ew_ret=ew_ret, halted=self.ctl.get_bool("halt.firm"))
        self.activity.log("think", "Head", f"Proposal: {r['plan']['action']}. {r['plan']['why']}")
        return 1, "ok", r["plan"]["action"]

    def job_runner(self):
        from firm.live.books import books
        from firm.live.runner import LIVE_DIR, RunnerBusy, run_books

        try:
            out = run_books(books(), LIVE_DIR / "firm.db", LIVE_DIR, control=self.ctl)
        except RunnerBusy:
            return 0, "skipped", "already running"
        errors = [r for r in out if "error" in r]
        msg = "; ".join(f"{r['book']}: " + (r["error"] if "error" in r else f"+{r['days']} days to {r['last_bar']}")
                        for r in out)
        return sum(r.get("days", 0) for r in out), ("partial" if errors else "ok"), msg

    def done_today(self, job: str) -> bool:
        row = self.db.conn.execute(
            "SELECT 1 FROM recorder_runs WHERE job=? AND status IN ('ok','partial')"
            " AND started >= ? LIMIT 1", (job, now().date().isoformat())
        ).fetchone()
        return row is not None

    def run_forever(self) -> None:
        if not self.ctl.service_should_run("recorder"):
            log.info("recorder is switched off on the Control page; exiting")
            return
        log.info("recorder started, data in %s", RECORDER_DB_PATH.parent)
        # Catch announcements made while we were off.
        self.run_job("announcements", lambda: self.job_announcements(now().date() - timedelta(days=1)))
        try:
            while True:
                if not self.ctl.service_should_run("recorder"):
                    log.info("recorder stopped from the Control page")
                    return
                self.ctl.beat("recorder")
                try:
                    self.tick()
                    self.handle_commands()
                except Exception:
                    log.exception("tick failed")
                time.sleep(TICK_S)
        finally:
            keep_awake(False)

    def backfill(self, bar_days: int, ann_days: int, yahoo_since: date) -> None:
        end = now().date() + timedelta(days=1)
        symbols = nifty100() + INDEXES
        log.info("backfill: %d symbols", len(symbols))
        self.run_job("backfill_1m", lambda: summarize(
            {"1m": record_bars(self.fyers(), "1m", "1m", symbols, end - timedelta(days=bar_days), end)}))
        self.run_job("backfill_1d", lambda: summarize(
            {"1d": record_bars(self.fyers(), "1d", "1d", symbols, date(2021, 1, 1), end)}))
        self.run_job("backfill_1d_yahoo", lambda: summarize(
            {"1d_yahoo": record_bars(YahooData(), "1d_yahoo", "1d", symbols, yahoo_since, end)}))

        today = now().date()
        self.backfill_announcements(today - timedelta(days=ann_days - 1), today)

    def backfill_announcements(self, start: date, end: date) -> None:
        """Fetch NSE announcements day by day, newest first, politely."""
        days = [end - timedelta(days=i) for i in range((end - start).days + 1)]

        def announcements():
            total, failed = 0, []
            for i, day in enumerate(days):
                try:
                    total += self.job_announcements(day)[0]
                except Exception as e:
                    failed.append(f"{day}: {type(e).__name__}")
                time.sleep(2)  # be polite to NSE
                if i % 30 == 29:
                    log.info("announcements backfill: %d/%d days done (at %s), %d new",
                             i + 1, len(days), day, total)
            return total, ("partial" if failed else "ok"), \
                f"{start} to {end}, {len(days)} days; failed {len(failed)} {failed[:5]}"

        self.run_job("backfill_announcements", announcements)


def folder_size_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6 if p.exists() else 0.0


def status() -> None:
    db = RecorderDB(RECORDER_DB_PATH)
    print("Last run per job:")
    for job, started, st, rows, msg in db.last_runs():
        print(f"  {job:24} {started}  {st:11} rows={rows:<8} {msg[:90]}")
    n_news = db.conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    n_ann = db.conn.execute("SELECT COUNT(*) FROM announcements").fetchone()[0]
    print(f"\nStored: {n_news} news items, {n_ann} announcements")
    print(f"Disk: bars {folder_size_mb(BARS_DIR):.0f} MB, options {folder_size_mb(OPTIONS_DIR):.0f} MB, "
          f"recorder.db {RECORDER_DB_PATH.stat().st_size / 1e6 if RECORDER_DB_PATH.exists() else 0:.0f} MB")


def install_startup() -> None:
    pythonw = ROOT / ".venv" / "Scripts" / "pythonw.exe"
    STARTUP_FILE.write_text(
        f'@echo off\r\ncd /d "{ROOT}"\r\nstart "" "{pythonw}" "{ROOT / "scripts" / "recorder.py"}" run\r\n',
        encoding="utf-8",
    )
    print(f"Installed: {STARTUP_FILE}\nThe recorder will start at every Windows login (no window).")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    once = sub.add_parser("once")
    once.add_argument("job", choices=["news", "announcements", "board_meetings", "options", "bars", "runner",
                                      "researcher", "desk", "results", "backup", "holidays", "head"])
    bf = sub.add_parser("backfill")
    bf.add_argument("--bar-days", type=int, default=365)
    bf.add_argument("--ann-days", type=int, default=365)
    bf.add_argument("--yahoo-since", type=date.fromisoformat, default=date(2005, 1, 1))
    ba = sub.add_parser("backfill-announcements")
    ba.add_argument("--start", type=date.fromisoformat, required=True)
    ba.add_argument("--end", type=date.fromisoformat, required=True)
    bb = sub.add_parser("backfill-board-meetings")
    bb.add_argument("--start", type=date.fromisoformat, default=date(2011, 1, 1))
    sub.add_parser("backfill-results")
    sub.add_parser("status")
    sub.add_parser("install-startup")
    sub.add_parser("uninstall-startup")
    args = p.parse_args()

    if args.cmd == "status":
        status()
        return 0
    if args.cmd == "install-startup":
        install_startup()
        return 0
    if args.cmd == "uninstall-startup":
        STARTUP_FILE.unlink(missing_ok=True)
        print("Removed startup entry.")
        return 0

    setup_logging()
    if args.cmd == "run":
        lock = socket.socket()
        try:
            lock.bind(("127.0.0.1", LOCK_PORT))  # held for the process lifetime
        except OSError:
            log.info("recorder already running; exiting")
            return 0
        from firm.ops import write_pid

        write_pid("recorder")
        Recorder().run_forever()
    elif args.cmd == "once":
        rec = Recorder()
        status_ = rec.run_job(args.job, getattr(rec, f"job_{args.job}"))
        return 0 if status_ in ("ok", "partial", "skipped") else 1
    elif args.cmd == "backfill":
        Recorder().backfill(args.bar_days, args.ann_days, args.yahoo_since)
    elif args.cmd == "backfill-announcements":
        Recorder().backfill_announcements(args.start, args.end)
    elif args.cmd == "backfill-board-meetings":
        Recorder().backfill_board_meetings(args.start)
    elif args.cmd == "backfill-results":
        Recorder().backfill_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
