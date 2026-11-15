"""'What do I need to do today?' for the Home page. Plain logic, no Streamlit, no network."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from datetime import time as dtime


@dataclass(frozen=True)
class Task:
    title: str
    detail: str
    done: bool
    action: str = ""        # "login" = offer the Fyers login button
    info: bool = False      # a note, not something to tick off


@dataclass(frozen=True)
class Coming:
    day: date
    text: str


def todays_tasks(now: datetime, is_trading_day: bool, fyers_ok: bool | None, sandbox_launch: date,
                 swing_start: date) -> list[Task]:
    """fyers_ok: True/False once checked, None if not checked (outside the hours it matters)."""
    tasks: list[Task] = []
    today = now.date()
    if is_trading_day and now.time() < dtime(15, 30):
        tasks.append(Task(
            "Log in to Fyers",
            "Needed once every trading morning, before 9:15. It takes about a minute. Without it the system "
            "cannot get live prices, so the practice accounts sit the day out.",
            done=bool(fyers_ok), action="login"))
    if today == sandbox_launch:
        tasks.append(Task("Today the practice accounts start for real",
                          "Four make-believe accounts begin their 6-month test. Nothing else to do.", True, info=True))
    if today == swing_start:
        tasks.append(Task("Today the main swing experiment starts",
                          "It decides at the market close tonight and places its first paper orders tomorrow.", True,
                          info=True))
    if not tasks:
        tasks.append(Task("Nothing to do today",
                          "The system runs by itself. Come back whenever you like to see how it's going.", True,
                          info=True))
    return tasks


def coming_up(today: date, sandbox_launch: date, swing_start: date) -> list[Coming]:
    items = [Coming(sandbox_launch, "The four practice accounts (intraday trading) start their live test."),
             Coming(swing_start, "The main swing experiment opens its first paper positions the next morning.")]
    return [c for c in sorted(items, key=lambda c: c.day) if c.day >= today]


def greeting(now: datetime) -> str:
    h = now.hour
    return "Good morning" if h < 12 else "Good afternoon" if h < 17 else "Good evening"


def days_until(today: date, day: date) -> str:
    n = (day - today).days
    return "today" if n == 0 else "tomorrow" if n == 1 else f"in {n} days"
