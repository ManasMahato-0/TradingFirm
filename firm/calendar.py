"""NSE trading calendar: weekends plus exchange holidays (cash market).

Holidays come from NSE's website and are cached per year. If NSE is unreachable and
no cache exists, weekdays are assumed to be trading days and `known` reports False.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta

from firm.config import UNIVERSE_DIR

HOLIDAY_URL = "https://www.nseindia.com/api/holiday-master"
HOLIDAY_REFERER = "https://www.nseindia.com/resources/exchange-communication-holidays"


def _cache_path(year: int):
    return UNIVERSE_DIR / f"nse_holidays_{year}.json"


def fetch_nse_holidays() -> dict[int, list[str]]:
    from firm.recorders.news import NSEAnnouncements

    nse = NSEAnnouncements()
    nse._warm_up()
    r = nse.session.get(HOLIDAY_URL, params={"type": "trading"},
                        headers={"Referer": HOLIDAY_REFERER}, timeout=30)
    r.raise_for_status()
    by_year: dict[int, list[str]] = {}
    for row in r.json().get("CM", []):
        d = datetime.strptime(row["tradingDate"], "%d-%b-%Y").date()
        by_year.setdefault(d.year, []).append(d.isoformat())
    return by_year


class TradingCalendar:
    def __init__(self, holidays: dict[int, set[date]] | None = None, refresh: bool = True) -> None:
        self.holidays: dict[int, set[date]] = dict(holidays or {})
        self._refresh = refresh and holidays is None

    def _year(self, year: int) -> set[date] | None:
        if year in self.holidays:
            return self.holidays[year]
        path = _cache_path(year)
        if path.exists():
            self.holidays[year] = {date.fromisoformat(s) for s in json.loads(path.read_text())}
            return self.holidays[year]
        if self._refresh:
            self._refresh = False  # one attempt per process
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                for y, days in fetch_nse_holidays().items():
                    _cache_path(y).write_text(json.dumps(sorted(days)))
            except Exception:
                return None
            return self._year(year)
        return None

    def known(self, year: int) -> bool:
        return self._year(year) is not None

    def is_trading_day(self, d: date) -> bool:
        return d.weekday() < 5 and d not in (self._year(d.year) or set())

    def next_trading_day(self, d: date) -> date:
        n = d + timedelta(days=1)
        while not self.is_trading_day(n):
            n += timedelta(days=1)
        return n

    def is_month_end(self, d: date) -> bool:
        return self.next_trading_day(d).month != d.month
