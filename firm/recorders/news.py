"""News headlines (RSS) and NSE corporate announcements."""
from __future__ import annotations

import calendar
import hashlib
import json
from datetime import date, datetime, timezone

import feedparser
import requests

from firm.config import IST
from firm.universe import USER_AGENT

FEEDS = {
    "et_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "et_stocks": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "bs_markets": "https://www.business-standard.com/rss/markets-106.rss",
    "mint_markets": "https://www.livemint.com/rss/markets",
    "gnews_markets": "https://news.google.com/rss/search?q=NSE+OR+Sensex+OR+Nifty+stocks&hl=en-IN&gl=IN&ceid=IN:en",
}


def now_ist() -> str:
    return datetime.now(IST).isoformat(timespec="seconds")


def parse_feed(source: str, content: bytes, seen_at: str) -> list[dict]:
    items = []
    for e in feedparser.parse(content).entries:
        title = (e.get("title") or "").strip()
        if not title:
            continue
        link = e.get("link") or ""
        published = None
        if e.get("published_parsed"):
            ts = calendar.timegm(e.published_parsed)  # feedparser normalises to UTC
            published = datetime.fromtimestamp(ts, timezone.utc).astimezone(IST).isoformat()
        items.append({
            "id": hashlib.sha1((link or title).encode("utf-8")).hexdigest(),
            "source": source,
            "title": title,
            "summary": (e.get("summary") or "")[:2000],
            "link": link,
            "published": published,
            "first_seen": seen_at,
        })
    return items


def fetch_news(session: requests.Session | None = None) -> tuple[list[dict], dict[str, str]]:
    s = session or requests.Session()
    seen_at = now_ist()
    items, errors = [], {}
    for name, url in FEEDS.items():
        try:
            r = s.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
            r.raise_for_status()
            items.extend(parse_feed(name, r.content, seen_at))
        except Exception as e:
            errors[name] = f"{type(e).__name__}: {e}"[:200]
    return items, errors


class NSEAnnouncements:
    """NSE's website API. Unofficial: needs cookies from the home page; poll politely."""

    HOME = "https://www.nseindia.com"
    URL = "https://www.nseindia.com/api/corporate-announcements"
    REFERER = "https://www.nseindia.com/companies-listing/corporate-filings-announcements"

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT, "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9",
        })
        self._warm = False

    def _warm_up(self) -> None:
        self.session.get(self.HOME, timeout=20)
        self._warm = True

    BOARD_URL = "https://www.nseindia.com/api/corporate-board-meetings"
    BOARD_REFERER = "https://www.nseindia.com/companies-listing/corporate-filings-board-meetings"

    def _get_list(self, url: str, referer: str, start: date, end: date) -> list[dict]:
        params = {"index": "equities", "from_date": start.strftime("%d-%m-%Y"),
                  "to_date": end.strftime("%d-%m-%Y")}
        for attempt in range(2):
            if not self._warm:
                self._warm_up()
            r = self.session.get(url, params=params, headers={"Referer": referer}, timeout=30)
            if r.status_code in (401, 403) and attempt == 0:
                self._warm = False  # cookies expired: refresh once
                continue
            r.raise_for_status()
            data = r.json()
            return data if isinstance(data, list) else []
        return []

    def fetch(self, day: date) -> list[dict]:
        """Corporate announcements disseminated on `day`."""
        return self._get_list(self.URL, self.REFERER, day, day)

    def fetch_board_meetings(self, start: date, end: date) -> list[dict]:
        """Board meetings scheduled between start and end (by meeting date)."""
        return self._get_list(self.BOARD_URL, self.BOARD_REFERER, start, end)


def _parse_nse_time(s: str | None) -> str | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%d-%b-%Y %H:%M:%S").replace(tzinfo=IST).isoformat()
    except ValueError:
        return None


def parse_board_meeting(m: dict, seen_at: str) -> dict:
    desc = m.get("bm_desc") or ""
    purpose = m.get("bm_purpose") or ""
    try:
        meeting = datetime.strptime(m.get("bm_date") or "", "%d-%b-%Y").date().isoformat()
    except ValueError:
        meeting = None
    key = f"{m.get('bm_symbol')}|{m.get('bm_date')}|{m.get('bm_timestamp')}"
    return {
        "id": hashlib.sha1(key.encode("utf-8")).hexdigest(),
        "symbol": m.get("bm_symbol"),
        "meeting_date": meeting,
        "purpose": purpose,
        "description": desc,
        "is_results": int("financial result" in desc.lower() or "result" in purpose.lower()),
        "intimated_at": _parse_nse_time(m.get("bm_timestamp")),
        "first_seen": seen_at,
        "raw": json.dumps(m, ensure_ascii=False),
    }


def parse_announcement(a: dict, seen_at: str) -> dict:
    return {
        "id": str(a.get("seq_id") or hashlib.sha1(json.dumps(a, sort_keys=True).encode()).hexdigest()),
        "symbol": a.get("symbol"),
        "company": a.get("sm_name"),
        "category": a.get("desc"),
        "text": a.get("attchmntText"),
        "attachment": a.get("attchmntFile"),
        "has_xbrl": int(bool(a.get("hasXbrl"))),
        "industry": a.get("smIndustry"),
        # Older filings carry exchdisstime = "-" (truthy but unparseable): fall back to an_dt.
        "exchange_time": _parse_nse_time(a.get("exchdisstime")) or _parse_nse_time(a.get("an_dt")),
        "first_seen": seen_at,
        "raw": json.dumps(a, ensure_ascii=False),
    }
