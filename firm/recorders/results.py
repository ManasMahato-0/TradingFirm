"""Quarterly results from NSE's structured feed and the XBRL files behind it.

The feed lists one row per results filing (with its public time) and links an XBRL file
holding the numbers. XBRL files exist from about mid-2018. We keep, per company and
quarter, the FIRST filing's numbers (later revisions are ignored, so the data is what
the market saw at the time). Amounts are stored in Rs crore.
"""
from __future__ import annotations

import re
import time
from datetime import date, datetime

from firm.config import IST
from firm.recorders.news import NSEAnnouncements, now_ist

LIST_URL = "https://www.nseindia.com/api/corporates-financial-results"       # to Dec-2024 quarter
INTEGRATED_URL = "https://www.nseindia.com/api/integrated-filing-results"    # from Apr 2025 (SEBI integrated filing)
INTEGRATED_REFERER = "https://www.nseindia.com/companies-listing/corporate-integrated-filing"
INTEGRATED_TYPE = "Integrated Filing- Financials"
LIST_REFERER = "https://www.nseindia.com/companies-listing/corporate-filings-financial-results"
CRORE = 1e7

# Tag fallbacks: companies, banks and finance companies use different taxonomies.
REVENUE_TAGS = ["RevenueFromOperations", "Income"]
INCOME_TAGS = ["Income", "RevenueFromOperations"]
PBT_TAGS = ["ProfitBeforeTax", "ProfitLossFromOrdinaryActivitiesBeforeTax", "ProfitBeforeExceptionalItemsAndTax"]
PAT_TAGS = ["ProfitOrLossAttributableToOwnersOfParent",
            "ProfitLossAfterTaxesMinorityInterestAndShareOfProfitLossOfAssociates",
            "ProfitLossForPeriod", "ProfitLossForThePeriod", "ProfitLossFromOrdinaryActivitiesAfterTax"]
EPS_TAGS = ["BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
            "BasicEarningsPerShareAfterExtraordinaryItems", "BasicEarningsLossPerShareFromContinuingOperations",
            "BasicEarningsPerShareBeforeExtraordinaryItems"]

_VALUE = re.compile(r"""<(?:[\w-]+:)?(?P<tag>[A-Za-z0-9]+)\b(?P<attrs>[^>]*)>\s*(?P<val>[-+]?[\d.,]+)\s*<""")
_DATE = re.compile(r"<(?:[\w-]+:)?(?P<tag>DateOfStartOfReportingPeriod|DateOfEndOfReportingPeriod)[^>]*>\s*"
                   r"(?P<val>\d{4}-\d{2}-\d{2})\s*<")


def _num(s: str) -> float | None:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def parse_xbrl(text: str) -> dict:
    """Current-quarter ('OneD' context) numbers from one results XBRL file, in Rs crore
    (EPS in Rs). Missing items are None."""
    vals: dict[str, float] = {}
    for m in _VALUE.finditer(text):
        if 'contextRef="OneD"' in m.group("attrs") and m.group("tag") not in vals:
            v = _num(m.group("val"))
            if v is not None:
                vals[m.group("tag")] = v
    dates = {m.group("tag"): m.group("val") for m in _DATE.finditer(text)}

    def pick(tags, crore=True):
        for t in tags:
            if t in vals:
                return vals[t] / CRORE if crore else vals[t]
        return None

    return {"income": pick(INCOME_TAGS), "revenue": pick(REVENUE_TAGS), "pbt": pick(PBT_TAGS),
            "pat": pick(PAT_TAGS), "eps": pick(EPS_TAGS, crore=False),
            "start": dates.get("DateOfStartOfReportingPeriod"), "end": dates.get("DateOfEndOfReportingPeriod")}


def _iso(day: str | None) -> str | None:
    try:
        return datetime.strptime(day or "", "%d-%b-%Y").date().isoformat()
    except ValueError:
        return None


def _public_time(row: dict) -> str | None:
    for key in ("exchdisstime", "broadCastDate"):
        try:
            return datetime.strptime(row[key], "%d-%b-%Y %H:%M:%S").replace(tzinfo=IST).isoformat(timespec="seconds")
        except (KeyError, TypeError, ValueError):
            continue
    return None


def usable_row(row: dict) -> bool:
    """A quarterly, non-cumulative filing that links an XBRL file."""
    return (row.get("period") == "Quarterly" and row.get("cumulative") == "Non-cumulative"
            and bool(row.get("xbrl")) and not row["xbrl"].endswith("-"))


def row_to_record(row: dict, numbers: dict, seen_at: str) -> dict | None:
    """Combine a feed row and parsed numbers. None if it is not a clean 3-month quarter."""
    to_date, pub = _iso(row.get("toDate")), _public_time(row)
    if to_date is None or pub is None or numbers.get("pat") is None and numbers.get("revenue") is None:
        return None
    from_date = _iso(row.get("fromDate"))
    if from_date:  # the feed's own period; half-year or full-year rows are not a quarter
        span = (date.fromisoformat(to_date) - date.fromisoformat(from_date)).days
        if not 80 <= span <= 100:
            return None
    return {"symbol": row["symbol"], "to_date": to_date, "consolidated": int(row.get("consolidated") == "Consolidated"),
            "from_date": _iso(row.get("fromDate")), "broadcast": pub, "audited": row.get("audited"),
            "seq": row.get("seqNumber"), "income": numbers["income"], "revenue": numbers["revenue"],
            "pbt": numbers["pbt"], "pat": numbers["pat"], "eps": numbers["eps"], "xbrl_url": row["xbrl"],
            "first_seen": seen_at}


def normalise_integrated(r: dict) -> dict | None:
    """Integrated-filing row -> the same shape as the older feed's rows."""
    link = r.get("xbrl") or ""
    if not link or link.endswith("-") or link.endswith("/null") or r.get("consolidated") not in ("Consolidated", "Standalone"):
        return None
    return {"symbol": r.get("symbol"), "toDate": r.get("qe_Date"), "fromDate": None,
            "consolidated": r["consolidated"], "exchdisstime": r.get("broadcast_Date"),
            "broadCastDate": r.get("broadcast_Date"), "audited": r.get("audited"),
            "seqNumber": r.get("seq_Id"), "xbrl": link, "period": "Quarterly", "cumulative": "Non-cumulative"}


class NSEResults(NSEAnnouncements):
    """NSE results feed + XBRL downloads. Polite: one request at a time with a pause."""

    def __init__(self, pause: float = 1.0) -> None:
        super().__init__()
        self.pause = pause

    def _get(self, url: str, params: dict | None = None, referer: str = LIST_REFERER):
        for attempt in range(3):
            if not self._warm:
                self._warm_up()
            r = self.session.get(url, params=params, headers={"Referer": referer}, timeout=40)
            if r.status_code in (401, 403) and attempt < 2:
                self._warm = False
                time.sleep(2)
                continue
            r.raise_for_status()
            time.sleep(self.pause)
            return r
        raise RuntimeError("NSE refused the request")

    def list_symbol(self, symbol: str) -> list[dict]:
        j = self._get(LIST_URL, {"index": "equities", "symbol": symbol, "period": "Quarterly"}).json()
        return [r for r in j if usable_row(r)] if isinstance(j, list) else []

    def list_range(self, start: date, end: date) -> list[dict]:
        j = self._get(LIST_URL, {"index": "equities", "period": "Quarterly",
                                 "from_date": start.strftime("%d-%m-%Y"), "to_date": end.strftime("%d-%m-%Y")}).json()
        return [r for r in j if usable_row(r)] if isinstance(j, list) else []

    def _integrated(self, params: dict) -> list[dict]:
        rows, page = [], 0
        while True:
            j = self._get(INTEGRATED_URL, {"index": "equities", "type": INTEGRATED_TYPE, **params,
                                           **({"page": page} if page else {})}, referer=INTEGRATED_REFERER).json()
            data = j.get("data", []) if isinstance(j, dict) else []
            rows += [n for n in map(normalise_integrated, data) if n]
            total = j.get("totalCount", len(data)) if isinstance(j, dict) else 0
            page += 1
            if not data or len(rows) >= total or page > 40:
                return rows

    def list_integrated_symbol(self, symbol: str) -> list[dict]:
        return self._integrated({"symbol": symbol})

    def list_integrated_range(self, start: date, end: date) -> list[dict]:
        return self._integrated({"from_date": start.strftime("%d-%m-%Y"), "to_date": end.strftime("%d-%m-%Y")})

    def numbers(self, xbrl_url: str) -> dict:
        return parse_xbrl(self._get(xbrl_url.replace("nsearchives.nseindia.in", "nsearchives.nseindia.com")).text)


def collect(nse: NSEResults, rows: list[dict], known: set, symbols: set[str] | None = None,
            log=None, workers: int = 4) -> list[dict]:
    """Download and parse XBRL for rows not yet stored. Oldest filing first, so the first
    filing of a quarter is the one kept. Consolidated numbers are fetched first; standalone
    only for quarters where no consolidated result is available. A few downloads run in
    parallel (NSE tolerates this; each request still pauses)."""
    from concurrent.futures import ThreadPoolExecutor

    rows = sorted((r for r in rows if symbols is None or r["symbol"] in symbols), key=lambda r: _public_time(r) or "")

    def one(r: dict) -> dict | None:
        try:
            return row_to_record(r, nse.numbers(r["xbrl"]), now_ist())
        except Exception as e:  # one bad file must not stop the run
            if log:
                log(f"{r['symbol']} {r.get('toDate')}: {type(e).__name__}: {e}")
            return None

    def stage(batch: list[dict]) -> list[dict]:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return [rec for rec in pool.map(one, batch) if rec]

    def first_per_key(batch: list[dict]) -> list[dict]:
        seen, out = set(), []
        for r in batch:
            key = (r["symbol"], _iso(r.get("toDate")), int(r.get("consolidated") == "Consolidated"))
            if key not in known and key not in seen:
                seen.add(key)
                out.append(r)
        return out

    cons = stage(first_per_key([r for r in rows if r.get("consolidated") == "Consolidated"]))
    have_cons = {(r["symbol"], r["to_date"]) for r in cons} | {(s, d) for s, d, c in known if c == 1}
    solo = [r for r in first_per_key([r for r in rows if r.get("consolidated") != "Consolidated"])
            if (r["symbol"], _iso(r.get("toDate"))) not in have_cons]
    return cons + stage(solo)
