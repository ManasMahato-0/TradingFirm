"""NSE daily F&O bhavcopy: one file per trading day with open/high/low/close/settle, volume and open
interest for EVERY option contract. Free, no login. We keep Nifty and Bank Nifty index options that
actually traded, one small parquet per day under data/fo/raw, so the download is resumable.

Two formats: the old one (to early July 2024) and NSE's UDiFF format (from then on). Both are
mapped to the same columns. Volume units differ (old: contracts, new: quantity), so volume is only
used as 'did it trade', never compared across the switch.
"""
from __future__ import annotations

import io
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from firm.config import DATA_DIR
from firm.recorders.news import NSEAnnouncements

FO_DIR = DATA_DIR / "fo" / "raw"
NEW_FORMAT_FROM = date(2024, 7, 8)
UNDERLYINGS = ("NIFTY", "BANKNIFTY")
REFERER = "https://www.nseindia.com/all-reports-derivatives"
COLUMNS = ["date", "symbol", "expiry", "strike", "opt_type", "open", "high", "low", "close", "settle",
           "volume", "oi", "fmt"]


def new_url(day: date) -> str:
    return f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{day:%Y%m%d}_F_0000.csv.zip"


def old_url(day: date) -> str:
    mon = day.strftime("%b").upper()
    return (f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{day:%Y}/{mon}/"
            f"fo{day:%d}{mon}{day:%Y}bhav.csv.zip")


def parse_bhav(df: pd.DataFrame) -> pd.DataFrame:
    """Index options of Nifty and Bank Nifty that traded, in the common columns."""
    if "TckrSymb" in df.columns:                                     # UDiFF format
        d = df[(df["TckrSymb"].isin(UNDERLYINGS)) & (df["FinInstrmTp"] == "IDO")]
        out = pd.DataFrame({
            "date": pd.to_datetime(d["TradDt"]).dt.date, "symbol": d["TckrSymb"],
            "expiry": pd.to_datetime(d["XpryDt"]).dt.date, "strike": d["StrkPric"].astype(float),
            "opt_type": d["OptnTp"], "open": d["OpnPric"], "high": d["HghPric"], "low": d["LwPric"],
            "close": d["ClsPric"], "settle": d["SttlmPric"], "volume": d["TtlTradgVol"], "oi": d["OpnIntrst"],
            "fmt": "udiff"})
    else:                                                            # old format
        d = df[(df["SYMBOL"].isin(UNDERLYINGS)) & (df["INSTRUMENT"] == "OPTIDX")]
        out = pd.DataFrame({
            "date": pd.to_datetime(d["TIMESTAMP"], format="%d-%b-%Y").dt.date, "symbol": d["SYMBOL"],
            "expiry": pd.to_datetime(d["EXPIRY_DT"], format="%d-%b-%Y").dt.date,
            "strike": d["STRIKE_PR"].astype(float), "opt_type": d["OPTION_TYP"], "open": d["OPEN"],
            "high": d["HIGH"], "low": d["LOW"], "close": d["CLOSE"], "settle": d["SETTLE_PR"],
            "volume": d["CONTRACTS"], "oi": d["OPEN_INT"], "fmt": "old"})
    out = out[(out["volume"] > 0) | (out["oi"] > 0)]                  # drop contracts that never existed in practice
    return out[COLUMNS].reset_index(drop=True)


class FOFetcher(NSEAnnouncements):
    def __init__(self, pause: float = 0.6, root: Path = FO_DIR) -> None:
        super().__init__()
        self.pause, self.root = pause, root
        root.mkdir(parents=True, exist_ok=True)
        self._no_data = root / "no_data.txt"

    def _skip_days(self) -> set[date]:
        if not self._no_data.exists():
            return set()
        return {date.fromisoformat(x) for x in self._no_data.read_text().split() if x}

    def fetch_day(self, day: date) -> pd.DataFrame | None:
        """None if NSE has no file for that day (holiday)."""
        urls = [new_url(day), old_url(day)] if day >= NEW_FORMAT_FROM else [old_url(day), new_url(day)]
        for url in urls:
            for attempt in range(3):
                if not self._warm:
                    self._warm_up()
                r = self.session.get(url, headers={"Referer": REFERER}, timeout=60)
                time.sleep(self.pause)
                if r.status_code in (401, 403) and attempt < 2:
                    self._warm = False
                    time.sleep(3)
                    continue
                break
            if r.status_code == 200 and r.content[:2] == b"PK":
                z = zipfile.ZipFile(io.BytesIO(r.content))
                return parse_bhav(pd.read_csv(z.open(z.namelist()[0])))
        return None

    def download(self, start: date, end: date, log=print) -> dict:
        """Fetch every missing weekday in [start, end]; resumable (done days are kept on disk)."""
        skip = self._skip_days()
        got = missing = failed = 0
        d = start
        while d <= end:
            path = self.root / f"{d.isoformat()}.parquet"
            if d.weekday() < 5 and d not in skip and not path.exists():
                try:
                    df = self.fetch_day(d)
                except Exception as e:
                    failed += 1
                    log(f"{d}: {type(e).__name__}: {str(e)[:100]}")
                    df = pd.DataFrame()
                    d += timedelta(days=1)
                    continue
                if df is None or df.empty:
                    with open(self._no_data, "a") as f:
                        f.write(d.isoformat() + "\n")
                    missing += 1
                else:
                    df.to_parquet(path, index=False)
                    got += 1
                if (got + missing) % 50 == 0:
                    log(f"{d}: {got} days saved, {missing} with no file (holidays), {failed} errors")
            d += timedelta(days=1)
        return {"saved": got, "no_file": missing, "errors": failed}


def load_options(start: date | None = None, end: date | None = None, root: Path = FO_DIR) -> pd.DataFrame:
    """All stored option rows in [start, end]."""
    frames = []
    for f in sorted(root.glob("*.parquet")):
        d = date.fromisoformat(f.stem)
        if (start and d < start) or (end and d > end):
            continue
        frames.append(pd.read_parquet(f))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
