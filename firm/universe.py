"""Stock universe: Nifty 100 constituents, refreshed from NSE monthly."""
from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

from firm.config import UNIVERSE_DIR

NIFTY100_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv"
INDEXES = ["NIFTY50", "BANKNIFTY", "INDIAVIX"]
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)


def nifty100(refresh_days: int = 30, root: Path | None = None) -> list[str]:
    path = (root or UNIVERSE_DIR) / "nifty100.csv"
    stale = not path.exists() or (time.time() - path.stat().st_mtime) > refresh_days * 86400
    if stale:
        try:
            r = requests.get(NIFTY100_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
            r.raise_for_status()
            pd.read_csv(io.StringIO(r.text))  # validate before overwriting
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(r.text, encoding="utf-8")
        except Exception:
            if not path.exists():
                raise
    df = pd.read_csv(path)
    return df.loc[df["Series"] == "EQ", "Symbol"].astype(str).tolist()


def sectors(root: Path | None = None) -> dict[str, str]:
    """Symbol -> NSE industry, from the stored Nifty 100 list."""
    nifty100(root=root)  # ensure the file exists
    df = pd.read_csv((root or UNIVERSE_DIR) / "nifty100.csv")
    return dict(zip(df["Symbol"].astype(str), df["Industry"].astype(str)))


def company_names(root: Path | None = None) -> dict[str, str]:
    """Symbol -> company name, e.g. 'TCS' -> 'Tata Consultancy Services Ltd.'"""
    nifty100(root=root)
    df = pd.read_csv((root or UNIVERSE_DIR) / "nifty100.csv")
    return dict(zip(df["Symbol"], df["Company Name"]))
