"""Option chain snapshots: one parquet file per underlying per day."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd

from firm.config import OPTIONS_DIR

# Underlying -> how many nearest expiries to record.
UNDERLYINGS = {"NIFTY50": 2, "BANKNIFTY": 1}
STRIKECOUNT = 20  # strikes each side of ATM


def parse_chain(data: dict, underlying: str, expiry: str, snap_ts: datetime) -> pd.DataFrame:
    chain = data.get("optionsChain", [])
    spot = next((r.get("ltp") for r in chain if r.get("strike_price") == -1), None)
    vix = (data.get("indiavixData") or {}).get("ltp")
    rows = [
        {
            "ts": snap_ts, "underlying": underlying, "expiry": expiry,
            "symbol": r.get("symbol"), "option_type": r.get("option_type"),
            "strike": float(r.get("strike_price")),
            "ltp": r.get("ltp"), "bid": r.get("bid"), "ask": r.get("ask"),
            "oi": r.get("oi"), "prev_oi": r.get("prev_oi"), "volume": r.get("volume"),
            "spot": spot, "vix": vix,
        }
        for r in chain
        if r.get("option_type") in ("CE", "PE")
    ]
    return pd.DataFrame(rows)


def snapshot(fyers, underlying: str, n_expiries: int, snap_ts: datetime,
             strikecount: int = STRIKECOUNT) -> pd.DataFrame:
    expiries = fyers.option_chain(underlying, 1, "").get("expiryData", [])[:n_expiries]
    frames = []
    for e in expiries:
        data = fyers.option_chain(underlying, strikecount, str(e["expiry"]))
        expiry = datetime.strptime(e["date"], "%d-%m-%Y").date().isoformat()
        frames.append(parse_chain(data, underlying, expiry, snap_ts))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save_snapshot(df: pd.DataFrame, underlying: str, day: date, root: Path | None = None) -> int:
    if df.empty:
        return 0
    path = (root or OPTIONS_DIR) / underlying / f"{day.isoformat()}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df.to_parquet(path, index=False)
    return len(df)


def record_option_chains(fyers, snap_ts: datetime, root: Path | None = None) -> int:
    rows = 0
    for underlying, n in UNDERLYINGS.items():
        df = snapshot(fyers, underlying, n, snap_ts)
        save_snapshot(df, underlying, snap_ts.date(), root)
        rows += len(df)
    return rows
