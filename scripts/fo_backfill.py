"""Download NSE daily F&O bhavcopy (Nifty and Bank Nifty options) into data/fo/raw. Resumable.

    .venv\Scripts\python scripts\fo_backfill.py --start 2019-01-01
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from firm.config import LOG_DIR  # noqa: E402
from firm.recorders.fo import FOFetcher  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=date.fromisoformat, default=date(2019, 1, 1))
    ap.add_argument("--end", type=date.fromisoformat, default=date.today())
    args = ap.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=LOG_DIR / "fo_backfill.log", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    log = logging.info
    log(f"fo backfill {args.start} to {args.end}")
    res = FOFetcher().download(args.start, args.end, log=log)
    log(f"fo backfill finished: {res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
