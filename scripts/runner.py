"""Live paper runner. Normally started by the recorder each evening after prices arrive.

    .venv\\Scripts\\python scripts\\runner.py run                      # process new days (live books)
    .venv\\Scripts\\python scripts\\runner.py dry-run --start 2026-06-01  # rehearsal in data\\dryrun (wiped first)
    .venv\\Scripts\\python scripts\\runner.py status
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from firm.live.books import books  # noqa: E402
from firm.live.runner import DRYRUN_DIR, LIVE_DIR, RunnerBusy, run_books  # noqa: E402
from firm.storage import Storage  # noqa: E402


def show(results: list[dict]) -> None:
    for r in results:
        if "error" in r:
            print(f"  {r['book']:24} ERROR {r['error']}")
        else:
            print(f"  {r['book']:24} +{r['days']} days, up to {r['last_bar']}, equity Rs {r['equity']:,.0f}"
                  f" (data to {r['data_last']})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    dr = sub.add_parser("dry-run")
    dr.add_argument("--start", type=date.fromisoformat, required=True)
    sub.add_parser("status")
    args = p.parse_args()

    if args.cmd == "status":
        st = Storage(LIVE_DIR / "firm.db")
        for row in st.conn.execute("SELECT name, status, last_bar, last_run FROM books"):
            print("  ", *row)
        return 0

    try:
        if args.cmd == "run":
            show(run_books(books(), LIVE_DIR / "firm.db", LIVE_DIR))
        else:
            if DRYRUN_DIR.exists():
                shutil.rmtree(DRYRUN_DIR)
            show(run_books(books(args.start), DRYRUN_DIR / "firm.db", DRYRUN_DIR))
    except RunnerBusy as e:
        print(e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
