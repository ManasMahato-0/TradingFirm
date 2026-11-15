r"""Start the dashboard at every Windows login (no window), like the recorder and sandbox.

    .venv\Scripts\python scripts\dashboard_startup.py install
    .venv\Scripts\python scripts\dashboard_startup.py uninstall

Then http://localhost:8501 is always there. It only listens on this computer.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STARTUP_FILE = Path(os.getenv("APPDATA", "")) / r"Microsoft\Windows\Start Menu\Programs\Startup\TradingFirmDashboard.cmd"


def command() -> str:
    pythonw = ROOT / ".venv" / "Scripts" / "pythonw.exe"
    return (f'@echo off\r\ncd /d "{ROOT}"\r\n'
            f'start "" "{pythonw}" -m streamlit run dashboard/app.py --server.port 8501 --server.headless true\r\n')


def main(argv: list[str]) -> int:
    action = argv[0] if argv else ""
    if action == "install":
        STARTUP_FILE.write_text(command(), encoding="utf-8")
        print(f"Installed: {STARTUP_FILE}\nThe dashboard will start at every Windows login (no window).")
    elif action == "uninstall":
        STARTUP_FILE.unlink(missing_ok=True)
        print("Removed the dashboard startup entry.")
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
