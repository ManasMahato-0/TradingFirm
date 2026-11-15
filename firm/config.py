"""Firm-wide configuration: teams, capital, risk limits."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta, timezone
from pathlib import Path

from firm.costs import Segment
from firm.risk import RiskLimits

ROOT = Path(__file__).resolve().parent.parent
# Set FIRM_DATA_DIR to keep market data on another drive.
DATA_DIR = Path(os.getenv("FIRM_DATA_DIR", ROOT / "data"))
LOG_DIR = ROOT / "logs"
DB_PATH = DATA_DIR / "firm.db"
RECORDER_DB_PATH = DATA_DIR / "recorder.db"
BARS_DIR = DATA_DIR / "bars"
OPTIONS_DIR = DATA_DIR / "options"
UNIVERSE_DIR = DATA_DIR / "universe"

IST = timezone(timedelta(hours=5, minutes=30))  # India has no DST

STARTING_CAPITAL = 100_000.0  # ₹1 lakh paper money per team


@dataclass(frozen=True)
class TeamConfig:
    key: str
    display_name: str
    limits: RiskLimits
    starting_capital: float = STARTING_CAPITAL


TEAMS: dict[str, TeamConfig] = {
    t.key: t
    for t in [
        TeamConfig(
            key="swing_stocks",
            display_name="Swing Stocks",
            limits=RiskLimits(
                allowed_segments=frozenset({Segment.EQ_DELIVERY}),
                allow_short=False,
                max_position_pct=0.20,
                max_gross_exposure_pct=1.00,
                max_daily_loss_pct=0.03,
                max_orders_per_day=20,
                max_data_age_seconds=3 * 24 * 3600,  # daily bars, weekends
            ),
        ),
        TeamConfig(
            key="intraday_stocks",
            display_name="Intraday Stocks",
            limits=RiskLimits(
                allowed_segments=frozenset({Segment.EQ_INTRADAY}),
                allow_short=True,
                max_position_pct=0.25,
                max_gross_exposure_pct=1.00,
                max_daily_loss_pct=0.02,
                max_orders_per_day=50,
                max_data_age_seconds=120,
            ),
        ),
        TeamConfig(
            key="intraday_options",
            display_name="Intraday Options",
            limits=RiskLimits(
                allowed_segments=frozenset({Segment.OPTIONS}),
                allow_short=False,  # buy-only: no option writing
                max_position_pct=0.10,
                max_gross_exposure_pct=0.30,
                max_daily_loss_pct=0.03,
                max_orders_per_day=40,
                max_data_age_seconds=60,
            ),
        ),
        TeamConfig(
            key="overnight_options",
            display_name="Overnight Options",
            limits=RiskLimits(
                allowed_segments=frozenset({Segment.OPTIONS}),
                allow_short=False,
                max_position_pct=0.10,
                max_gross_exposure_pct=0.30,
                max_daily_loss_pct=0.04,
                max_orders_per_day=20,
                max_data_age_seconds=120,
            ),
        ),
    ]
}
