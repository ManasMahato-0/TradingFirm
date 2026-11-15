"""Trial registry: every backtest is logged, so luck can be discounted and the
locked test period can only be opened once per strategy version."""
from __future__ import annotations

import hashlib
import inspect
import json
import sqlite3
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    track TEXT NOT NULL,      -- swing | intraday
    strategy TEXT NOT NULL,
    version TEXT NOT NULL,    -- hash of strategy code + params
    params TEXT,
    kind TEXT NOT NULL,       -- research | walkforward | final_exam
    start TEXT,
    end TEXT,
    metrics TEXT
);
CREATE INDEX IF NOT EXISTS trials_track ON trials(track, kind);
"""


def strategy_version(strategy) -> str:
    """Hash of the strategy class source, any modules it declares in `depends_on`
    (e.g. feature code), and its params."""
    cls = type(strategy)
    try:
        src = inspect.getsource(cls)
        for mod in getattr(strategy, "depends_on", ()):
            src += inspect.getsource(mod)
    except (OSError, TypeError):
        src = cls.__qualname__
    blob = src + json.dumps(getattr(strategy, "params", {}), sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def _clean(v):
    return None if isinstance(v, float) and v != v else v  # NaN -> null


class TrialRegistry:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.executescript(SCHEMA)

    def log(self, track: str, strategy: str, version: str, params: dict, kind: str,
            start, end, metrics: dict) -> int:
        cur = self.conn.execute(
            "INSERT INTO trials (ts, track, strategy, version, params, kind, start, end, metrics)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), track, strategy, version,
             json.dumps(params, sort_keys=True, default=str), kind, str(start), str(end),
             json.dumps({k: _clean(v) for k, v in metrics.items()}, default=str)),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def trial_stats(self, track: str) -> tuple[int, float]:
        """(number of distinct strategy versions tried, variance of their per-period Sharpe)."""
        rows = self.conn.execute(
            "SELECT version, metrics FROM trials WHERE track = ? AND kind IN ('research','walkforward')"
            " ORDER BY id", (track,)
        ).fetchall()
        latest: dict[str, float] = {}
        for version, m in rows:
            sr = json.loads(m or "{}").get("sharpe_per_period")
            if sr is not None:
                latest[version] = sr
        srs = list(latest.values())
        if len(srs) < 2:
            return len(srs), 0.0
        mean = sum(srs) / len(srs)
        return len(srs), sum((s - mean) ** 2 for s in srs) / (len(srs) - 1)

    def has_final_exam(self, version: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM trials WHERE version = ? AND kind = 'final_exam' LIMIT 1", (version,)
        ).fetchone() is not None
