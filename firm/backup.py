"""Back up the data that only exists on this laptop (prices, filings, results, books, settings).

Code is on GitHub; this protects everything else. Rules:
- SQLite databases are copied with SQLite's own backup call (safe while a program is writing to them).
  recorder.db (the big one) keeps ONE rolling copy; the small databases also keep dated copies for KEEP_DAYS.
- Plain files (bars, option chains, universe lists, research) are mirrored: new or changed files are copied,
  nothing is ever deleted from the backup.
- `.env` (login keys) is NEVER copied. Rebuildable things (cache, replay and dry-run folders, pid files) are skipped.
"""
from __future__ import annotations

import shutil
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

DATABASES = ["recorder.db", "control.db", "research.db", "live/firm.db"]
BIG = {"recorder.db"}                      # one rolling copy, no dated copies
MIRROR_DIRS = ["bars", "fo", "options", "universe", "research", "live"]
SKIP_SUFFIXES = (".db", ".db-wal", ".db-shm", ".lock", ".tmp")
KEEP_DAYS = 14
HEADROOM = 1.15


@dataclass
class BackupResult:
    status: str                             # ok | error
    message: str
    files_copied: int = 0
    bytes_copied: int = 0
    databases: list[str] = field(default_factory=list)


def default_destination() -> Path:
    """D: is a separate drive letter on most laptops; otherwise the home folder. The owner can change it."""
    d = Path("D:/")
    return (d / "TradingFirmBackups") if d.exists() else (Path.home() / "TradingFirmBackups")


def _snapshot(src: Path, dst: Path) -> None:
    """Consistent copy of a live SQLite database; replaces dst only when the copy is complete."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    tmp.unlink(missing_ok=True)
    with closing(sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=60)) as a, closing(sqlite3.connect(tmp)) as b:
        a.backup(b)
    with closing(sqlite3.connect(f"file:{tmp}?mode=ro", uri=True)) as chk:
        chk.execute("SELECT count(*) FROM sqlite_master").fetchone()      # opens and reads: not a broken file
    tmp.replace(dst)


def _mirror(src_dir: Path, dst_dir: Path) -> tuple[int, int]:
    n = size = 0
    for f in src_dir.rglob("*"):
        if not f.is_file() or f.name.endswith(SKIP_SUFFIXES) or f.name.startswith(".env"):
            continue
        out = dst_dir / f.relative_to(src_dir)
        s = f.stat()
        if out.exists() and out.stat().st_size == s.st_size and out.stat().st_mtime >= s.st_mtime - 2:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)
        n += 1
        size += s.st_size
    return n, size


def _prune(folder: Path, keep_days: int, today: date) -> None:
    for f in folder.glob("*.db"):
        stamp = f.stem.rsplit("_", 1)[-1]
        try:
            age = (today - date.fromisoformat(stamp)).days
        except ValueError:
            continue
        if age > keep_days:
            f.unlink(missing_ok=True)


def needed_bytes(data_dir: Path) -> int:
    total = 0
    for rel in DATABASES:
        p = data_dir / rel
        if p.exists():
            total += p.stat().st_size
    for d in MIRROR_DIRS:
        p = data_dir / d
        if p.exists():
            total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.name.endswith(SKIP_SUFFIXES) and not f.name.startswith(".env"))
    return total


def run_backup(data_dir: Path, dest: Path, today: date, keep_days: int = KEEP_DAYS) -> BackupResult:
    data_dir, dest = Path(data_dir), Path(dest)
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return BackupResult("error", f"Cannot use the backup folder {dest} ({type(e).__name__}). Pick another in Settings.")
    if dest.resolve() == data_dir.resolve() or data_dir.resolve() in dest.resolve().parents:
        return BackupResult("error", "The backup folder must be outside the data folder.")
    free = shutil.disk_usage(dest).free
    if not (dest / "latest").exists():
        need = int(needed_bytes(data_dir) * HEADROOM)                    # first run copies everything
    else:                                                                 # later runs: room for one new database copy
        biggest = max([(data_dir / r).stat().st_size for r in DATABASES if (data_dir / r).exists()] or [0])
        need = int(biggest * HEADROOM) + 200_000_000
    if free < need:
        return BackupResult("error", f"Not enough free space at {dest} ({free // 2**20} MB free, about "
                                     f"{need // 2**20} MB needed). Pick another folder in Settings.")
    res = BackupResult("ok", "")
    try:
        for rel in DATABASES:
            src = data_dir / rel
            if not src.exists():
                continue
            name = rel.replace("/", "_")
            _snapshot(src, dest / "latest" / name)
            res.databases.append(rel)
            if rel not in BIG:
                _snapshot(src, dest / "dated" / f"{Path(name).stem}_{today.isoformat()}.db")
        _prune(dest / "dated", keep_days, today)
        for d in MIRROR_DIRS:
            if (data_dir / d).exists():
                n, s = _mirror(data_dir / d, dest / "files" / d)
                res.files_copied += n
                res.bytes_copied += s
    except Exception as e:
        return BackupResult("error", f"Backup stopped: {type(e).__name__}: {str(e)[:150]}", res.files_copied,
                            res.bytes_copied, res.databases)
    res.message = (f"{len(res.databases)} databases and {res.files_copied} new or changed files "
                   f"({res.bytes_copied / 2**20:.0f} MB) saved to {dest}.")
    return res
