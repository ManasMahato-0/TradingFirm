"""Backups: consistent database copies, mirrored files, secrets never copied, safe on bad destinations."""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta

from firm.backup import run_backup

TODAY = date(2026, 11, 16)


def make_data(root):
    for rel in ("recorder.db", "control.db", "live/firm.db"):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(p) as c:
            c.execute("CREATE TABLE t (x)")
            c.execute("INSERT INTO t VALUES (?)", (rel,))
    (root / "bars" / "1m").mkdir(parents=True)
    (root / "bars" / "1m" / "A.parquet").write_bytes(b"A" * 100)
    (root / "cache").mkdir()
    (root / "cache" / "junk.bin").write_bytes(b"x")
    (root / "universe").mkdir()
    (root / "universe" / "n.csv").write_text("a,b")


def test_first_backup_copies_databases_and_files_but_not_cache(tmp_path):
    data, dest = tmp_path / "data", tmp_path / "bk"
    make_data(data)
    r = run_backup(data, dest, TODAY)
    assert r.status == "ok" and r.files_copied == 2
    assert sqlite3.connect(dest / "latest" / "recorder.db").execute("SELECT x FROM t").fetchone() == ("recorder.db",)
    assert (dest / "latest" / "live_firm.db").exists()
    assert (dest / "dated" / f"control_{TODAY}.db").exists()
    assert not list((dest).rglob("recorder_2026*")), "the big database keeps only one rolling copy"
    assert (dest / "files" / "bars" / "1m" / "A.parquet").read_bytes() == b"A" * 100
    assert not (dest / "files" / "cache").exists()


def test_second_run_copies_only_what_changed_and_never_deletes(tmp_path):
    data, dest = tmp_path / "data", tmp_path / "bk"
    make_data(data)
    run_backup(data, dest, TODAY)
    assert run_backup(data, dest, TODAY).files_copied == 0
    (data / "bars" / "1m" / "B.parquet").write_bytes(b"B")
    (data / "bars" / "1m" / "A.parquet").unlink()
    r = run_backup(data, dest, TODAY + timedelta(days=1))
    assert r.files_copied == 1 and (dest / "files" / "bars" / "1m" / "A.parquet").exists()


def test_old_dated_copies_are_pruned(tmp_path):
    data, dest = tmp_path / "data", tmp_path / "bk"
    make_data(data)
    run_backup(data, dest, TODAY - timedelta(days=30))
    run_backup(data, dest, TODAY)
    names = {p.name for p in (dest / "dated").glob("*.db")}
    assert f"control_{TODAY}.db" in names and f"control_{TODAY - timedelta(days=30)}.db" not in names


def test_env_file_is_never_copied(tmp_path):
    data, dest = tmp_path / "data", tmp_path / "bk"
    make_data(data)
    (data / ".env").write_text("FYERS_SECRET_KEY=shh")
    (data / "universe" / ".env").write_text("FYERS_SECRET_KEY=shh")
    run_backup(data, dest, TODAY)
    assert "shh" not in "".join(p.read_text(errors="ignore") for p in dest.rglob("*") if p.is_file() and p.suffix != ".db")
