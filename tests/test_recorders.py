from datetime import date, datetime

import pandas as pd

from firm.config import IST
from firm.recorders.bars import load_bars, record_bars, save_bars
from firm.recorders.db import RecorderDB
from firm.recorders.news import parse_announcement, parse_feed
from firm.recorders.options import parse_chain, save_snapshot


def bars(start, periods, freq="1min", base=100.0):
    idx = pd.date_range(start, periods=periods, freq=freq, tz="Asia/Kolkata", name="ts")
    return pd.DataFrame({"open": base, "high": base + 1, "low": base - 1, "close": base,
                         "volume": 10}, index=idx)


def test_save_bars_merges_dedupes_and_splits_by_month(tmp_path):
    first = bars("2026-10-31 15:28", 3)
    assert save_bars(first, "1m", "M&M", tmp_path) == 3
    # Overlaps one bar, adds one bar in November.
    second = bars("2026-10-31 15:29", 2, base=200.0)
    second.index = [second.index[0], pd.Timestamp("2026-11-02 09:15", tz="Asia/Kolkata")]
    second.index.name = "ts"
    assert save_bars(second, "1m", "M&M", tmp_path) == 1
    assert sorted(p.name for p in (tmp_path / "1m" / "M&M").iterdir()) == ["2026-10.parquet", "2026-11.parquet"]
    df = load_bars("1m", "M&M", root=tmp_path)
    assert len(df) == 4
    assert df.loc["2026-10-31 15:29", "close"] == 200.0  # newest copy wins
    assert len(load_bars("1m", "M&M", start=date(2026, 11, 1), root=tmp_path)) == 1


def test_record_bars_collects_errors_per_symbol(tmp_path):
    class Source:
        def history(self, symbol, interval, start, end):
            if symbol == "BAD":
                raise ValueError("boom")
            return bars("2026-11-03 09:15", 2)

    r = record_bars(Source(), "1m", "1m", ["GOOD", "BAD"], date(2026, 11, 3), date(2026, 11, 4), tmp_path)
    assert r["rows"] == 2 and r["symbols_ok"] == 1 and "BAD" in r["errors"]


def test_parse_chain_skips_underlying_row():
    data = {
        "optionsChain": [
            {"strike_price": -1, "option_type": "", "ltp": 25144.3, "symbol": "NSE:NIFTY50-INDEX"},
            {"strike_price": 25050, "option_type": "PE", "ltp": 92.65, "bid": 92.6, "ask": 93.8,
             "oi": 100, "prev_oi": 50, "volume": 7, "symbol": "NSE:NIFTY26N1725050PE"},
        ],
        "indiavixData": {"ltp": 15.5},
    }
    ts = datetime(2026, 11, 13, 10, 0, tzinfo=IST)
    df = parse_chain(data, "NIFTY50", "2026-11-17", ts)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["strike"] == 25050 and row["spot"] == 25144.3 and row["vix"] == 15.5


def test_save_snapshot_appends(tmp_path):
    df = pd.DataFrame({"ts": [1], "strike": [100.0]})
    save_snapshot(df, "NIFTY50", date(2026, 11, 13), tmp_path)
    assert save_snapshot(df, "NIFTY50", date(2026, 11, 13), tmp_path) == 2


RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
<item><title>Infosys wins big deal</title><link>https://x.test/a</link>
<pubDate>Sat, 14 Nov 2026 14:45:57 +0530</pubDate><description>d</description></item>
<item><title></title><link>https://x.test/empty</link></item>
</channel></rss>"""


def test_parse_feed_and_dedupe():
    items = parse_feed("et", RSS, "2026-11-14T15:00:00+05:30")
    assert len(items) == 1
    assert items[0]["published"].startswith("2026-11-14T14:45:57")
    db = RecorderDB(":memory:")
    assert db.insert_news(items) == 1
    assert db.insert_news(items) == 0  # same item again is ignored


def test_parse_announcement():
    a = {"seq_id": "106860651", "symbol": "ABHISHEK", "sm_name": "Abhishek Corporation Limited",
         "desc": "Outcome of Board Meeting", "attchmntText": "results", "hasXbrl": True,
         "exchdisstime": "14-Nov-2026 15:32:19", "attchmntFile": "https://x.test/f.pdf"}
    row = parse_announcement(a, "2026-11-14T15:35:00+05:30")
    assert row["id"] == "106860651" and row["has_xbrl"] == 1
    assert row["exchange_time"] == "2026-11-14T15:32:19+05:30"
    db = RecorderDB(":memory:")
    assert db.insert_announcements([row, row]) == 1


def test_old_announcement_without_dissemination_time_falls_back_to_an_dt():
    from firm.recorders.news import parse_announcement

    row = parse_announcement({"symbol": "TCS", "desc": "x", "exchdisstime": "-", "an_dt": "31-Jul-2020 23:38:14",
                              "seq_id": "1", "attchmntText": "t"}, "2026-11-15T00:00:00+05:30")
    assert row["exchange_time"] == "2020-07-31T23:38:14+05:30"
