import sqlite3

import numpy as np
import pandas as pd

from firm.features.fundamentals import (
    describe_result, fundamental_panel, quarter_label, quarter_measures,
)
from firm.recorders.db import RecorderDB
from firm.recorders.results import normalise_integrated, parse_xbrl, row_to_record, usable_row

XML = """<xbrli:xbrl>
<in-capmkt:RevenueFromOperations contextRef="OneD" unitRef="INR" decimals="-5">731880000000</in-capmkt:RevenueFromOperations>
<in-capmkt:Income contextRef="OneD" unitRef="INR">745250000000</in-capmkt:Income>
<in-capmkt:ProfitBeforeTax decimals="-5" contextRef="OneD" unitRef="INR">186380000000</in-capmkt:ProfitBeforeTax>
<in-capmkt:ProfitLossForPeriod contextRef="OneD" unitRef="INR">139340000000</in-capmkt:ProfitLossForPeriod>
<in-capmkt:ProfitOrLossAttributableToOwnersOfParent contextRef="OneD" unitRef="INR">138840000000</in-capmkt:ProfitOrLossAttributableToOwnersOfParent>
<in-capmkt:BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations contextRef="OneD" unitRef="INR">38.37</in-capmkt:BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations>
<in-capmkt:RevenueFromOperations contextRef="FourD" unitRef="INR">1454630000000</in-capmkt:RevenueFromOperations>
</xbrli:xbrl>"""

BANK_XML = """<xbrli:xbrl>
<in-capmkt:Income contextRef="OneD">1293586900000</in-capmkt:Income>
<in-capmkt:ProfitLossFromOrdinaryActivitiesBeforeTax contextRef="OneD">278254800000</in-capmkt:ProfitLossFromOrdinaryActivitiesBeforeTax>
<in-capmkt:ProfitLossAfterTaxesMinorityInterestAndShareOfProfitLossOfAssociates contextRef="OneD">207000300000</in-capmkt:ProfitLossAfterTaxesMinorityInterestAndShareOfProfitLossOfAssociates>
<in-capmkt:BasicEarningsPerShareAfterExtraordinaryItems contextRef="OneD">13.43</in-capmkt:BasicEarningsPerShareAfterExtraordinaryItems>
</xbrli:xbrl>"""


def test_parse_xbrl_uses_current_quarter_and_crore():
    n = parse_xbrl(XML)
    assert n["revenue"] == 73188.0 and n["income"] == 74525.0 and n["pbt"] == 18638.0
    assert n["pat"] == 13884.0            # owners' share preferred over total profit
    assert n["eps"] == 38.37              # EPS is not scaled to crore


def test_parse_xbrl_bank_fallbacks():
    n = parse_xbrl(BANK_XML)
    assert n["revenue"] == n["income"] == 129358.69      # no RevenueFromOperations: total income
    assert n["pbt"] == 27825.48 and n["pat"] == 20700.03 and n["eps"] == 13.43


def test_row_to_record_quarter_length_and_public_time():
    row = {"symbol": "TCS", "toDate": "31-Dec-2024", "fromDate": "01-Oct-2024", "consolidated": "Consolidated",
           "exchdisstime": "09-Jan-2025 21:40:06", "xbrl": "https://x/a.xml"}
    rec = row_to_record(row, parse_xbrl(XML), "now")
    assert rec["to_date"] == "2024-12-31" and rec["consolidated"] == 1
    assert rec["broadcast"] == "2025-01-09T21:40:06+05:30"
    year = dict(row, fromDate="01-Apr-2024")                           # nine months is not a quarter
    assert row_to_record(year, parse_xbrl(XML), "now") is None
    assert row_to_record(row, parse_xbrl("<x/>"), "now") is None       # nothing parsed


def test_usable_and_integrated_rows():
    assert not usable_row({"period": "Quarterly", "cumulative": "Non-cumulative", "xbrl": "https://x/corporate/xbrl/-"})
    assert usable_row({"period": "Quarterly", "cumulative": "Non-cumulative", "xbrl": "https://x/a.xml"})
    r = {"symbol": "TCS", "qe_Date": "30-SEP-2026", "consolidated": "Consolidated", "seq_Id": "1",
         "broadcast_Date": "08-Oct-2026 22:07:33", "xbrl": "https://x/INTEGRATED.xml"}
    n = normalise_integrated(r)
    assert n["toDate"] == "30-SEP-2026" and usable_row(n)
    assert normalise_integrated(dict(r, xbrl="https://x/corporate/null")) is None
    assert normalise_integrated(dict(r, consolidated=None)) is None


def make_results(n_quarters=14, growth=0.10, symbol="AAA", cons=1, jump_last=None, noise=0.0):
    dates = pd.date_range("2019-03-31", periods=n_quarters, freq="QE")
    rng = np.random.default_rng(0)
    rows = []
    for i, d in enumerate(dates):
        base = 100 * (1 + growth) ** (i / 4)
        pat = base * 0.1 * (1 + noise * rng.normal())
        if jump_last and i == n_quarters - 1:
            pat *= jump_last
        rows.append({"symbol": symbol, "to_date": d.date().isoformat(), "consolidated": cons,
                     "broadcast": (d + pd.Timedelta(days=40)).strftime("%Y-%m-%dT16:00:00+05:30"),
                     "revenue": base, "pat": pat, "eps": pat / 10})
    df = pd.DataFrame(rows)
    df["to_date"] = pd.to_datetime(df["to_date"])
    df["broadcast"] = pd.to_datetime(df["broadcast"], utc=True).dt.tz_convert("Asia/Kolkata")
    return df


def test_yoy_growth_margin_and_surprise():
    m = quarter_measures(make_results(jump_last=2.0)).sort_values("to_date")
    assert m.rev_yoy.iloc[:4].isna().all()                      # no year-ago quarter yet
    assert np.allclose(m.rev_yoy.iloc[4:12], 0.10, atol=1e-6)
    assert np.allclose(m.margin_chg.iloc[4:12], 0.0, atol=1e-9)   # margin unchanged
    assert m.pat_yoy.iloc[-1] > 0.8                              # profit doubled in the last quarter
    # Noisy ~10% growth for years, then profit doubles: the surprise is large and positive.
    n = quarter_measures(make_results(n_quarters=16, jump_last=2.5, noise=0.03)).sort_values("to_date")
    assert n.pat_surprise.iloc[:8].isna().all()                  # needs 4+ earlier YoY values
    assert n.pat_surprise.iloc[-1] > 2
    assert abs(n.pat_surprise.iloc[-2]) < 3                      # ordinary quarters are not extreme


def test_surprise_uses_only_past_values():
    full = quarter_measures(make_results(n_quarters=16, jump_last=3.0, noise=0.03)).sort_values("to_date").reset_index(drop=True)
    part = quarter_measures(make_results(n_quarters=16, jump_last=3.0, noise=0.03).iloc[:15]).sort_values("to_date").reset_index(drop=True)
    pd.testing.assert_frame_equal(full.iloc[:15], part, check_dtype=False)   # the last quarter changes nothing before it


def test_panel_is_point_in_time():
    res = make_results(n_quarters=8)
    m = quarter_measures(res)
    last = m.sort_values("to_date").iloc[-1]
    pub = last.broadcast
    idx = pd.bdate_range(pub.date() - pd.Timedelta(days=3), periods=10, tz="Asia/Kolkata")
    panel = fundamental_panel(idx, m, ["AAA"])
    day = pd.Timestamp(pub.date()).tz_localize("Asia/Kolkata")
    nxt = idx[idx > day][0]
    assert panel["rev_yoy"].loc[nxt, "AAA"] == last.rev_yoy       # usable the day AFTER publication
    assert panel["rev_yoy"].loc[day, "AAA"] != last.rev_yoy       # never on the day itself


def test_panel_goes_stale():
    m = quarter_measures(make_results(n_quarters=8))
    last_pub = m.broadcast.max()
    idx = pd.bdate_range(last_pub.date() + pd.Timedelta(days=250), periods=3, tz="Asia/Kolkata")
    assert fundamental_panel(idx, m, ["AAA"])["rev_yoy"]["AAA"].isna().all()


def test_quarter_label_and_description():
    assert quarter_label("2026-09-30") == "Q2 FY27" and quarter_label("2026-03-31") == "Q4 FY26"
    assert quarter_label("2026-06-30") == "Q1 FY27" and quarter_label("2026-12-31") == "Q3 FY27"
    db = RecorderDB(":memory:")
    cur = {"symbol": "AAA", "to_date": "2026-09-30", "consolidated": 1, "from_date": "2026-07-01",
           "broadcast": "2026-10-08T22:07:33+05:30", "audited": "x", "seq": "1", "income": 1.0, "revenue": 1100.0,
           "pbt": 1.0, "pat": 150.0, "eps": 3.5, "xbrl_url": "u", "first_seen": "x"}
    old = dict(cur, to_date="2025-09-30", revenue=1000.0, pat=100.0, eps=2.5)
    db.insert_results([old, cur])
    text = describe_result(db.conn, cur)
    assert "Q2 FY27" in text and "+10.0%" in text and "+50.0%" in text and "consolidated" in text
    first = describe_result(db.conn, old)
    assert "No year-ago quarter" in first
    assert db.insert_results([dict(cur, pat=999.0)]) == 0           # first filing wins, revisions ignored
    assert db.conn.execute("SELECT pat FROM results WHERE to_date='2026-09-30'").fetchone()[0] == 150.0


def test_sort_measure_rule_on_synthetic_signal():
    import importlib.util
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("fundamentals_sort", Path(__file__).resolve().parent.parent
                                                  / "scripts" / "fundamentals_sort.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fundamentals_sort"] = mod
    spec.loader.exec_module(mod)

    rng = np.random.default_rng(1)
    months = pd.date_range("2020-01-31", periods=40, freq="ME")
    cols = [f"S{i}" for i in range(60)]
    vals = pd.DataFrame(rng.normal(size=(40, 60)), index=months, columns=cols)
    noise = pd.DataFrame(rng.normal(0, 0.05, size=(40, 60)), index=months, columns=cols)
    good = mod.sort_measure(vals, noise + 0.02 * vals)          # return rises with the measure
    assert good["pass"] and good["rank_corr"] >= 0.7 and good["spread_pct_per_month"] > 1
    useless = mod.sort_measure(vals, noise)                      # pure noise
    assert not useless["pass"]
    wrong_way = mod.sort_measure(vals, noise - 0.02 * vals)
    assert not wrong_way["pass"] and wrong_way["spread_pct_per_month"] < 0
    few = mod.sort_measure(vals.iloc[:, :10], noise.iloc[:, :10])  # too few stocks to form groups
    assert few["months"] == 0 and not few["pass"]
