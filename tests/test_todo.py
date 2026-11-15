from datetime import date, datetime

from firm.config import IST
from firm.todo import coming_up, days_until, greeting, todays_tasks

LAUNCH, SWING = date(2026, 11, 23), date(2026, 11, 30)


def at(d, h, m=0):
    return datetime(d.year, d.month, d.day, h, m, tzinfo=IST)


def test_on_a_trading_morning_the_one_thing_to_do_is_the_fyers_login():
    (t,) = todays_tasks(at(date(2026, 11, 24), 8, 30), True, False, LAUNCH, SWING)
    assert t.title == "Log in to Fyers" and not t.done and t.action == "login"
    (t,) = todays_tasks(at(date(2026, 11, 24), 8, 30), True, True, LAUNCH, SWING)
    assert t.done                                                          # ticked off once logged in


def test_weekends_holidays_and_evenings_have_nothing_to_do():
    for now, trading in [(at(date(2026, 11, 21), 10), False), (at(date(2026, 11, 24), 19), True)]:
        (t,) = todays_tasks(now, trading, None, LAUNCH, SWING)
        assert t.title == "Nothing to do today" and t.done and t.info


def test_launch_days_get_a_friendly_note_next_to_the_login():
    tasks = todays_tasks(at(LAUNCH, 8, 30), True, False, LAUNCH, SWING)
    assert [t.title for t in tasks] == ["Log in to Fyers", "Today the practice accounts start for real"]
    tasks = todays_tasks(at(SWING, 18), True, None, LAUNCH, SWING)
    assert any("swing experiment starts" in t.title for t in tasks)


def test_coming_up_lists_only_future_events_in_order():
    assert [c.day for c in coming_up(date(2026, 11, 16), LAUNCH, SWING)] == [LAUNCH, SWING]
    assert [c.day for c in coming_up(date(2026, 11, 25), LAUNCH, SWING)] == [SWING]
    assert coming_up(date(2026, 12, 1), LAUNCH, SWING) == []


def test_small_wording_helpers():
    assert greeting(at(date(2026, 11, 16), 8)) == "Good morning" and greeting(at(date(2026, 11, 16), 21)) == "Good evening"
    assert days_until(date(2026, 11, 16), date(2026, 11, 23)) == "in 7 days"
    assert days_until(date(2026, 11, 16), date(2026, 11, 17)) == "tomorrow"
