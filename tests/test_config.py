"""Tests for config module — is_market_open(), NSE holidays, VIX tiers, V2 constants."""

from datetime import date, datetime, timezone, timedelta

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import is_market_open

IST = timezone(timedelta(hours=5, minutes=30))


def _ist(year=2026, month=2, day=9, hour=10, minute=0):
    """Create a timezone-aware IST datetime."""
    return datetime(year, month, day, hour, minute, tzinfo=IST)


def test_is_market_open_weekday_trading_hours():
    # Monday 10:00 IST -> open
    is_open, msg = is_market_open(_now=_ist(day=9, hour=10, minute=0))  # Mon
    assert is_open is True
    assert "open" in msg.lower()


def test_is_market_open_weekend():
    # Saturday -> closed
    is_open, msg = is_market_open(_now=_ist(day=14, hour=10, minute=0))  # Sat
    assert is_open is False
    assert "saturday" in msg.lower()


def test_is_market_open_before_open():
    # Monday 8:00 IST -> closed
    is_open, msg = is_market_open(_now=_ist(day=9, hour=8, minute=0))  # Mon
    assert is_open is False
    assert "9:15" in msg


def test_is_market_open_after_close():
    # Monday 16:00 IST -> closed
    is_open, msg = is_market_open(_now=_ist(day=9, hour=16, minute=0))  # Mon
    assert is_open is False
    assert "3:30" in msg.lower() or "15:30" in msg.lower() or "after" in msg.lower()


def test_is_market_open_boundary_open():
    # 9:15 AM exactly -> open
    is_open, msg = is_market_open(_now=_ist(day=9, hour=9, minute=15))  # Mon
    assert is_open is True


def test_is_market_open_boundary_close():
    # 3:30 PM exactly -> closed (>= market_close)
    is_open, msg = is_market_open(_now=_ist(day=9, hour=15, minute=30))  # Mon
    assert is_open is False


# --- V2 constants tests ---

def test_nse_holidays_2026_count():
    from config import NSE_HOLIDAYS_2026
    assert len(NSE_HOLIDAYS_2026) == 16


def test_republic_day_is_holiday():
    from config import NSE_HOLIDAYS_2026
    assert date(2026, 1, 26) in NSE_HOLIDAYS_2026


def test_is_trading_day_weekend():
    from config import is_trading_day
    assert is_trading_day(date(2026, 3, 7)) is False  # Saturday


def test_is_trading_day_holiday():
    from config import is_trading_day
    assert is_trading_day(date(2026, 1, 26)) is False  # Republic Day


def test_is_trading_day_normal():
    from config import is_trading_day
    assert is_trading_day(date(2026, 3, 9)) is True  # Monday, not holiday


def test_get_vix_tier_normal():
    from config import get_vix_tier
    tier = get_vix_tier(15.0)
    assert tier["size_multiplier"] == 1.00
    assert tier["iron_condor"] is True


def test_get_vix_tier_crisis():
    from config import get_vix_tier
    tier = get_vix_tier(30.0)
    assert tier["size_multiplier"] == 0.00
    assert tier["iron_condor"] is False


def test_get_vix_tier_extreme_low():
    from config import get_vix_tier
    tier = get_vix_tier(10.0)
    assert tier["size_multiplier"] == 0.50


def test_capital_allocation_sums_to_100():
    from config import (ALLOC_EQUITY_MAX, ALLOC_SPREADS_MAX,
                        ALLOC_IRON_CONDOR_MAX, ALLOC_MOMENTUM_MAX, ALLOC_CASH_MIN)
    total = ALLOC_EQUITY_MAX + ALLOC_SPREADS_MAX + ALLOC_IRON_CONDOR_MAX + ALLOC_MOMENTUM_MAX + ALLOC_CASH_MIN
    assert total == 1.0


# --- Trading day helper tests ---

def test_trading_days_between_normal_week():
    from config import trading_days_between
    # Mon Mar 9 to Fri Mar 13 = 3 trading days (Tue 10 is Holi, so Wed, Thu, Fri)
    assert trading_days_between(date(2026, 3, 9), date(2026, 3, 13)) == 3


def test_trading_days_between_over_weekend():
    from config import trading_days_between
    # Fri Mar 6 to Mon Mar 9 = 1 trading day (Mon)
    assert trading_days_between(date(2026, 3, 6), date(2026, 3, 9)) == 1


def test_trading_days_between_over_holiday():
    from config import trading_days_between
    # Day before Republic Day (Jan 23 Fri) to Jan 27 (Tue)
    # Jan 24 Sat, Jan 25 Sun, Jan 26 Holiday, Jan 27 Tue = 1 trading day
    assert trading_days_between(date(2026, 1, 23), date(2026, 1, 27)) == 1


def test_trading_days_between_same_day():
    from config import trading_days_between
    assert trading_days_between(date(2026, 3, 9), date(2026, 3, 9)) == 0


def test_add_trading_days_simple():
    from config import add_trading_days
    # Mon + 3 trading days = Fri (Tue 10 is Holi, so Wed=1, Thu=2, Fri=3)
    assert add_trading_days(date(2026, 3, 9), 3) == date(2026, 3, 13)


def test_add_trading_days_over_weekend():
    from config import add_trading_days
    # Thu + 2 trading days = Mon (skips Sat, Sun)
    assert add_trading_days(date(2026, 3, 12), 2) == date(2026, 3, 16)


def test_add_trading_days_over_holiday():
    from config import add_trading_days
    # Fri Jan 23 + 1 = Tue Jan 27 (skips Sat, Sun, Republic Day Mon)
    assert add_trading_days(date(2026, 1, 23), 1) == date(2026, 1, 27)


def test_add_trading_days_negative():
    from config import add_trading_days
    # Wed Mar 11 - 2 trading days: Tue 10 is Holi (skip), Mon=1, Fri=2
    assert add_trading_days(date(2026, 3, 11), -2) == date(2026, 3, 6)


def test_next_trading_day_from_friday():
    from config import next_trading_day
    assert next_trading_day(date(2026, 3, 6)) == date(2026, 3, 9)  # Skip weekend


def test_next_trading_day_from_holiday():
    from config import next_trading_day
    # Jan 26 (Mon holiday) -> Jan 27 (Tue)
    assert next_trading_day(date(2026, 1, 26)) == date(2026, 1, 27)


def test_prev_trading_day_from_monday():
    from config import prev_trading_day
    assert prev_trading_day(date(2026, 3, 9)) == date(2026, 3, 6)  # Previous Friday


def test_is_market_open_holiday():
    # Republic Day (Monday) at 10:00 AM should be closed
    is_open, msg = is_market_open(_now=_ist(year=2026, month=1, day=26, hour=10))
    assert is_open is False
    assert "holiday" in msg.lower() or "closed" in msg.lower()
