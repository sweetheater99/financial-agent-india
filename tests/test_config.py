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
