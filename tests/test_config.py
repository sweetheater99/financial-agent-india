"""Tests for config.is_market_open()."""

from datetime import datetime, timezone, timedelta

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
