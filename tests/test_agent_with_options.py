"""Tests for agent_with_options.py pure functions."""

import re
from datetime import date, timedelta
import calendar

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_with_options import get_nearest_expiry


def test_nearest_expiry_format():
    result = get_nearest_expiry()
    # Should match DDMONYYYY pattern like 27FEB2026
    assert re.match(r"^\d{2}[A-Z]{3}\d{4}$", result), f"Unexpected format: {result}"


def test_nearest_expiry_is_thursday():
    result = get_nearest_expiry()
    # Parse back to date
    d = _parse_expiry(result)
    assert d.weekday() == 3, f"Expected Thursday (3), got weekday {d.weekday()} for {result}"


def test_nearest_expiry_is_future():
    result = get_nearest_expiry()
    d = _parse_expiry(result)
    assert d >= date.today(), f"Expiry {result} is in the past"


def test_nearest_expiry_is_last_thursday():
    result = get_nearest_expiry()
    d = _parse_expiry(result)
    # Verify it's the last Thursday of its month
    last_day = calendar.monthrange(d.year, d.month)[1]
    last_date = date(d.year, d.month, last_day)
    while last_date.weekday() != 3:
        last_date -= timedelta(days=1)
    assert d == last_date, f"{result} ({d}) is not the last Thursday ({last_date})"


def _parse_expiry(expiry_str: str) -> date:
    """Parse DDMONYYYY format back to a date object."""
    from datetime import datetime
    return datetime.strptime(expiry_str, "%d%b%Y").date()
