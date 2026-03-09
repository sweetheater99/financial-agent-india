"""Tests for time_intel.py — expiry detection, gaps, no-trade zones, IV."""

from datetime import date, time
from time_intel import (
    is_expiry_day,
    get_no_trade_zone,
    compute_gap,
    is_iv_elevated,
    format_time_context_for_claude,
)


# ── is_expiry_day ──────────────────────────────────────────────

def test_regular_thursday_is_weekly():
    # 2026-03-12 is a Thursday, not the last one in March
    d = date(2026, 3, 12)
    assert is_expiry_day(d) == "weekly"


def test_last_thursday_is_monthly():
    # 2026-03-26 is the last Thursday of March 2026
    d = date(2026, 3, 26)
    assert is_expiry_day(d) == "monthly"


def test_non_thursday_returns_none():
    # 2026-03-09 is a Monday
    d = date(2026, 3, 9)
    assert is_expiry_day(d) is None


def test_friday_returns_none():
    d = date(2026, 3, 13)  # Friday
    assert is_expiry_day(d) is None


def test_holiday_shift_weekly():
    """If Thursday is a holiday, Wednesday becomes weekly expiry."""
    thursday = date(2026, 3, 12)
    wednesday = date(2026, 3, 11)
    holidays = [thursday]
    assert is_expiry_day(wednesday, holidays) == "weekly"
    # The Thursday itself should not be expiry
    assert is_expiry_day(thursday, holidays) is None


def test_holiday_shift_monthly():
    """If last Thursday is a holiday, previous Wednesday becomes monthly expiry."""
    last_thursday = date(2026, 3, 26)
    wednesday = date(2026, 3, 25)
    holidays = [last_thursday]
    assert is_expiry_day(wednesday, holidays) == "monthly"


def test_february_last_thursday():
    # Feb 2026: last Thursday is Feb 26
    d = date(2026, 2, 26)
    assert is_expiry_day(d) == "monthly"


# ── get_no_trade_zone ──────────────────────────────────────────

def test_opening_auction():
    assert get_no_trade_zone(time(9, 15)) == "opening_auction"
    assert get_no_trade_zone(time(9, 20)) == "opening_auction"
    assert get_no_trade_zone(time(9, 29, 59)) == "opening_auction"


def test_post_lunch():
    assert get_no_trade_zone(time(13, 0)) == "post_lunch"
    assert get_no_trade_zone(time(13, 30)) == "post_lunch"
    assert get_no_trade_zone(time(13, 59)) == "post_lunch"


def test_closing_auction():
    assert get_no_trade_zone(time(15, 15)) == "closing_auction"
    assert get_no_trade_zone(time(15, 29)) == "closing_auction"


def test_normal_trading_hours():
    assert get_no_trade_zone(time(10, 0)) is None
    assert get_no_trade_zone(time(11, 30)) is None
    assert get_no_trade_zone(time(14, 30)) is None


def test_boundary_not_in_zone():
    # 9:30 is exactly outside opening auction
    assert get_no_trade_zone(time(9, 30)) is None
    # 14:00 is exactly outside post lunch
    assert get_no_trade_zone(time(14, 0)) is None
    # 15:30 is exactly outside closing auction
    assert get_no_trade_zone(time(15, 30)) is None


# ── compute_gap ────────────────────────────────────────────────

def test_gap_up():
    result = compute_gap(100.0, 102.0)
    assert result["direction"] == "up"
    assert result["gap_pct"] == 2.0
    assert result["signed_pct"] == 2.0


def test_gap_down():
    result = compute_gap(100.0, 97.5)
    assert result["direction"] == "down"
    assert result["gap_pct"] == 2.5
    assert result["signed_pct"] == -2.5


def test_no_gap():
    result = compute_gap(100.0, 100.0)
    assert result["gap_pct"] == 0.0
    assert result["direction"] == "up"  # zero treated as up


def test_gap_zero_prev_close():
    result = compute_gap(0.0, 100.0)
    assert result["gap_pct"] == 0.0


# ── is_iv_elevated ─────────────────────────────────────────────

def test_iv_elevated_true():
    # 25 is more than 20% above 20
    assert is_iv_elevated(25.0, 20.0) is True


def test_iv_elevated_false():
    # 23 is only 15% above 20, not > 20%
    assert is_iv_elevated(23.0, 20.0) is False


def test_iv_elevated_exact_boundary():
    # Exactly 20% above → 24.0. Should NOT be elevated (need strictly >20%)
    assert is_iv_elevated(24.0, 20.0) is False


def test_iv_elevated_zero_avg():
    assert is_iv_elevated(10.0, 0.0) is False


# ── format_time_context_for_claude ─────────────────────────────

def test_format_empty_when_nothing_noteworthy():
    # Monday, normal hours, no gap, normal VIX
    result = format_time_context_for_claude(
        current_time=time(11, 0),
        d=date(2026, 3, 9),  # Monday
        gap={"gap_pct": 0.05, "direction": "up", "signed_pct": 0.05},
        vix=15.0,
        avg_vix=14.5,
        holidays=None,
    )
    assert result == ""


def test_format_includes_expiry():
    result = format_time_context_for_claude(
        current_time=time(11, 0),
        d=date(2026, 3, 12),  # Thursday
        gap=None,
        vix=None,
        avg_vix=None,
    )
    assert "expiry" in result.lower()


def test_format_includes_no_trade_zone():
    result = format_time_context_for_claude(
        current_time=time(9, 20),
        d=date(2026, 3, 9),
        gap=None,
        vix=None,
        avg_vix=None,
    )
    assert "opening" in result.lower() or "No-trade" in result


def test_format_includes_gap():
    gap = compute_gap(100.0, 102.5)
    result = format_time_context_for_claude(
        current_time=time(11, 0),
        d=date(2026, 3, 9),
        gap=gap,
        vix=None,
        avg_vix=None,
    )
    assert "gap" in result.lower()


def test_format_includes_iv_elevated():
    result = format_time_context_for_claude(
        current_time=time(11, 0),
        d=date(2026, 3, 9),
        gap=None,
        vix=25.0,
        avg_vix=20.0,
    )
    assert "IV elevated" in result or "elevated" in result.lower()
