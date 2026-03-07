"""Tests for IV crush event strategy."""

from datetime import date, timedelta
from unittest.mock import patch, MagicMock


def _make_portfolio(positions=None, capital=100000):
    return {
        "positions": positions or [],
        "available_capital": capital,
        "capital": capital,
        "stats": {"total_pnl": 0},
    }


def _make_iv_crush_position(event_date="2026-04-09", net_credit=3000,
                             max_loss=4500, entry_date="2026-04-07",
                             max_hold_date="2026-04-10",
                             underlying_at_entry=23000):
    """Create a sample IV crush position for testing."""
    return {
        "id": f"iv_crush_{entry_date}_RBI_POLICY",
        "strategy": "iv_crush",
        "symbol": "NIFTY",
        "instrument": "IV_CRUSH",
        "entry_date": entry_date,
        "expiry": "16APR2026",
        "quantity": 75,
        "direction": "neutral",
        "entry_price": 0,
        "call_short": {"strike": 23300, "entry_premium": 60, "option_type": "CE"},
        "call_long": {"strike": 23500, "entry_premium": 25, "option_type": "CE"},
        "put_short": {"strike": 22700, "entry_premium": 55, "option_type": "PE"},
        "put_long": {"strike": 22500, "entry_premium": 20, "option_type": "PE"},
        "net_credit": net_credit,
        "max_profit": net_credit,
        "max_loss": max_loss,
        "allocated": 2250,
        "max_hold_date": max_hold_date,
        "underlying_at_entry": underlying_at_entry,
        "vix_at_entry": 18.5,
        "status": "open",
        "regime_at_entry": "EVENT_PLAY",
        "iv_percentile_at_entry": 0,
        "event_type": "RBI_POLICY",
        "event_date": event_date,
        "strategy_tag": "iv_crush",
        "index_name": "NIFTY",
    }


# --- Test 1: Upcoming events found ---

def test_upcoming_events_found():
    """Mock today near an RBI date, verify event returned."""
    from iv_crush import get_upcoming_events

    # 2026-04-07 is 2 trading days before RBI on 2026-04-09
    events = get_upcoming_events(days_ahead=3, _today=date(2026, 4, 7))
    rbi_events = [e for e in events if e["type"] == "RBI_POLICY"]
    assert len(rbi_events) >= 1
    assert rbi_events[0]["date"] == "2026-04-09"
    assert rbi_events[0]["days_until"] > 0
    assert rbi_events[0]["days_until"] <= 3


# --- Test 2: No upcoming events ---

def test_no_upcoming_events():
    """Mock today far from any event, verify empty."""
    from iv_crush import get_upcoming_events

    # 2026-03-15 is a Sunday (won't matter), 2026-03-16 is a Monday
    # Far from RBI (next is 2026-04-09) and budget (2026-02-01 already passed)
    events = get_upcoming_events(days_ahead=2, _today=date(2026, 3, 16))

    # Filter to only RBI/BUDGET (monthly expiry might show up)
    high_impact = [e for e in events if e["type"] in ("RBI_POLICY", "BUDGET")]
    assert len(high_impact) == 0


# --- Test 3: Entry conditions met ---

def test_entry_conditions_met():
    """All conditions good, verify trade setup returned."""
    from iv_crush import should_enter_iv_crush

    portfolio = _make_portfolio()
    # 2026-04-07 is 2 trading days before RBI on 2026-04-09
    result = should_enter_iv_crush(
        smart_api=None,
        _today=date(2026, 4, 7),
        _vix=18.5,
        _regime="SIDEWAYS",
        _portfolio=portfolio,
    )

    assert result is not None
    assert result["event"] == "RBI_POLICY"
    assert result["event_date"] == "2026-04-09"
    assert result["strategy"] == "IV_CRUSH_CONDOR"
    assert result["vix_at_signal"] == 18.5
    assert result["max_hold_date"] is not None


# --- Test 4: Entry blocked - low VIX ---

def test_entry_blocked_low_vix():
    """VIX < 15, verify None returned."""
    from iv_crush import should_enter_iv_crush

    portfolio = _make_portfolio()
    result = should_enter_iv_crush(
        smart_api=None,
        _today=date(2026, 4, 7),
        _vix=12.0,  # too low
        _regime="SIDEWAYS",
        _portfolio=portfolio,
    )
    assert result is None


# --- Test 5: Entry blocked - VOLATILE regime ---

def test_entry_blocked_volatile_regime():
    """VOLATILE regime, verify None."""
    from iv_crush import should_enter_iv_crush

    portfolio = _make_portfolio()
    result = should_enter_iv_crush(
        smart_api=None,
        _today=date(2026, 4, 7),
        _vix=20.0,
        _regime="VOLATILE",
        _portfolio=portfolio,
    )
    assert result is None


# --- Test 6: Entry blocked - existing position ---

def test_entry_blocked_existing_position():
    """Already have IV crush position, verify None."""
    from iv_crush import should_enter_iv_crush

    existing_pos = _make_iv_crush_position()
    portfolio = _make_portfolio(positions=[existing_pos])

    result = should_enter_iv_crush(
        smart_api=None,
        _today=date(2026, 4, 7),
        _vix=18.5,
        _regime="SIDEWAYS",
        _portfolio=portfolio,
    )
    assert result is None


# --- Test 7: Exit after event ---

def test_exit_after_event():
    """Event passed, verify exit signal."""
    from iv_crush import check_iv_crush_exit

    pos = _make_iv_crush_position(
        event_date="2026-04-09",
        net_credit=3000,
        max_hold_date="2026-04-10",
    )
    # Set _current_value to simulate modest profit (20% -- below early profit threshold)
    pos["_current_value"] = 2400  # (3000-2400)/3000 = 20% profit

    should_exit, reason = check_iv_crush_exit(
        pos, smart_api=None,
        _today="2026-04-10",  # day after event
    )
    assert should_exit is True
    assert "event_passed" in reason or "max_hold" in reason


# --- Test 8: Exit early profit ---

def test_exit_early_profit():
    """40%+ profit before event, verify exit."""
    from iv_crush import check_iv_crush_exit

    pos = _make_iv_crush_position(
        event_date="2026-04-09",
        net_credit=3000,
        max_hold_date="2026-04-10",
    )
    # Simulate 50% profit: cost to close is 1500, credit was 3000
    pos["_current_value"] = 1500  # (3000-1500)/3000 = 50% profit

    should_exit, reason = check_iv_crush_exit(
        pos, smart_api=None,
        _today="2026-04-08",  # before event
    )
    assert should_exit is True
    assert "early_profit" in reason


# --- Test 9: Emergency loss exit ---

def test_exit_emergency_loss():
    """50%+ loss, verify emergency exit."""
    from iv_crush import check_iv_crush_exit

    pos = _make_iv_crush_position(
        event_date="2026-04-09",
        net_credit=3000,
        max_hold_date="2026-04-10",
    )
    # Simulate 60% loss: cost to close is 4800, credit was 3000
    # pnl = 3000 - 4800 = -1800, pnl_pct = -60%
    pos["_current_value"] = 4800

    should_exit, reason = check_iv_crush_exit(
        pos, smart_api=None,
        _today="2026-04-08",  # before event
    )
    assert should_exit is True
    assert "emergency_loss" in reason


# --- Test 10: Hold before event ---

def test_hold_before_event():
    """Event not yet passed, modest P&L, verify no exit."""
    from iv_crush import check_iv_crush_exit

    pos = _make_iv_crush_position(
        event_date="2026-04-09",
        net_credit=3000,
        max_hold_date="2026-04-10",
    )
    # Simulate small profit: 15%
    pos["_current_value"] = 2550  # (3000-2550)/3000 = 15%

    should_exit, reason = check_iv_crush_exit(
        pos, smart_api=None,
        _today="2026-04-08",  # before event, within thresholds
    )
    assert should_exit is False
    assert reason == "hold"


# --- Test: CASH regime also blocked ---

def test_entry_blocked_cash_regime():
    """CASH regime should also block entry."""
    from iv_crush import should_enter_iv_crush

    portfolio = _make_portfolio()
    result = should_enter_iv_crush(
        smart_api=None,
        _today=date(2026, 4, 7),
        _vix=20.0,
        _regime="CASH",
        _portfolio=portfolio,
    )
    assert result is None


# --- Test: Monthly expiry detected ---

def test_monthly_expiry_event_detected():
    """Monthly expiry (last Thursday) should be detected as an event."""
    from iv_crush import get_upcoming_events

    # 2026-03-24 is 2 trading days before 2026-03-26 (last Thursday of March)
    events = get_upcoming_events(days_ahead=3, _today=date(2026, 3, 24))
    expiry_events = [e for e in events if e["type"] == "MONTHLY_EXPIRY"]
    assert len(expiry_events) >= 1
    assert expiry_events[0]["date"] == "2026-03-26"


# --- Test: Open IV crush condor ---

def test_open_iv_crush_condor():
    """Test opening an IV crush condor position."""
    from iv_crush import open_iv_crush_condor

    portfolio = _make_portfolio(capital=100000)
    crush_setup = {
        "event": "RBI_POLICY",
        "event_date": "2026-04-09",
        "strategy": "IV_CRUSH_CONDOR",
        "days_until_event": 2,
        "vix_at_signal": 18.5,
        "target_expiry": "2026-04-16",
        "max_hold_date": "2026-04-10",
        "otm_range_min": 300,
        "otm_range_max": 500,
        "size_multiplier": 0.5,
    }
    condor_data = {
        "call_short_strike": 23300,
        "call_long_strike": 23500,
        "put_short_strike": 22700,
        "put_long_strike": 22500,
        "call_short_premium": 60,
        "call_long_premium": 25,
        "put_short_premium": 55,
        "put_long_premium": 20,
        "net_credit": 5250,
        "max_loss": 9750,
        "width": 200,
        "lot_size": 75,
        "spot": 23000,
    }

    pos = open_iv_crush_condor(portfolio, crush_setup, condor_data)
    assert pos is not None
    assert pos["instrument"] == "IV_CRUSH"
    assert pos["event_type"] == "RBI_POLICY"
    assert pos["event_date"] == "2026-04-09"
    assert pos["strategy_tag"] == "iv_crush"
    assert pos["net_credit"] > 0
    assert portfolio["available_capital"] < 100000
