"""Tests for credit spread IV routing and exit logic."""
import pytest
import sys
import os
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_iv_routing_high_iv_returns_credit():
    from paper_trade import _select_spread_type
    result = _select_spread_type("bullish", iv_percentile=75, vix=20)
    assert result == "credit"


def test_iv_routing_low_iv_returns_debit():
    from paper_trade import _select_spread_type
    result = _select_spread_type("bullish", iv_percentile=40, vix=20)
    assert result == "debit"


def test_iv_routing_overlap_prefers_debit():
    from paper_trade import _select_spread_type
    result = _select_spread_type("bullish", iv_percentile=60, vix=20)
    assert result == "debit"


def test_iv_routing_vix_too_low_for_credit():
    from paper_trade import _select_spread_type
    result = _select_spread_type("bullish", iv_percentile=75, vix=12)
    assert result is None  # IV high blocks debit, VIX low blocks credit


def test_iv_routing_vix_too_high_for_credit():
    from paper_trade import _select_spread_type
    result = _select_spread_type("bullish", iv_percentile=75, vix=30)
    assert result is None  # IV high blocks debit, VIX too high blocks credit


def test_credit_spread_exit_target():
    from paper_trade import check_spread_exit
    pos = {
        "instrument": "SPREAD",
        "spread_type": "credit",
        "spread_direction": "bullish",
        "direction": "bullish",
        "net_credit": 3000,
        "max_profit": 3000,
        "max_loss": 4500,
        "allocated": 4500,
        "net_debit": 0,
        "long_leg": {"strike": 22100, "entry_premium": 65, "option_type": "PE"},
        "short_leg": {"strike": 22300, "entry_premium": 130, "option_type": "PE"},
        "underlying_at_entry": 22500,
        "atr_at_entry": 100,
        "spread_width": 200,
        "quantity": 75,
        "entry_date": "2026-03-01",
        "max_hold_date": "2026-03-20",
        "expiry": "26MAR2026",
    }
    # Short dropped to 30, long dropped to 10: cost_to_close = (30-10)*75 = 1500
    # P&L = 3000 - 1500 = 1500 = 50% of max profit -> EXIT
    exit_reason = check_spread_exit(
        pos, underlying_ltp=22550, today="2026-03-10",
        long_premium=10, short_premium=30,
    )
    assert exit_reason == "credit_target"


def test_credit_spread_exit_stoploss():
    from paper_trade import check_spread_exit
    pos = {
        "instrument": "SPREAD",
        "spread_type": "credit",
        "spread_direction": "bullish",
        "direction": "bullish",
        "net_credit": 3000,
        "max_profit": 3000,
        "max_loss": 4500,
        "allocated": 4500,
        "net_debit": 0,
        "long_leg": {"strike": 22100, "entry_premium": 65, "option_type": "PE"},
        "short_leg": {"strike": 22300, "entry_premium": 130, "option_type": "PE"},
        "underlying_at_entry": 22500,
        "atr_at_entry": 100,
        "spread_width": 200,
        "quantity": 75,
        "entry_date": "2026-03-01",
        "max_hold_date": "2026-03-20",
        "expiry": "26MAR2026",
    }
    # Short went deep ITM: short=300, long=180: cost_to_close = (300-180)*75 = 9000
    # P&L = 3000 - 9000 = -6000 = 2x credit received -> STOP
    exit_reason = check_spread_exit(
        pos, underlying_ltp=22100, today="2026-03-10",
        long_premium=180, short_premium=300,
    )
    assert exit_reason == "credit_stoploss"


def test_credit_spread_no_exit_when_profitable():
    from paper_trade import check_spread_exit
    pos = {
        "instrument": "SPREAD",
        "spread_type": "credit",
        "spread_direction": "bullish",
        "direction": "bullish",
        "net_credit": 3000,
        "max_profit": 3000,
        "max_loss": 4500,
        "allocated": 4500,
        "net_debit": 0,
        "long_leg": {"strike": 22100, "entry_premium": 65, "option_type": "PE"},
        "short_leg": {"strike": 22300, "entry_premium": 130, "option_type": "PE"},
        "underlying_at_entry": 22500,
        "atr_at_entry": 100,
        "spread_width": 200,
        "quantity": 75,
        "entry_date": "2026-03-01",
        "max_hold_date": "2026-03-20",
        "expiry": "26MAR2026",
    }
    # Mildly profitable but not at 50% target yet
    # short=100, long=55: cost_to_close=(100-55)*75=3375, P&L=3000-3375=-375 (small loss)
    exit_reason = check_spread_exit(
        pos, underlying_ltp=22450, today="2026-03-04",
        long_premium=55, short_premium=100,
    )
    assert exit_reason is None
