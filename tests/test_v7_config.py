# tests/test_v7_config.py
"""Tests for V7-specific configuration."""
import pytest
from v7.config_v7 import (
    WATCHLIST, CAPITAL, PHASE_TIMES,
    RISK_LIMITS, THETA_LIMITS, BROKERAGE,
    is_15min_boundary, get_conviction_risk_pct,
)
from datetime import time


def test_watchlist_has_10_instruments():
    assert len(WATCHLIST) == 10
    assert "NIFTY" in [w["symbol"] for w in WATCHLIST]
    assert "BANKNIFTY" in [w["symbol"] for w in WATCHLIST]


def test_watchlist_has_lot_sizes():
    nifty = next(w for w in WATCHLIST if w["symbol"] == "NIFTY")
    assert nifty["lot_size"] == 75
    assert nifty["type"] == "index"


def test_capital_defaults():
    assert CAPITAL["initial"] == 300_000
    assert CAPITAL["cash_reserve_pct"] == 0.20


def test_phase_times():
    assert PHASE_TIMES["opening_read_end"] == time(9, 45)
    assert PHASE_TIMES["active_start"] == time(9, 45)
    assert PHASE_TIMES["wind_down_start"] == time(14, 30)


def test_risk_limits():
    assert RISK_LIMITS["max_daily_risk_pct"] == 4.0
    assert RISK_LIMITS["max_per_trade_risk_pct"] == 1.5
    assert RISK_LIMITS["max_trades_per_day"] == 4
    assert RISK_LIMITS["max_concurrent_positions"] == 4
    assert RISK_LIMITS["survival_mode_threshold_pct"] == 5.0
    assert RISK_LIMITS["full_stop_threshold_pct"] == 8.0


def test_theta_limits():
    assert THETA_LIMITS["max_margin_pct"] == 0.40
    assert THETA_LIMITS["min_vix"] == 14.0
    assert THETA_LIMITS["max_vix"] == 20.0
    assert THETA_LIMITS["profit_target_pct"] == 0.50


def test_brokerage():
    assert BROKERAGE["flat_per_order"] == 20.0
    assert BROKERAGE["min_trade_value"] == 2000.0


def test_is_15min_boundary():
    assert is_15min_boundary(time(10, 0)) is True
    assert is_15min_boundary(time(10, 15)) is True
    assert is_15min_boundary(time(10, 30)) is True
    assert is_15min_boundary(time(10, 45)) is True
    assert is_15min_boundary(time(10, 3)) is False
    assert is_15min_boundary(time(10, 14)) is False
    assert is_15min_boundary(time(10, 1)) is True
    assert is_15min_boundary(time(10, 16)) is True


def test_get_conviction_risk_pct():
    assert get_conviction_risk_pct("high") == 2.0
    assert get_conviction_risk_pct("medium") == 1.5
    assert get_conviction_risk_pct("low") == 0.75
