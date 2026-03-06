"""Tests for risk_guardrails module."""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from risk_guardrails import (
    check_correlation_guard,
    check_daily_loss_limit,
    check_drawdown_circuit_breaker,
    check_event_calendar,
    check_portfolio_heat,
    get_consecutive_losses,
    run_all_guards,
)

SYMBOL_SECTOR = {
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking",
    "TCS": "IT", "INFY": "IT",
    "RELIANCE": "Energy",
    "MARUTI": "Auto",
    "SUNPHARMA": "Pharma",
}

CAPITAL = 100_000


def _make_portfolio(
    positions=None, closed_trades=None, capital=CAPITAL, total_pnl=0
):
    return {
        "capital": capital,
        "available_capital": capital + total_pnl - sum(
            p.get("allocated", 0)
            for p in (positions or [])
            if p.get("status") == "open"
        ),
        "positions": positions or [],
        "closed_trades": closed_trades or [],
        "stats": {"total_pnl": total_pnl},
    }


def _make_closed(pnl, exit_date="2026-03-05", symbol="TEST"):
    return {
        "symbol": symbol, "direction": "bullish",
        "entry_date": "2026-03-01", "exit_date": exit_date,
        "pnl": pnl, "pnl_pct": pnl / 1000 * 100, "exit_reason": "test",
    }


def _make_position(symbol, direction="bullish", allocated=10000,
                   entry_price=100, stoploss_price=95, status="open"):
    return {
        "symbol": symbol, "direction": direction, "instrument": "EQ",
        "entry_price": entry_price, "stoploss_price": stoploss_price,
        "allocated": allocated, "status": status, "entry_date": "2026-03-01",
    }


# ── 1. Monthly drawdown blocks when > 8% ────────────────────────────────────

class TestDrawdownCircuitBreaker:
    def test_monthly_drawdown_blocks(self):
        """Monthly drawdown > 8% of capital blocks all trades."""
        # -9000 on 100k capital = 9% > 8%
        closed = [_make_closed(-9000, exit_date="2026-03-04")]
        portfolio = _make_portfolio(closed_trades=closed, total_pnl=-9000)
        blocked, reason, mult = check_drawdown_circuit_breaker(portfolio, _today="2026-03-06")
        assert blocked is True
        assert "Monthly drawdown" in reason

    def test_monthly_drawdown_allows(self):
        """Monthly drawdown < 8% allows trades."""
        closed = [_make_closed(-5000, exit_date="2026-03-04")]
        portfolio = _make_portfolio(closed_trades=closed, total_pnl=-5000)
        blocked, reason, mult = check_drawdown_circuit_breaker(portfolio, _today="2026-03-06")
        assert blocked is False

    def test_weekly_drawdown_reduces_allocation(self):
        """Weekly drawdown > 4% reduces allocation by 50%."""
        # -5000 on 100k capital = 5% > 4%
        closed = [_make_closed(-5000, exit_date="2026-03-04")]  # Wednesday
        portfolio = _make_portfolio(closed_trades=closed, total_pnl=-5000)
        # 2026-03-06 is Friday, week started Monday 2026-03-02
        blocked, reason, mult = check_drawdown_circuit_breaker(portfolio, _today="2026-03-06")
        assert blocked is False
        assert mult == 0.5

    def test_consecutive_3_losses_pauses_1_day(self):
        """3 consecutive losses triggers 1-day cooling off."""
        closed = [
            _make_closed(-500, exit_date="2026-03-05"),
            _make_closed(-500, exit_date="2026-03-05"),
            _make_closed(-500, exit_date="2026-03-05"),
        ]
        portfolio = _make_portfolio(closed_trades=closed, total_pnl=-1500)
        # Same day as last exit → should be blocked (resume next trading day)
        blocked, reason, mult = check_drawdown_circuit_breaker(portfolio, _today="2026-03-05")
        assert blocked is True
        assert "consecutive losses" in reason.lower()
        assert "1-day" in reason

    def test_consecutive_5_losses_pauses_2_days(self):
        """5 consecutive losses triggers 2-day cooling off."""
        closed = [
            _make_closed(-200, exit_date="2026-03-04"),
            _make_closed(-200, exit_date="2026-03-04"),
            _make_closed(-200, exit_date="2026-03-04"),
            _make_closed(-200, exit_date="2026-03-04"),
            _make_closed(-200, exit_date="2026-03-04"),
        ]
        portfolio = _make_portfolio(closed_trades=closed, total_pnl=-1000)
        # Day after last exit
        blocked, reason, mult = check_drawdown_circuit_breaker(portfolio, _today="2026-03-05")
        assert blocked is True
        assert "2-day" in reason

    def test_no_constraints_passes(self):
        """No drawdown issues allows trading."""
        closed = [_make_closed(500, exit_date="2026-03-04")]
        portfolio = _make_portfolio(closed_trades=closed, total_pnl=500)
        blocked, reason, mult = check_drawdown_circuit_breaker(portfolio, _today="2026-03-06")
        assert blocked is False
        assert mult == 1.0


# ── 2. Daily loss limit ─────────────────────────────────────────────────────

class TestDailyLossLimit:
    def test_daily_loss_blocks(self):
        """Daily loss > 2% of capital blocks new trades."""
        closed = [_make_closed(-2500, exit_date="2026-03-06")]
        portfolio = _make_portfolio(closed_trades=closed, total_pnl=-2500)
        blocked, reason = check_daily_loss_limit(portfolio, _today="2026-03-06")
        assert blocked is True
        assert "Daily loss" in reason

    def test_daily_loss_allows_early(self):
        """Daily loss < 2% allows trades."""
        closed = [_make_closed(-500, exit_date="2026-03-06")]
        portfolio = _make_portfolio(closed_trades=closed, total_pnl=-500)
        blocked, reason = check_daily_loss_limit(portfolio, _today="2026-03-06")
        assert blocked is False

    def test_daily_loss_ignores_other_days(self):
        """Losses from other days don't count."""
        closed = [_make_closed(-5000, exit_date="2026-03-05")]
        portfolio = _make_portfolio(closed_trades=closed, total_pnl=-5000)
        blocked, reason = check_daily_loss_limit(portfolio, _today="2026-03-06")
        assert blocked is False


# ── 3. Correlation guard ────────────────────────────────────────────────────

class TestCorrelationGuard:
    def test_sector_blocks_third_position(self):
        """3rd position in same sector is blocked."""
        positions = [
            _make_position("HDFCBANK"),
            _make_position("ICICIBANK"),
        ]
        portfolio = _make_portfolio(positions=positions)
        blocked, reason, mult = check_correlation_guard(
            portfolio, "SBIN", "bullish", SYMBOL_SECTOR
        )
        assert blocked is True
        assert "sector" in reason.lower()

    def test_sector_allows_different_sector(self):
        """Position in different sector is allowed."""
        positions = [
            _make_position("HDFCBANK"),
            _make_position("ICICIBANK"),
        ]
        portfolio = _make_portfolio(positions=positions)
        blocked, reason, mult = check_correlation_guard(
            portfolio, "TCS", "bullish", SYMBOL_SECTOR
        )
        assert blocked is False

    def test_second_in_sector_reduces_allocation(self):
        """2nd position in same sector reduces allocation 25%."""
        positions = [_make_position("HDFCBANK")]
        portfolio = _make_portfolio(positions=positions)
        blocked, reason, mult = check_correlation_guard(
            portfolio, "ICICIBANK", "bullish", SYMBOL_SECTOR
        )
        assert blocked is False
        assert mult == 0.75

    def test_direction_blocks_fourth_same_direction(self):
        """4th same-direction position is blocked."""
        positions = [
            _make_position("HDFCBANK", direction="bullish"),
            _make_position("TCS", direction="bullish"),
            _make_position("RELIANCE", direction="bullish"),
        ]
        portfolio = _make_portfolio(positions=positions)
        blocked, reason, mult = check_correlation_guard(
            portfolio, "MARUTI", "bullish", SYMBOL_SECTOR
        )
        assert blocked is True
        assert "direction" in reason.lower()

    def test_direction_allows_different_direction(self):
        """Position in different direction is allowed."""
        positions = [
            _make_position("HDFCBANK", direction="bullish"),
            _make_position("TCS", direction="bullish"),
            _make_position("RELIANCE", direction="bullish"),
        ]
        portfolio = _make_portfolio(positions=positions)
        blocked, reason, mult = check_correlation_guard(
            portfolio, "MARUTI", "bearish", SYMBOL_SECTOR
        )
        assert blocked is False

    def test_max_positions_cap(self):
        """Total position cap is enforced."""
        positions = [_make_position(f"SYM{i}") for i in range(8)]
        portfolio = _make_portfolio(positions=positions)
        blocked, reason, mult = check_correlation_guard(
            portfolio, "NEW", "bullish", SYMBOL_SECTOR
        )
        assert blocked is True
        assert "Max open positions" in reason


# ── 4. Portfolio heat ────────────────────────────────────────────────────────

class TestPortfolioHeat:
    def test_heat_calculation_correct(self):
        """Portfolio heat is sum of (allocated * risk_pct)."""
        # 2 positions: allocated=10000, entry=100, sl=95 → risk 5%
        # Heat per = 10000 * 0.05 = 500 each, total = 1000
        positions = [
            _make_position("HDFCBANK", entry_price=100, stoploss_price=95),
            _make_position("TCS", entry_price=100, stoploss_price=95),
        ]
        portfolio = _make_portfolio(positions=positions)
        # New position: 10000 * 0.05 = 500, total heat = 1500
        # 6% of 100k = 6000, so should pass
        blocked, reason = check_portfolio_heat(portfolio, 10000, new_risk_pct=0.05)
        assert blocked is False

    def test_heat_blocks_when_exceeding(self):
        """Portfolio heat blocks when exceeding 6% of capital."""
        # 5 positions each: allocated=20000, entry=100, sl=95 → risk 5%
        # Heat per = 20000 * 0.05 = 1000 each, total = 5000
        positions = [
            _make_position(f"SYM{i}", allocated=20000, entry_price=100, stoploss_price=95)
            for i in range(5)
        ]
        portfolio = _make_portfolio(positions=positions)
        # Adding 20000 * 0.05 = 1000, total = 6000 → exactly at limit
        # Try to add another which pushes over
        blocked, reason = check_portfolio_heat(portfolio, 25000, new_risk_pct=0.05)
        assert blocked is True
        assert "heat" in reason.lower()

    def test_heat_allows_within_limit(self):
        """Portfolio heat allows when within limit."""
        portfolio = _make_portfolio()
        blocked, reason = check_portfolio_heat(portfolio, 10000, new_risk_pct=0.03)
        assert blocked is False


# ── 5. Consecutive loss counter ──────────────────────────────────────────────

class TestConsecutiveLosses:
    def test_counts_losses(self):
        closed = [
            _make_closed(500),
            _make_closed(-100),
            _make_closed(-100),
            _make_closed(-100),
        ]
        assert get_consecutive_losses(closed) == 3

    def test_resets_on_win(self):
        closed = [
            _make_closed(-100),
            _make_closed(-100),
            _make_closed(500),
        ]
        assert get_consecutive_losses(closed) == 0

    def test_empty_list(self):
        assert get_consecutive_losses([]) == 0

    def test_all_losses(self):
        closed = [_make_closed(-100) for _ in range(7)]
        assert get_consecutive_losses(closed) == 7


# ── 6. Event calendar ───────────────────────────────────────────────────────

class TestEventCalendar:
    def test_budget_day_blocks(self):
        """Budget day blocks all trades."""
        blocked, reason, mult = check_event_calendar("TCS", _today="2026-02-01")
        assert blocked is True
        assert "Budget" in reason

    def test_rbi_policy_blocks(self):
        """RBI policy day blocks all trades."""
        blocked, reason, mult = check_event_calendar("TCS", _today="2026-04-09")
        assert blocked is True
        assert "RBI" in reason

    def test_monthly_expiry_reduces(self):
        """Last Thursday of month reduces allocation 50%."""
        # 2026-03-26 is the last Thursday of March 2026
        blocked, reason, mult = check_event_calendar("TCS", _today="2026-03-26")
        assert blocked is False
        assert mult == 0.5
        assert "expiry" in reason.lower()

    def test_weekly_expiry_reduces(self):
        """Regular Thursday reduces allocation 25%."""
        # 2026-03-05 is a Thursday but not last Thursday of March
        blocked, reason, mult = check_event_calendar("TCS", _today="2026-03-05")
        assert blocked is False
        assert mult == 0.75

    def test_normal_day_passes(self):
        """Normal non-Thursday, non-event day passes cleanly."""
        # 2026-03-04 is a Wednesday
        blocked, reason, mult = check_event_calendar("TCS", _today="2026-03-04")
        assert blocked is False
        assert mult == 1.0

    def test_last_thursday_detection(self):
        """Verify last Thursday detection for multiple months."""
        from risk_guardrails import _last_thursday_of_month
        from datetime import date as d
        # Jan 2026: last Thursday is Jan 29
        assert _last_thursday_of_month(2026, 1) == d(2026, 1, 29)
        # Feb 2026: last Thursday is Feb 26
        assert _last_thursday_of_month(2026, 2) == d(2026, 2, 26)
        # Dec 2026: last Thursday is Dec 31
        assert _last_thursday_of_month(2026, 12) == d(2026, 12, 31)


# ── 7. Integration: run_all_guards ───────────────────────────────────────────

class TestRunAllGuards:
    def test_all_pass(self):
        """When no constraints violated, everything passes."""
        portfolio = _make_portfolio()
        blocked, reason, mult = run_all_guards(
            portfolio, "TCS", "bullish", 10000, SYMBOL_SECTOR,
            new_risk_pct=0.03, _today="2026-03-04",  # Wednesday, no events
        )
        assert blocked is False
        assert mult == 1.0

    def test_drawdown_blocks_propagates(self):
        """Monthly drawdown block propagates through run_all_guards."""
        closed = [_make_closed(-9000, exit_date="2026-03-04")]
        portfolio = _make_portfolio(closed_trades=closed, total_pnl=-9000)
        blocked, reason, mult = run_all_guards(
            portfolio, "TCS", "bullish", 10000, SYMBOL_SECTOR,
            _today="2026-03-04",
        )
        assert blocked is True
        assert "Drawdown" in reason

    def test_multipliers_compound(self):
        """Multiple reduction factors compound."""
        # Weekly drawdown reduction (0.5) + Thursday weekly expiry (0.75)
        # Use loss on a previous day to avoid daily loss limit trigger
        closed = [_make_closed(-5000, exit_date="2026-03-04")]  # loss was yesterday
        portfolio = _make_portfolio(closed_trades=closed, total_pnl=-5000)
        blocked, reason, mult = run_all_guards(
            portfolio, "TCS", "bullish", 5000, SYMBOL_SECTOR,
            new_risk_pct=0.03, _today="2026-03-05",  # Thursday
        )
        assert blocked is False
        # 0.5 (weekly DD) * 0.75 (Thursday) = 0.375
        assert abs(mult - 0.375) < 0.01
