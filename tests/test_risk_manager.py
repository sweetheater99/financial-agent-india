import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from datetime import date, timedelta
from risk_manager import RiskManager


CAPITAL = 1_000_000.0  # 10 lakh


# ── record_trade_result ──────────────────────────────────────────────────────

def test_record_trade_result_updates_all_counters():
    rm = RiskManager(CAPITAL)
    rm.record_trade_result(-5000.0, "equity")
    assert rm._daily_pnl == -5000.0
    assert rm._weekly_pnl == -5000.0
    assert rm._monthly_pnl == -5000.0
    assert rm._total_pnl == -5000.0
    assert rm.current_capital == CAPITAL - 5000.0


def test_record_trade_result_accumulates():
    rm = RiskManager(CAPITAL)
    rm.record_trade_result(-2000.0)
    rm.record_trade_result(-3000.0)
    assert rm._daily_pnl == -5000.0
    assert rm._total_pnl == -5000.0
    assert rm.current_capital == CAPITAL - 5000.0


def test_record_trade_result_tracks_consecutive_losses():
    rm = RiskManager(CAPITAL)
    rm.record_trade_result(-1000.0)
    rm.record_trade_result(-1000.0)
    assert rm._consecutive_losses == 2
    # Win resets the counter
    rm.record_trade_result(5000.0)
    assert rm._consecutive_losses == 0


def test_record_trade_result_tracks_strategy_losses():
    rm = RiskManager(CAPITAL)
    rm.record_trade_result(-1000.0, "spread")
    rm.record_trade_result(-1000.0, "spread")
    assert rm._strategy_losses["spread"] == 2
    # Different strategy is independent
    rm.record_trade_result(-1000.0, "equity")
    assert rm._strategy_losses["equity"] == 1
    assert rm._strategy_losses["spread"] == 2


def test_record_trade_result_strategy_win_resets():
    rm = RiskManager(CAPITAL)
    rm.record_trade_result(-1000.0, "spread")
    rm.record_trade_result(-1000.0, "spread")
    rm.record_trade_result(2000.0, "spread")
    assert rm._strategy_losses.get("spread", 0) == 0


# ── Drawdown circuit breakers ────────────────────────────────────────────────

def test_check_daily_halt_triggers_at_3pct():
    rm = RiskManager(CAPITAL)
    # Loss of 3% of capital = 30,000
    rm.record_trade_result(-30_001.0)
    assert rm.check_daily_halt() is True


def test_check_daily_halt_no_trigger_below_3pct():
    rm = RiskManager(CAPITAL)
    rm.record_trade_result(-29_000.0)
    assert rm.check_daily_halt() is False


def test_check_weekly_reduce_triggers_at_5pct():
    rm = RiskManager(CAPITAL)
    rm.record_trade_result(-50_001.0)
    assert rm.check_weekly_reduce() is True


def test_check_weekly_reduce_no_trigger_below_5pct():
    rm = RiskManager(CAPITAL)
    # 5% of 1M = 50k, but capital drops after loss, so use a safe margin
    rm.record_trade_result(-40_000.0)
    assert rm.check_weekly_reduce() is False


def test_check_monthly_halt_triggers_at_8pct():
    rm = RiskManager(CAPITAL)
    rm.record_trade_result(-80_001.0)
    assert rm.check_monthly_halt() is True


def test_check_monthly_halt_no_trigger_below_8pct():
    rm = RiskManager(CAPITAL)
    # 8% of 1M = 80k, but capital drops after loss, so use a safe margin
    rm.record_trade_result(-70_000.0)
    assert rm.check_monthly_halt() is False


def test_check_total_stop_triggers_at_15pct():
    rm = RiskManager(CAPITAL)
    rm.record_trade_result(-150_001.0)
    assert rm.check_total_stop() is True


def test_check_total_stop_no_trigger_below_15pct():
    rm = RiskManager(CAPITAL)
    rm.record_trade_result(-140_000.0)
    assert rm.check_total_stop() is False


# ── Strategy pause ───────────────────────────────────────────────────────────

def test_is_strategy_paused_after_3_consecutive_losses():
    rm = RiskManager(CAPITAL)
    rm.record_trade_result(-1000.0, "spread")
    rm.record_trade_result(-1000.0, "spread")
    rm.record_trade_result(-1000.0, "spread")
    assert rm.is_strategy_paused("spread") is True


def test_is_strategy_paused_resets_after_win():
    rm = RiskManager(CAPITAL)
    rm.record_trade_result(-1000.0, "spread")
    rm.record_trade_result(-1000.0, "spread")
    rm.record_trade_result(-1000.0, "spread")
    assert rm.is_strategy_paused("spread") is True
    rm.record_trade_result(500.0, "spread")
    assert rm.is_strategy_paused("spread") is False


def test_is_strategy_paused_unknown_strategy():
    rm = RiskManager(CAPITAL)
    assert rm.is_strategy_paused("unknown") is False


# ── Cash mode ────────────────────────────────────────────────────────────────

def test_enter_cash_mode_and_is_cash_mode():
    rm = RiskManager(CAPITAL)
    assert rm.is_cash_mode() is False
    rm.enter_cash_mode(5)
    assert rm.is_cash_mode() is True


def test_cash_days_remaining():
    rm = RiskManager(CAPITAL)
    rm.enter_cash_mode(5, _today=date(2026, 3, 2))
    # On the same day, should have 5 days remaining
    assert rm.cash_days_remaining(_today=date(2026, 3, 2)) == 5
    # Next day, 4 remaining
    assert rm.cash_days_remaining(_today=date(2026, 3, 3)) == 4
    # On the last day
    assert rm.cash_days_remaining(_today=date(2026, 3, 6)) == 1
    # After expiry
    assert rm.cash_days_remaining(_today=date(2026, 3, 7)) == 0


def test_is_cash_mode_expires():
    rm = RiskManager(CAPITAL)
    rm.enter_cash_mode(3, _today=date(2026, 3, 2))
    assert rm.is_cash_mode(_today=date(2026, 3, 4)) is True
    assert rm.is_cash_mode(_today=date(2026, 3, 5)) is False


def test_cash_days_remaining_no_cash_mode():
    rm = RiskManager(CAPITAL)
    assert rm.cash_days_remaining() == 0


# ── get_risk_state ───────────────────────────────────────────────────────────

def test_get_risk_state_format():
    rm = RiskManager(CAPITAL)
    rm.record_trade_result(-50_000.0)
    rm.record_trade_result(-10_000.0)
    state = rm.get_risk_state()
    assert "monthly_drawdown_pct" in state
    assert "consecutive_losses" in state
    # monthly drawdown = -60000 / 1_000_000 * 100 = -6.0
    assert abs(state["monthly_drawdown_pct"] - (-6.0)) < 0.01
    assert state["consecutive_losses"] == 2


# ── can_open_position ────────────────────────────────────────────────────────

def test_can_open_position_blocks_max_concurrent():
    rm = RiskManager(CAPITAL)
    positions = [
        {"strategy": "equity", "sector": f"sector_{i}", "max_loss": 5000.0}
        for i in range(8)
    ]
    ok, reason = rm.can_open_position(positions)
    assert ok is False
    assert "concurrent" in reason.lower() or "position" in reason.lower()


def test_can_open_position_allows_below_max():
    rm = RiskManager(CAPITAL)
    positions = [
        {"strategy": "equity", "sector": f"sector_{i}", "max_loss": 5000.0}
        for i in range(5)
    ]
    ok, reason = rm.can_open_position(positions, new_sector="sector_99")
    assert ok is True


def test_can_open_position_blocks_max_same_sector():
    rm = RiskManager(CAPITAL)
    positions = [
        {"strategy": "equity", "sector": "IT", "max_loss": 5000.0},
        {"strategy": "equity", "sector": "IT", "max_loss": 5000.0},
    ]
    ok, reason = rm.can_open_position(positions, new_sector="IT")
    assert ok is False
    assert "sector" in reason.lower()


def test_can_open_position_allows_different_sector():
    rm = RiskManager(CAPITAL)
    positions = [
        {"strategy": "equity", "sector": "IT", "max_loss": 5000.0},
        {"strategy": "equity", "sector": "IT", "max_loss": 5000.0},
    ]
    ok, reason = rm.can_open_position(positions, new_sector="BANK")
    assert ok is True


def test_can_open_position_correlation_guard():
    rm = RiskManager(CAPITAL)
    # MAX_SIMULTANEOUS_LOSS_PCT = 10 => 10% of 1M = 100,000
    positions = [
        {"strategy": "equity", "sector": "IT", "max_loss": 50_000.0},
        {"strategy": "equity", "sector": "BANK", "max_loss": 40_000.0},
    ]
    # Adding another 15k would make total 105k > 100k
    ok, reason = rm.can_open_position(positions, new_sector="AUTO", new_max_loss=15_000.0)
    assert ok is False
    assert "loss" in reason.lower() or "correlation" in reason.lower()


def test_can_open_position_correlation_guard_allows():
    rm = RiskManager(CAPITAL)
    positions = [
        {"strategy": "equity", "sector": "IT", "max_loss": 30_000.0},
        {"strategy": "equity", "sector": "BANK", "max_loss": 30_000.0},
    ]
    ok, reason = rm.can_open_position(positions, new_sector="AUTO", new_max_loss=10_000.0)
    assert ok is True


# ── get_size_multiplier ──────────────────────────────────────────────────────

def test_get_size_multiplier_normal():
    rm = RiskManager(CAPITAL)
    assert rm.get_size_multiplier() == 1.0


def test_get_size_multiplier_weekly_loss():
    rm = RiskManager(CAPITAL)
    rm.record_trade_result(-50_001.0)
    assert rm.get_size_multiplier() == 0.5


# ── Reset methods ────────────────────────────────────────────────────────────

def test_reset_daily_clears_daily_only():
    rm = RiskManager(CAPITAL)
    rm.record_trade_result(-10_000.0)
    rm.reset_daily()
    assert rm._daily_pnl == 0.0
    assert rm._weekly_pnl == -10_000.0
    assert rm._monthly_pnl == -10_000.0
    assert rm._total_pnl == -10_000.0


def test_reset_weekly_clears_weekly_only():
    rm = RiskManager(CAPITAL)
    rm.record_trade_result(-10_000.0)
    rm.reset_weekly()
    assert rm._weekly_pnl == 0.0
    assert rm._daily_pnl == -10_000.0
    assert rm._monthly_pnl == -10_000.0


def test_reset_monthly_clears_monthly_only():
    rm = RiskManager(CAPITAL)
    rm.record_trade_result(-10_000.0)
    rm.reset_monthly()
    assert rm._monthly_pnl == 0.0
    assert rm._daily_pnl == -10_000.0
    assert rm._weekly_pnl == -10_000.0
