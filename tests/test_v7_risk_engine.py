# tests/test_v7_risk_engine.py
"""Tests for V7 Risk Engine — pure rules, no Claude."""
import json
import pytest
from datetime import date, time
from pathlib import Path
from v7.risk_engine import RiskEngine
from v7.types import (
    Conviction, Position, RiskBudget, PacingStatus, Setup, SetupType,
)


@pytest.fixture
def tmp_state_dir(tmp_path):
    return tmp_path / "v7_state"


@pytest.fixture
def engine(tmp_state_dir):
    return RiskEngine(capital=300_000, state_dir=tmp_state_dir)


# ── Per-Trade Sizing ────────────────────────────────────────────────────

def test_risk_amount_high_conviction(engine):
    amount = engine.risk_amount_for_conviction(Conviction.HIGH)
    assert amount == 6000.0  # 2% of 300k

def test_risk_amount_medium_conviction(engine):
    amount = engine.risk_amount_for_conviction(Conviction.MEDIUM)
    assert amount == 4500.0  # 1.5% of 300k

def test_risk_amount_low_conviction(engine):
    amount = engine.risk_amount_for_conviction(Conviction.LOW)
    assert amount == 2250.0  # 0.75% of 300k

def test_risk_amount_ahead_pacing_reduces_25pct(engine):
    engine.set_pacing(PacingStatus.AHEAD)
    amount = engine.risk_amount_for_conviction(Conviction.HIGH)
    assert amount == 4500.0  # 2% * 0.75 = 1.5% of 300k

def test_risk_amount_behind_pacing_no_increase(engine):
    engine.set_pacing(PacingStatus.BEHIND)
    amount = engine.risk_amount_for_conviction(Conviction.HIGH)
    assert amount == 6000.0  # no change

def test_lot_calculation(engine):
    lots = engine.calculate_lots(risk_amount=4500.0, premium=60.0, lot_size=75)
    assert lots == 1

def test_lot_calculation_rounds_down(engine):
    lots = engine.calculate_lots(risk_amount=4500.0, premium=25.0, lot_size=75)
    assert lots == 2

def test_lot_calculation_zero_if_too_expensive(engine):
    lots = engine.calculate_lots(risk_amount=4500.0, premium=100.0, lot_size=75)
    assert lots == 0

# ── Daily Limits ────────────────────────────────────────────────────────

def test_daily_loss_cap_blocks(engine):
    engine.record_daily_pnl(-6500.0)
    allowed, reason = engine.can_open_trade()
    assert allowed is False
    assert "daily loss" in reason.lower()

def test_daily_loss_under_cap_allows(engine):
    engine.record_daily_pnl(-5000.0)
    allowed, _ = engine.can_open_trade()
    assert allowed is True

def test_consecutive_sl_hits_blocks(engine):
    engine.record_sl_hit()
    engine.record_sl_hit()
    engine.record_sl_hit()
    allowed, reason = engine.can_open_trade()
    assert allowed is False
    assert "sl" in reason.lower() or "stop" in reason.lower()

def test_max_trades_blocks(engine):
    for _ in range(4):
        engine.record_trade_opened()
    allowed, reason = engine.can_open_trade()
    assert allowed is False
    assert "max trades" in reason.lower()

def test_margin_utilization_blocks(engine):
    engine.set_margin_used(0.75)
    allowed, reason = engine.can_open_trade()
    assert allowed is False
    assert "margin" in reason.lower()

def test_margin_under_threshold_allows(engine):
    engine.set_margin_used(0.60)
    allowed, _ = engine.can_open_trade()
    assert allowed is True

# ── Concurrent Risk Budget ──────────────────────────────────────────────

def test_concurrent_risk_allows(engine):
    can = engine.can_allocate_risk(4500.0, current_risk=0.0)
    assert can is True

def test_concurrent_risk_blocks_when_full(engine):
    can = engine.can_allocate_risk(4500.0, current_risk=10000.0)
    assert can is False

def test_concurrent_risk_allows_after_close(engine):
    can = engine.can_allocate_risk(4500.0, current_risk=4500.0)
    assert can is True

# ── Correlation Check ───────────────────────────────────────────────────

def test_correlation_blocks_third_same_direction(engine):
    positions = [
        Position(symbol="NIFTY", instrument="NIFTY CE", direction="bullish",
                 entry_price=100, quantity=75, lot_size=75, allocated=7500,
                 stoploss=80, target=150, entry_date=date(2026, 3, 11), setup_id="N1"),
        Position(symbol="HDFCBANK", instrument="HDFCBANK CE", direction="bullish",
                 entry_price=50, quantity=550, lot_size=550, allocated=27500,
                 stoploss=40, target=70, entry_date=date(2026, 3, 11), setup_id="H1"),
    ]
    blocked, reason = engine.check_correlation(positions, new_direction="bullish")
    assert blocked is True
    assert "correlation" in reason.lower()

def test_correlation_allows_different_direction(engine):
    positions = [
        Position(symbol="NIFTY", instrument="NIFTY CE", direction="bullish",
                 entry_price=100, quantity=75, lot_size=75, allocated=7500,
                 stoploss=80, target=150, entry_date=date(2026, 3, 11), setup_id="N1"),
        Position(symbol="HDFCBANK", instrument="HDFCBANK CE", direction="bullish",
                 entry_price=50, quantity=550, lot_size=550, allocated=27500,
                 stoploss=40, target=70, entry_date=date(2026, 3, 11), setup_id="H1"),
    ]
    blocked, _ = engine.check_correlation(positions, new_direction="bearish")
    assert blocked is False

# ── Daily Reset ─────────────────────────────────────────────────────────

def test_daily_reset(engine):
    engine.record_daily_pnl(-5000)
    engine.record_sl_hit()
    engine.record_sl_hit()
    engine.record_trade_opened()
    engine.reset_daily()
    allowed, _ = engine.can_open_trade()
    assert allowed is True

# ── Survival Mode ───────────────────────────────────────────────────────

def test_survival_mode_activates_at_5pct(engine):
    engine.update_mtd_pnl(-15000.0)
    assert engine.survival_mode is True
    allowed, reason = engine.can_open_trade()
    assert allowed is False
    assert "survival" in reason.lower()

def test_survival_allows_theta(engine):
    engine.update_mtd_pnl(-15000.0)
    allowed, _ = engine.can_open_theta()
    assert allowed is True

def test_full_stop_at_8pct(engine):
    engine.update_mtd_pnl(-24000.0)
    assert engine.full_stop is True
    allowed, reason = engine.can_open_trade()
    assert allowed is False
    assert "full stop" in reason.lower()

def test_full_stop_blocks_theta(engine):
    engine.update_mtd_pnl(-24000.0)
    allowed, reason = engine.can_open_theta()
    assert allowed is False

def test_recovery_exits_survival(engine):
    engine.update_mtd_pnl(-15000.0)
    assert engine.survival_mode is True
    engine.update_mtd_pnl(-5000.0)
    assert engine.survival_mode is False

def test_drawdown_reduces_sizing(engine):
    engine.update_mtd_pnl(-10000.0)  # 3.3% → reduce 25%
    amount = engine.risk_amount_for_conviction(Conviction.HIGH)
    assert amount == 4500.0  # 2% * 0.75

# ── F&O Ban List ────────────────────────────────────────────────────────

def test_fo_ban_blocks(engine):
    engine.update_fo_ban_list(["DELTACORP", "IBULHSGFIN"])
    assert engine.is_fo_banned("DELTACORP") is True
    assert engine.is_fo_banned("deltacorp") is True
    assert engine.is_fo_banned("NIFTY") is False

def test_pre_trade_check_blocks_banned(engine):
    engine.update_fo_ban_list(["DELTACORP"])
    allowed, reason, _ = engine.pre_trade_check(
        symbol="DELTACORP", conviction=Conviction.MEDIUM,
        direction="bullish", trade_value=5000.0,
        open_positions=[], current_risk=0.0,
    )
    assert allowed is False
    assert "ban" in reason.lower()

# ── Chop Detection ──────────────────────────────────────────────────────

def test_chop_detected_whipsaws(engine):
    engine.update_chop_signals(whipsaw_count=3, opening_range_pct=0.5, first_hour_volume_ratio=0.8)
    choppy, reason = engine.is_choppy()
    assert choppy is True
    assert "whipsaw" in reason.lower()

def test_chop_detected_narrow_range(engine):
    engine.update_chop_signals(whipsaw_count=0, opening_range_pct=0.2, first_hour_volume_ratio=0.8)
    choppy, reason = engine.is_choppy()
    assert choppy is True
    assert "narrow" in reason.lower()

def test_chop_detected_low_volume(engine):
    engine.update_chop_signals(whipsaw_count=0, opening_range_pct=0.5, first_hour_volume_ratio=0.4)
    choppy, reason = engine.is_choppy()
    assert choppy is True
    assert "volume" in reason.lower()

def test_no_chop_normal_market(engine):
    engine.update_chop_signals(whipsaw_count=1, opening_range_pct=0.8, first_hour_volume_ratio=1.2)
    choppy, _ = engine.is_choppy()
    assert choppy is False

# ── Brokerage Optimization ──────────────────────────────────────────────

def test_min_trade_value_blocks_small(engine):
    ok, reason = engine.check_min_trade_value(1500.0)
    assert ok is False
    assert "small" in reason.lower()

def test_min_trade_value_allows_large(engine):
    ok, _ = engine.check_min_trade_value(5000.0)
    assert ok is True

# ── Pre-Trade Master Check ──────────────────────────────────────────────

def test_pre_trade_check_passes_clean(engine):
    allowed, reason, risk = engine.pre_trade_check(
        symbol="NIFTY", conviction=Conviction.MEDIUM,
        direction="bullish", trade_value=5000.0,
        open_positions=[], current_risk=0.0,
    )
    assert allowed is True
    assert risk == 4500.0

def test_pre_trade_check_blocks_risk_full(engine):
    allowed, reason, risk = engine.pre_trade_check(
        symbol="NIFTY", conviction=Conviction.HIGH,
        direction="bullish", trade_value=5000.0,
        open_positions=[], current_risk=11000.0,
    )
    assert allowed is False
    assert "risk budget" in reason.lower()

# ── Persistence ─────────────────────────────────────────────────────────

def test_state_persists_across_instances(tmp_state_dir):
    e1 = RiskEngine(capital=300_000, state_dir=tmp_state_dir)
    e1.record_daily_pnl(-5000.0)
    e1.record_sl_hit()
    e1.record_sl_hit()
    e1.record_trade_opened()
    e1.update_mtd_pnl(-10000.0)
    e1.update_fo_ban_list(["DELTACORP"])

    e2 = RiskEngine(capital=300_000, state_dir=tmp_state_dir)
    assert e2._daily_pnl == -5000.0
    assert e2._sl_hits_today == 2
    assert e2._trades_today == 1
    assert e2.is_fo_banned("DELTACORP") is True

def test_monthly_reset(engine):
    engine.update_mtd_pnl(-20000.0)
    assert engine.survival_mode is True
    engine.reset_monthly()
    assert engine.survival_mode is False
    assert engine.full_stop is False
    allowed, _ = engine.can_open_trade()
    assert allowed is True

# ── State Summary ───────────────────────────────────────────────────────

def test_get_state_summary(engine):
    engine.record_daily_pnl(-2000.0)
    engine.record_sl_hit()
    engine.update_mtd_pnl(-5000.0)
    summary = engine.get_state_summary()
    assert summary["daily_pnl"] == -2000.0
    assert summary["sl_hits_today"] == 1
    assert summary["mtd_pnl"] == -5000.0
    assert "pacing" in summary
    assert "fo_ban_list" in summary
