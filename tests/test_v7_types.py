# tests/test_v7_types.py
"""Tests for V7 shared types."""
import pytest
from datetime import date, time
from v7.types import (
    DayPhase, DayClassification, Conviction, SetupType,
    Setup, Playbook, Position, TradeResult, CarryRules,
    RiskBudget, PacingStatus, MarketContext, KeyLevels,
)


def test_day_phase_from_time_premarket():
    assert DayPhase.from_time(time(8, 45)) == DayPhase.PRE_MARKET


def test_day_phase_from_time_opening_read():
    assert DayPhase.from_time(time(9, 20)) == DayPhase.OPENING_READ


def test_day_phase_from_time_active():
    assert DayPhase.from_time(time(11, 0)) == DayPhase.ACTIVE_TRADING


def test_day_phase_from_time_wind_down():
    assert DayPhase.from_time(time(14, 45)) == DayPhase.WIND_DOWN


def test_day_phase_from_time_post_close():
    assert DayPhase.from_time(time(15, 20)) == DayPhase.POST_CLOSE


def test_day_phase_from_time_outside_hours():
    assert DayPhase.from_time(time(7, 0)) == DayPhase.OUTSIDE_HOURS
    assert DayPhase.from_time(time(16, 0)) == DayPhase.OUTSIDE_HOURS


def test_setup_creation():
    s = Setup(
        id="N1",
        priority=1,
        type=SetupType.BREAKOUT_LONG,
        symbol="NIFTY",
        trigger_level=24350.0,
        trigger_condition="15-min close above with volume > 1.5x",
        instrument="NIFTY CE",
        strike_logic="slightly OTM, delta 0.40-0.50",
        target=24500.0,
        stoploss=24280.0,
        max_risk_pct=1.5,
    )
    assert s.id == "N1"
    assert s.type == SetupType.BREAKOUT_LONG
    assert s.fired is False
    assert s.cancelled is False


def test_setup_to_dict_roundtrip():
    s = Setup(
        id="N1", priority=1, type=SetupType.BREAKOUT_LONG,
        symbol="NIFTY", trigger_level=24350.0,
        trigger_condition="15-min close above",
        instrument="NIFTY CE", strike_logic="delta 0.45",
        target=24500.0, stoploss=24280.0, max_risk_pct=1.5,
    )
    d = s.to_dict()
    s2 = Setup.from_dict(d)
    assert s2.id == s.id
    assert s2.trigger_level == s.trigger_level
    assert s2.type == s.type


def test_position_pnl_long():
    p = Position(
        symbol="NIFTY", instrument="NIFTY CE 24400",
        direction="bullish", entry_price=100.0,
        quantity=75, lot_size=75, allocated=7500.0,
        stoploss=80.0, target=150.0,
        entry_date=date(2026, 3, 11), setup_id="N1",
    )
    assert p.unrealized_pnl(120.0) == 75 * 20.0  # 1500
    assert p.unrealized_pnl(80.0) == 75 * -20.0   # -1500


def test_position_to_dict_roundtrip():
    p = Position(
        symbol="NIFTY", instrument="NIFTY CE 24400",
        direction="bullish", entry_price=100.0,
        quantity=75, lot_size=75, allocated=7500.0,
        stoploss=80.0, target=150.0,
        entry_date=date(2026, 3, 11), setup_id="N1",
    )
    d = p.to_dict()
    p2 = Position.from_dict(d)
    assert p2.symbol == p.symbol
    assert p2.entry_price == p.entry_price
    assert p2.stoploss == p.stoploss


def test_trade_result_creation():
    tr = TradeResult(
        symbol="NIFTY", instrument="NIFTY CE 24400",
        direction="bullish", entry_price=100.0, exit_price=150.0,
        quantity=75, entry_date=date(2026, 3, 11),
        exit_date=date(2026, 3, 11), exit_reason="target",
        pnl=3750.0, pnl_pct=50.0, costs=120.0,
        setup_id="N1", setup_type=SetupType.BREAKOUT_LONG,
        entry_grade="A", exit_grade="A",
    )
    assert tr.pnl == 3750.0
    assert tr.exit_reason == "target"


def test_risk_budget_can_allocate():
    rb = RiskBudget(
        max_capital_at_risk_today_pct=4.0,
        max_trades_today=4,
        max_per_trade_risk_pct=1.5,
        survival_mode=False,
    )
    assert rb.can_allocate(4500, 0, 300_000) is True
    assert rb.can_allocate(4500, 10000, 300_000) is False


def test_risk_budget_can_enter_trade():
    rb = RiskBudget(
        max_capital_at_risk_today_pct=4.0,
        max_trades_today=4,
        max_per_trade_risk_pct=1.5,
        survival_mode=False,
    )
    assert rb.can_enter_trade(
        new_risk=4500, current_risk=0, capital=300_000,
        trades_today=0, consecutive_sl_hits=0, daily_pnl=-1000,
    ) is True
    assert rb.can_enter_trade(
        new_risk=4500, current_risk=0, capital=300_000,
        trades_today=4, consecutive_sl_hits=0, daily_pnl=0,
    ) is False
    assert rb.can_enter_trade(
        new_risk=4500, current_risk=0, capital=300_000,
        trades_today=1, consecutive_sl_hits=3, daily_pnl=0,
    ) is False
    assert rb.can_enter_trade(
        new_risk=4500, current_risk=0, capital=300_000,
        trades_today=1, consecutive_sl_hits=0, daily_pnl=-6500,
    ) is False


def test_risk_budget_survival_mode():
    rb = RiskBudget(
        max_capital_at_risk_today_pct=4.0,
        max_trades_today=4,
        max_per_trade_risk_pct=1.5,
        survival_mode=True,
    )
    assert rb.allows_directional() is False
    assert rb.allows_theta() is True


def test_risk_budget_full_stop():
    rb = RiskBudget(
        max_capital_at_risk_today_pct=4.0,
        max_trades_today=4,
        max_per_trade_risk_pct=1.5,
        survival_mode=False,
        pacing_status=PacingStatus.FULL_STOP,
    )
    assert rb.allows_directional() is False
    assert rb.allows_theta() is False


def test_playbook_serialization():
    pb = Playbook(
        date=date(2026, 3, 11),
        day_classification=DayClassification.LIKELY_TREND_UP,
        nifty_bias="bullish",
        nifty_setups=[],
        stock_plans=[],
        risk_budget=RiskBudget(
            max_capital_at_risk_today_pct=4.0,
            max_trades_today=4,
            max_per_trade_risk_pct=1.5,
            survival_mode=False,
        ),
        no_trade_conditions=["VIX > 22"],
        carry_rules=CarryRules(
            min_profit_pct=1.5, max_vix=20.0, min_dte=3,
            max_hedge_cost=500.0, never_carry=["expiry_day"],
        ),
    )
    d = pb.to_dict()
    pb2 = Playbook.from_dict(d)
    assert pb2.date == pb.date
    assert pb2.day_classification == pb.day_classification
