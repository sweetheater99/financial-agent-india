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


# ── New Position field tests ────────────────────────────────────────────


def _make_position(**kwargs) -> Position:
    defaults = dict(
        symbol="NIFTY", instrument="NIFTY CE 24400",
        direction="bullish", entry_price=100.0,
        quantity=75, lot_size=75, allocated=7500.0,
        stoploss=80.0, target=150.0,
        entry_date=date(2026, 3, 11), setup_id="N1",
    )
    defaults.update(kwargs)
    return Position(**defaults)


def test_position_entry_time():
    """Position accepts an entry_time field."""
    p = _make_position(entry_time=time(9, 30))
    assert p.entry_time == time(9, 30)


def test_position_entry_time_defaults_none():
    """entry_time defaults to None when not provided."""
    p = _make_position()
    assert p.entry_time is None


def test_position_age_minutes():
    """age_minutes returns elapsed minutes since entry."""
    p = _make_position(entry_time=time(9, 30))
    assert p.age_minutes(time(10, 0)) == 30
    assert p.age_minutes(time(9, 30)) == 0


def test_position_age_minutes_no_entry_time():
    """age_minutes returns 0 when entry_time is None."""
    p = _make_position()
    assert p.age_minutes(time(10, 0)) == 0


def test_position_age_minutes_negative_clamped():
    """age_minutes never returns negative values."""
    p = _make_position(entry_time=time(10, 0))
    assert p.age_minutes(time(9, 30)) == 0


def test_position_premium_health():
    """premium_health returns ratio of current to entry premium."""
    p = _make_position(entry_price=100.0)
    assert p.premium_health(80.0) == pytest.approx(0.8)
    assert p.premium_health(100.0) == pytest.approx(1.0)
    assert p.premium_health(150.0) == pytest.approx(1.5)


def test_position_premium_health_zero_entry():
    """premium_health returns 1.0 when entry_price is zero."""
    p = _make_position(entry_price=0.0, allocated=0.0, stoploss=0.0, target=0.0)
    assert p.premium_health(50.0) == 1.0


def test_position_initial_quantity_auto_set():
    """initial_quantity is set from quantity when not provided."""
    p = _make_position(quantity=75)
    assert p.initial_quantity == 75


def test_position_initial_quantity_explicit():
    """initial_quantity can be explicitly set."""
    p = _make_position(quantity=75, initial_quantity=150)
    assert p.initial_quantity == 150


def test_position_partial_exit_done_defaults_false():
    """partial_exit_done defaults to False."""
    p = _make_position()
    assert p.partial_exit_done is False


def test_position_health_score_defaults_100():
    """health_score defaults to 100.0."""
    p = _make_position()
    assert p.health_score == 100.0


def test_position_to_dict_new_fields():
    """New fields are included in to_dict output."""
    p = _make_position(
        entry_time=time(9, 30),
        initial_quantity=150,
        partial_exit_done=True,
        health_score=75.0,
    )
    d = p.to_dict()
    assert d["entry_time"] == "09:30:00"
    assert d["initial_quantity"] == 150
    assert d["partial_exit_done"] is True
    assert d["health_score"] == 75.0


def test_position_to_dict_entry_time_none():
    """entry_time serializes as None when not set."""
    p = _make_position()
    d = p.to_dict()
    assert d["entry_time"] is None


def test_position_from_dict_new_fields():
    """New fields deserialize correctly via from_dict."""
    d = {
        "symbol": "NIFTY", "instrument": "NIFTY CE 24400",
        "direction": "bullish", "entry_price": 100.0,
        "quantity": 75, "lot_size": 75, "allocated": 7500.0,
        "stoploss": 80.0, "target": 150.0,
        "entry_date": "2026-03-11", "setup_id": "N1",
        "peak_price": 100.0,
        "entry_time": "09:30:00",
        "initial_quantity": 150,
        "partial_exit_done": True,
        "health_score": 75.0,
    }
    p = Position.from_dict(d)
    assert p.entry_time == time(9, 30)
    assert p.initial_quantity == 150
    assert p.partial_exit_done is True
    assert p.health_score == 75.0


def test_position_from_dict_missing_new_fields():
    """from_dict handles missing new fields with safe defaults."""
    d = {
        "symbol": "NIFTY", "instrument": "NIFTY CE 24400",
        "direction": "bullish", "entry_price": 100.0,
        "quantity": 75, "lot_size": 75, "allocated": 7500.0,
        "stoploss": 80.0, "target": 150.0,
        "entry_date": "2026-03-11", "setup_id": "N1",
    }
    p = Position.from_dict(d)
    assert p.entry_time is None
    assert p.initial_quantity == 75  # falls back to quantity
    assert p.partial_exit_done is False
    assert p.health_score == 100.0


def test_position_roundtrip_with_new_fields():
    """Full roundtrip: new fields survive to_dict -> from_dict."""
    p = _make_position(
        entry_time=time(10, 15),
        initial_quantity=150,
        partial_exit_done=True,
        health_score=60.0,
    )
    p2 = Position.from_dict(p.to_dict())
    assert p2.entry_time == p.entry_time
    assert p2.initial_quantity == p.initial_quantity
    assert p2.partial_exit_done == p.partial_exit_done
    assert p2.health_score == p.health_score
