# tests/test_v7_integration.py
"""Integration test: core V7 infrastructure works together."""
import pytest
from datetime import date, time
from pathlib import Path
from v7.types import (
    Playbook, Setup, SetupType, Position, TradeResult,
    RiskBudget, CarryRules, DayClassification, DayPhase,
)
from v7.config_v7 import WATCHLIST, RISK_LIMITS, is_15min_boundary
from v7.state import StateManager
from v7.margin import MarginTracker


@pytest.fixture
def state(tmp_path):
    return StateManager(tmp_path / "v7")


def test_full_day_lifecycle(state):
    """Simulate a complete trading day through the core layer."""

    # 1. Create morning playbook
    playbook = Playbook(
        date=date(2026, 3, 11),
        day_classification=DayClassification.LIKELY_TREND_UP,
        nifty_bias="bullish",
        nifty_setups=[
            Setup(id="N1", priority=1, type=SetupType.BREAKOUT_LONG,
                  symbol="NIFTY", trigger_level=24350.0,
                  trigger_condition="15-min close above with volume",
                  instrument="NIFTY CE", strike_logic="delta 0.45",
                  target=24500.0, stoploss=24280.0, max_risk_pct=1.5),
        ],
        stock_plans=[
            Setup(id="H1", priority=2, type=SetupType.SUPPORT_BOUNCE,
                  symbol="HDFCBANK", trigger_level=1625.0,
                  trigger_condition="15-min close above 1625",
                  instrument="HDFCBANK CE", strike_logic="ATM",
                  target=1660.0, stoploss=1610.0, max_risk_pct=1.0),
        ],
        risk_budget=RiskBudget(max_capital_at_risk_today_pct=4.0,
                               max_trades_today=4, max_per_trade_risk_pct=1.5),
        no_trade_conditions=["VIX > 22"],
        carry_rules=CarryRules(),
    )

    # 2. Save and reload playbook
    state.save_playbook(playbook)
    loaded = state.load_playbook(today=date(2026, 3, 11))
    assert loaded is not None
    assert len(loaded.active_setups()) == 2

    # 3. Simulate a trade entry
    margin = MarginTracker(capital=300_000)
    position = Position(
        symbol="NIFTY", instrument="NIFTY CE 24400",
        direction="bullish", entry_price=80.0,
        quantity=75, lot_size=75, allocated=6000.0,
        stoploss=60.0, target=130.0,
        entry_date=date(2026, 3, 11), setup_id="N1",
    )
    margin.add_position("NIFTY CE 24400", margin=6000)
    assert margin.can_add(5000)  # can add another trade

    # 4. Save position
    state.save_positions([position])
    loaded_pos = state.load_positions()
    assert len(loaded_pos) == 1
    assert loaded_pos[0].unrealized_pnl(100.0) == 75 * 20  # 1500

    # 5. Simulate trade exit
    result = TradeResult(
        symbol="NIFTY", instrument="NIFTY CE 24400",
        direction="bullish", entry_price=80.0, exit_price=120.0,
        quantity=75, entry_date=date(2026, 3, 11),
        exit_date=date(2026, 3, 11), exit_reason="target",
        pnl=3000.0, pnl_pct=50.0, costs=80.0,
        setup_id="N1", setup_type=SetupType.BREAKOUT_LONG,
    )
    state.append_trade(result)
    state.save_positions([])  # no open positions
    margin.remove_position("NIFTY CE 24400")

    # 6. Verify final state
    assert len(state.load_positions()) == 0
    assert len(state.load_trade_history()) == 1
    assert margin.used_margin() == 0

    # 7. Update daily state
    daily = state.load_daily_state(today=date(2026, 3, 11))
    daily["trades_today"] = 1
    daily["daily_pnl"] = 3000.0
    state.save_daily_state(daily)

    reloaded = state.load_daily_state(today=date(2026, 3, 11))
    assert reloaded["trades_today"] == 1
    assert reloaded["daily_pnl"] == 3000.0


def test_phase_transitions():
    """Verify day phases are correct for key times."""
    assert DayPhase.from_time(time(8, 45)) == DayPhase.PRE_MARKET
    assert DayPhase.from_time(time(9, 15)) == DayPhase.OPENING_READ
    assert DayPhase.from_time(time(9, 45)) == DayPhase.ACTIVE_TRADING
    assert DayPhase.from_time(time(14, 30)) == DayPhase.WIND_DOWN
    assert DayPhase.from_time(time(15, 15)) == DayPhase.POST_CLOSE
    assert DayPhase.from_time(time(15, 30)) == DayPhase.POST_CLOSE


def test_risk_budget_concurrent_limit():
    """4% concurrent risk at 3L = 12K. Can't exceed."""
    rb = RiskBudget(max_capital_at_risk_today_pct=4.0)
    assert rb.can_allocate(4500, 0, 300_000)       # 4500 < 12000
    assert rb.can_allocate(4500, 4500, 300_000)     # 9000 < 12000
    assert rb.can_allocate(4500, 9000, 300_000) is False  # 13500 > 12000
