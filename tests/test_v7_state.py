# tests/test_v7_state.py
"""Tests for V7 state persistence."""
import json
import pytest
from datetime import date, time
from pathlib import Path
from v7.state import StateManager
from v7.types import (
    Playbook, Setup, SetupType, Position, TradeResult,
    RiskBudget, CarryRules, DayClassification, Conviction,
)


@pytest.fixture
def tmp_state_dir(tmp_path):
    return tmp_path / "v7_state"


@pytest.fixture
def state(tmp_state_dir):
    return StateManager(tmp_state_dir)


@pytest.fixture
def sample_playbook():
    return Playbook(
        date=date(2026, 3, 11),
        day_classification=DayClassification.LIKELY_TREND_UP,
        nifty_bias="bullish",
        nifty_setups=[
            Setup(id="N1", priority=1, type=SetupType.BREAKOUT_LONG,
                  symbol="NIFTY", trigger_level=24350.0,
                  trigger_condition="15-min close above",
                  instrument="NIFTY CE", strike_logic="delta 0.45",
                  target=24500.0, stoploss=24280.0, max_risk_pct=1.5),
        ],
        stock_plans=[],
        risk_budget=RiskBudget(),
        no_trade_conditions=["VIX > 22"],
        carry_rules=CarryRules(),
    )


@pytest.fixture
def sample_position():
    return Position(
        symbol="NIFTY", instrument="NIFTY CE 24400",
        direction="bullish", entry_price=100.0,
        quantity=75, lot_size=75, allocated=7500.0,
        stoploss=80.0, target=150.0,
        entry_date=date(2026, 3, 11), setup_id="N1",
    )


def test_state_creates_directory(state, tmp_state_dir):
    assert tmp_state_dir.exists()


def test_save_and_load_playbook(state, sample_playbook):
    state.save_playbook(sample_playbook)
    loaded = state.load_playbook()
    assert loaded is not None
    assert loaded.date == sample_playbook.date
    assert loaded.day_classification == sample_playbook.day_classification
    assert len(loaded.nifty_setups) == 1


def test_load_playbook_returns_none_if_missing(state):
    assert state.load_playbook() is None


def test_load_playbook_returns_none_if_stale(state, sample_playbook):
    state.save_playbook(sample_playbook)
    loaded = state.load_playbook(today=date(2026, 3, 12))
    assert loaded is None


def test_save_and_load_positions(state, sample_position):
    state.save_positions([sample_position])
    loaded = state.load_positions()
    assert len(loaded) == 1
    assert loaded[0].symbol == "NIFTY"
    assert loaded[0].entry_price == 100.0


def test_load_positions_empty_if_missing(state):
    assert state.load_positions() == []


def test_save_and_load_daily_state(state):
    daily = {
        "date": "2026-03-11",
        "trades_today": 2,
        "sl_hits_today": 1,
        "daily_pnl": -1500.0,
        "current_risk": 4500.0,
    }
    state.save_daily_state(daily)
    loaded = state.load_daily_state()
    assert loaded["trades_today"] == 2
    assert loaded["daily_pnl"] == -1500.0


def test_daily_state_resets_on_new_day(state):
    daily = {"date": "2026-03-10", "trades_today": 3, "sl_hits_today": 2,
             "daily_pnl": -2000.0, "current_risk": 0.0}
    state.save_daily_state(daily)
    loaded = state.load_daily_state(today=date(2026, 3, 11))
    assert loaded["trades_today"] == 0


def test_save_and_load_trade_history(state):
    tr = TradeResult(
        symbol="NIFTY", instrument="NIFTY CE 24400",
        direction="bullish", entry_price=100.0, exit_price=150.0,
        quantity=75, entry_date=date(2026, 3, 11),
        exit_date=date(2026, 3, 11), exit_reason="target",
        pnl=3750.0, pnl_pct=50.0, costs=120.0,
        setup_id="N1", setup_type=SetupType.BREAKOUT_LONG,
    )
    state.append_trade(tr)
    history = state.load_trade_history()
    assert len(history) == 1
    assert history[0].pnl == 3750.0


def test_append_trade_preserves_existing(state):
    for i in range(3):
        tr = TradeResult(
            symbol=f"SYM{i}", instrument=f"OPT{i}",
            direction="bullish", entry_price=100.0, exit_price=110.0,
            quantity=75, entry_date=date(2026, 3, 11),
            exit_date=date(2026, 3, 11), exit_reason="target",
            pnl=750.0, pnl_pct=10.0, costs=40.0,
            setup_id=f"S{i}", setup_type=SetupType.BREAKOUT_LONG,
        )
        state.append_trade(tr)
    assert len(state.load_trade_history()) == 3


def test_save_and_load_level_memory(state):
    levels = {
        "NIFTY": {
            "levels": [
                {"price": 24000, "type": "support", "strength": 3,
                 "source": "tested 3x", "last_tested": "2026-03-10"},
            ],
            "oi_walls": {"call_max_oi_strike": 24500, "put_max_oi_strike": 24000},
        }
    }
    state.save_level_memory(levels)
    loaded = state.load_level_memory()
    assert "NIFTY" in loaded
    assert loaded["NIFTY"]["levels"][0]["strength"] == 3


def test_save_and_load_edge_tracker(state):
    edge = {
        "overall_win_rate": 0.55,
        "by_strategy": {
            "momentum_breakout": {"trades": 45, "win_rate": 0.58, "avg_rr": 1.8},
        },
        "by_instrument": {
            "NIFTY": {"trades": 35, "net_pnl": 12000},
        },
        "by_time": {
            "9:45-11:00": {"trades": 40, "win_rate": 0.60},
        },
    }
    state.save_edge_tracker(edge)
    loaded = state.load_edge_tracker()
    assert loaded["overall_win_rate"] == 0.55
    assert loaded["by_strategy"]["momentum_breakout"]["trades"] == 45


def test_save_and_load_monthly_state(state):
    monthly = {
        "month": "2026-03",
        "mtd_pnl": 5000.0,
        "mtd_pnl_pct": 1.67,
        "trades_this_month": 15,
        "survival_mode": False,
    }
    state.save_monthly_state(monthly)
    loaded = state.load_monthly_state()
    assert loaded["mtd_pnl"] == 5000.0
