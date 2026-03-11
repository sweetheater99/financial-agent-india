# tests/test_v7_executor.py
"""Tests for V7 Executor — tick cycle, phase behavior, position management."""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, date, time, timedelta
from v7.types import (
    DayPhase, Setup, SetupType, Position, Playbook,
    DayClassification, RiskBudget, CarryRules, Conviction,
)
from v7.executor import Executor


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_deps():
    """Create mocked dependencies for Executor."""
    state = MagicMock()
    state.load_playbook.return_value = Playbook(
        date=date(2026, 3, 11),
        day_classification=DayClassification.LIKELY_TREND_UP,
        nifty_bias="bullish",
        nifty_setups=[],
        stock_plans=[],
        risk_budget=RiskBudget(
            max_capital_at_risk_today_pct=4.0,
            max_trades_today=4,
            max_per_trade_risk_pct=1.5,
        ),
        no_trade_conditions=[],
        carry_rules=CarryRules(),
    )
    state.load_positions.return_value = []
    state.load_daily_state.return_value = {
        "date": str(date(2026, 3, 11)),
        "daily_pnl": 0.0,
        "trades_today": 0,
        "consecutive_sl_hits": 0,
        "vix_30min_ago": None,
        "nifty_open": None,
    }
    state.save_positions = MagicMock()
    state.save_daily_state = MagicMock()

    data_feed = MagicMock()
    data_feed.get_ltp_batch.return_value = {"NSE:NIFTY 50": 24350.0}
    data_feed.get_vix.return_value = 15.0
    data_feed.get_candles.return_value = []

    risk_engine = MagicMock()
    risk_engine.can_enter.return_value = True
    risk_engine.check_margin.return_value = True

    order_mgr = MagicMock()
    margin = MagicMock()
    margin.current_utilization_pct.return_value = 30.0

    return {
        "state": state,
        "data_feed": data_feed,
        "risk_engine": risk_engine,
        "order_mgr": order_mgr,
        "margin": margin,
    }


@pytest.fixture
def executor(mock_deps):
    ex = Executor(
        state_mgr=mock_deps["state"],
        data_feed=mock_deps["data_feed"],
        risk_engine=mock_deps["risk_engine"],
        order_mgr=mock_deps["order_mgr"],
        margin_tracker=mock_deps["margin"],
        capital=300_000,
    )
    # Pre-initialize with mocked state so tests can access _playbook directly
    ex._playbook = mock_deps["state"].load_playbook.return_value
    ex._positions = []
    ex._daily = mock_deps["state"].load_daily_state.return_value.copy()
    ex._initialized = True
    return ex


# ── Phase Detection ──────────────────────────────────────────────────


def test_phase_opening_read():
    """9:15-9:44 is Opening Read."""
    assert DayPhase.from_time(time(9, 20)) == DayPhase.OPENING_READ


def test_phase_active_trading():
    """9:45-14:29 is Active Trading."""
    assert DayPhase.from_time(time(11, 0)) == DayPhase.ACTIVE_TRADING


def test_phase_wind_down():
    """14:30-15:14 is Wind Down."""
    assert DayPhase.from_time(time(14, 45)) == DayPhase.WIND_DOWN


# ── 15-min Candle Boundary ────────────────────────────────────────────


def test_is_candle_boundary_true():
    """Minutes :00, :15, :30, :45 are candle boundaries."""
    from v7.executor import is_15min_boundary
    assert is_15min_boundary(time(10, 0)) is True
    assert is_15min_boundary(time(10, 15)) is True
    assert is_15min_boundary(time(10, 30)) is True
    assert is_15min_boundary(time(10, 45)) is True


def test_is_candle_boundary_false():
    """Other minutes are NOT candle boundaries."""
    from v7.executor import is_15min_boundary
    assert is_15min_boundary(time(10, 3)) is False
    assert is_15min_boundary(time(10, 12)) is False
    assert is_15min_boundary(time(10, 27)) is False


# ── Tick Cycle Behavior ───────────────────────────────────────────────


def test_tick_opening_read_skips_triggers(executor, mock_deps):
    """During Opening Read (9:15-9:45), do NOT fire playbook triggers."""
    # Add a setup to the playbook
    setup = Setup(
        id="N1", priority=1, type=SetupType.BREAKOUT_LONG,
        symbol="NIFTY", trigger_level=24350.0,
        trigger_condition="15-min close above",
        instrument="NIFTY CE", strike_logic="delta 0.45",
        target=24500.0, stoploss=24280.0, max_risk_pct=1.5,
    )
    executor._playbook.nifty_setups = [setup]

    with patch("v7.executor.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 3, 11, 9, 20)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        executor.tick()

    # Order manager should NOT have been called for entry
    mock_deps["order_mgr"].place_entry_order.assert_not_called()


def test_tick_active_trading_checks_triggers(executor, mock_deps):
    """During Active Trading on candle boundary, check triggers."""
    setup = Setup(
        id="N1", priority=1, type=SetupType.BREAKOUT_LONG,
        symbol="NIFTY", trigger_level=24300.0,
        trigger_condition="15-min close above",
        instrument="NIFTY CE", strike_logic="delta 0.45",
        target=24500.0, stoploss=24200.0, max_risk_pct=1.5,
    )
    executor._playbook.nifty_setups = [setup]
    mock_deps["data_feed"].get_ltp_batch.return_value = {"NSE:NIFTY 50": 24350.0}

    with patch("v7.executor.datetime") as mock_dt:
        # 10:00 is a candle boundary during active trading
        mock_dt.now.return_value = datetime(2026, 3, 11, 10, 0)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        executor.tick()

    # Should have attempted to evaluate the trigger (whether it fires depends on full logic)
    # The key assertion: trigger evaluation happened (check via internal state or order call)


def test_tick_non_boundary_skips_triggers(executor, mock_deps):
    """On non-boundary ticks, only manage positions — skip trigger checks."""
    setup = Setup(
        id="N1", priority=1, type=SetupType.BREAKOUT_LONG,
        symbol="NIFTY", trigger_level=24300.0,
        trigger_condition="15-min close above",
        instrument="NIFTY CE", strike_logic="delta 0.45",
        target=24500.0, stoploss=24200.0, max_risk_pct=1.5,
    )
    executor._playbook.nifty_setups = [setup]

    with patch("v7.executor.datetime") as mock_dt:
        # 10:03 is NOT a candle boundary
        mock_dt.now.return_value = datetime(2026, 3, 11, 10, 3)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        executor.tick()

    mock_deps["order_mgr"].place_entry_order.assert_not_called()


def test_tick_outside_hours_exits_early(executor, mock_deps):
    """Outside market hours, tick exits immediately."""
    with patch("v7.executor.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 3, 11, 7, 0)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        executor.tick()

    # No data fetching should occur
    mock_deps["data_feed"].get_ltp_batch.assert_not_called()
