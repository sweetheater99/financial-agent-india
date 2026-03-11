# tests/test_v7_plan3_integration.py
"""Integration tests for V7 Plan 3: Executor + ThetaEngine + OrderManager working together.

Covers:
- Expiry day behavior (tighter SL, no carry, 1PM cutoff)
- Position rolling conditions
- Theta engine integration with executor tick cycle
- Recovery on restart with carried positions
- Stale playbook protect-only mode
"""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, date, time, timedelta

from v7.types import (
    DayPhase, Setup, SetupType, Position, Playbook,
    DayClassification, RiskBudget, CarryRules, Conviction,
)
from v7.executor import Executor
from v7.order_manager import OrderResult, OrderSide
from v7.theta_engine import ThetaEngine, CondorPosition, CondorLeg


# ── Shared Fixtures ───────────────────────────────────────────────────


@pytest.fixture
def mock_deps():
    """Shared mocked dependencies for Executor."""
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
    order_mgr.place_entry_order.return_value = OrderResult(
        order_id="entry_1", filled=True, fill_price=100.5, fill_quantity=75,
    )
    order_mgr.place_exit_order.return_value = OrderResult(
        order_id="exit_1", filled=True, fill_price=99.5, fill_quantity=75,
    )
    order_mgr.place_sl_order.return_value = OrderResult(
        order_id="sl_1", filled=False,
    )
    order_mgr.is_sl_order_live.return_value = True

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
    ex._playbook = mock_deps["state"].load_playbook.return_value
    ex._positions = []
    ex._daily = mock_deps["state"].load_daily_state.return_value.copy()
    ex._initialized = True
    return ex


def _make_position(**kwargs) -> Position:
    defaults = dict(
        symbol="NIFTY",
        instrument="NIFTY2530624400CE",
        direction="bullish",
        entry_price=100.0,
        quantity=75,
        lot_size=75,
        allocated=7500.0,
        stoploss=80.0,
        target=150.0,
        entry_date=date(2026, 3, 12),
        setup_id="N1",
    )
    defaults.update(kwargs)
    return Position(**defaults)


# ── Expiry Day Behavior ───────────────────────────────────────────────


def test_expiry_day_no_new_positions_after_1pm(executor, mock_deps):
    """On Thursday after 1:00 PM, no new positions should be entered."""
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
        # Thursday 1:15 PM — after 1 PM cutoff
        mock_dt.now.return_value = datetime(2026, 3, 12, 13, 15)  # Thursday
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        executor.tick()

    mock_deps["order_mgr"].place_entry_order.assert_not_called()


def test_expiry_day_tighter_sl(executor, mock_deps):
    """On expiry day, SL distance should be halved."""
    pos = _make_position(entry_price=100.0, stoploss=80.0)
    # Normal SL distance = 20. On expiry, should be 10 → SL at 90.0
    adjusted_sl = executor._expiry_adjusted_sl(pos)
    assert adjusted_sl == 90.0


def test_expiry_day_tighter_sl_bearish(executor, mock_deps):
    """On expiry day, bearish position SL tightens upward."""
    pos = _make_position(direction="bearish", entry_price=100.0, stoploss=120.0)
    # Normal distance = 20. On expiry half = 10 → SL at 110.0
    adjusted_sl = executor._expiry_adjusted_sl(pos)
    assert adjusted_sl == 110.0


def test_expiry_day_no_carry(executor, mock_deps):
    """No overnight carry on expiry day (Thursday)."""
    pos = _make_position(entry_date=date(2026, 3, 12))
    carry_rules = CarryRules()

    # On non-expiry day passes (assuming profit ok and vix ok)
    # On Thursday should fail regardless
    with patch("v7.executor.date") as mock_date:
        mock_date.today.return_value = date(2026, 3, 12)  # Thursday
        result = executor._meets_carry_criteria(pos, 110.0, carry_rules)

    assert result is False


def test_expiry_day_is_expiry_day_method(executor):
    """_is_expiry_day returns True on Thursday, False otherwise."""
    with patch("v7.executor.date") as mock_date:
        mock_date.today.return_value = date(2026, 3, 12)  # Thursday
        assert executor._is_expiry_day() is True

    with patch("v7.executor.date") as mock_date:
        mock_date.today.return_value = date(2026, 3, 11)  # Wednesday
        assert executor._is_expiry_day() is False


def test_expiry_cutoff_after_1pm(executor):
    """_is_expiry_cutoff returns True when hour >= 13 on Thursday."""
    with patch("v7.executor.date") as mock_date:
        mock_date.today.return_value = date(2026, 3, 12)  # Thursday
        assert executor._is_expiry_cutoff(datetime(2026, 3, 12, 13, 0)) is True
        assert executor._is_expiry_cutoff(datetime(2026, 3, 12, 12, 59)) is False


def test_expiry_cutoff_not_active_on_non_expiry_day(executor):
    """_is_expiry_cutoff returns False on non-Thursday even after 1PM."""
    with patch("v7.executor.date") as mock_date:
        mock_date.today.return_value = date(2026, 3, 11)  # Wednesday
        assert executor._is_expiry_cutoff(datetime(2026, 3, 11, 14, 0)) is False


# ── Theta Engine Integration ──────────────────────────────────────────


def test_executor_calls_theta_tick(executor, mock_deps):
    """Executor should call theta_engine.tick() each tick during active hours."""
    theta_mock = MagicMock()
    executor._theta_engine = theta_mock

    with patch("v7.executor.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 3, 11, 11, 0)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        executor.tick()

    theta_mock.tick.assert_called_once()


def test_executor_theta_not_called_outside_hours(executor, mock_deps):
    """Theta engine tick should not be called outside market hours."""
    theta_mock = MagicMock()
    executor._theta_engine = theta_mock

    with patch("v7.executor.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 3, 11, 7, 0)  # pre-market
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        executor.tick()

    theta_mock.tick.assert_not_called()


def test_executor_set_theta_engine(executor):
    """set_theta_engine attaches the engine and tick() calls it."""
    theta_mock = MagicMock()
    executor.set_theta_engine(theta_mock)
    assert executor._theta_engine is theta_mock


def test_executor_theta_tick_called_in_wind_down(executor, mock_deps):
    """Theta engine should also tick during wind down phase."""
    theta_mock = MagicMock()
    executor._theta_engine = theta_mock
    # Ensure wind down doesn't crash: provide a non-Thursday playbook so carry rules work
    executor._playbook = None  # no playbook → wind down clears positions immediately

    with patch("v7.executor.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 3, 11, 14, 45)  # Wind Down
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        executor.tick()

    theta_mock.tick.assert_called_once()


# ── Recovery on Restart ───────────────────────────────────────────────


def test_initialize_verifies_sl_orders(mock_deps):
    """On restart, carried positions with dead SL orders should be logged."""
    pos = _make_position(
        entry_date=date(2026, 3, 10),
        carried=True,
        sl_order_id="sl_old_123",
    )
    mock_deps["state"].load_positions.return_value = [pos]
    mock_deps["order_mgr"].is_sl_order_live.return_value = False
    # Playbook is for today so it won't be cleared
    mock_deps["state"].load_playbook.return_value = Playbook(
        date=date.today(),
        day_classification=DayClassification.LIKELY_TREND_UP,
        nifty_bias="bullish",
        nifty_setups=[],
        stock_plans=[],
        risk_budget=RiskBudget(),
        no_trade_conditions=[],
        carry_rules=CarryRules(),
    )

    ex = Executor(
        state_mgr=mock_deps["state"],
        data_feed=mock_deps["data_feed"],
        risk_engine=mock_deps["risk_engine"],
        order_mgr=mock_deps["order_mgr"],
        margin_tracker=mock_deps["margin"],
        capital=300_000,
    )
    ex.initialize()

    # Position should still be present and sl_order_id preserved
    assert len(ex._positions) == 1
    assert ex._positions[0].sl_order_id == "sl_old_123"
    mock_deps["order_mgr"].is_sl_order_live.assert_called_once_with("sl_old_123")


def test_initialize_loads_positions_and_playbook(mock_deps):
    """initialize() loads playbook and positions from state."""
    pos1 = _make_position(setup_id="N1")
    pos2 = _make_position(setup_id="N2", instrument="NIFTY2530624500CE")
    mock_deps["state"].load_positions.return_value = [pos1, pos2]
    # Playbook for today
    mock_deps["state"].load_playbook.return_value = Playbook(
        date=date.today(),
        day_classification=DayClassification.LIKELY_TREND_UP,
        nifty_bias="bullish",
        nifty_setups=[],
        stock_plans=[],
        risk_budget=RiskBudget(),
        no_trade_conditions=[],
        carry_rules=CarryRules(),
    )

    ex = Executor(
        state_mgr=mock_deps["state"],
        data_feed=mock_deps["data_feed"],
        risk_engine=mock_deps["risk_engine"],
        order_mgr=mock_deps["order_mgr"],
        margin_tracker=mock_deps["margin"],
    )
    ex.initialize()

    assert ex._initialized is True
    assert len(ex._positions) == 2
    assert ex._playbook is not None


def test_stale_playbook_protect_only(mock_deps):
    """If playbook is from a previous day, clear it (protect-only mode)."""
    mock_deps["state"].load_playbook.return_value = Playbook(
        date=date(2026, 3, 10),  # yesterday relative to test
        day_classification=DayClassification.LIKELY_TREND_UP,
        nifty_bias="bullish",
        nifty_setups=[],
        stock_plans=[],
        risk_budget=RiskBudget(),
        no_trade_conditions=[],
        carry_rules=CarryRules(),
    )
    with patch("v7.executor.date") as mock_date:
        mock_date.today.return_value = date(2026, 3, 11)

        ex = Executor(
            state_mgr=mock_deps["state"],
            data_feed=mock_deps["data_feed"],
            risk_engine=mock_deps["risk_engine"],
            order_mgr=mock_deps["order_mgr"],
            margin_tracker=mock_deps["margin"],
        )
        ex.initialize()

    assert ex._playbook is None


# ── Position Rolling ──────────────────────────────────────────────────


def test_should_roll_conditions(executor, mock_deps):
    """Position eligible for rolling: 20-40% loss, DTE < 5, VIX normal."""
    executor._vix = 18.0
    pos = _make_position(entry_price=100.0, stoploss=80.0, entry_date=date(2026, 3, 9))
    # Current price: 70 → 30% loss, DTE=3 < 5
    assert executor._should_roll(pos, current_price=70.0, dte=3) is True


def test_should_not_roll_loss_too_small(executor, mock_deps):
    """Don't roll if loss < 20% (position still viable, let it recover)."""
    executor._vix = 18.0
    pos = _make_position(entry_price=100.0, stoploss=80.0, entry_date=date(2026, 3, 9))
    # Current price: 90 → 10% loss → too small
    assert executor._should_roll(pos, current_price=90.0, dte=3) is False


def test_should_not_roll_loss_too_large(executor, mock_deps):
    """Don't roll if loss > 50% (thesis is wrong — just exit)."""
    executor._vix = 18.0
    pos = _make_position(entry_price=100.0, stoploss=80.0, entry_date=date(2026, 3, 9))
    # Current price: 45 → 55% loss → too large
    assert executor._should_roll(pos, current_price=45.0, dte=3) is False


def test_should_not_roll_dte_high(executor, mock_deps):
    """Don't roll if DTE >= 5 (theta not accelerating yet — wait)."""
    executor._vix = 18.0
    pos = _make_position(entry_price=100.0, stoploss=80.0, entry_date=date(2026, 3, 9))
    # DTE=7 → too far from expiry
    assert executor._should_roll(pos, current_price=70.0, dte=7) is False


def test_should_not_roll_high_vix(executor, mock_deps):
    """Don't roll when VIX > 22 — next expiry premium will be too expensive."""
    executor._vix = 24.0  # high VIX
    pos = _make_position(entry_price=100.0, stoploss=80.0, entry_date=date(2026, 3, 9))
    assert executor._should_roll(pos, current_price=70.0, dte=3) is False


def test_should_roll_boundary_exactly_20pct_loss(executor, mock_deps):
    """At exactly 20% loss: eligible for rolling."""
    executor._vix = 18.0
    pos = _make_position(entry_price=100.0, stoploss=80.0, entry_date=date(2026, 3, 9))
    # Current: 80 → 20% loss, DTE=4
    assert executor._should_roll(pos, current_price=80.0, dte=4) is True


def test_should_roll_bearish_position(executor, mock_deps):
    """Rolling logic works for bearish positions (loss goes up in price)."""
    executor._vix = 18.0
    pos = _make_position(
        direction="bearish", entry_price=100.0, stoploss=120.0,
        entry_date=date(2026, 3, 9),
    )
    # Current: 130 → 30% loss for bearish, DTE=3
    assert executor._should_roll(pos, current_price=130.0, dte=3) is True


# ── Executor + ThetaEngine Full Integration ──────────────────────────


def _make_theta_deps():
    """Create mocked deps for ThetaEngine."""
    state = MagicMock()
    state.load_theta_state.return_value = None
    state.save_theta_state = MagicMock()

    data_feed = MagicMock()
    data_feed.get_vix.return_value = 16.0
    data_feed.get_ltp_batch.return_value = {"NSE:NIFTY 50": 24350.0}
    data_feed.get_option_chain.return_value = []

    order_mgr = MagicMock()
    order_mgr.place_exit_order.return_value = OrderResult(
        order_id="theta_exit_1", filled=True, fill_price=12.0, fill_quantity=75,
    )

    strike_selector = MagicMock()
    margin = MagicMock()
    margin.current_utilization_pct.return_value = 25.0

    return {
        "state": state,
        "data_feed": data_feed,
        "order_mgr": order_mgr,
        "strike_selector": strike_selector,
        "margin": margin,
    }


def test_executor_theta_shared_order_manager(mock_deps):
    """Executor and ThetaEngine should use the same order manager instance."""
    theta_deps = _make_theta_deps()

    theta = ThetaEngine(
        data_feed=theta_deps["data_feed"],
        order_mgr=mock_deps["order_mgr"],  # same order manager as executor
        state_mgr=theta_deps["state"],
        strike_selector=theta_deps["strike_selector"],
        margin_tracker=theta_deps["margin"],
        capital=300_000,
    )

    ex = Executor(
        state_mgr=mock_deps["state"],
        data_feed=mock_deps["data_feed"],
        risk_engine=mock_deps["risk_engine"],
        order_mgr=mock_deps["order_mgr"],
        margin_tracker=mock_deps["margin"],
        capital=300_000,
    )
    ex._playbook = mock_deps["state"].load_playbook.return_value
    ex._positions = []
    ex._daily = {}
    ex._initialized = True
    ex.set_theta_engine(theta)

    # Both should share the same underlying order_mgr
    assert ex._orders is theta._orders


def test_full_tick_with_theta_no_condor(executor, mock_deps):
    """Full tick during active trading when theta has no condor — no orders."""
    theta_deps = _make_theta_deps()
    theta_deps["data_feed"].get_vix.return_value = 25.0  # too high, won't enter

    theta = ThetaEngine(
        data_feed=theta_deps["data_feed"],
        order_mgr=theta_deps["order_mgr"],
        state_mgr=theta_deps["state"],
        strike_selector=theta_deps["strike_selector"],
        margin_tracker=theta_deps["margin"],
        capital=300_000,
    )
    executor.set_theta_engine(theta)

    with patch("v7.executor.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 3, 9, 11, 0)  # Monday active
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        executor.tick()

    # Theta engine tick was called — but no condor (VIX too high), no exit orders
    theta_deps["order_mgr"].place_exit_order.assert_not_called()


def test_full_tick_with_theta_has_condor_wednesday(executor, mock_deps):
    """On Wednesday, if theta has an open condor, it should close."""
    theta_deps = _make_theta_deps()

    theta = ThetaEngine(
        data_feed=theta_deps["data_feed"],
        order_mgr=theta_deps["order_mgr"],
        state_mgr=theta_deps["state"],
        strike_selector=theta_deps["strike_selector"],
        margin_tracker=theta_deps["margin"],
        capital=300_000,
    )

    # Inject an open condor
    theta._condor = CondorPosition(
        entry_date=date(2026, 3, 9),
        expiry_date=date(2026, 3, 12),
        short_ce=CondorLeg("NIFTY2531224700CE", 24700, 30.0, 75, "CE"),
        long_ce=CondorLeg("NIFTY2531224900CE", 24900, 10.0, 75, "CE"),
        short_pe=CondorLeg("NIFTY2531224000PE", 24000, 30.0, 75, "PE"),
        long_pe=CondorLeg("NIFTY2531223800PE", 23800, 10.0, 75, "PE"),
        net_credit=40.0,
    )
    # Make profit NOT at target so time management triggers close
    theta_deps["data_feed"].get_ltp_batch.return_value = {
        "NFO:NIFTY2531224700CE": 28.0,  # short CE still expensive
        "NFO:NIFTY2531224900CE": 9.0,
        "NFO:NIFTY2531224000PE": 27.0,  # short PE still expensive
        "NFO:NIFTY2531223800PE": 8.0,
    }

    executor.set_theta_engine(theta)

    with patch("v7.executor.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 3, 11, 11, 0)  # Wednesday active
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        executor.tick()

    # On Wednesday, theta should have closed the condor (4 legs closed)
    assert theta_deps["order_mgr"].place_exit_order.call_count == 4
    assert theta._condor is None


def test_executor_tick_expiry_thursday_no_entry_but_theta_still_ticks(executor, mock_deps):
    """On Thursday after 1PM, no new directional entries, but theta still ticks."""
    setup = Setup(
        id="N1", priority=1, type=SetupType.BREAKOUT_LONG,
        symbol="NIFTY", trigger_level=24300.0,
        trigger_condition="15-min close above",
        instrument="NIFTY CE", strike_logic="delta 0.45",
        target=24500.0, stoploss=24200.0, max_risk_pct=1.5,
    )
    executor._playbook.nifty_setups = [setup]
    mock_deps["data_feed"].get_ltp_batch.return_value = {"NSE:NIFTY 50": 24350.0}

    theta_mock = MagicMock()
    executor._theta_engine = theta_mock

    with patch("v7.executor.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 3, 12, 13, 30)  # Thursday 1:30PM
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        executor.tick()

    mock_deps["order_mgr"].place_entry_order.assert_not_called()
    theta_mock.tick.assert_called_once()
