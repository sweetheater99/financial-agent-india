# tests/test_v7_health_integration.py
"""Integration tests for the full tick cycle with the health engine active.

Tests call _manage_positions(now) directly with mocked LTP, candles, and orders,
then verify health-based exit or hold behaviour end-to-end.
"""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, date, time

from v7.types import Position
from v7.executor import Executor


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_executor():
    """Build a fully-initialised Executor with all deps mocked."""
    data = MagicMock()
    orders = MagicMock()
    state = MagicMock()
    risk = MagicMock()
    margin = MagicMock()
    strategist = MagicMock()

    state.load_positions.return_value = []
    state.load_playbook.return_value = None
    state.load_daily_state.return_value = {}

    executor = Executor(
        state_mgr=state,
        data_feed=data,
        risk_engine=risk,
        order_mgr=orders,
        margin_tracker=margin,
        capital=300_000,
        strategist=strategist,
    )
    executor._initialized = True
    executor._capital = 300_000
    executor._vix = 15.0
    executor._quotes = {}
    executor._daily = {"closed_trades": []}
    return executor


def _falling_candles(n: int = 8, base_close: float = 1400.0) -> list:
    """Generate n candles with declining closes and steady volume."""
    candles = []
    for i in range(n):
        close = base_close - i * 2          # declining: 1400, 1398, 1396, …
        open_ = close + 1
        high = open_ + 1
        low = close - 1
        vol = 1000
        candles.append([i, open_, high, low, close, vol])
    return candles


def _rising_candles(n: int = 8, base_close: float = 23900.0) -> list:
    """Generate n candles with rising closes and increasing volume."""
    candles = []
    for i in range(n):
        close = base_close + i * 20         # rising: 23900, 23920, …
        open_ = close - 5
        high = close + 5
        low = open_ - 2
        vol = 1000 + i * 200                # increasing volume
        candles.append([i, open_, high, low, close, vol])
    return candles


# ── Test 1: Dying position → health exit ─────────────────────────────────


class TestDyingPositionGetsHealthExited:
    """Position that's been open 3 hours, underlying near SL, premium eroded,
    momentum against → should receive a health exit.

    Score breakdown with these inputs
    ----------------------------------
    pos.entry_price = 1420.0  (underlying scale — used by score_progress)
    pos.target      = 1460.0
    pos.stoploss    = 1380.0
    underlying LTP  = 1390.0  → below entry (no progress, -0.75 R)
    option LTP      = 10.0    → 37% of option entry (15.75) → score_premium ≈ 26
    candles         = 8 falling candles → score_momentum = 0, score_volume = 50 (neutral)
    sl_distance     = (1390-1380)/(1420-1380) = 0.25 → score_sl ≈ 25

    progress_ratio  = (1390-1420)/(1460-1420) = -30/40 = -0.75 → score_progress = 0
    composite ≈ 0×0.30 + 0×0.25 + 26×0.20 + 50×0.15 + 25×0.10 = 0+0+5.2+7.5+2.5 = 15.2 < 30
    """

    def test_dying_position_gets_health_exited(self):
        executor = _make_executor()

        # Position uses underlying-scale entry_price (1420) so score_progress is
        # computed correctly: underlying at 1390 is BELOW entry → progress_ratio < 0 → score = 0.
        # The original option entry premium (15.75) is patched into premium_health so
        # score_premium reflects 37% erosion (10.0 / 15.75 ≈ 0.635 → score ≈ 26),
        # without conflating option and underlying price scales.
        pos = Position(
            symbol="RELIANCE",
            instrument="RELIANCE26MAR1420CE",
            direction="bullish",
            entry_price=1420.0,    # underlying entry level for score_progress
            quantity=500,
            lot_size=250,
            allocated=7875.0,      # 500 × 15.75 original option premium
            stoploss=1380.0,
            target=1460.0,
            entry_date=date(2026, 3, 18),
            setup_id="S3",
            entry_time=time(10, 0),
        )
        # Patch premium_health to simulate option-premium erosion (15.75 → 10.0).
        pos.premium_health = lambda current_premium: current_premium / 15.75

        executor._positions = [pos]
        executor._daily = {"closed_trades": []}

        # Underlying LTP: 1390 (below entry 1420 → negative progress → score_progress = 0)
        # Option LTP: 10.0 (feeds into score_premium via patched premium_health)
        executor._quotes = {
            "NSE:RELIANCE": 1390.0,
            "NFO:RELIANCE26MAR1420CE": 10.0,
        }

        # 8 falling candles: declining closes, steady volume
        # → score_momentum = 0 (all moves against bullish), score_volume ≈ 50 (neutral)
        falling = _falling_candles(n=8, base_close=1400.0)
        executor._data.get_candles.return_value = falling

        # Exit order mock
        exit_result = MagicMock()
        exit_result.filled = True
        exit_result.fill_price = 10.0
        exit_result.order_id = "EXIT_HEALTH_001"
        executor._orders.place_exit_order.return_value = exit_result
        executor._orders.cancel_order = MagicMock()
        executor._state.append_trade = MagicMock()

        # Silence Telegram
        with patch("v7.telegram.TelegramAlerter", MagicMock()):
            # Current time: 1 PM (3 hours after 10 AM entry)
            now = datetime(2026, 3, 18, 13, 0)
            executor._manage_positions(now)

        # Position should be removed (health exit triggered)
        assert len(executor._positions) == 0, (
            f"Expected position to be removed by health exit, "
            f"but {len(executor._positions)} position(s) remain. "
            f"health_score={pos.health_score}"
        )

        # Confirm health score was indeed low (< 30 to trigger exit)
        assert pos.health_score < 30, (
            f"Expected health_score < 30 for a dying position, got {pos.health_score}"
        )

        # Confirm an exit order was placed
        executor._orders.place_exit_order.assert_called_once()

        # Confirm symbol added to cooldown
        assert "RELIANCE" in executor._daily.get("cooldown_symbols", []), (
            "Expected RELIANCE to be in cooldown_symbols after health exit"
        )


# ── Test 2: Healthy position → holds ─────────────────────────────────────


class TestHealthyPositionHolds:
    """Position progressing well after 1 hour: rising candles, premium up,
    underlying 62% toward target → health score > 60, position held."""

    def test_healthy_position_holds(self):
        executor = _make_executor()

        # Position setup: 1 hour old, progressing toward target
        pos = Position(
            symbol="NIFTY",
            instrument="NIFTY26MAR23000CE",
            direction="bullish",
            entry_price=73.5,
            quantity=65,
            lot_size=65,
            allocated=4777.5,
            stoploss=23800.0,
            target=24200.0,
            entry_date=date(2026, 3, 18),
            setup_id="N1",
            entry_time=time(10, 0),
        )

        executor._positions = [pos]
        executor._daily = {"closed_trades": []}

        # Underlying LTP: 24050 = 62% of the way from entry(~23900) to target(24200)
        # (If we assume the position was entered when spot was near 23900,
        #  which is above SL=23800 and heading toward 24200.)
        # Option LTP: 95.0 (premium UP from 73.5 → health = 95/73.5 ≈ 1.29 → premium score = 100)
        executor._quotes = {
            "NSE:NIFTY 50": 24050.0,           # index alias
            "NFO:NIFTY26MAR23000CE": 95.0,
        }

        # 8 rising candles with increasing volume
        rising = _rising_candles(n=8, base_close=23900.0)
        executor._data.get_candles.return_value = rising

        # Exit order mock (should NOT be called)
        exit_result = MagicMock()
        exit_result.filled = True
        exit_result.fill_price = 95.0
        exit_result.order_id = "EXIT_SHOULD_NOT_HAPPEN"
        executor._orders.place_exit_order.return_value = exit_result
        executor._orders.cancel_order = MagicMock()
        executor._state.append_trade = MagicMock()

        # Silence Telegram
        with patch("v7.telegram.TelegramAlerter", MagicMock()):
            # Current time: 11 AM (1 hour after 10 AM entry)
            now = datetime(2026, 3, 18, 11, 0)
            executor._manage_positions(now)

        # Position should still be held
        assert len(executor._positions) == 1, (
            f"Expected position to be held, but {len(executor._positions)} remain. "
            f"health_score={pos.health_score}"
        )

        # Health score should be > 60
        assert pos.health_score > 60, (
            f"Expected health_score > 60 for a healthy position, got {pos.health_score}"
        )

        # No exit order should have been placed
        executor._orders.place_exit_order.assert_not_called()
