# tests/test_v7_executor_health.py
"""Tests for V7 Executor health features: partial exit, profit ratchet,
concentration guard, entry_time tracking."""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, date, time

from v7.types import (
    Position, Setup, SetupType, Playbook, DayClassification,
    RiskBudget, CarryRules, Conviction,
)
from v7.executor import Executor


# ── Helpers ──────────────────────────────────────────────────────────


def _make_executor():
    data = MagicMock()
    orders = MagicMock()
    state = MagicMock()
    strategist = MagicMock()
    state.load_positions.return_value = []
    state.load_playbook.return_value = None
    state.load_daily_state.return_value = {}
    risk = MagicMock()
    margin = MagicMock()
    margin.utilization_pct.return_value = 30.0
    executor = Executor(
        state_mgr=state, data_feed=data, risk_engine=risk,
        order_mgr=orders, margin_tracker=margin, capital=300_000,
        strategist=strategist,
    )
    executor._initialized = True
    executor._capital = 300_000
    executor._vix = 15.0
    executor._quotes = {}
    executor._daily = {"closed_trades": []}
    return executor


def _make_position(**kwargs) -> Position:
    defaults = dict(
        symbol="RELIANCE", instrument="RELIANCE26MAR1420CE",
        direction="bullish", entry_price=15.0, quantity=500,
        lot_size=250, allocated=7500.0,
        stoploss=1380.0, target=1460.0,
        entry_date=date(2026, 3, 18), setup_id="S3",
        entry_time=time(10, 0), initial_quantity=500,
    )
    defaults.update(kwargs)
    return Position(**defaults)


# ── TestPartialExit ──────────────────────────────────────────────────


class TestPartialExit:

    def test_partial_exit_at_1rr(self):
        """At 1:1 R:R, 50% of position is exited (returns 250 from 500)."""
        ex = _make_executor()
        pos = _make_position(
            entry_price=15.0, quantity=500, lot_size=250,
            stoploss=1380.0, target=1460.0, initial_quantity=500,
        )
        # entry_price on underlying implied by stoploss/target context:
        # risk = |entry_underlying - stoploss|. For bullish, entry ~ 1420, SL=1380 → risk=40
        # 1:1 R:R → underlying at 1420+40 = 1460 (target)
        # We set underlying_ltp = 1460 to hit 1:1
        underlying_ltp = 1460.0
        option_ltp = 20.0  # option premium (doesn't matter for trigger check)
        now = datetime(2026, 3, 18, 11, 0)

        # Mock exit order
        result_mock = MagicMock()
        result_mock.filled = True
        result_mock.fill_price = 20.0
        ex._orders.place_exit_order.return_value = result_mock

        qty_exited = ex._check_partial_exit(pos, underlying_ltp, option_ltp, now)
        assert qty_exited == 250  # 50% of 500

    def test_no_partial_if_already_done(self):
        """partial_exit_done=True -> returns 0."""
        ex = _make_executor()
        pos = _make_position(partial_exit_done=True)
        qty = ex._check_partial_exit(pos, 1460.0, 20.0, datetime(2026, 3, 18, 11, 0))
        assert qty == 0

    def test_no_partial_if_below_1rr(self):
        """Below 1:1 R:R -> returns 0."""
        ex = _make_executor()
        pos = _make_position()
        # underlying at 1410 — risk=40 from entry 1420, needs 1460 for 1:1
        qty = ex._check_partial_exit(pos, 1410.0, 20.0, datetime(2026, 3, 18, 11, 0))
        assert qty == 0


# ── TestProfitRatchet ────────────────────────────────────────────────


class TestProfitRatchet:

    def test_default_trail_when_no_profit(self):
        """Before breakeven -> 1.5 (default)."""
        ex = _make_executor()
        # entry_price=1420 (underlying entry), SL still at original 1380 (below entry)
        pos = _make_position(entry_price=1420.0, stoploss=1380.0)
        # underlying at 1400 — SL < entry_price → not at breakeven
        result = ex._get_trailing_multiplier(pos, 1400.0)
        assert result == 1.5

    def test_tighter_trail_at_1r(self):
        """After breakeven, 1R profit -> 1.2 (breakeven_to_1r band)."""
        ex = _make_executor()
        # Bullish: entry=1420, SL moved to breakeven (1420). Target=1460.
        # risk = |1420 - 1420| computed via target-SL = 40.
        # underlying=1460: profit = 1460-1420 = 40 = 1R → breakeven_to_1r
        pos = _make_position(entry_price=1420.0, stoploss=1420.0)
        result = ex._get_trailing_multiplier(pos, 1460.0)
        assert result == 1.2

    def test_very_tight_trail_above_2r(self):
        """After breakeven, 2R+ profit -> 0.5."""
        ex = _make_executor()
        # SL at breakeven (1420), target=1460, risk=40.
        # underlying=1500: profit=80 → 2R → above_2r
        pos = _make_position(entry_price=1420.0, stoploss=1420.0)
        result = ex._get_trailing_multiplier(pos, 1500.0)
        assert result == 0.5


# ── TestConcentrationGuard ───────────────────────────────────────────


class TestConcentrationGuard:

    def test_block_duplicate_symbol(self):
        """Holding RELIANCE -> _has_symbol_concentration('RELIANCE') is True."""
        ex = _make_executor()
        ex._positions = [_make_position(symbol="RELIANCE")]
        assert ex._has_symbol_concentration("RELIANCE") is True

    def test_allow_different_symbol(self):
        """Holding RELIANCE -> _has_symbol_concentration('SBIN') is False."""
        ex = _make_executor()
        ex._positions = [_make_position(symbol="RELIANCE")]
        assert ex._has_symbol_concentration("SBIN") is False

    def test_block_cooldown_symbol(self):
        """cooldown_symbols has RELIANCE -> _is_symbol_cooled_down('RELIANCE') is True."""
        ex = _make_executor()
        ex._daily = {"cooldown_symbols": ["RELIANCE"]}
        assert ex._is_symbol_cooled_down("RELIANCE") is True

    def test_allow_non_cooldown_symbol(self):
        """cooldown_symbols has RELIANCE -> _is_symbol_cooled_down('SBIN') is False."""
        ex = _make_executor()
        ex._daily = {"cooldown_symbols": ["RELIANCE"]}
        assert ex._is_symbol_cooled_down("SBIN") is False


# ── TestEntryTimeTracking ────────────────────────────────────────────


class TestEntryTimeTracking:

    def test_entry_sets_time(self):
        """After _enter_position, pos has entry_time matching now.time()."""
        ex = _make_executor()
        now = datetime(2026, 3, 18, 10, 30)

        # Create a setup
        setup = Setup(
            id="S1", priority=1, type=SetupType.BREAKOUT_LONG,
            symbol="RELIANCE", trigger_level=1420.0,
            trigger_condition="breakout above R1",
            instrument="auto", strike_logic="delta 0.45",
            target=1460.0, stoploss=1380.0, max_risk_pct=1.5,
        )

        # Mock data feed for option chain + strike selection
        chain_mock = [{"strike": 1420, "tradingsymbol": "RELIANCE26MAR1420CE",
                       "premium": 15.0, "delta": 0.45, "lot_size": 250}]
        ex._data.get_option_chain.return_value = chain_mock

        # Mock strike selector and telegram (both are locally imported)
        with patch("v7.strike_selector.select_directional_strike") as mock_strike, \
             patch("v7.telegram.TelegramAlerter", MagicMock()), \
             patch("v7.telegram.AlertLevel", MagicMock()):
            mock_strike.return_value = {
                "tradingsymbol": "RELIANCE26MAR1420CE",
                "premium": 15.0, "strike": 1420, "delta": 0.45,
                "lot_size": 250,
            }

            # Mock order result
            order_result = MagicMock()
            order_result.filled = True
            order_result.fill_price = 15.0
            ex._orders.place_entry_order.return_value = order_result

            ex._enter_position(setup, 1425.0, now)

        assert len(ex._positions) == 1
        pos = ex._positions[0]
        assert pos.entry_time == time(10, 30)
