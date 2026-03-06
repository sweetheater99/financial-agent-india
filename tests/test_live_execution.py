"""
Tests for the live execution bridge.

All tests mock the SmartAPI — no real API calls. LIVE_MODE is never set to True.
"""

import pytest
from unittest.mock import MagicMock, patch, call

import config
from live_execution import (
    LiveExecutor,
    _validate_order_value,
    _validate_price_deviation,
    _determine_exchange,
    _determine_product_type,
    _get_trading_symbol,
    _record_failure,
    _reset_failure_count,
    execute_live_entry,
    execute_live_exit,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_state():
    """Ensure LIVE_MODE is off and failure counter is reset for every test."""
    original_live_mode = config.LIVE_MODE
    config.LIVE_MODE = False
    _reset_failure_count()
    yield
    config.LIVE_MODE = original_live_mode
    _reset_failure_count()


@pytest.fixture
def mock_smart_api():
    api = MagicMock()
    api.placeOrder.return_value = "ORDER123"
    api.orderBook.return_value = {
        "data": [
            {
                "orderid": "ORDER123",
                "orderstatus": "complete",
                "averageprice": "100.50",
                "filledshares": "10",
                "tradingsymbol": "RELIANCE-EQ",
                "quantity": "10",
                "price": "100",
            }
        ]
    }
    api.holding.return_value = {"data": []}
    api.gttCreateRule.return_value = {"data": {"id": "GTT456"}}
    api.gttCancelRule.return_value = {"status": True}
    return api


@pytest.fixture
def eq_position():
    return {
        "symbol": "RELIANCE",
        "token": "2885",
        "direction": "bullish",
        "instrument": "EQ",
        "entry_price": 100.0,
        "ltp_at_entry": 100.0,
        "quantity": 10,
        "allocated": 1000.0,
        "stoploss_price": 95.0,
        "target_price": 110.0,
        "status": "open",
    }


@pytest.fixture
def opt_position():
    return {
        "symbol": "NIFTY",
        "token": "12345",
        "direction": "bearish",
        "instrument": "OPT",
        "trading_symbol": "NIFTY25MAR24000PE",
        "entry_price": 150.0,
        "ltp_at_entry": 150.0,
        "quantity": 75,
        "allocated": 11250.0,
        "stoploss_price": 90.0,
        "target_price": 225.0,
        "status": "open",
    }


# ---------------------------------------------------------------------------
# Test 1: Entry order params correct for equity buy
# ---------------------------------------------------------------------------

class TestEntryEquity:
    def test_entry_places_buy_order_with_correct_params(self, mock_smart_api, eq_position):
        executor = LiveExecutor(mock_smart_api, dry_run=False)
        result = executor.execute_entry(eq_position)

        mock_smart_api.placeOrder.assert_called_once()
        params = mock_smart_api.placeOrder.call_args[0][0]

        assert params["transactiontype"] == "BUY"
        assert params["tradingsymbol"] == "RELIANCE-EQ"
        assert params["symboltoken"] == "2885"
        assert params["exchange"] == "NSE"
        assert params["producttype"] == "DELIVERY"
        assert params["ordertype"] == "MARKET"
        assert params["quantity"] == 10

    def test_entry_returns_fill_info(self, mock_smart_api, eq_position):
        executor = LiveExecutor(mock_smart_api, dry_run=False)
        result = executor.execute_entry(eq_position)

        assert result["order_id"] == "ORDER123"
        assert result["status"] == "filled"
        assert result["fill_price"] == 100.50
        assert result["fill_qty"] == 10


# ---------------------------------------------------------------------------
# Test 2: Entry order params correct for option buy
# ---------------------------------------------------------------------------

class TestEntryOption:
    def test_option_entry_uses_nfo_exchange(self, mock_smart_api, opt_position):
        executor = LiveExecutor(mock_smart_api, dry_run=False)
        result = executor.execute_entry(opt_position)

        params = mock_smart_api.placeOrder.call_args[0][0]
        assert params["exchange"] == "NFO"
        assert params["producttype"] == "CARRYFORWARD"
        assert params["tradingsymbol"] == "NIFTY25MAR24000PE"
        assert params["quantity"] == 75


# ---------------------------------------------------------------------------
# Test 3: Exit order places sell
# ---------------------------------------------------------------------------

class TestExit:
    def test_exit_places_sell_order(self, mock_smart_api, eq_position):
        executor = LiveExecutor(mock_smart_api, dry_run=False)
        result = executor.execute_exit(eq_position, exit_price=110.0, reason="target")

        params = mock_smart_api.placeOrder.call_args[0][0]
        assert params["transactiontype"] == "SELL"
        assert params["tradingsymbol"] == "RELIANCE-EQ"
        assert params["quantity"] == 10

    def test_exit_returns_fill_info(self, mock_smart_api, eq_position):
        executor = LiveExecutor(mock_smart_api, dry_run=False)
        result = executor.execute_exit(eq_position, exit_price=110.0, reason="target")

        assert result["order_id"] == "ORDER123"
        assert result["status"] == "filled"
        assert result["fill_price"] == 100.50


# ---------------------------------------------------------------------------
# Test 4: Max order value cap blocks large orders
# ---------------------------------------------------------------------------

class TestOrderValueCap:
    def test_large_order_blocked(self, mock_smart_api, eq_position):
        eq_position["entry_price"] = 5000.0
        eq_position["quantity"] = 10  # 50,000 > 15,000 cap

        executor = LiveExecutor(mock_smart_api, dry_run=False)
        result = executor.execute_entry(eq_position)

        assert result["status"] == "blocked"
        assert "exceeds cap" in result["rejection_reason"]
        mock_smart_api.placeOrder.assert_not_called()

    def test_small_order_allowed(self, mock_smart_api, eq_position):
        # 100 * 10 = 1000, under 15000 cap
        executor = LiveExecutor(mock_smart_api, dry_run=False)
        result = executor.execute_entry(eq_position)

        assert result["status"] == "filled"
        mock_smart_api.placeOrder.assert_called_once()

    def test_validate_order_value_pass(self):
        ok, reason = _validate_order_value({"entry_price": 100, "quantity": 10})
        assert ok is True

    def test_validate_order_value_fail(self):
        ok, reason = _validate_order_value({"entry_price": 5000, "quantity": 10})
        assert ok is False


# ---------------------------------------------------------------------------
# Test 5: Price deviation check blocks bad prices
# ---------------------------------------------------------------------------

class TestPriceDeviation:
    def test_deviation_within_limit(self):
        ok, _ = _validate_price_deviation(100.0, 101.0)  # 1% deviation
        assert ok is True

    def test_deviation_exceeds_limit(self):
        ok, reason = _validate_price_deviation(100.0, 110.0)  # 10% deviation
        assert ok is False
        assert "deviation" in reason.lower()

    def test_entry_blocked_on_deviation(self, mock_smart_api, eq_position):
        eq_position["entry_price"] = 100.0
        eq_position["ltp_at_entry"] = 110.0  # 10% off

        executor = LiveExecutor(mock_smart_api, dry_run=False)
        result = executor.execute_entry(eq_position)

        assert result["status"] == "blocked"
        mock_smart_api.placeOrder.assert_not_called()


# ---------------------------------------------------------------------------
# Test 6: Kill switch triggers after 3 consecutive failures
# ---------------------------------------------------------------------------

class TestKillSwitch:
    def test_kill_switch_disables_live_mode(self, mock_smart_api, eq_position):
        mock_smart_api.placeOrder.side_effect = Exception("API error")
        config.LIVE_MODE = False  # will be set by kill switch check

        executor = LiveExecutor(mock_smart_api, dry_run=False)

        # First two failures: LIVE_MODE should still be whatever it was
        executor.execute_entry(eq_position)
        executor.execute_entry(eq_position)
        # Still False (was never True in test)

        # Third failure triggers kill switch
        result = executor.execute_entry(eq_position)
        assert config.LIVE_MODE is False  # kill switch sets it to False
        assert result.get("kill_switch") is True

    def test_success_resets_failure_count(self, mock_smart_api, eq_position):
        # Two failures
        mock_smart_api.placeOrder.side_effect = Exception("fail")
        executor = LiveExecutor(mock_smart_api, dry_run=False)
        executor.execute_entry(eq_position)
        executor.execute_entry(eq_position)

        # Success resets counter
        mock_smart_api.placeOrder.side_effect = None
        mock_smart_api.placeOrder.return_value = "ORDER123"
        executor.execute_entry(eq_position)

        # Two more failures should NOT trigger kill switch (counter reset)
        mock_smart_api.placeOrder.side_effect = Exception("fail")
        result1 = executor.execute_entry(eq_position)
        result2 = executor.execute_entry(eq_position)
        assert result1.get("kill_switch") is None or result1.get("kill_switch") is False
        assert result2.get("kill_switch") is None or result2.get("kill_switch") is False


# ---------------------------------------------------------------------------
# Test 7: Slippage calculation correct
# ---------------------------------------------------------------------------

class TestSlippage:
    def test_entry_slippage_calculation(self, mock_smart_api, eq_position):
        # Fill at 100.50, expected entry at 100.0 => slippage = 0.50
        executor = LiveExecutor(mock_smart_api, dry_run=False)
        result = executor.execute_entry(eq_position)

        assert result["slippage"] == 0.50

    def test_exit_slippage_calculation(self, mock_smart_api, eq_position):
        # Fill at 100.50, expected exit at 110.0 => slippage = -9.50
        executor = LiveExecutor(mock_smart_api, dry_run=False)
        result = executor.execute_exit(eq_position, exit_price=110.0, reason="target")

        assert result["slippage"] == round(100.50 - 110.0, 2)


# ---------------------------------------------------------------------------
# Test 8: Reconciliation detects paper-only position
# ---------------------------------------------------------------------------

class TestReconciliation:
    def test_paper_only_position_detected(self, mock_smart_api, eq_position):
        eq_position["live_order_id"] = "ORDER123"
        mock_smart_api.holding.return_value = {"data": []}  # no broker holdings

        executor = LiveExecutor(mock_smart_api, dry_run=False)
        portfolio = {"positions": [eq_position]}
        discrepancies = executor.reconcile_positions(portfolio)

        assert len(discrepancies) == 1
        assert "PAPER ONLY" in discrepancies[0]

    def test_broker_only_position_detected(self, mock_smart_api):
        mock_smart_api.holding.return_value = {
            "data": [{"tradingsymbol": "TCS-EQ", "quantity": "5"}]
        }

        executor = LiveExecutor(mock_smart_api, dry_run=False)
        portfolio = {"positions": []}
        discrepancies = executor.reconcile_positions(portfolio)

        assert len(discrepancies) == 1
        assert "BROKER ONLY" in discrepancies[0]

    def test_qty_mismatch_detected(self, mock_smart_api, eq_position):
        eq_position["live_order_id"] = "ORDER123"
        eq_position["quantity"] = 10
        mock_smart_api.holding.return_value = {
            "data": [{"tradingsymbol": "RELIANCE-EQ", "quantity": "5"}]
        }

        executor = LiveExecutor(mock_smart_api, dry_run=False)
        portfolio = {"positions": [eq_position]}
        discrepancies = executor.reconcile_positions(portfolio)

        assert len(discrepancies) == 1
        assert "QTY MISMATCH" in discrepancies[0]

    def test_no_discrepancies_when_matching(self, mock_smart_api, eq_position):
        eq_position["live_order_id"] = "ORDER123"
        eq_position["quantity"] = 10
        mock_smart_api.holding.return_value = {
            "data": [{"tradingsymbol": "RELIANCE-EQ", "quantity": "10"}]
        }

        executor = LiveExecutor(mock_smart_api, dry_run=False)
        portfolio = {"positions": [eq_position]}
        discrepancies = executor.reconcile_positions(portfolio)

        assert len(discrepancies) == 0


# ---------------------------------------------------------------------------
# Test 9: Dry run mode doesn't call placeOrder
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_entry(self, mock_smart_api, eq_position):
        executor = LiveExecutor(mock_smart_api, dry_run=True)
        result = executor.execute_entry(eq_position)

        assert result["status"] == "dry_run"
        mock_smart_api.placeOrder.assert_not_called()

    def test_dry_run_exit(self, mock_smart_api, eq_position):
        executor = LiveExecutor(mock_smart_api, dry_run=True)
        result = executor.execute_exit(eq_position, exit_price=110.0, reason="target")

        assert result["status"] == "dry_run"
        mock_smart_api.placeOrder.assert_not_called()

    def test_dry_run_gtt(self, mock_smart_api, eq_position):
        executor = LiveExecutor(mock_smart_api, dry_run=True)
        result = executor.place_stoploss_gtt(eq_position)

        assert result["status"] == "dry_run"
        mock_smart_api.gttCreateRule.assert_not_called()

    def test_execute_live_entry_noop_when_mode_off(self, mock_smart_api, eq_position):
        """execute_live_entry returns None when LIVE_MODE is False."""
        result = execute_live_entry(mock_smart_api, eq_position)
        assert result is None
        mock_smart_api.placeOrder.assert_not_called()


# ---------------------------------------------------------------------------
# Test 10: GTT stoploss fallback to regular SL
# ---------------------------------------------------------------------------

class TestGTTStoploss:
    def test_gtt_placed_successfully(self, mock_smart_api, eq_position):
        executor = LiveExecutor(mock_smart_api, dry_run=False)
        result = executor.place_stoploss_gtt(eq_position)

        assert result["status"] == "placed"
        assert result["type"] == "gtt"
        assert result["order_id"] == "GTT456"
        mock_smart_api.gttCreateRule.assert_called_once()

    def test_gtt_fallback_to_regular_sl(self, mock_smart_api, eq_position):
        mock_smart_api.gttCreateRule.side_effect = Exception("GTT not supported")
        # placeOrder is the fallback for regular SL
        mock_smart_api.placeOrder.return_value = "SL789"

        executor = LiveExecutor(mock_smart_api, dry_run=False)
        result = executor.place_stoploss_gtt(eq_position)

        assert result["status"] == "placed"
        assert result["type"] == "regular_sl"
        # placeOrder called for fallback SL
        assert mock_smart_api.placeOrder.called

    def test_update_stoploss_cancels_old(self, mock_smart_api, eq_position):
        eq_position["live_sl_order_id"] = "GTT_OLD"
        eq_position["live_sl_type"] = "gtt"

        executor = LiveExecutor(mock_smart_api, dry_run=False)
        result = executor.update_stoploss(eq_position, new_sl_price=97.0)

        mock_smart_api.gttCancelRule.assert_called_once_with({"id": "GTT_OLD"})
        assert result["status"] == "placed"


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_determine_exchange_eq(self):
        assert _determine_exchange({"instrument": "EQ"}) == "NSE"

    def test_determine_exchange_opt(self):
        assert _determine_exchange({"instrument": "OPT"}) == "NFO"

    def test_determine_exchange_spread(self):
        assert _determine_exchange({"instrument": "SPREAD"}) == "NFO"

    def test_determine_product_type_eq(self):
        assert _determine_product_type({"instrument": "EQ"}) == "DELIVERY"

    def test_determine_product_type_opt(self):
        assert _determine_product_type({"instrument": "OPT"}) == "CARRYFORWARD"

    def test_get_trading_symbol_eq(self):
        assert _get_trading_symbol({"instrument": "EQ", "symbol": "RELIANCE"}) == "RELIANCE-EQ"

    def test_get_trading_symbol_eq_already_suffixed(self):
        assert _get_trading_symbol({"instrument": "EQ", "symbol": "RELIANCE-EQ"}) == "RELIANCE-EQ"

    def test_get_trading_symbol_opt(self):
        pos = {"instrument": "OPT", "trading_symbol": "NIFTY25MAR24000PE"}
        assert _get_trading_symbol(pos) == "NIFTY25MAR24000PE"


# ---------------------------------------------------------------------------
# Test check_fills
# ---------------------------------------------------------------------------

class TestCheckFills:
    def test_filled_order(self, mock_smart_api):
        executor = LiveExecutor(mock_smart_api, dry_run=False)
        result = executor.check_fills("ORDER123")

        assert result["status"] == "filled"
        assert result["fill_price"] == 100.50
        assert result["fill_qty"] == 10

    def test_rejected_order(self, mock_smart_api):
        mock_smart_api.orderBook.return_value = {
            "data": [
                {
                    "orderid": "ORDER999",
                    "orderstatus": "rejected",
                    "text": "Insufficient margin",
                }
            ]
        }
        executor = LiveExecutor(mock_smart_api, dry_run=False)
        result = executor.check_fills("ORDER999")

        assert result["status"] == "rejected"
        assert result["rejection_reason"] == "Insufficient margin"

    def test_order_not_found(self, mock_smart_api):
        executor = LiveExecutor(mock_smart_api, dry_run=False)
        result = executor.check_fills("NONEXISTENT")

        assert result["status"] == "not_found"

    def test_empty_order_book(self, mock_smart_api):
        mock_smart_api.orderBook.return_value = {"data": None}
        executor = LiveExecutor(mock_smart_api, dry_run=False)
        result = executor.check_fills("ORDER123")

        assert result["status"] == "unknown"


# ---------------------------------------------------------------------------
# Test order book
# ---------------------------------------------------------------------------

class TestOrderBook:
    def test_get_order_book(self, mock_smart_api):
        executor = LiveExecutor(mock_smart_api, dry_run=False)
        orders = executor.get_order_book()

        assert len(orders) == 1
        assert orders[0]["orderid"] == "ORDER123"

    def test_get_order_book_empty(self, mock_smart_api):
        mock_smart_api.orderBook.return_value = {"data": None}
        executor = LiveExecutor(mock_smart_api, dry_run=False)
        orders = executor.get_order_book()

        assert orders == []
