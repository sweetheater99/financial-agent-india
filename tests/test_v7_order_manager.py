# tests/test_v7_order_manager.py
"""Tests for V7 Order Manager — order placement, SL orders, fill tracking."""
import pytest
from unittest.mock import MagicMock, patch, call
from datetime import datetime, date
from v7.order_manager import OrderManager, OrderResult, OrderType, OrderSide


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_kite():
    kite = MagicMock()
    kite.place_order = MagicMock(return_value="order_123")
    kite.order_history = MagicMock(return_value=[
        {"status": "COMPLETE", "average_price": 105.0, "filled_quantity": 75}
    ])
    kite.orders = MagicMock(return_value=[
        {"order_id": "order_123", "status": "COMPLETE", "average_price": 105.0}
    ])
    kite.cancel_order = MagicMock()
    kite.modify_order = MagicMock()
    return kite


@pytest.fixture
def order_mgr(mock_kite):
    with patch("v7.order_manager.get_kite", return_value=mock_kite):
        mgr = OrderManager(dry_run=False)
        mgr._kite = mock_kite
        return mgr


# ── Entry Orders ──────────────────────────────────────────────────────


def test_place_entry_limit_order(order_mgr, mock_kite):
    """Entry: limit order at bid+0.50 for buy."""
    result = order_mgr.place_entry_order(
        tradingsymbol="NIFTY2530624400CE",
        exchange="NFO",
        side=OrderSide.BUY,
        quantity=75,
        limit_price=104.50,
    )
    assert result.order_id == "order_123"
    mock_kite.place_order.assert_called_once()
    call_kwargs = mock_kite.place_order.call_args[1]
    assert call_kwargs["order_type"] == "LIMIT"
    assert call_kwargs["price"] == 104.50
    assert call_kwargs["quantity"] == 75


def test_entry_limit_timeout_converts_to_market(order_mgr, mock_kite):
    """If limit not filled in 30s, modify to market."""
    # First call: place limit. order_history returns OPEN (not filled).
    mock_kite.order_history.return_value = [
        {"status": "OPEN", "average_price": 0, "filled_quantity": 0}
    ]
    # After modify, order fills
    mock_kite.modify_order = MagicMock()

    result = order_mgr.place_entry_order(
        tradingsymbol="NIFTY2530624400CE",
        exchange="NFO",
        side=OrderSide.BUY,
        quantity=75,
        limit_price=104.50,
        timeout_seconds=0,  # immediate timeout for testing
    )
    # Should have tried to modify to market
    mock_kite.modify_order.assert_called_once()
    modify_kwargs = mock_kite.modify_order.call_args[1]
    assert modify_kwargs["order_type"] == "MARKET"


# ── Exit Orders ───────────────────────────────────────────────────────


def test_place_exit_limit_order(order_mgr, mock_kite):
    """Exit: limit order at ask-0.50 for sell."""
    result = order_mgr.place_exit_order(
        tradingsymbol="NIFTY2530624400CE",
        exchange="NFO",
        side=OrderSide.SELL,
        quantity=75,
        limit_price=149.50,
    )
    assert result.order_id == "order_123"
    call_kwargs = mock_kite.place_order.call_args[1]
    assert call_kwargs["order_type"] == "LIMIT"


def test_sl_exit_timeout_shorter(order_mgr, mock_kite):
    """SL exit: 15s timeout (not 30s), then market."""
    mock_kite.order_history.return_value = [
        {"status": "OPEN", "average_price": 0, "filled_quantity": 0}
    ]
    result = order_mgr.place_exit_order(
        tradingsymbol="NIFTY2530624400CE",
        exchange="NFO",
        side=OrderSide.SELL,
        quantity=75,
        limit_price=79.50,
        is_sl_exit=True,
        timeout_seconds=0,  # immediate for testing
    )
    mock_kite.modify_order.assert_called_once()


# ── Exchange SL Orders ────────────────────────────────────────────────


def test_place_exchange_sl_order(order_mgr, mock_kite):
    """Place SL order on exchange (for overnight carry)."""
    result = order_mgr.place_sl_order(
        tradingsymbol="NIFTY2530624400CE",
        exchange="NFO",
        side=OrderSide.SELL,
        quantity=75,
        trigger_price=80.0,
        limit_price=79.0,
    )
    assert result.order_id == "order_123"
    call_kwargs = mock_kite.place_order.call_args[1]
    assert call_kwargs["order_type"] == "SL"
    assert call_kwargs["trigger_price"] == 80.0
    assert call_kwargs["price"] == 79.0


def test_place_exchange_slm_order(order_mgr, mock_kite):
    """Place SL-M (market) order on exchange."""
    result = order_mgr.place_sl_order(
        tradingsymbol="NIFTY2530624400CE",
        exchange="NFO",
        side=OrderSide.SELL,
        quantity=75,
        trigger_price=80.0,
        limit_price=None,  # None = SL-M
    )
    call_kwargs = mock_kite.place_order.call_args[1]
    assert call_kwargs["order_type"] == "SL-M"
    assert call_kwargs["trigger_price"] == 80.0


# ── SL Order Management ──────────────────────────────────────────────


def test_update_sl_order(order_mgr, mock_kite):
    """Update SL: cancel old, place new."""
    new_result = order_mgr.update_sl_order(
        old_order_id="old_sl_123",
        tradingsymbol="NIFTY2530624400CE",
        exchange="NFO",
        side=OrderSide.SELL,
        quantity=75,
        new_trigger_price=90.0,
        new_limit_price=89.0,
    )
    # Should cancel old order first
    mock_kite.cancel_order.assert_called_once()
    # Then place new SL
    assert new_result.order_id == "order_123"


def test_check_sl_order_live(order_mgr, mock_kite):
    """Verify SL order is still active on exchange."""
    mock_kite.order_history.return_value = [
        {"status": "TRIGGER PENDING", "average_price": 0}
    ]
    assert order_mgr.is_sl_order_live("sl_order_123") is True


def test_check_sl_order_triggered(order_mgr, mock_kite):
    """SL order has been triggered (filled)."""
    mock_kite.order_history.return_value = [
        {"status": "COMPLETE", "average_price": 79.5, "filled_quantity": 75}
    ]
    assert order_mgr.is_sl_order_live("sl_order_123") is False


def test_check_sl_order_cancelled(order_mgr, mock_kite):
    """SL order was cancelled."""
    mock_kite.order_history.return_value = [
        {"status": "CANCELLED", "average_price": 0}
    ]
    assert order_mgr.is_sl_order_live("sl_order_123") is False


# ── Fill Verification ─────────────────────────────────────────────────


def test_get_fill_price(order_mgr, mock_kite):
    """Get actual fill price from order history."""
    mock_kite.order_history.return_value = [
        {"status": "COMPLETE", "average_price": 105.25, "filled_quantity": 75}
    ]
    fill = order_mgr.get_fill_info("order_123")
    assert fill["price"] == 105.25
    assert fill["quantity"] == 75
    assert fill["filled"] is True


def test_get_fill_not_filled(order_mgr, mock_kite):
    """Order not yet filled."""
    mock_kite.order_history.return_value = [
        {"status": "OPEN", "average_price": 0, "filled_quantity": 0}
    ]
    fill = order_mgr.get_fill_info("order_123")
    assert fill["filled"] is False


# ── Dry Run Mode ──────────────────────────────────────────────────────


def test_dry_run_no_kite_calls():
    """In dry_run mode, no actual Kite calls are made."""
    mgr = OrderManager(dry_run=True)
    result = mgr.place_entry_order(
        tradingsymbol="NIFTY2530624400CE",
        exchange="NFO",
        side=OrderSide.BUY,
        quantity=75,
        limit_price=104.50,
    )
    assert result.order_id.startswith("DRY_")
    assert result.filled is True  # simulated fill


def test_dry_run_sl_order():
    """Dry run SL order returns simulated ID."""
    mgr = OrderManager(dry_run=True)
    result = mgr.place_sl_order(
        tradingsymbol="NIFTY2530624400CE",
        exchange="NFO",
        side=OrderSide.SELL,
        quantity=75,
        trigger_price=80.0,
        limit_price=79.0,
    )
    assert result.order_id.startswith("DRY_SL_")
