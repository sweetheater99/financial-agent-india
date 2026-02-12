"""Tests for tools.py handler functions."""

from unittest.mock import MagicMock

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import handle_place_order


# ---------------------------------------------------------------------------
# handle_place_order validation
# ---------------------------------------------------------------------------

def _base_params(**overrides):
    params = {
        "symbol": "RELIANCE-EQ",
        "token": "2885",
        "transaction_type": "BUY",
        "quantity": 10,
        "order_type": "MARKET",
        "price": 0,
        "exchange": "NSE",
        "product_type": "DELIVERY",
    }
    params.update(overrides)
    return params


def test_place_order_rejects_zero_qty():
    result = handle_place_order(MagicMock(), _base_params(quantity=0))
    assert result["status"] == "REJECTED"
    assert "quantity" in result["error"].lower() or "0" in result["error"]


def test_place_order_rejects_negative_qty():
    result = handle_place_order(MagicMock(), _base_params(quantity=-5))
    assert result["status"] == "REJECTED"


def test_place_order_rejects_limit_no_price():
    result = handle_place_order(MagicMock(), _base_params(order_type="LIMIT", price=0))
    assert result["status"] == "REJECTED"
    assert "price" in result["error"].lower()


def test_place_order_dry_run():
    result = handle_place_order(MagicMock(), _base_params(), dry_run=True)
    assert result["status"] == "DRY_RUN"
    assert "SIMULATED" in result["message"]


def test_place_order_validation_passes():
    mock_api = MagicMock()
    mock_api.placeOrder.return_value = "ORDER123"
    result = handle_place_order(mock_api, _base_params(), dry_run=False)
    assert result["status"] == "PLACED"
    assert result["order_id"] == "ORDER123"
    mock_api.placeOrder.assert_called_once()
