# tests/test_v7_data_feed.py
"""Tests for V7 unified data feed.

These tests use mocks — actual API calls tested via integration tests.
"""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from v7.data_feed import DataFeed, DataFeedError, ProtectOnlyMode


def test_data_feed_init():
    feed = DataFeed(use_kite=False, use_angelone=False)
    assert feed.mode == "offline"


def test_data_feed_kite_mode():
    import sys
    mock_kite_module = MagicMock()
    mock_kite_module.get_kite.return_value = MagicMock()
    with patch.dict(sys.modules, {"kite_data": mock_kite_module}):
        feed = DataFeed(use_kite=True, use_angelone=False)
        assert feed.mode == "kite"


def test_data_feed_protect_only_on_kite_failure():
    import sys
    mock_kite_module = MagicMock()
    mock_kite_module.get_kite.side_effect = Exception("Token expired")
    mock_connect_module = MagicMock()
    mock_connect_module.get_session.return_value = MagicMock()
    with patch.dict(sys.modules, {"kite_data": mock_kite_module, "connect": mock_connect_module}):
        feed = DataFeed(use_kite=True, use_angelone=True)
        assert feed.mode == "protect_only"


def test_batch_ltp_returns_dict():
    feed = DataFeed(use_kite=False, use_angelone=False)
    with patch.object(feed, "_fetch_ltp_batch") as mock:
        mock.return_value = {"NIFTY": 24200.0, "BANKNIFTY": 52000.0}
        prices = feed.get_batch_ltp(["NIFTY", "BANKNIFTY"])
        assert prices["NIFTY"] == 24200.0


def test_cannot_trade_in_protect_only():
    feed = DataFeed(use_kite=False, use_angelone=False)
    feed.mode = "protect_only"
    assert feed.can_trade() is False


def test_can_trade_in_kite_mode():
    feed = DataFeed(use_kite=False, use_angelone=False)
    feed.mode = "kite"
    assert feed.can_trade() is True


def test_get_candles_raises_in_protect_only():
    feed = DataFeed(use_kite=False, use_angelone=False)
    feed.mode = "protect_only"
    with pytest.raises(ProtectOnlyMode):
        feed.get_candles("NIFTY", interval="5minute", days=1)


def test_get_option_chain_raises_in_protect_only():
    feed = DataFeed(use_kite=False, use_angelone=False)
    feed.mode = "protect_only"
    with pytest.raises(ProtectOnlyMode):
        feed.get_option_chain("NIFTY", "26MAR2026")
