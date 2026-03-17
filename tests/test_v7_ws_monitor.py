"""Tests for V7 WebSocket monitor."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from v7.ws_monitor import WSMonitor


@pytest.fixture
def monitor(tmp_path):
    """Create a WSMonitor with temp data dir."""
    m = WSMonitor(data_dir=tmp_path, paper=True)
    return m


@pytest.fixture
def positions_file(tmp_path):
    """Write sample positions to temp dir."""
    positions = [
        {
            "symbol": "NIFTY",
            "instrument": "FUT",
            "tradingsymbol": "NIFTY25MARFUT",
            "exchange": "NFO",
            "direction": "bullish",
            "entry_price": 24000,
            "stoploss": 23850,
            "target": 24300,
            "setup_id": "S1",
        },
        {
            "symbol": "BANKNIFTY",
            "instrument": "FUT",
            "tradingsymbol": "BANKNIFTY25MARFUT",
            "exchange": "NFO",
            "direction": "bearish",
            "entry_price": 52000,
            "stoploss": 52300,
            "target": 51500,
            "setup_id": "S2",
        },
    ]
    (tmp_path / "positions.json").write_text(json.dumps(positions))
    return positions


class TestLoadPositions:
    def test_no_file(self, monitor):
        assert monitor._load_positions() == []

    def test_with_positions(self, monitor, positions_file):
        loaded = monitor._load_positions()
        assert len(loaded) == 2
        assert loaded[0]["symbol"] == "NIFTY"


class TestCheckSLTarget:
    def test_bullish_sl_hit(self, monitor):
        monitor._token_to_positions = {
            100: [{"symbol": "NIFTY", "direction": "bullish",
                   "stoploss": 23850, "target": 24300, "setup_id": "S1"}]
        }
        triggered = monitor._check_sl_target(100, 23840)
        assert len(triggered) == 1
        assert triggered[0]["type"] == "stoploss"
        assert triggered[0]["ltp"] == 23840

    def test_bullish_target_hit(self, monitor):
        monitor._token_to_positions = {
            100: [{"symbol": "NIFTY", "direction": "bullish",
                   "stoploss": 23850, "target": 24300, "setup_id": "S1"}]
        }
        triggered = monitor._check_sl_target(100, 24300)
        assert len(triggered) == 1
        assert triggered[0]["type"] == "target"

    def test_bearish_sl_hit(self, monitor):
        monitor._token_to_positions = {
            200: [{"symbol": "BANKNIFTY", "direction": "bearish",
                   "stoploss": 52300, "target": 51500, "setup_id": "S2"}]
        }
        triggered = monitor._check_sl_target(200, 52350)
        assert len(triggered) == 1
        assert triggered[0]["type"] == "stoploss"

    def test_bearish_target_hit(self, monitor):
        monitor._token_to_positions = {
            200: [{"symbol": "BANKNIFTY", "direction": "bearish",
                   "stoploss": 52300, "target": 51500, "setup_id": "S2"}]
        }
        triggered = monitor._check_sl_target(200, 51400)
        assert len(triggered) == 1
        assert triggered[0]["type"] == "target"

    def test_no_trigger_in_range(self, monitor):
        monitor._token_to_positions = {
            100: [{"symbol": "NIFTY", "direction": "bullish",
                   "stoploss": 23850, "target": 24300, "setup_id": "S1"}]
        }
        triggered = monitor._check_sl_target(100, 24100)
        assert len(triggered) == 0

    def test_no_double_trigger(self, monitor):
        monitor._token_to_positions = {
            100: [{"symbol": "NIFTY", "direction": "bullish",
                   "stoploss": 23850, "target": 24300, "setup_id": "S1"}]
        }
        # First tick triggers
        triggered1 = monitor._check_sl_target(100, 23840)
        assert len(triggered1) == 1

        # Second tick should NOT trigger again
        triggered2 = monitor._check_sl_target(100, 23830)
        assert len(triggered2) == 0

    def test_unknown_token(self, monitor):
        monitor._token_to_positions = {}
        triggered = monitor._check_sl_target(999, 24000)
        assert len(triggered) == 0


class TestDailyStateUpdate:
    def test_sl_hit_increments_counter(self, monitor):
        event = {
            "symbol": "NIFTY", "type": "stoploss",
            "ltp": 23840, "sl": 23850, "target": 24300,
            "direction": "bullish", "setup_id": "S1",
            "timestamp": "2025-01-15T10:30:00",
        }
        monitor._update_daily_state_on_trigger(event)
        daily = monitor._load_daily_state()
        assert daily["sl_hits_today"] == 1
        assert len(daily["ws_triggers"]) == 1

    def test_target_hit_no_sl_increment(self, monitor):
        event = {
            "symbol": "NIFTY", "type": "target",
            "ltp": 24300, "sl": 23850, "target": 24300,
            "direction": "bullish", "setup_id": "S1",
            "timestamp": "2025-01-15T11:00:00",
        }
        monitor._update_daily_state_on_trigger(event)
        daily = monitor._load_daily_state()
        assert daily.get("sl_hits_today", 0) == 0
        assert len(daily["ws_triggers"]) == 1


class TestWSState:
    def test_save_and_load(self, monitor):
        monitor._save_ws_state({"active": True, "subscribed_tokens": 3})
        path = monitor.data_dir / "ws_state.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["active"] is True
        assert data["subscribed_tokens"] == 3
