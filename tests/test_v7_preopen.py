"""Tests for V7 pre-open session command."""
from unittest.mock import MagicMock, patch
import pytest

from v7.telegram import format_preopen_alert


class TestFormatPreopenAlert:
    def test_confirmed_gap_up(self):
        data = {
            "nifty_preopen_price": 24150,
            "prev_close": 24000,
            "actual_gap_pct": 0.63,
            "expected_gap_pct": 0.55,
            "gap_direction": "gap_up",
            "gap_confirmation": "confirmed",
        }
        msg = format_preopen_alert(data)
        assert "24,150" in msg
        assert "+0.63%" in msg
        assert "CONFIRMED" in msg
        assert "gap up" in msg

    def test_divergent_gap(self):
        data = {
            "nifty_preopen_price": 23950,
            "prev_close": 24000,
            "actual_gap_pct": -0.21,
            "expected_gap_pct": 0.40,
            "gap_direction": "flat",
            "gap_confirmation": "divergent",
        }
        msg = format_preopen_alert(data)
        assert "DIVERGENT" in msg
        assert "-0.21%" in msg

    def test_partial_confirmation(self):
        data = {
            "nifty_preopen_price": 24030,
            "prev_close": 24000,
            "actual_gap_pct": 0.13,
            "expected_gap_pct": 0.50,
            "gap_direction": "flat",
            "gap_confirmation": "partial",
        }
        msg = format_preopen_alert(data)
        assert "PARTIAL" in msg


class TestCmdPreopen:
    def _make_components(self, ltp=24150.0, prev_close=24000.0, gift_nifty_str="+0.35% gap at 24084"):
        """Build mock components dict."""
        data_feed = MagicMock()
        data_feed.can_trade.return_value = True
        data_feed.get_batch_ltp.return_value = {"NIFTY 50": ltp}
        data_feed.get_candles.return_value = [
            {"open": prev_close - 50, "high": prev_close + 20, "low": prev_close - 60, "close": prev_close},
            {"open": prev_close, "high": prev_close + 30, "low": prev_close - 10, "close": prev_close + 10},
        ]

        state = MagicMock()
        playbook = MagicMock()
        playbook.market_context = {"gift_nifty": gift_nifty_str}
        state.load_playbook.return_value = playbook
        state.load_daily_state.return_value = {"date": "2025-01-15"}

        telegram = MagicMock()

        return {
            "data_feed": data_feed,
            "state": state,
            "telegram": telegram,
        }

    def test_confirmed_gap(self):
        from v7.main import cmd_preopen
        components = self._make_components(ltp=24100, prev_close=24000, gift_nifty_str="+0.35%")
        cmd_preopen(components)

        # Check daily state was saved with preopen data
        state = components["state"]
        state.save_daily_state.assert_called_once()
        saved = state.save_daily_state.call_args[0][0]
        assert saved["preopen"]["gap_confirmation"] == "confirmed"
        assert saved["preopen"]["actual_gap_pct"] == 0.42  # (24100-24000)/24000*100

        # Check telegram was called
        components["telegram"].send.assert_called_once()

    def test_divergent_gap(self):
        from v7.main import cmd_preopen
        components = self._make_components(ltp=23950, prev_close=24000, gift_nifty_str="+0.50%")
        cmd_preopen(components)

        state = components["state"]
        saved = state.save_daily_state.call_args[0][0]
        assert saved["preopen"]["gap_confirmation"] == "divergent"

    def test_no_data_feed_skips(self):
        from v7.main import cmd_preopen
        components = self._make_components()
        components["data_feed"] = None
        cmd_preopen(components)
        # Should not crash, should not send telegram
        components["telegram"].send.assert_not_called()

    def test_no_playbook_still_works(self):
        from v7.main import cmd_preopen
        components = self._make_components(ltp=24100, prev_close=24000)
        components["state"].load_playbook.return_value = None
        cmd_preopen(components)

        state = components["state"]
        saved = state.save_daily_state.call_args[0][0]
        # With no GIFT Nifty expectation, expected_gap_pct = 0
        assert saved["preopen"]["expected_gap_pct"] == 0.0
        # Actual gap is positive, expected is 0 → divergent since actual > 0.1 but expected ~ 0
        assert saved["preopen"]["gap_confirmation"] in ("confirmed", "divergent")

    def test_flat_gap_classified_correctly(self):
        from v7.main import cmd_preopen
        components = self._make_components(ltp=24005, prev_close=24000, gift_nifty_str="+0.01%")
        cmd_preopen(components)

        state = components["state"]
        saved = state.save_daily_state.call_args[0][0]
        assert saved["preopen"]["gap_direction"] == "flat"
        assert saved["preopen"]["gap_confirmation"] == "confirmed"
