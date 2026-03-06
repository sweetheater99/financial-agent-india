"""Tests for gap detection and Friday risk reduction."""
from datetime import date


class TestGapDetection:
    def test_gap_through_sl_bullish(self):
        """Bullish position that gapped below SL should be flagged."""
        from paper_trade import detect_gap_through_sl
        pos = {
            "direction": "bullish",
            "entry_price": 100.0,
            "stoploss_price": 95.0,
        }
        # LTP at open is 92 — gapped through 95 SL
        assert detect_gap_through_sl(pos, current_ltp=92.0) is True

    def test_no_gap_bullish(self):
        """Bullish position above SL should not be flagged."""
        from paper_trade import detect_gap_through_sl
        pos = {
            "direction": "bullish",
            "entry_price": 100.0,
            "stoploss_price": 95.0,
        }
        assert detect_gap_through_sl(pos, current_ltp=97.0) is False

    def test_gap_through_sl_bearish(self):
        """Bearish position that gapped above SL should be flagged."""
        from paper_trade import detect_gap_through_sl
        pos = {
            "direction": "bearish",
            "entry_price": 100.0,
            "stoploss_price": 105.0,
        }
        assert detect_gap_through_sl(pos, current_ltp=108.0) is True

    def test_no_gap_bearish(self):
        from paper_trade import detect_gap_through_sl
        pos = {
            "direction": "bearish",
            "entry_price": 100.0,
            "stoploss_price": 105.0,
        }
        assert detect_gap_through_sl(pos, current_ltp=103.0) is False


class TestFridayRisk:
    def test_is_friday_afternoon(self):
        from paper_trade import _is_friday_afternoon
        # Friday 2:30 PM
        from datetime import datetime
        assert _is_friday_afternoon(weekday=4, hour=14, minute=30) is True

    def test_not_friday(self):
        from paper_trade import _is_friday_afternoon
        assert _is_friday_afternoon(weekday=3, hour=14, minute=30) is False

    def test_friday_morning(self):
        from paper_trade import _is_friday_afternoon
        assert _is_friday_afternoon(weekday=4, hour=10, minute=0) is False

    def test_friday_2pm_boundary(self):
        from paper_trade import _is_friday_afternoon
        assert _is_friday_afternoon(weekday=4, hour=14, minute=0) is True
