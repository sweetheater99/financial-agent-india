"""Tests for V7 market intelligence module."""
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from v7.market_intel import (
    fetch_us_close,
    fetch_gift_nifty,
    fetch_fii_dii_flows,
    fetch_watchlist_news,
    get_economic_events,
    fetch_fo_ban_list,
    fetch_nse_corporate_actions,
    gather_premarket_intel,
    _cache_set,
    _cache_get,
    CACHE_DIR,
)


class TestUSClose:
    @patch("global_intel.fetch_us_markets")
    def test_returns_data(self, mock_fetch):
        mock_fetch.return_value = {"sp500_pct_change": 0.5, "nasdaq_pct_change": -0.2}
        result = fetch_us_close()
        assert result["sp500_pct_change"] == 0.5
        assert result["nasdaq_pct_change"] == -0.2

    @patch("global_intel.fetch_us_markets", side_effect=Exception("network"))
    def test_fallback_on_error(self, mock_fetch):
        result = fetch_us_close()
        assert result["sp500_pct_change"] == 0.0


class TestGiftNifty:
    def test_no_prev_close(self):
        result = fetch_gift_nifty(0.0)
        assert "unavailable" in result

    @patch("global_intel.fetch_gift_nifty_gap")
    def test_with_gap(self, mock_fetch):
        mock_fetch.return_value = {"gift_nifty_gap_pct": 0.35, "gift_nifty_ltp": 24150}
        result = fetch_gift_nifty(24000)
        assert "24150" in result
        assert "0.35" in result


class TestFIIDII:
    @patch("global_intel.fetch_fii_dii")
    def test_returns_formatted(self, mock_fetch):
        mock_fetch.return_value = {"fii_net": -1500, "dii_net": 2000}
        result = fetch_fii_dii_flows()
        assert "FII" in result
        assert "DII" in result
        assert "-1,500" in result

    @patch("global_intel.fetch_fii_dii", side_effect=Exception("fail"))
    def test_fallback(self, mock_fetch):
        result = fetch_fii_dii_flows()
        assert result == "unavailable"


class TestEconomicEvents:
    @patch("v7.market_intel._cache_get", return_value=None)
    def test_returns_lists(self, mock_cache):
        today, week = get_economic_events()
        assert isinstance(today, list)
        assert isinstance(week, list)

    @patch("time_intel.is_expiry_day", return_value="weekly")
    @patch("v7.market_intel._cache_get", return_value=None)
    def test_expiry_detection(self, mock_cache, mock_expiry):
        today, _ = get_economic_events()
        assert any("expiry" in e.lower() for e in today)


class TestFOBanList:
    @patch("v7.market_intel._cache_get", return_value=["IDEA", "RBLBANK"])
    def test_cached(self, mock_cache):
        result = fetch_fo_ban_list()
        assert "IDEA" in result
        assert "RBLBANK" in result


class TestWatchlistNews:
    @patch("v7.market_intel._cache_get", return_value={"RELIANCE": ["Reliance Q3 profit up 10%"]})
    def test_cached(self, mock_cache):
        result = fetch_watchlist_news(["RELIANCE"])
        assert "RELIANCE" in result
        assert len(result["RELIANCE"]) == 1


class TestCache:
    def test_roundtrip(self, tmp_path):
        with patch("v7.market_intel.CACHE_DIR", tmp_path):
            _cache_set("test_key", {"foo": "bar"})
            result = _cache_get("test_key", max_age_hours=1)
            assert result == {"foo": "bar"}

    def test_expired(self, tmp_path):
        with patch("v7.market_intel.CACHE_DIR", tmp_path):
            _cache_set("old_key", {"foo": "bar"})
            # 0 hours max age = always expired
            result = _cache_get("old_key", max_age_hours=0)
            assert result is None


class TestGatherPremarketIntel:
    @patch("v7.market_intel.fetch_us_close", return_value={"sp500_pct_change": 0.5})
    @patch("v7.market_intel.fetch_gift_nifty", return_value="+0.3% gap")
    @patch("v7.market_intel.fetch_fii_dii_flows", return_value="FII: -500Cr, DII: +800Cr")
    @patch("v7.market_intel.get_economic_events", return_value=(["Expiry day"], ["RBI Thursday"]))
    @patch("v7.market_intel.fetch_nse_corporate_actions", return_value=[])
    @patch("v7.market_intel.fetch_watchlist_news", return_value={})
    @patch("v7.market_intel.classify_news_batch", return_value={})
    @patch("v7.market_intel.fetch_fo_ban_list", return_value=["IDEA"])
    def test_returns_all_keys(self, *mocks):
        result = gather_premarket_intel()
        required_keys = [
            "us_close", "gift_nifty", "prev_vix", "fii_dii",
            "events_today", "events_this_week", "level_memory",
            "edge_tracker", "risk_state", "fo_ban_list", "recent_lessons",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    @patch("v7.market_intel.fetch_us_close", return_value={})
    @patch("v7.market_intel.fetch_gift_nifty", return_value="unavailable")
    @patch("v7.market_intel.fetch_fii_dii_flows", return_value="unavailable")
    @patch("v7.market_intel.get_economic_events", return_value=([], []))
    @patch("v7.market_intel.fetch_nse_corporate_actions", return_value=[])
    @patch("v7.market_intel.fetch_watchlist_news", return_value={})
    @patch("v7.market_intel.classify_news_batch", return_value={})
    @patch("v7.market_intel.fetch_fo_ban_list", return_value=[])
    def test_graceful_all_failures(self, *mocks):
        """Even if everything fails, returns valid dict with all keys."""
        result = gather_premarket_intel()
        assert isinstance(result["us_close"], dict)
        assert isinstance(result["events_today"], list)
        assert isinstance(result["fo_ban_list"], list)

    @patch("v7.market_intel.fetch_us_close", return_value={})
    @patch("v7.market_intel.fetch_gift_nifty", return_value="unavailable")
    @patch("v7.market_intel.fetch_fii_dii_flows", return_value="unavailable")
    @patch("v7.market_intel.get_economic_events", return_value=([], []))
    @patch("v7.market_intel.fetch_nse_corporate_actions", return_value=[])
    @patch("v7.market_intel.fetch_watchlist_news", return_value={"RELIANCE": ["Reliance profit up"]})
    @patch("v7.market_intel.classify_news_batch", return_value={
        "RELIANCE": {"sentiment": "positive", "has_earnings_soon": True,
                     "has_policy_impact": False, "key_headline": "Reliance profit up"}
    })
    @patch("v7.market_intel.fetch_fo_ban_list", return_value=[])
    def test_news_added_to_events(self, *mocks):
        result = gather_premarket_intel()
        # Should have earnings event and positive sentiment event
        assert any("earnings" in e.lower() for e in result["events_today"])
        assert any("reliance" in e.lower() for e in result["events_today"])


class TestSetupPerformanceSummary:
    def test_basic(self):
        from v7.market_intel import setup_performance_summary
        edge_tracker = MagicMock()
        edge_tracker.all_symbol_setup_combos.return_value = {
            "RELIANCE:breakout_long": {
                "trades": 4, "wins": 0, "win_rate": 0.0,
                "net_pnl": -4550, "weighted_win_rate": 0.0,
            },
            "SBIN:breakout_short": {
                "trades": 1, "wins": 1, "win_rate": 1.0,
                "net_pnl": 675, "weighted_win_rate": 1.0,
            },
        }
        text = setup_performance_summary(edge_tracker)
        assert "RELIANCE:breakout_long" in text
        assert "0W/4L" in text
        assert "SBIN:breakout_short" in text
        assert "1W/0L" in text

    def test_empty_combos(self):
        from v7.market_intel import setup_performance_summary
        edge_tracker = MagicMock()
        edge_tracker.all_symbol_setup_combos.return_value = {}
        text = setup_performance_summary(edge_tracker)
        assert "No setup performance data yet." in text

    def test_exception_returns_empty_string(self):
        from v7.market_intel import setup_performance_summary
        edge_tracker = MagicMock()
        edge_tracker.all_symbol_setup_combos.side_effect = Exception("boom")
        text = setup_performance_summary(edge_tracker)
        assert text == ""

    def test_trend_arrows(self):
        from v7.market_intel import setup_performance_summary
        edge_tracker = MagicMock()
        edge_tracker.all_symbol_setup_combos.return_value = {
            "A:x": {"trades": 1, "wins": 1, "win_rate": 1.0, "net_pnl": 100, "weighted_win_rate": 0.8},
            "B:y": {"trades": 1, "wins": 0, "win_rate": 0.0, "net_pnl": -100, "weighted_win_rate": 0.1},
            "C:z": {"trades": 2, "wins": 1, "win_rate": 0.5, "net_pnl": 0, "weighted_win_rate": 0.4},
        }
        text = setup_performance_summary(edge_tracker)
        assert "↑" in text
        assert "↓" in text
        assert "→" in text
