"""Tests for premarket_briefing module."""

import re
from datetime import date, datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# --- Helpers ---

def _mock_smart_api():
    """Return a mock SmartAPI object."""
    api = MagicMock()
    api.ltpData.return_value = {"data": {"ltp": 22500.0}}
    api.getCandleData.return_value = {"data": _sample_candles()}
    return api


def _sample_candles(n=50):
    """Generate n sample Nifty daily candles."""
    base = 22000
    candles = []
    for i in range(n):
        o = base + i * 10
        h = o + 50
        l = o - 30
        c = o + 20
        candles.append([f"2026-01-{(i+1):02d}T00:00:00", o, h, l, c, 100000])
    return candles


def _sample_option_chain():
    """Return a minimal option chain list."""
    return [
        {
            "strikePrice": 22000,
            "CE": {"openInterest": 50000},
            "PE": {"openInterest": 40000},
        },
        {
            "strikePrice": 22500,
            "CE": {"openInterest": 80000},
            "PE": {"openInterest": 90000},
        },
        {
            "strikePrice": 23000,
            "CE": {"openInterest": 30000},
            "PE": {"openInterest": 60000},
        },
    ]


def _macro_result(**overrides):
    """Return a typical macro context dict."""
    base = {
        "sp500_pct_change": 0.3,
        "nasdaq_pct_change": 0.5,
        "gift_nifty_gap_pct": 0.2,
        "gift_nifty_ltp": 22545.0,
        "hang_seng_pct": 0.8,
        "nikkei_pct": -0.2,
        "fii_net_crores": -2340,
        "dii_net_crores": 1890,
        "hard_gate": "NONE",
        "hard_gate_reason": "Global cues neutral",
        "fetched_at": "2026-03-07T08:00:00",
    }
    base.update(overrides)
    return base


def _regime_result(**overrides):
    """Return a typical regime dict."""
    base = {
        "regime": "TRENDING_UP",
        "confidence": 0.78,
        "reason": "ADX 28.5>20 for 3+ bars, price above EMA20",
        "iv_percentile": 50,
    }
    base.update(overrides)
    return base


# --- Tests ---

class TestAllDataAvailable:
    """Test with all data sources returning valid data."""

    @patch("premarket_briefing.fetch_option_chain", return_value=_sample_option_chain())
    @patch("premarket_briefing.fetch_macro_context", return_value=_macro_result())
    @patch("premarket_briefing.classify_regime", return_value=_regime_result())
    @patch("premarket_briefing.get_ltp", return_value=14.2)
    @patch("premarket_briefing.fetch_daily_candles", return_value=_sample_candles())
    @patch("premarket_briefing.time.sleep")
    def test_all_sections_present(self, mock_sleep, mock_candles, mock_ltp,
                                   mock_regime, mock_macro, mock_chain):
        from premarket_briefing import generate_premarket_briefing
        msg = generate_premarket_briefing(_mock_smart_api())

        # All section headers present
        assert "Pre-Market Briefing" in msg
        assert "Global Overnight" in msg
        assert "FII/DII Flow" in msg
        assert "Market Regime" in msg
        assert "OI Signals" in msg
        assert "Gates Active" in msg
        assert "Events Today" in msg
        assert "Playbook" in msg

    @patch("premarket_briefing.fetch_option_chain", return_value=_sample_option_chain())
    @patch("premarket_briefing.fetch_macro_context", return_value=_macro_result())
    @patch("premarket_briefing.classify_regime", return_value=_regime_result())
    @patch("premarket_briefing.get_ltp", return_value=14.2)
    @patch("premarket_briefing.fetch_daily_candles", return_value=_sample_candles())
    @patch("premarket_briefing.time.sleep")
    def test_global_data_in_message(self, mock_sleep, mock_candles, mock_ltp,
                                     mock_regime, mock_macro, mock_chain):
        from premarket_briefing import generate_premarket_briefing
        msg = generate_premarket_briefing(_mock_smart_api())

        assert "S&P 500: +0.3%" in msg
        assert "Nasdaq: +0.5%" in msg
        assert "Hang Seng +0.8%" in msg

    @patch("premarket_briefing.fetch_option_chain", return_value=_sample_option_chain())
    @patch("premarket_briefing.fetch_macro_context", return_value=_macro_result())
    @patch("premarket_briefing.classify_regime", return_value=_regime_result())
    @patch("premarket_briefing.get_ltp", return_value=14.2)
    @patch("premarket_briefing.fetch_daily_candles", return_value=_sample_candles())
    @patch("premarket_briefing.time.sleep")
    def test_pcr_and_max_pain_in_message(self, mock_sleep, mock_candles, mock_ltp,
                                          mock_regime, mock_macro, mock_chain):
        from premarket_briefing import generate_premarket_briefing
        msg = generate_premarket_briefing(_mock_smart_api())

        assert "PCR:" in msg
        assert "Max Pain:" in msg


class TestGlobalIntelNone:
    """Test graceful degradation when global intel returns None."""

    @patch("premarket_briefing.fetch_option_chain", return_value=None)
    @patch("premarket_briefing.fetch_macro_context", side_effect=Exception("network error"))
    @patch("premarket_briefing.classify_regime", return_value=_regime_result())
    @patch("premarket_briefing.get_ltp", return_value=14.2)
    @patch("premarket_briefing.fetch_daily_candles", return_value=_sample_candles())
    @patch("premarket_briefing.time.sleep")
    def test_graceful_degradation(self, mock_sleep, mock_candles, mock_ltp,
                                   mock_regime, mock_macro, mock_chain):
        from premarket_briefing import generate_premarket_briefing
        msg = generate_premarket_briefing(_mock_smart_api())

        # Should still have all section headers
        assert "Pre-Market Briefing" in msg
        assert "Global Overnight" in msg
        assert "Data unavailable" in msg
        assert "Market Regime: TRENDING_UP" in msg  # regime still works

    @patch("premarket_briefing.fetch_option_chain", side_effect=Exception("fail"))
    @patch("premarket_briefing.fetch_macro_context", side_effect=Exception("fail"))
    @patch("premarket_briefing.classify_regime", side_effect=Exception("fail"))
    @patch("premarket_briefing.get_ltp", side_effect=Exception("fail"))
    @patch("premarket_briefing.fetch_daily_candles", side_effect=Exception("fail"))
    @patch("premarket_briefing.time.sleep")
    def test_all_sources_fail(self, mock_sleep, mock_candles, mock_ltp,
                               mock_regime, mock_macro, mock_chain):
        from premarket_briefing import generate_premarket_briefing
        msg = generate_premarket_briefing(_mock_smart_api())

        # Should still produce a valid message with section headers
        assert "Pre-Market Briefing" in msg
        assert "Playbook" in msg


class TestActiveHardGates:
    """Test that active hard gates are displayed correctly."""

    @patch("premarket_briefing.fetch_option_chain", return_value=None)
    @patch("premarket_briefing.fetch_macro_context", return_value=_macro_result(
        hard_gate="BLOCK_BULLISH", hard_gate_reason="S&P 500 -2.5% crash"
    ))
    @patch("premarket_briefing.classify_regime", return_value=_regime_result())
    @patch("premarket_briefing.get_ltp", return_value=14.2)
    @patch("premarket_briefing.fetch_daily_candles", return_value=_sample_candles())
    @patch("premarket_briefing.time.sleep")
    def test_block_bullish_gate(self, mock_sleep, mock_candles, mock_ltp,
                                 mock_regime, mock_macro, mock_chain):
        from premarket_briefing import generate_premarket_briefing
        msg = generate_premarket_briefing(_mock_smart_api())

        assert "BLOCK_BULLISH" in msg
        assert "S&P 500 -2.5% crash" in msg

    @patch("premarket_briefing.fetch_option_chain", return_value=None)
    @patch("premarket_briefing.fetch_macro_context", return_value=_macro_result(
        hard_gate="BLOCK_ALL", hard_gate_reason="S&P 500 -3.5% severe crash"
    ))
    @patch("premarket_briefing.classify_regime", return_value=_regime_result())
    @patch("premarket_briefing.get_ltp", return_value=14.2)
    @patch("premarket_briefing.fetch_daily_candles", return_value=_sample_candles())
    @patch("premarket_briefing.time.sleep")
    def test_block_all_gate(self, mock_sleep, mock_candles, mock_ltp,
                             mock_regime, mock_macro, mock_chain):
        from premarket_briefing import generate_premarket_briefing
        msg = generate_premarket_briefing(_mock_smart_api())

        assert "BLOCK_ALL" in msg
        # Playbook should show equity blocked
        assert "blocked by macro gate" in msg


class TestEventCalendar:
    """Test event detection for RBI policy, budget, expiry."""

    def test_rbi_policy_today(self):
        from premarket_briefing import _section_events
        # Use an actual RBI date from config
        result = _section_events(date(2026, 4, 9))
        assert "RBI Policy" in result

    def test_budget_today(self):
        from premarket_briefing import _section_events
        result = _section_events(date(2026, 2, 1))
        assert "Union Budget" in result

    def test_weekly_expiry_thursday(self):
        from premarket_briefing import _section_events
        # 2026-03-05 is a Thursday
        result = _section_events(date(2026, 3, 5))
        assert "Weekly Expiry" in result

    def test_no_events(self):
        from premarket_briefing import _section_events
        # 2026-03-09 is a Monday, no events
        result = _section_events(date(2026, 3, 9))
        assert "Events Today: None" in result

    def test_next_event_shown(self):
        from premarket_briefing import _section_events
        # Before first RBI date, should show next upcoming
        result = _section_events(date(2026, 3, 1))
        assert "Next:" in result


class TestRegimePlaybook:
    """Test that regime correctly maps to playbook recommendations."""

    def test_sideways_enables_condors(self):
        from premarket_briefing import _section_playbook
        # VIX in condor range (12-18)
        result = _section_playbook("SIDEWAYS", 15.0, "NONE")
        assert "Iron condors: \u2705" in result

    def test_trending_disables_condors(self):
        from premarket_briefing import _section_playbook
        result = _section_playbook("TRENDING_UP", 15.0, "NONE")
        assert "Iron condors: \u274c" in result
        assert "not sideways" in result

    def test_trending_up_enables_equity(self):
        from premarket_briefing import _section_playbook
        result = _section_playbook("TRENDING_UP", 15.0, "NONE")
        assert "Equity longs: \u2705" in result

    def test_trending_down_disables_equity(self):
        from premarket_briefing import _section_playbook
        result = _section_playbook("TRENDING_DOWN", 15.0, "NONE")
        assert "Equity longs: \u274c" in result

    def test_block_all_disables_equity(self):
        from premarket_briefing import _section_playbook
        result = _section_playbook("TRENDING_UP", 15.0, "BLOCK_ALL")
        assert "Equity longs: \u274c" in result

    def test_low_vix_disables_spreads(self):
        from premarket_briefing import _section_playbook
        result = _section_playbook("SIDEWAYS", 10.0, "NONE")
        assert "Credit spreads: \u274c" in result

    def test_high_vix_disables_condors(self):
        from premarket_briefing import _section_playbook
        result = _section_playbook("SIDEWAYS", 22.0, "NONE")
        assert "Iron condors: \u274c" in result


class TestHTMLValidity:
    """Test that the message has valid HTML (no unclosed tags)."""

    @patch("premarket_briefing.fetch_option_chain", return_value=_sample_option_chain())
    @patch("premarket_briefing.fetch_macro_context", return_value=_macro_result())
    @patch("premarket_briefing.classify_regime", return_value=_regime_result())
    @patch("premarket_briefing.get_ltp", return_value=14.2)
    @patch("premarket_briefing.fetch_daily_candles", return_value=_sample_candles())
    @patch("premarket_briefing.time.sleep")
    def test_no_unclosed_tags(self, mock_sleep, mock_candles, mock_ltp,
                               mock_regime, mock_macro, mock_chain):
        from premarket_briefing import generate_premarket_briefing
        msg = generate_premarket_briefing(_mock_smart_api())

        # Count opening and closing <b> tags
        open_b = msg.count("<b>")
        close_b = msg.count("</b>")
        assert open_b == close_b, f"Unclosed <b> tags: {open_b} open vs {close_b} close"

        # No other HTML tags used, but check for common ones
        for tag in ["<i>", "<a ", "<code>"]:
            if tag in msg:
                close_tag = tag.replace("<", "</").split()[0] + ">"
                assert msg.count(tag) == msg.count(close_tag), f"Unclosed {tag} tags"

    @patch("premarket_briefing.fetch_option_chain", side_effect=Exception("fail"))
    @patch("premarket_briefing.fetch_macro_context", side_effect=Exception("fail"))
    @patch("premarket_briefing.classify_regime", side_effect=Exception("fail"))
    @patch("premarket_briefing.get_ltp", side_effect=Exception("fail"))
    @patch("premarket_briefing.fetch_daily_candles", side_effect=Exception("fail"))
    @patch("premarket_briefing.time.sleep")
    def test_no_unclosed_tags_on_failure(self, mock_sleep, mock_candles, mock_ltp,
                                          mock_regime, mock_macro, mock_chain):
        from premarket_briefing import generate_premarket_briefing
        msg = generate_premarket_briefing(_mock_smart_api())

        open_b = msg.count("<b>")
        close_b = msg.count("</b>")
        assert open_b == close_b
