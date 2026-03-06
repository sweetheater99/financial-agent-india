"""Tests for multi_timeframe.py — uses mocked yfinance data."""

import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import pandas as pd
import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_timeframe import (
    check_weekly_trend,
    check_multi_timeframe,
    _cache,
    _CACHE_TTL,
)


def _make_weekly_df(closes: list[float]) -> pd.DataFrame:
    """Create a mock weekly OHLCV DataFrame from a list of close prices."""
    n = len(closes)
    dates = pd.date_range(end="2026-03-06", periods=n, freq="7D")
    df = pd.DataFrame({
        "Open": [c * 0.99 for c in closes],
        "High": [c * 1.02 for c in closes],
        "Low": [c * 0.97 for c in closes],
        "Close": closes,
        "Volume": [1000000] * n,
    }, index=dates)
    return df


def _bullish_closes() -> list[float]:
    """Generate 30 weeks of steadily rising prices (bullish weekly trend)."""
    return [100 + i * 2 for i in range(30)]


def _bearish_closes() -> list[float]:
    """Generate 30 weeks of steadily falling prices (bearish weekly trend)."""
    return [200 - i * 2 for i in range(30)]


def _neutral_closes() -> list[float]:
    """Generate 30 weeks of sideways prices (neutral weekly trend)."""
    base = 150
    # Oscillate around base
    return [base + (3 if i % 2 == 0 else -3) for i in range(30)]


class TestBullishDailyBullishWeekly:
    """Test 1: Bullish daily + bullish weekly -> aligned."""

    @patch("multi_timeframe._get_weekly_data")
    def test_aligned(self, mock_get):
        _cache.clear()
        mock_get.return_value = _make_weekly_df(_bullish_closes())
        result = check_weekly_trend("RELIANCE", "bullish")
        assert result["aligned"] is True
        assert result["weekly_trend"] == "bullish"
        assert result["confidence"] > 0


class TestBullishDailyBearishWeekly:
    """Test 2: Bullish daily + bearish weekly -> not aligned."""

    @patch("multi_timeframe._get_weekly_data")
    def test_not_aligned(self, mock_get):
        _cache.clear()
        mock_get.return_value = _make_weekly_df(_bearish_closes())
        result = check_weekly_trend("RELIANCE", "bullish")
        assert result["aligned"] is False
        assert result["weekly_trend"] == "bearish"


class TestNeutralWeekly:
    """Test 3: Neutral weekly -> aligned (benefit of doubt)."""

    @patch("multi_timeframe._get_weekly_data")
    def test_neutral_gives_benefit(self, mock_get):
        _cache.clear()
        # Use very short data to get neutral (insufficient for strong signals)
        mock_get.return_value = None  # no data -> neutral
        result = check_weekly_trend("RELIANCE", "bullish")
        assert result["aligned"] is True
        assert result["weekly_trend"] == "neutral"


class TestCacheWorks:
    """Test 4: Cache works (2nd call doesn't re-fetch)."""

    @patch("multi_timeframe._get_weekly_data")
    def test_cache(self, mock_get):
        _cache.clear()
        mock_get.return_value = _make_weekly_df(_bullish_closes())

        # First call
        result1 = check_weekly_trend("HDFCBANK", "bullish")
        assert mock_get.call_count == 1

        # Second call should use cache
        result2 = check_weekly_trend("HDFCBANK", "bullish")
        assert mock_get.call_count == 1  # no additional fetch

        assert result1["weekly_rsi"] == result2["weekly_rsi"]


class TestHandlesNoData:
    """Test 5: Handles no data gracefully."""

    @patch("multi_timeframe._get_weekly_data")
    def test_no_data(self, mock_get):
        _cache.clear()
        mock_get.return_value = None

        aligned, reason, confidence = check_multi_timeframe("UNKNOWN", "bullish")
        assert aligned is True  # don't block on missing data
        assert confidence == 0.0
