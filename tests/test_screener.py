"""Tests for screener.py pure functions."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from screener import get_field, extract_underlying, score_signals, clear_signal_cache, _signal_cache


# ---------------------------------------------------------------------------
# get_field
# ---------------------------------------------------------------------------

def test_get_field_first_match():
    record = {"percentChange": 3.5, "perChange": 2.1}
    assert get_field(record, "percentChange", "perChange") == 3.5


def test_get_field_skips_none():
    record = {"percentChange": None, "perChange": 2.1}
    assert get_field(record, "percentChange", "perChange") == 2.1


def test_get_field_zero_not_skipped():
    record = {"percentChange": 0, "perChange": 2.1}
    assert get_field(record, "percentChange", "perChange") == 0


def test_get_field_all_missing():
    record = {"unrelated": "value"}
    assert get_field(record, "percentChange", "perChange") is None


# ---------------------------------------------------------------------------
# extract_underlying
# ---------------------------------------------------------------------------

def test_extract_underlying_futures():
    assert extract_underlying("RELIANCE26FEB26FUT") == "RELIANCE"


def test_extract_underlying_ampersand():
    assert extract_underlying("M&M26FEB26FUT") == "M&M"


def test_extract_underlying_index():
    assert extract_underlying("BANKNIFTY26FEB26FUT") == "BANKNIFTY"


def test_extract_underlying_clean():
    # No digits — returns as-is via fallback
    assert extract_underlying("RELIANCE") == "RELIANCE"


# ---------------------------------------------------------------------------
# score_signals
# ---------------------------------------------------------------------------

def test_score_signals_empty():
    assert score_signals({}) == []


def test_score_signals_single_bullish():
    signals = {
        "LongBuildUp": [{"tradingSymbol": "RELIANCE26FEB26FUT", "percentChange": 2.5}],
    }
    result = score_signals(signals)
    assert len(result) == 1
    assert result[0]["symbol"] == "RELIANCE"
    assert result[0]["direction"] == "bullish"
    assert result[0]["score"] == 2.0  # LongBuildUp weight


def test_score_signals_multi_category_bonus():
    # 3 bullish signals -> 1.5x multiplier
    signals = {
        "LongBuildUp": [{"tradingSymbol": "TCS26FEB26FUT", "percentChange": 1.0}],
        "PercPriceGainers": [{"tradingSymbol": "TCS26FEB26FUT", "percentChange": 1.0}],
        "ShortCovering": [{"tradingSymbol": "TCS26FEB26FUT", "percentChange": 1.0}],
    }
    result = score_signals(signals)
    assert len(result) == 1
    # Base: 2.0 + 1.0 + 1.5 = 4.5, then * 1.5 = 6.75
    assert result[0]["score"] == 6.75
    assert result[0]["direction"] == "bullish"


def test_score_signals_mixed_direction():
    signals = {
        "LongBuildUp": [{"tradingSymbol": "INFY26FEB26FUT", "percentChange": 2.0}],
        "ShortBuildUp": [{"tradingSymbol": "INFY26FEB26FUT", "percentChange": -1.0}],
    }
    result = score_signals(signals)
    assert len(result) == 1
    # bullish=2.0, bearish=2.0 -> tie goes to bullish (>=)
    assert result[0]["direction"] == "bullish"


def test_score_signals_sorted_by_score():
    signals = {
        "PercPriceGainers": [
            {"tradingSymbol": "TCS26FEB26FUT", "percentChange": 1.0},
        ],
        "LongBuildUp": [
            {"tradingSymbol": "RELIANCE26FEB26FUT", "percentChange": 2.0},
        ],
    }
    result = score_signals(signals)
    assert len(result) == 2
    # RELIANCE: LongBuildUp=2.0, TCS: PercPriceGainers=1.0
    assert result[0]["symbol"] == "RELIANCE"
    assert result[1]["symbol"] == "TCS"


# ---------------------------------------------------------------------------
# clear_signal_cache
# ---------------------------------------------------------------------------

def test_clear_signal_cache():
    _signal_cache["data"] = {"some": "data"}
    _signal_cache["timestamp"] = "now"
    clear_signal_cache()
    assert _signal_cache["data"] is None
    assert _signal_cache["timestamp"] is None
