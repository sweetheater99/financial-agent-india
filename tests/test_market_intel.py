"""Tests for market_intel.py — VWAP, OI classification, OI S/R, formatting."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from market_intel import (
    compute_vwap_from_candles,
    classify_oi_change,
    get_oi_support_resistance,
    format_market_intel_for_claude,
)


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------

def test_compute_vwap():
    """Verify VWAP against manual calculation."""
    candles = [
        {"high": 110, "low": 90, "close": 100, "volume": 1000},
        {"high": 120, "low": 100, "close": 110, "volume": 2000},
    ]
    # tp1 = (110+90+100)/3 = 100.0,  tp2 = (120+100+110)/3 = 110.0
    # vwap = (100*1000 + 110*2000) / (1000+2000) = 320000/3000 = 106.666...
    expected = (100.0 * 1000 + 110.0 * 2000) / 3000
    result = compute_vwap_from_candles(candles)
    assert result is not None
    assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"


def test_vwap_empty_candles():
    """Empty candle list returns None."""
    assert compute_vwap_from_candles([]) is None


def test_vwap_zero_volume():
    """All-zero volume returns None (division by zero guard)."""
    candles = [
        {"high": 100, "low": 90, "close": 95, "volume": 0},
        {"high": 105, "low": 92, "close": 98, "volume": 0},
    ]
    assert compute_vwap_from_candles(candles) is None


# ---------------------------------------------------------------------------
# OI Classification
# ---------------------------------------------------------------------------

def test_classify_oi_change():
    """All 4 OI patterns."""
    assert classify_oi_change(5.0, 2.0) == "long_buildup"
    assert classify_oi_change(5.0, -2.0) == "short_buildup"
    assert classify_oi_change(-5.0, 2.0) == "short_covering"
    assert classify_oi_change(-5.0, -2.0) == "long_unwinding"


def test_classify_oi_change_zero():
    """Zero values treated as non-negative."""
    assert classify_oi_change(0.0, 0.0) == "long_buildup"
    assert classify_oi_change(0.0, -1.0) == "short_buildup"
    assert classify_oi_change(-1.0, 0.0) == "short_covering"


# ---------------------------------------------------------------------------
# OI Support / Resistance
# ---------------------------------------------------------------------------

def test_get_oi_support_resistance():
    option_chain = {
        "CE": [
            {"strike": 22000, "oi": 500000},
            {"strike": 22500, "oi": 800000},
            {"strike": 23000, "oi": 600000},
        ],
        "PE": [
            {"strike": 21000, "oi": 400000},
            {"strike": 21500, "oi": 900000},
            {"strike": 22000, "oi": 700000},
        ],
    }
    result = get_oi_support_resistance(option_chain)
    assert result["resistance_strike"] == 22500
    assert result["max_call_oi"] == 800000
    assert result["support_strike"] == 21500
    assert result["max_put_oi"] == 900000


def test_get_oi_support_resistance_empty():
    """Empty option chain returns zeros."""
    result = get_oi_support_resistance({"CE": [], "PE": []})
    assert result["support_strike"] == 0
    assert result["resistance_strike"] == 0


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def test_format_market_intel_for_claude():
    oi_change = {"oi_change_pct": 5.0, "price_change_pct": 2.0}
    oi_sr = {
        "support_strike": 21500,
        "resistance_strike": 22500,
        "max_put_oi": 900000,
        "max_call_oi": 800000,
    }
    result = format_market_intel_for_claude(
        vwap=22100.0, ltp=22200.0, oi_change=oi_change, oi_sr=oi_sr,
    )

    assert "VWAP: 22100.00" in result
    assert "LTP: 22200.00" in result
    assert "BULLISH" in result
    assert "long_buildup" in result
    assert "Support 21500" in result
    assert "Resistance 22500" in result


def test_format_market_intel_for_claude_none_values():
    """Handles None gracefully."""
    result = format_market_intel_for_claude(vwap=None, ltp=100.0, oi_change=None, oi_sr=None)
    assert "unavailable" in result
