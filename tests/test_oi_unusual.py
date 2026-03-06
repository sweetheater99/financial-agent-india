"""Tests for oi_unusual module."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from oi_unusual import detect_unusual_oi, get_oi_signal_for_trade


def _make_chain(strikes_data):
    """Helper to build option chain rows."""
    chain = []
    for strike, ce_oi, pe_oi in strikes_data:
        chain.append({
            "strikePrice": strike,
            "CE": {"openInterest": ce_oi, "totalTradedVolume": 0},
            "PE": {"openInterest": pe_oi, "totalTradedVolume": 0},
        })
    return chain


def test_empty_chain():
    """Empty chain should return safe defaults."""
    result = detect_unusual_oi([])
    assert result["has_unusual_activity"] is False
    assert result["oi_skew"] == "balanced"


def test_detect_call_spike():
    """Single massive call OI should be detected."""
    chain = _make_chain([
        (24000, 5000, 2000),
        (24500, 5000, 2000),
        (25000, 50000, 2000),  # 10x spike in calls
        (25500, 5000, 2000),
    ])
    result = detect_unusual_oi(chain, "TEST")
    assert result["has_unusual_activity"] is True
    assert len(result["unusual_calls"]) >= 1
    assert result["unusual_calls"][0]["strike"] == 25000


def test_detect_put_spike():
    """Massive put OI spike detection."""
    chain = _make_chain([
        (23000, 1000, 80000),  # huge put OI
        (23500, 1000, 3000),
        (24000, 1000, 3000),
        (24500, 1000, 3000),
    ])
    result = detect_unusual_oi(chain, "TEST")
    assert result["has_unusual_activity"] is True
    assert len(result["unusual_puts"]) >= 1
    assert result["unusual_puts"][0]["strike"] == 23000


def test_balanced_no_unusual():
    """Evenly distributed OI should not flag unusual activity."""
    chain = _make_chain([
        (24000, 5000, 5000),
        (24500, 5000, 5000),
        (25000, 5000, 5000),
        (25500, 5000, 5000),
    ])
    result = detect_unusual_oi(chain, "TEST")
    assert result["has_unusual_activity"] is False
    assert result["oi_skew"] == "balanced"


def test_call_heavy_skew():
    """Call-heavy OI should flag call_heavy skew."""
    chain = _make_chain([
        (24000, 20000, 2000),
        (24500, 20000, 2000),
        (25000, 20000, 2000),
    ])
    result = detect_unusual_oi(chain, "TEST")
    assert result["oi_skew"] == "call_heavy"


def test_put_heavy_skew():
    """Put-heavy OI should flag put_heavy skew."""
    chain = _make_chain([
        (24000, 2000, 20000),
        (24500, 2000, 20000),
        (25000, 2000, 20000),
    ])
    result = detect_unusual_oi(chain, "TEST")
    assert result["oi_skew"] == "put_heavy"


def test_signal_bullish_confirming():
    """Put support below spot confirms bullish trade."""
    unusual = {
        "has_unusual_activity": True,
        "unusual_calls": [],
        "unusual_puts": [{"strike": 23500, "oi": 80000, "vs_avg": 4.0, "signal": "support"}],
    }
    result = get_oi_signal_for_trade(unusual, "bullish", 24000)
    assert result["signal"] == "confirming"
    assert result["key_level"] == 23500


def test_signal_bullish_cautionary():
    """Call resistance above spot cautions bullish trade."""
    unusual = {
        "has_unusual_activity": True,
        "unusual_calls": [{"strike": 24500, "oi": 60000, "vs_avg": 3.5, "signal": "resistance"}],
        "unusual_puts": [],
    }
    result = get_oi_signal_for_trade(unusual, "bullish", 24000)
    assert result["signal"] == "cautionary"
    assert result["key_level"] == 24500


def test_signal_no_activity():
    """No unusual activity returns neutral."""
    result = get_oi_signal_for_trade({"has_unusual_activity": False}, "bullish", 24000)
    assert result["signal"] == "neutral"
