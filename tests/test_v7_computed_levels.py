"""Tests for v7/computed_levels.py — pivot points, VWAP, and level formatting."""
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from v7.computed_levels import ComputedLevels, LevelComputer


def test_pivot_computation():
    """PDH=24500, PDL=24200, PDC=24350 -> PP=24350, R1=24500, S1=24200, R2=24650, S2=24050."""
    lc = LevelComputer()
    levels = lc.compute_pivots("NIFTY", pdh=24500, pdl=24200, pdc=24350)

    assert levels.pdh == 24500
    assert levels.pdl == 24200
    assert levels.pdc == 24350
    # PP = (24500 + 24200 + 24350) / 3 = 73050 / 3 = 24350
    assert levels.pp == 24350.0
    # R1 = 2*24350 - 24200 = 48700 - 24200 = 24500
    assert levels.r1 == 24500.0
    # S1 = 2*24350 - 24500 = 48700 - 24500 = 24200
    assert levels.s1 == 24200.0
    # R2 = 24350 + (24500 - 24200) = 24350 + 300 = 24650
    assert levels.r2 == 24650.0
    # S2 = 24350 - (24500 - 24200) = 24350 - 300 = 24050
    assert levels.s2 == 24050.0


def test_from_daily_candle():
    """Verify classmethod produces correct pivots."""
    levels = ComputedLevels.from_daily_candle("BANKNIFTY", high=52000, low=51500, close=51800)

    pp = (52000 + 51500 + 51800) / 3
    assert abs(levels.pp - pp) < 0.01
    assert abs(levels.r1 - (2 * pp - 51500)) < 0.01
    assert abs(levels.r2 - (pp + (52000 - 51500))) < 0.01
    assert abs(levels.s1 - (2 * pp - 52000)) < 0.01
    assert abs(levels.s2 - (pp - (52000 - 51500))) < 0.01
    assert levels.symbol == "BANKNIFTY"


def test_vwap_single_candle():
    """One candle H=100, L=90, C=95, V=1000 -> VWAP = 95."""
    lc = LevelComputer()
    lc.compute_pivots("TEST", pdh=100, pdl=90, pdc=95)
    lc.init_vwap("TEST")

    vwap = lc.update_vwap("TEST", high=100, low=90, close=95, volume=1000)
    # TP = (100 + 90 + 95) / 3 = 95.0
    assert abs(vwap - 95.0) < 0.01

    levels = lc.get_levels("TEST")
    assert levels is not None
    assert abs(levels.vwap - 95.0) < 0.01
    # Single candle: TP == VWAP, so std dev = 0, bands == VWAP
    assert abs(levels.vwap_upper - 95.0) < 0.01
    assert abs(levels.vwap_lower - 95.0) < 0.01


def test_vwap_multiple_candles():
    """3 candles, verify cumulative VWAP against manual calculation."""
    lc = LevelComputer()
    lc.compute_pivots("TEST", pdh=100, pdl=90, pdc=95)
    lc.init_vwap("TEST")

    # Candle 1: H=100, L=90, C=95, V=1000 -> TP=95
    lc.update_vwap("TEST", 100, 90, 95, 1000)
    # Candle 2: H=102, L=96, C=100, V=2000 -> TP=99.333
    lc.update_vwap("TEST", 102, 96, 100, 2000)
    # Candle 3: H=105, L=98, C=103, V=1500 -> TP=102
    vwap = lc.update_vwap("TEST", 105, 98, 103, 1500)

    # Manual: sum(TP*V) = 95*1000 + 99.333*2000 + 102*1500
    tp1, tp2, tp3 = 95.0, (102 + 96 + 100) / 3, (105 + 98 + 103) / 3
    cum_tp_vol = tp1 * 1000 + tp2 * 2000 + tp3 * 1500
    cum_vol = 1000 + 2000 + 1500
    expected_vwap = cum_tp_vol / cum_vol

    assert abs(vwap - expected_vwap) < 0.01


def test_vwap_bands():
    """Verify upper/lower bands are computed correctly with multiple candles."""
    lc = LevelComputer()
    lc.compute_pivots("TEST", pdh=100, pdl=90, pdc=95)
    lc.init_vwap("TEST")

    # Two candles with different TPs to get non-zero std dev
    lc.update_vwap("TEST", 100, 90, 95, 1000)   # TP=95
    lc.update_vwap("TEST", 110, 100, 108, 1000)  # TP=106

    levels = lc.get_levels("TEST")
    assert levels is not None

    # VWAP should be between the two TPs
    assert 95 < levels.vwap < 106
    # Upper band should be above VWAP
    assert levels.vwap_upper > levels.vwap
    # Lower band should be below VWAP
    assert levels.vwap_lower < levels.vwap
    # Bands should be symmetric
    assert abs((levels.vwap_upper - levels.vwap) - (levels.vwap - levels.vwap_lower)) < 0.01


def test_vwap_init_resets():
    """After init_vwap, previous state is cleared."""
    lc = LevelComputer()
    lc.compute_pivots("TEST", pdh=100, pdl=90, pdc=95)
    lc.init_vwap("TEST")

    lc.update_vwap("TEST", 100, 90, 95, 1000)
    lc.update_vwap("TEST", 110, 100, 108, 1000)

    # Reset
    lc.init_vwap("TEST")
    vwap = lc.update_vwap("TEST", 100, 90, 95, 1000)

    # Should be fresh — single candle VWAP
    assert abs(vwap - 95.0) < 0.01


def test_roundtrip():
    """to_dict -> from_dict preserves all fields."""
    original = ComputedLevels(
        symbol="NIFTY",
        pdh=24500, pdl=24200, pdc=24350,
        pp=24350, r1=24500, r2=24650, s1=24200, s2=24050,
        vwap=24380, vwap_upper=24420, vwap_lower=24340,
    )
    d = original.to_dict()
    restored = ComputedLevels.from_dict(d)

    assert restored.symbol == original.symbol
    assert restored.pdh == original.pdh
    assert restored.pdl == original.pdl
    assert restored.pdc == original.pdc
    assert restored.pp == original.pp
    assert restored.r1 == original.r1
    assert restored.r2 == original.r2
    assert restored.s1 == original.s1
    assert restored.s2 == original.s2
    assert restored.vwap == original.vwap
    assert restored.vwap_upper == original.vwap_upper
    assert restored.vwap_lower == original.vwap_lower


def test_format_for_prompt():
    """Verify output structure from format_for_prompt."""
    lc = LevelComputer()
    lc.compute_pivots("NIFTY", 24500, 24200, 24350)
    lc.compute_pivots("BANKNIFTY", 52000, 51500, 51800)

    result = lc.format_for_prompt(["NIFTY", "BANKNIFTY", "RELIANCE"])

    # Should have entries for computed symbols only
    assert "NIFTY" in result
    assert "BANKNIFTY" in result
    assert "RELIANCE" not in result

    # Each entry should be a dict with expected keys
    nifty = result["NIFTY"]
    assert nifty["symbol"] == "NIFTY"
    assert nifty["pp"] == 24350.0
    assert "vwap" in nifty
    assert "r1" in nifty
    assert "s1" in nifty


def test_get_levels_none():
    """get_levels returns None for unknown symbol."""
    lc = LevelComputer()
    assert lc.get_levels("UNKNOWN") is None


def test_update_vwap_auto_init():
    """update_vwap auto-initializes if init_vwap not called."""
    lc = LevelComputer()
    # No init_vwap call — should still work
    vwap = lc.update_vwap("TEST", 100, 90, 95, 1000)
    assert abs(vwap - 95.0) < 0.01
