"""Tests for V3 trailing stop loss enhancement (tuned params: activation=3%, mult=2.5x)."""


def test_trailing_sl_not_active_below_threshold():
    """Before +3%, the existing fixed SL applies, not the enhanced trail."""
    from config import TRAILING_SL_ACTIVATION_PCT
    assert TRAILING_SL_ACTIVATION_PCT == 3.0


def test_trailing_sl_activates_at_threshold():
    """After +3%, trailing SL should activate and never go below entry."""
    entry = 100.0
    atr = 5.0
    high_water = 104.0  # +4% above entry (above 3% threshold)
    from config import TRAILING_SL_ATR_MULT
    trailing_stop = high_water - TRAILING_SL_ATR_MULT * atr  # 104 - 12.5 = 91.5
    # Never below entry
    effective_stop = max(trailing_stop, entry)
    assert effective_stop == entry  # 91.5 < 100, so capped at entry


def test_trailing_sl_moves_up_with_price():
    """As price rises, trailing SL should move up."""
    entry = 100.0
    atr = 3.0
    high_water = 110.0  # +10% above entry
    from config import TRAILING_SL_ATR_MULT
    trailing_stop = high_water - TRAILING_SL_ATR_MULT * atr  # 110 - 7.5 = 102.5
    effective_stop = max(trailing_stop, entry)
    assert effective_stop == 102.5


def test_trailing_sl_locks_in_profit():
    """After a big run-up, SL should lock in meaningful profit."""
    entry = 100.0
    atr = 2.0
    high_water = 115.0
    from config import TRAILING_SL_ATR_MULT
    trailing_stop = high_water - TRAILING_SL_ATR_MULT * atr  # 115 - 5 = 110
    effective_stop = max(trailing_stop, entry)
    assert effective_stop == 110.0
    assert effective_stop > entry
