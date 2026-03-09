import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_compute_exit_signals_basic():
    """_compute_exit_signals returns expected signal structure."""
    from paper_trade import _compute_exit_signals
    pos = {
        "instrument": "EQ", "direction": "bullish",
        "entry_price": 100, "target_price": 105,
        "stoploss_price": 97, "atr_at_entry": 2.0,
        "peak_price": 104, "entry_date": "2026-03-01",
        "max_hold_date": "2026-03-15",
    }
    signals = _compute_exit_signals(pos, current_ltp=102, pnl_pct=2.0)
    assert "mechanical_sl" in signals
    assert "trailing_stop" in signals
    assert "target" in signals
    assert "time_pressure" in signals
    assert isinstance(signals["mechanical_sl"], dict)
    assert "fired" in signals["mechanical_sl"]

def test_mechanical_sl_fires():
    """Mechanical SL fires when price hits stoploss."""
    from paper_trade import _compute_exit_signals
    pos = {
        "instrument": "EQ", "direction": "bullish",
        "entry_price": 100, "stoploss_price": 97,
        "atr_at_entry": 2.0, "peak_price": 100,
        "target_price": 105, "entry_date": "2026-03-01",
        "max_hold_date": "2026-03-15",
    }
    signals = _compute_exit_signals(pos, current_ltp=96.5, pnl_pct=-3.5)
    assert signals["mechanical_sl"]["fired"] is True

def test_target_signal_fires():
    """Target signal fires when price hits target."""
    from paper_trade import _compute_exit_signals
    pos = {
        "instrument": "EQ", "direction": "bullish",
        "entry_price": 100, "target_price": 105,
        "stoploss_price": 97, "atr_at_entry": 2.0,
        "peak_price": 106, "entry_date": "2026-03-01",
        "max_hold_date": "2026-03-15",
    }
    signals = _compute_exit_signals(pos, current_ltp=106, pnl_pct=6.0)
    assert signals["target"]["fired"] is True

def test_format_signals():
    """_format_signals_for_claude produces readable text."""
    from paper_trade import _format_signals_for_claude
    signals = {
        "mechanical_sl": {"fired": False, "sl_price": 97, "distance_pct": 5.0},
        "trailing_stop": {"fired": True, "trail_price": 101.5, "distance": 0.5},
        "target": {"fired": False, "target_price": 105, "distance_pct": 3.0},
    }
    text = _format_signals_for_claude(signals)
    assert "trailing_stop" in text
    assert "FIRED" in text
    assert "mechanical_sl" not in text  # Should be excluded

def test_bearish_sl_fires():
    """Bearish SL fires when price goes up past SL."""
    from paper_trade import _compute_exit_signals
    pos = {
        "instrument": "EQ", "direction": "bearish",
        "entry_price": 100, "stoploss_price": 103,
        "atr_at_entry": 2.0, "peak_price": 98,
        "target_price": 95, "entry_date": "2026-03-01",
        "max_hold_date": "2026-03-15",
    }
    signals = _compute_exit_signals(pos, current_ltp=104, pnl_pct=-4.0)
    assert signals["mechanical_sl"]["fired"] is True
