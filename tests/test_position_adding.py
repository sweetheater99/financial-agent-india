import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_add_action_increases_position():
    """ADD action increases position quantity within safety limits."""
    from paper_trade import _handle_position_add
    pos = {
        "symbol": "MCX", "direction": "bullish", "instrument": "EQ",
        "entry_price": 100, "quantity": 10, "allocated": 1000,
        "atr_at_entry": 3.0, "_add_count": 0,
    }
    portfolio = {"available_capital": 50000, "capital": 100000, "positions": [pos]}
    result = _handle_position_add(pos, current_ltp=106, portfolio=portfolio)
    assert result is True
    assert pos["quantity"] > 10
    assert pos["_add_count"] == 1

def test_add_blocked_if_losing():
    """ADD blocked if position is in loss."""
    from paper_trade import _handle_position_add
    pos = {
        "symbol": "MCX", "direction": "bullish", "instrument": "EQ",
        "entry_price": 100, "quantity": 10, "allocated": 1000,
        "atr_at_entry": 3.0, "_add_count": 0,
    }
    portfolio = {"available_capital": 50000, "capital": 100000, "positions": [pos]}
    result = _handle_position_add(pos, current_ltp=98, portfolio=portfolio)
    assert result is False
    assert pos["quantity"] == 10

def test_add_blocked_if_already_added():
    """Maximum 1 add per position."""
    from paper_trade import _handle_position_add
    pos = {
        "symbol": "MCX", "direction": "bullish", "instrument": "EQ",
        "entry_price": 100, "quantity": 15, "allocated": 1500,
        "atr_at_entry": 3.0, "_add_count": 1,
    }
    portfolio = {"available_capital": 50000, "capital": 100000, "positions": [pos]}
    result = _handle_position_add(pos, current_ltp=108, portfolio=portfolio)
    assert result is False

def test_add_blocked_if_insufficient_capital():
    """ADD blocked when not enough capital."""
    from paper_trade import _handle_position_add
    pos = {
        "symbol": "MCX", "direction": "bullish", "instrument": "EQ",
        "entry_price": 100, "quantity": 10, "allocated": 1000,
        "atr_at_entry": 3.0, "_add_count": 0,
    }
    portfolio = {"available_capital": 100, "capital": 100000, "positions": [pos]}
    result = _handle_position_add(pos, current_ltp=106, portfolio=portfolio)
    assert result is False

def test_add_bearish_position():
    """ADD works for bearish positions when profitable."""
    from paper_trade import _handle_position_add
    pos = {
        "symbol": "MCX", "direction": "bearish", "instrument": "EQ",
        "entry_price": 100, "quantity": 10, "allocated": 1000,
        "atr_at_entry": 3.0, "_add_count": 0,
    }
    portfolio = {"available_capital": 50000, "capital": 100000, "positions": [pos]}
    result = _handle_position_add(pos, current_ltp=95, portfolio=portfolio)
    assert result is True
    assert pos["_add_count"] == 1
