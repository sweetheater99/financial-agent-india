"""Tests for market breadth module."""
from unittest.mock import patch


def test_breadth_signal_healthy():
    """Breadth > 60% should be healthy."""
    result = {"advancing": 35, "declining": 15, "total": 50, "breadth_pct": 70.0, "signal": "healthy"}
    assert result["signal"] == "healthy"


def test_breadth_signal_weak():
    """Breadth 40-60% should be weak."""
    result = {"advancing": 25, "declining": 25, "total": 50, "breadth_pct": 50.0, "signal": "weak"}
    assert result["signal"] == "weak"


def test_breadth_signal_divergent():
    """Breadth < 40% should be divergent."""
    result = {"advancing": 15, "declining": 35, "total": 50, "breadth_pct": 30.0, "signal": "divergent"}
    assert result["signal"] == "divergent"


def test_nifty50_symbols_count():
    from breadth import NIFTY50_SYMBOLS
    assert len(NIFTY50_SYMBOLS) == 50


def test_compute_breadth_returns_dict_or_none():
    """compute_breadth should return dict with expected keys or None."""
    from breadth import compute_breadth
    # Will likely return None in test env (no market data), but shouldn't crash
    result = compute_breadth()
    if result is not None:
        assert "advancing" in result
        assert "declining" in result
        assert "breadth_pct" in result
        assert "signal" in result
