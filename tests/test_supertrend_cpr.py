"""Tests for Supertrend and CPR indicators."""
import pandas as pd
import numpy as np


def _make_ohlc(n=30, base=24000, volatility=100):
    """Create synthetic OHLC data."""
    np.random.seed(42)
    closes = base + np.cumsum(np.random.randn(n) * volatility)
    highs = closes + abs(np.random.randn(n) * volatility * 0.5)
    lows = closes - abs(np.random.randn(n) * volatility * 0.5)
    opens = closes + np.random.randn(n) * volatility * 0.3
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
    })


class TestSupertrend:
    def test_returns_buy_or_sell(self):
        from indicators_v3 import compute_supertrend
        df = _make_ohlc(30)
        signal = compute_supertrend(df, period=10, multiplier=3)
        assert signal in ("buy", "sell")

    def test_uptrending_data(self):
        from indicators_v3 import compute_supertrend
        closes = [24000 + i * 50 for i in range(30)]
        df = pd.DataFrame({
            "open": [c - 10 for c in closes],
            "high": [c + 30 for c in closes],
            "low": [c - 30 for c in closes],
            "close": closes,
        })
        assert compute_supertrend(df) == "buy"

    def test_downtrending_data(self):
        from indicators_v3 import compute_supertrend
        closes = [26000 - i * 50 for i in range(30)]
        df = pd.DataFrame({
            "open": [c + 10 for c in closes],
            "high": [c + 30 for c in closes],
            "low": [c - 30 for c in closes],
            "close": closes,
        })
        assert compute_supertrend(df) == "sell"

    def test_insufficient_data(self):
        from indicators_v3 import compute_supertrend
        df = _make_ohlc(5)
        signal = compute_supertrend(df, period=10, multiplier=3)
        assert signal in ("buy", "sell", "unknown")


class TestCPR:
    def test_narrow_cpr(self):
        from indicators_v3 import compute_cpr
        result = compute_cpr(prev_high=24100, prev_low=24050, prev_close=24080)
        assert result["cpr_width_pct"] < 0.3
        assert result["day_type"] == "trending"

    def test_wide_cpr(self):
        from indicators_v3 import compute_cpr
        result = compute_cpr(prev_high=25000, prev_low=24000, prev_close=24200)
        assert result["cpr_width_pct"] > 0.8
        assert result["day_type"] == "sideways"

    def test_normal_cpr(self):
        from indicators_v3 import compute_cpr
        result = compute_cpr(prev_high=24600, prev_low=24000, prev_close=24100)
        assert result["day_type"] == "normal"

    def test_cpr_values(self):
        from indicators_v3 import compute_cpr
        result = compute_cpr(prev_high=24300, prev_low=24100, prev_close=24200)
        assert "pivot" in result
        assert "tc" in result
        assert "bc" in result
        assert result["tc"] >= result["bc"]
