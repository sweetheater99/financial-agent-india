import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np
from datetime import date


# Helper to create a simple Nifty DataFrame
def _make_nifty_df(n=60, base=22000, trend="flat", volatility=50):
    """Generate synthetic Nifty daily candles."""
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=n, freq="B")
    closes = np.full(n, base, dtype=float)

    if trend == "up":
        closes = base + np.cumsum(np.random.uniform(10, 30, n))
    elif trend == "down":
        closes = base - np.cumsum(np.random.uniform(10, 30, n))
    elif trend == "flat":
        closes = base + np.cumsum(np.random.uniform(-10, 10, n))
    elif trend == "volatile":
        closes = base + np.cumsum(np.random.uniform(-100, 100, n))

    highs = closes + np.random.uniform(20, volatility, n)
    lows = closes - np.random.uniform(20, volatility, n)
    opens = closes + np.random.uniform(-20, 20, n)

    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes
    }, index=dates)


def test_cash_regime_on_monthly_drawdown():
    from regime import classify_regime
    df = _make_nifty_df(60)
    risk_state = {"monthly_drawdown_pct": -9.0, "consecutive_losses": 2}
    result = classify_regime(df, vix=16, risk_state=risk_state)
    assert result["regime"] == "CASH"
    assert result["confidence"] == 1.0


def test_cash_regime_on_consecutive_losses():
    from regime import classify_regime
    df = _make_nifty_df(60)
    risk_state = {"monthly_drawdown_pct": -2.0, "consecutive_losses": 5}
    result = classify_regime(df, vix=16, risk_state=risk_state)
    assert result["regime"] == "CASH"


def test_cash_regime_on_high_vix():
    from regime import classify_regime
    df = _make_nifty_df(60)
    result = classify_regime(df, vix=26)
    assert result["regime"] == "CASH"


def test_volatile_regime_high_vix():
    from regime import classify_regime
    df = _make_nifty_df(60, volatility=30)
    result = classify_regime(df, vix=21)
    # VIX > 20 but < 25 (not CASH), should be VOLATILE
    assert result["regime"] == "VOLATILE"


def test_uncertain_regime_default():
    from regime import classify_regime
    # With insufficient data (< 30 rows), should default to UNCERTAIN
    df = _make_nifty_df(10)
    result = classify_regime(df, vix=15)
    assert result["regime"] == "UNCERTAIN"


def test_vix_to_iv_percentile_normal():
    from regime import vix_to_iv_percentile
    assert vix_to_iv_percentile(15) == 50
    assert vix_to_iv_percentile(10) == 10
    assert vix_to_iv_percentile(25) == 90


def test_vix_to_iv_percentile_boundaries():
    from regime import vix_to_iv_percentile
    assert vix_to_iv_percentile(12) == 25  # exactly at boundary
    assert vix_to_iv_percentile(18) == 70
    assert vix_to_iv_percentile(22) == 90


def test_classify_regime_returns_required_keys():
    from regime import classify_regime
    df = _make_nifty_df(60)
    result = classify_regime(df, vix=15)
    assert "regime" in result
    assert "confidence" in result
    assert "reason" in result
    assert "iv_percentile" in result


def test_classify_regime_valid_regimes():
    from regime import classify_regime
    df = _make_nifty_df(60)
    result = classify_regime(df, vix=15)
    valid = {"CASH", "TRENDING_UP", "TRENDING_DOWN", "SIDEWAYS", "VOLATILE", "UNCERTAIN"}
    assert result["regime"] in valid


def test_classify_regime_no_risk_state():
    from regime import classify_regime
    df = _make_nifty_df(60)
    # Should work without risk_state (default None)
    result = classify_regime(df, vix=15)
    assert result["regime"] in {"TRENDING_UP", "TRENDING_DOWN", "SIDEWAYS", "VOLATILE", "UNCERTAIN"}
