"""Tests for indicators_v4.py — MFI, OBV divergence, VWAP deviation, ADX, PCR, max pain, FII."""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indicators_v4 import (
    compute_adx,
    compute_fii_momentum,
    compute_maxpain_signal,
    compute_mfi,
    compute_obv_divergence,
    compute_pcr_signal,
    compute_vwap_deviation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ohlcv(
    n: int = 60,
    base: float = 100.0,
    trend: float = 0.5,
    vol: float = 2.0,
    base_volume: int = 100_000,
) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with controllable trend."""
    dates = pd.bdate_range("2025-01-01", periods=n)
    closes = [base + trend * i for i in range(n)]
    highs = [c + vol * 0.6 for c in closes]
    lows = [c - vol * 0.4 for c in closes]
    opens = [c - trend * 0.3 for c in closes]
    volumes = [base_volume] * n
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=dates,
    )


def make_divergence_df(price_trend: str, obv_trend: str, n: int = 30) -> pd.DataFrame:
    """Build a DataFrame where price and OBV trends can diverge.

    price_trend / obv_trend: 'up' or 'down'
    """
    dates = pd.bdate_range("2025-01-01", periods=n)
    if price_trend == "up":
        closes = [100 + i * 0.5 for i in range(n)]
    else:
        closes = [100 - i * 0.5 for i in range(n)]

    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    opens = closes[:]

    # Volume controls OBV direction: big volume on up/down days
    volumes = []
    for i in range(n):
        if i == 0:
            volumes.append(100_000)
            continue
        if obv_trend == "up":
            # Make close > prev_close so OBV adds volume
            closes[i] = closes[i - 1] + 0.01 if price_trend == "down" else closes[i]
            volumes.append(200_000)
        else:
            # Make close < prev_close so OBV subtracts volume
            closes[i] = closes[i - 1] - 0.01 if price_trend == "up" else closes[i]
            volumes.append(200_000)

    # Re-fix highs/lows after close adjustments
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=dates,
    )


# ===========================================================================
# 1. compute_mfi
# ===========================================================================

class TestMFI:
    def test_returns_series(self):
        df = make_ohlcv(30)
        result = compute_mfi(df, period=14)
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)

    def test_nan_warmup(self):
        df = make_ohlcv(30)
        result = compute_mfi(df, period=14)
        # First `period` bars should be NaN
        assert result.iloc[:14].isna().all()
        # After warmup, should have values
        assert result.iloc[14:].notna().all()

    def test_range_0_100(self):
        df = make_ohlcv(60, trend=0.5)
        result = compute_mfi(df, period=14)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_uptrend_high_mfi(self):
        df = make_ohlcv(60, trend=1.0)
        result = compute_mfi(df, period=14)
        # Strong uptrend should give high MFI
        assert result.iloc[-1] > 50

    def test_downtrend_low_mfi(self):
        df = make_ohlcv(60, trend=-1.0)
        result = compute_mfi(df, period=14)
        assert result.iloc[-1] < 50

    def test_insufficient_data(self):
        df = make_ohlcv(10)
        result = compute_mfi(df, period=14)
        assert result.isna().all()

    def test_custom_period(self):
        df = make_ohlcv(30)
        result = compute_mfi(df, period=5)
        assert result.iloc[:5].isna().all()
        assert result.iloc[5:].notna().all()


# ===========================================================================
# 2. compute_obv_divergence
# ===========================================================================

class TestOBVDivergence:
    def test_returns_series_of_strings(self):
        df = make_ohlcv(30)
        result = compute_obv_divergence(df, lookback=10)
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)
        valid_vals = {"bullish_div", "bearish_div", "none"}
        assert set(result.dropna().unique()).issubset(valid_vals)

    def test_warmup_none(self):
        df = make_ohlcv(30)
        result = compute_obv_divergence(df, lookback=10)
        # First lookback bars can't look back enough
        assert (result.iloc[:10] == "none").all()

    def test_no_divergence_aligned_trends(self):
        """When price and OBV both go up, no divergence."""
        df = make_ohlcv(40, trend=1.0)
        result = compute_obv_divergence(df, lookback=10)
        # Most should be 'none' when trends align
        assert (result.iloc[-5:] == "none").all() or (result.iloc[-5:] == "none").sum() >= 3

    def test_bearish_divergence_detection(self):
        """Price makes new high but OBV doesn't -> bearish."""
        n = 25
        dates = pd.bdate_range("2025-01-01", periods=n)
        # Price rises throughout
        closes = [100 + i * 0.5 for i in range(n)]
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        opens = closes[:]
        # Volume declining (OBV won't keep up even on up days)
        volumes = [200_000 - i * 8000 for i in range(n)]
        # Force last few closes down slightly so OBV loses
        for i in range(n - 5, n):
            closes[i] = closes[i - 1] - 0.01  # tiny down moves kill OBV
            volumes[i] = 500_000  # big volume on down moves
        # But price still near highs overall
        closes[-1] = max(closes) + 0.1
        highs = [max(c + 1, c) for c in closes]
        lows = [min(c - 1, c) for c in closes]

        df = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
            index=dates,
        )
        result = compute_obv_divergence(df, lookback=10)
        # Should detect at least one bearish divergence
        assert "bearish_div" in result.values

    def test_bullish_divergence_detection(self):
        """Price makes new low but OBV doesn't -> bullish."""
        n = 25
        dates = pd.bdate_range("2025-01-01", periods=n)
        # Price falling
        closes = [100 - i * 0.5 for i in range(n)]
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        opens = closes[:]
        # But big volume on up days in latter half
        volumes = [100_000] * n
        for i in range(n - 5, n):
            closes[i] = closes[i - 1] + 0.01  # tiny up moves
            volumes[i] = 500_000  # big volume adds to OBV
        # Price still near lows
        closes[-1] = min(closes[:n - 5]) - 0.1
        highs = [max(c + 1, c) for c in closes]
        lows = [min(c - 1, c) for c in closes]

        df = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
            index=dates,
        )
        result = compute_obv_divergence(df, lookback=10)
        assert "bullish_div" in result.values


# ===========================================================================
# 3. compute_vwap_deviation
# ===========================================================================

class TestVWAPDeviation:
    def test_returns_series(self):
        df = make_ohlcv(30)
        result = compute_vwap_deviation(df, atr_period=14)
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)

    def test_nan_warmup(self):
        df = make_ohlcv(30)
        result = compute_vwap_deviation(df, atr_period=14)
        assert result.iloc[:14].isna().all()
        assert result.iloc[14:].notna().all()

    def test_positive_above_vwap(self):
        """In uptrend, close should be above VWAP -> positive deviation."""
        df = make_ohlcv(60, trend=2.0)
        result = compute_vwap_deviation(df, atr_period=14)
        # Late bars should be above VWAP
        assert result.iloc[-1] > 0

    def test_insufficient_data(self):
        df = make_ohlcv(10)
        result = compute_vwap_deviation(df, atr_period=14)
        assert result.isna().all()


# ===========================================================================
# 4. compute_adx
# ===========================================================================

class TestADX:
    def test_returns_dict_of_series(self):
        df = make_ohlcv(60)
        result = compute_adx(df, period=14)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"adx", "di_plus", "di_minus"}
        for v in result.values():
            assert isinstance(v, pd.Series)
            assert len(v) == len(df)

    def test_adx_nan_warmup(self):
        df = make_ohlcv(60)
        result = compute_adx(df, period=14)
        # ADX needs 2*period warmup
        assert result["adx"].iloc[: 2 * 14].isna().all()
        # DI needs period warmup
        assert result["di_plus"].iloc[:14].isna().all()
        assert result["di_minus"].iloc[:14].isna().all()

    def test_adx_range(self):
        df = make_ohlcv(60, trend=1.0)
        result = compute_adx(df, period=14)
        valid_adx = result["adx"].dropna()
        assert (valid_adx >= 0).all()
        assert (valid_adx <= 100).all()

    def test_strong_trend_high_adx(self):
        """Strong trend should yield high ADX."""
        df = make_ohlcv(80, trend=3.0, vol=1.0)
        result = compute_adx(df, period=14)
        # ADX > 25 indicates strong trend
        assert result["adx"].iloc[-1] > 20

    def test_di_plus_gt_di_minus_in_uptrend(self):
        df = make_ohlcv(80, trend=2.0, vol=0.5)
        result = compute_adx(df, period=14)
        assert result["di_plus"].iloc[-1] > result["di_minus"].iloc[-1]

    def test_di_minus_gt_di_plus_in_downtrend(self):
        df = make_ohlcv(80, trend=-2.0, vol=0.5)
        result = compute_adx(df, period=14)
        assert result["di_minus"].iloc[-1] > result["di_plus"].iloc[-1]


# ===========================================================================
# 5. compute_pcr_signal
# ===========================================================================

class TestPCRSignal:
    def test_none_returns_zero(self):
        assert compute_pcr_signal(None) == 0.0

    def test_high_pcr_bullish(self):
        """PCR > 1.3 -> bullish (positive)."""
        sig = compute_pcr_signal(1.5)
        assert sig > 0
        assert sig <= 1.0

    def test_low_pcr_bearish(self):
        """PCR < 0.7 -> bearish (negative)."""
        sig = compute_pcr_signal(0.5)
        assert sig < 0
        assert sig >= -1.0

    def test_neutral_pcr(self):
        """PCR = 1.0 (middle) -> near zero."""
        sig = compute_pcr_signal(1.0)
        assert abs(sig) < 0.5

    def test_extreme_pcr(self):
        """Very high PCR -> clamp at 1.0."""
        sig = compute_pcr_signal(2.0)
        assert sig <= 1.0

    def test_very_low_pcr(self):
        sig = compute_pcr_signal(0.3)
        assert sig >= -1.0

    def test_boundary_1_3(self):
        """At exactly 1.3, should be at upper boundary of interpolation."""
        sig = compute_pcr_signal(1.3)
        assert sig > 0.8

    def test_boundary_0_7(self):
        sig = compute_pcr_signal(0.7)
        assert sig <= -0.9  # At 0.7 boundary, should be strongly bearish


# ===========================================================================
# 6. compute_maxpain_signal
# ===========================================================================

class TestMaxPainSignal:
    def test_none_returns_zero(self):
        assert compute_maxpain_signal(100.0, None) == 0.0

    def test_price_at_maxpain(self):
        """Within 1% -> strong signal ~0.8."""
        sig = compute_maxpain_signal(100.0, 100.5)
        assert abs(sig) >= 0.5

    def test_price_well_above_maxpain(self):
        """Price above max pain -> bearish (negative)."""
        sig = compute_maxpain_signal(110.0, 100.0)
        assert sig < 0

    def test_price_well_below_maxpain(self):
        """Price below max pain -> bullish (positive)."""
        sig = compute_maxpain_signal(90.0, 100.0)
        assert sig > 0

    def test_far_from_maxpain(self):
        """More than 3% away -> weak signal ~0.1."""
        sig = compute_maxpain_signal(110.0, 100.0)
        assert abs(sig) <= 0.3

    def test_range(self):
        """Should always be in [-1, 1]."""
        for price in [50, 90, 100, 110, 200]:
            sig = compute_maxpain_signal(price, 100.0)
            assert -1.0 <= sig <= 1.0


# ===========================================================================
# 7. compute_fii_momentum
# ===========================================================================

class TestFIIMomentum:
    def test_none_returns_zero(self):
        assert compute_fii_momentum(None) == 0.0

    def test_empty_returns_zero(self):
        assert compute_fii_momentum([]) == 0.0

    def test_positive_flows(self):
        sig = compute_fii_momentum([1000, 1500, 2000])
        assert sig > 0
        assert sig <= 1.0

    def test_negative_flows(self):
        sig = compute_fii_momentum([-1000, -1500, -2000])
        assert sig < 0
        assert sig >= -1.0

    def test_extreme_positive(self):
        """3000+ Cr average -> clamp at 1.0."""
        sig = compute_fii_momentum([5000, 5000, 5000])
        assert sig == 1.0

    def test_extreme_negative(self):
        sig = compute_fii_momentum([-5000, -5000, -5000])
        assert sig == -1.0

    def test_mixed_flows(self):
        sig = compute_fii_momentum([1000, -1000, 500])
        # Average is ~166, normalized by 3000 -> small positive
        assert 0 < sig < 0.5

    def test_range_clamped(self):
        for flows in [[10000, 10000, 10000], [-10000, -10000, -10000], [0, 0, 0]]:
            sig = compute_fii_momentum(flows)
            assert -1.0 <= sig <= 1.0
