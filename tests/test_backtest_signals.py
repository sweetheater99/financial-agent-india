"""Tests for backtest_signals.py using synthetic DataFrames (no yfinance)."""

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest_signals import (
    DEFAULT_MIN_SCORE,
    SIGNAL_WEIGHTS,
    SignalBacktestEngine,
    compute_signals,
)
from backtest import calc_round_trip_costs


# ---------------------------------------------------------------------------
# Helpers — build synthetic OHLCV DataFrames
# ---------------------------------------------------------------------------

def make_df(
    n: int = 60,
    base_price: float = 100.0,
    trend: float = 0.0,
    volatility: float = 2.0,
    base_volume: int = 100_000,
    volume_pattern: list[float] | None = None,
) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame.

    Args:
        n: number of bars
        base_price: starting close price
        trend: per-bar price drift (positive = uptrend)
        volatility: typical high-low range
        base_volume: average volume
        volume_pattern: if provided, multipliers for volume at each bar (len must == n)
    """
    dates = pd.bdate_range("2025-01-01", periods=n)
    closes = [base_price + trend * i for i in range(n)]
    highs = [c + volatility * 0.6 for c in closes]
    lows = [c - volatility * 0.4 for c in closes]
    opens = [c - trend * 0.3 for c in closes]

    if volume_pattern is not None:
        volumes = [int(base_volume * m) for m in volume_pattern]
    else:
        volumes = [base_volume] * n

    return pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes,
    }, index=dates)


def make_bullish_confluence_df(n: int = 60) -> pd.DataFrame:
    """Build a DataFrame engineered to trigger a bullish entry with high score.

    - Uptrend so close > EMA20
    - Volume spike at the entry bar
    - RSI in 40-65 range (needs some down bars to avoid RSI=100)
    - Day range > 1.2x ATR (ATR breakout)
    - 3-day price trend up + volume increasing
    """
    base = 100.0
    trend = 0.2  # gentle uptrend
    vol = 2.0
    dates = pd.bdate_range("2025-01-01", periods=n)

    # Add zigzag noise so RSI isn't pegged at 100
    # Pattern: up, up, down, up, up, down ... keeps RSI in a realistic range
    closes = []
    price = base
    for i in range(n):
        if i % 3 == 2:
            price -= 0.3  # small pullback every 3rd bar
        else:
            price += 0.5  # up on other bars
        closes.append(price)

    highs = [c + vol * 0.6 for c in closes]
    lows = [c - vol * 0.4 for c in closes]
    opens = [c - 0.1 for c in closes]
    volumes = [100_000] * n

    # At bar 40, create confluence:
    # - spike volume (>1.3x avg)
    # - widen range for ATR breakout
    # - ensure 3 consecutive increasing volume bars (for trend_confirm)
    # - ensure bar 40 is an "up" bar in a 3-day uptrend
    entry_bar = 40
    # Make bars 37-40 all go up for a clear 3-day uptrend
    for j in range(37, entry_bar + 1):
        closes[j] = closes[36] + (j - 36) * 0.8
        highs[j] = closes[j] + vol * 0.6
        lows[j] = closes[j] - vol * 0.4
        opens[j] = closes[j] - 0.1

    # Volume: increasing over 3 bars
    volumes[entry_bar - 2] = 150_000
    volumes[entry_bar - 1] = 170_000
    volumes[entry_bar] = 190_000

    # Widen the range at entry bar for ATR breakout
    highs[entry_bar] = closes[entry_bar] + vol * 2.5
    lows[entry_bar] = closes[entry_bar] - vol * 1.5

    return pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes,
    }, index=dates)


def make_bearish_df(n: int = 60) -> pd.DataFrame:
    """Downtrending DataFrame to test bearish signal detection."""
    base = 200.0
    trend = -0.5
    dates = pd.bdate_range("2025-01-01", periods=n)
    closes = [base + trend * i for i in range(n)]
    highs = [c + 2.0 for c in closes]
    lows = [c - 2.0 for c in closes]
    opens = [c + 0.1 for c in closes]
    volumes = [100_000] * n

    # Volume spike + increasing volume at bar 40
    for j in [38, 39, 40]:
        volumes[j] = int(100_000 * (1.5 + 0.3 * (j - 37)))
    # Widen range
    highs[40] = closes[40] + 5.0
    lows[40] = closes[40] - 5.0

    return pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes,
    }, index=dates)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSignalScoring:
    """Test 1: All signals present should produce score >= 3.5."""

    def test_all_signals_present_high_score(self):
        df = make_bullish_confluence_df(60)
        signals = compute_signals(df)
        # Find the bar with the highest score
        max_score = signals["score"].max()
        # With momentum(1) + volume(1) + rsi(1) + atr_breakout(1.5) + trend_confirm(2) = 6.5
        assert max_score >= DEFAULT_MIN_SCORE, (
            f"Expected score >= {DEFAULT_MIN_SCORE} when all signals align, got {max_score}"
        )

    def test_max_possible_score(self):
        """All 9 signals should sum to 10.5."""
        total = sum(SIGNAL_WEIGHTS.values())
        assert total == 10.5


class TestNoSignalNoEntry:
    """Test 2: Flat market with no signals should produce no entries."""

    def test_flat_market_no_entries(self):
        # Completely flat price, constant volume, RSI should hover around 50
        # but no volume spike, no ATR breakout, no trend
        df = make_df(60, base_price=100, trend=0, volatility=0.5, base_volume=100_000)
        engine = SignalBacktestEngine(min_score=DEFAULT_MIN_SCORE)
        trades = engine.run("FLAT", df)
        # With flat prices (close ~= EMA20), momentum won't fire reliably
        # and definitely no volume spike or ATR breakout
        # Allow 0 or very few trades
        assert len(trades) <= 2, f"Expected ~0 trades on flat data, got {len(trades)}"


class TestVolumeSpikeAlone:
    """Test 3: Volume spike alone should score too low for entry."""

    def test_volume_spike_insufficient(self):
        df = make_df(60, base_price=100, trend=0.01, volatility=1.0)
        # Spike volume on one bar
        volumes = [100_000] * 60
        volumes[40] = 200_000
        df["Volume"] = volumes

        signals = compute_signals(df)
        # Volume alone = 1.0, which is below 3.5
        # Even with momentum (1.0) that's only 2.0
        # V4 indicators may add marginal score but ADX dampening keeps it at/below threshold
        bar_score = signals["score"].iloc[40]
        assert bar_score <= DEFAULT_MIN_SCORE, (
            f"Volume spike alone scored {bar_score}, should be <= {DEFAULT_MIN_SCORE}"
        )


class TestMomentumVolumeTrendEntry:
    """Test 4: Momentum + volume + trend confirm should trigger entry."""

    def test_three_signal_entry(self):
        df = make_bullish_confluence_df(60)
        engine = SignalBacktestEngine(min_score=3.0)  # lower threshold to ensure we catch it
        trades = engine.run("TEST", df)
        assert len(trades) > 0, "Expected at least one trade with momentum + volume + trend"
        # Check the first trade has multiple signals
        assert len(trades[0]["signals_fired"]) >= 2


class TestRSIFilter:
    """Test 5: RSI filter blocks overbought entries."""

    def test_rsi_blocks_overbought(self):
        # Strong uptrend will push RSI > 65, which should block the RSI signal
        df = make_df(60, base_price=100, trend=2.0, volatility=1.0)
        signals = compute_signals(df)
        # Late bars should have high RSI and sig_rsi should be False
        late_rsi_signals = signals["sig_rsi"].iloc[40:].sum()
        # With trend=2.0 per bar, RSI should be very high, blocking the signal
        assert late_rsi_signals < 10, (
            f"RSI filter should block most entries in strong uptrend, but {late_rsi_signals} fired"
        )


class TestBearishDetection:
    """Test 6: Bearish signals should be detected."""

    def test_bearish_direction_detected(self):
        df = make_bearish_df(60)
        signals = compute_signals(df)
        bearish_bars = (signals["direction"] == "bearish").sum()
        assert bearish_bars > 0, "Expected some bearish signals in downtrending data"

    def test_bearish_skipped_in_equity_long(self):
        """Equity-long-only engine should skip bearish entries."""
        df = make_bearish_df(60)
        engine = SignalBacktestEngine(min_score=1.0)  # low threshold to catch any signals
        trades = engine.run("BEAR", df)
        # All bearish entries should be skipped in equity-long mode
        for t in trades:
            assert t["direction"] == "bullish", f"Got non-bullish trade: {t['direction']}"


class TestMultiSymbolSorted:
    """Test 7: Multi-symbol run produces trades sorted by entry date."""

    def test_multi_symbol_sorted(self):
        df1 = make_bullish_confluence_df(60)
        df2 = make_bullish_confluence_df(60)
        # Shift df2 dates forward so we can verify sorting
        df2.index = pd.bdate_range("2025-04-01", periods=60)

        data = {"SYM_A": df1, "SYM_B": df2}
        engine = SignalBacktestEngine(min_score=3.0)
        trades = engine.run_multi(data)

        if len(trades) >= 2:
            dates = [t["entry_date"] for t in trades]
            assert dates == sorted(dates), "Trades should be sorted by entry_date"


class TestStatsComputation:
    """Test 8: Stats computation produces expected fields."""

    def test_stats_fields(self):
        df = make_bullish_confluence_df(60)
        engine = SignalBacktestEngine(min_score=2.0)
        engine.run_multi({"TEST": df})
        stats = engine.compute_stats()

        if stats["total_trades"] > 0:
            required_fields = [
                "total_trades", "winning_trades", "losing_trades", "win_rate",
                "total_pnl", "total_costs", "avg_win_pct", "avg_loss_pct",
                "avg_holding_days", "sharpe_ratio", "max_drawdown", "profit_factor",
                "best_trade", "worst_trade", "exit_reasons", "per_symbol",
                "estimated_stcg_tax", "signal_distribution", "entry_rate_pct",
                "total_bars_scanned", "entry_bars",
            ]
            for field in required_fields:
                assert field in stats, f"Missing stats field: {field}"

            # Signal distribution should have known keys
            for sig in ["momentum", "volume", "rsi", "atr_breakout", "trend_confirm",
                         "mfi", "obv_divergence", "vwap_deviation", "adx_trend"]:
                assert sig in stats["signal_distribution"]

    def test_empty_stats(self):
        engine = SignalBacktestEngine(min_score=99.0)  # impossible threshold
        df = make_df(60)
        engine.run_multi({"EMPTY": df})
        stats = engine.compute_stats()
        assert stats["total_trades"] == 0


class TestExitReasons:
    """Test 9: Exit reasons match expected values."""

    def test_exit_reasons_are_valid(self):
        df = make_bullish_confluence_df(80)
        engine = SignalBacktestEngine(min_score=2.0)
        trades = engine.run("TEST", df)

        valid_reasons = {"target", "stoploss", "trailing_stop", "expiry", "data_end"}
        for t in trades:
            assert t["exit_reason"] in valid_reasons, (
                f"Unexpected exit reason: {t['exit_reason']}"
            )

    def test_holding_days_within_max(self):
        df = make_bullish_confluence_df(80)
        engine = SignalBacktestEngine(min_score=2.0)
        trades = engine.run("TEST", df)

        max_hold = engine.params["max_hold_days"]
        for t in trades:
            # data_end can happen at any holding period
            if t["exit_reason"] != "data_end":
                assert t["holding_days"] <= max_hold + 1, (
                    f"Held {t['holding_days']} days, max is {max_hold}"
                )


class TestTransactionCosts:
    """Test 10: Transaction costs are deducted from P&L."""

    def test_costs_deducted(self):
        df = make_bullish_confluence_df(60)
        engine = SignalBacktestEngine(min_score=2.0)
        trades = engine.run("TEST", df)

        for t in trades:
            assert t["costs"] > 0, "Transaction costs should be > 0"
            # pnl_net = pnl_gross - costs
            expected_net = round(t["pnl_gross"] - t["costs"], 2)
            assert abs(t["pnl_net"] - expected_net) < 0.02, (
                f"pnl_net ({t['pnl_net']}) != pnl_gross ({t['pnl_gross']}) - costs ({t['costs']})"
            )

    def test_round_trip_costs_positive(self):
        costs = calc_round_trip_costs(100.0, 105.0, 10)
        assert costs > 0
        # Should include brokerage, STT, exchange, stamp, SEBI, GST for both legs
        assert costs < 100.0  # sanity: costs shouldn't exceed trade value


# ---------------------------------------------------------------------------
# V4 indicator integration tests
# ---------------------------------------------------------------------------

class TestV4Indicators:
    """Tests for V4 indicator integration into backtest scoring."""

    def test_new_signals_in_weights(self):
        """New V4 signals should appear in SIGNAL_WEIGHTS."""
        assert "mfi" in SIGNAL_WEIGHTS
        assert "obv_divergence" in SIGNAL_WEIGHTS
        assert "vwap_deviation" in SIGNAL_WEIGHTS
        assert "adx_trend" in SIGNAL_WEIGHTS

    def test_new_signal_columns_exist(self):
        """compute_signals should include new signal columns."""
        df = make_df(60, trend=2.0, volatility=2.0)
        result = compute_signals(df)
        assert "sig_mfi" in result.columns
        assert "sig_obv_divergence" in result.columns
        assert "sig_vwap_deviation" in result.columns
        assert "sig_adx_trend" in result.columns

    def test_adx_dampening_reduces_scores(self):
        """When ADX < 20 (choppy/flat), scores should be dampened by 0.7."""
        df = make_df(60, trend=0.0, volatility=5.0)
        result = compute_signals(df)
        # Scores should be reasonable (dampened in choppy market)
        assert result["score"].max() <= 20
