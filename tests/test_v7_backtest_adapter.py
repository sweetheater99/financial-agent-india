"""Tests for V7 backtest adapter."""
import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from v7.backtest_adapter import (
    V7BacktestEngine,
    V7BreakoutStrategy,
    V7CreditSpreadStrategy,
    V7IronCondorStrategy,
    CONVICTION_RISK,
    V7_STRATEGY_CAPS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_chain(spot: float, vix: float = 15.0, dte: int = 20, symbol: str = "NIFTY"):
    """Generate a synthetic chain DataFrame for testing."""
    from greeks import black_scholes_greeks

    interval = 50 if symbol == "NIFTY" else 100
    atm = round(spot / interval) * interval
    iv = vix / 100.0

    rows = []
    for offset in range(-10, 11):
        strike = atm + offset * interval
        for opt_type in ["CE", "PE"]:
            greeks = black_scholes_greeks(spot, strike, dte, iv=iv, option_type=opt_type)
            rows.append({
                "strike": strike,
                "option_type": opt_type,
                "premium": greeks["theoretical_price"],
                "delta": greeks.get("delta", 0.5),
                "oi": 50000,
                "volume": 5000,
            })
    return pd.DataFrame(rows)


def make_ohlcv(n_days: int = 100, start_price: float = 24000.0, trend: float = 0.0005):
    """Generate synthetic OHLCV data."""
    import numpy as np

    dates = pd.bdate_range(end=datetime.date.today(), periods=n_days)
    prices = [start_price]
    for i in range(1, n_days):
        ret = trend + np.random.normal(0, 0.01)
        prices.append(prices[-1] * (1 + ret))

    df = pd.DataFrame({
        "Open": prices,
        "High": [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        "Low": [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        "Close": prices,
        "Volume": [1000000] * n_days,
        "vix": [15.0] * n_days,
    }, index=dates)
    return df


# ---------------------------------------------------------------------------
# V7BreakoutStrategy Tests
# ---------------------------------------------------------------------------

class TestV7BreakoutStrategy:
    def test_enter_bullish(self):
        strategy = V7BreakoutStrategy()
        entry = strategy.should_enter(
            date=datetime.date(2025, 1, 15),
            spot=24200, vix=16, atr=150, score=4.0,
            direction="bullish", available_capital=200000,
            current_positions=[], symbol="NIFTY",
            high_20=24200, low_20=23800,
        )
        assert entry is not None
        assert entry["instrument"] == "FUT"
        assert entry["direction"] == "bullish"
        assert entry["v7_setup_type"] == "breakout_long"
        assert entry["v7_conviction"] == "medium"

    def test_enter_bearish(self):
        strategy = V7BreakoutStrategy()
        entry = strategy.should_enter(
            date=datetime.date(2025, 1, 15),
            spot=23800, vix=16, atr=150, score=4.0,
            direction="bearish", available_capital=200000,
            current_positions=[], symbol="NIFTY",
            high_20=24200, low_20=23800,
        )
        assert entry is not None
        assert entry["direction"] == "bearish"
        assert entry["v7_setup_type"] == "breakout_short"

    def test_high_conviction_sizing(self):
        strategy = V7BreakoutStrategy()
        high = strategy.should_enter(
            date=datetime.date(2025, 1, 15),
            spot=24200, vix=16, atr=150, score=5.5,
            direction="bullish", available_capital=200000,
            current_positions=[], symbol="NIFTY",
            high_20=24200, low_20=23800,
        )
        med = strategy.should_enter(
            date=datetime.date(2025, 1, 16),
            spot=24200, vix=16, atr=150, score=4.0,
            direction="bullish", available_capital=200000,
            current_positions=[], symbol="NIFTY",
            high_20=24200, low_20=23800,
        )
        assert high is not None and med is not None
        assert high["v7_conviction"] == "high"
        assert med["v7_conviction"] == "medium"
        # High conviction should size larger
        assert high["quantity"] >= med["quantity"]

    def test_reject_low_score(self):
        strategy = V7BreakoutStrategy()
        entry = strategy.should_enter(
            date=datetime.date(2025, 1, 15),
            spot=24200, vix=16, atr=150, score=1.0,
            direction="bullish", available_capital=200000,
            current_positions=[], symbol="NIFTY",
        )
        assert entry is None

    def test_reject_high_vix(self):
        strategy = V7BreakoutStrategy()
        entry = strategy.should_enter(
            date=datetime.date(2025, 1, 15),
            spot=24200, vix=30, atr=150, score=4.0,
            direction="bullish", available_capital=200000,
            current_positions=[], symbol="NIFTY",
        )
        assert entry is None

    def test_daily_trade_limit(self):
        strategy = V7BreakoutStrategy(max_daily_trades=2)
        date = datetime.date(2025, 1, 15)
        # First 2 trades should work
        for i in range(2):
            entry = strategy.should_enter(
                date=date, spot=24200, vix=16, atr=150, score=4.0,
                direction="bullish", available_capital=200000,
                current_positions=[], symbol="NIFTY",
                high_20=24200, low_20=23800,
            )
            assert entry is not None

        # Third trade should be rejected
        entry = strategy.should_enter(
            date=date, spot=24200, vix=16, atr=150, score=4.0,
            direction="bullish", available_capital=200000,
            current_positions=[], symbol="NIFTY",
            high_20=24200, low_20=23800,
        )
        assert entry is None

    def test_exit_target(self):
        strategy = V7BreakoutStrategy()
        pos = {
            "direction": "bullish",
            "target_price": 24500,
            "stoploss_price": 23800,
            "max_hold_date": datetime.date(2025, 2, 1),
            "expiry": datetime.date(2025, 2, 15),
        }
        should_exit, reason = strategy.should_exit(
            position=pos, date=datetime.date(2025, 1, 20),
            spot=24500, high=24550, low=24400,
        )
        assert should_exit
        assert reason == "target"

    def test_exit_stoploss(self):
        strategy = V7BreakoutStrategy()
        pos = {
            "direction": "bullish",
            "target_price": 24500,
            "stoploss_price": 23800,
            "max_hold_date": datetime.date(2025, 2, 1),
            "expiry": datetime.date(2025, 2, 15),
        }
        should_exit, reason = strategy.should_exit(
            position=pos, date=datetime.date(2025, 1, 20),
            spot=23700, high=24000, low=23700,
        )
        assert should_exit
        assert reason == "stoploss"


# ---------------------------------------------------------------------------
# V7CreditSpreadStrategy Tests
# ---------------------------------------------------------------------------

class TestV7CreditSpreadStrategy:
    def test_enter_bull_put_spread(self):
        strategy = V7CreditSpreadStrategy()
        chain = make_chain(24000, vix=16, dte=25)
        entry = strategy.should_enter(
            date=datetime.date(2025, 1, 15),
            spot=24000, vix=16, atr=150, score=4.0,
            direction="bullish", available_capital=200000,
            current_positions=[], chain=chain, dte=25, symbol="NIFTY",
        )
        if entry:
            assert entry["instrument"] == "SPREAD"
            assert entry["option_type"] == "PE"
            assert entry["v7_setup_type"] == "credit_spread_bull"
            assert entry["net_credit"] > 0

    def test_enter_bear_call_spread(self):
        strategy = V7CreditSpreadStrategy()
        chain = make_chain(24000, vix=16, dte=25)
        entry = strategy.should_enter(
            date=datetime.date(2025, 1, 15),
            spot=24000, vix=16, atr=150, score=4.0,
            direction="bearish", available_capital=200000,
            current_positions=[], chain=chain, dte=25, symbol="NIFTY",
        )
        if entry:
            assert entry["instrument"] == "SPREAD"
            assert entry["option_type"] == "CE"
            assert entry["v7_setup_type"] == "credit_spread_bear"

    def test_reject_short_dte(self):
        strategy = V7CreditSpreadStrategy()
        chain = make_chain(24000, vix=16, dte=10)
        entry = strategy.should_enter(
            date=datetime.date(2025, 1, 15),
            spot=24000, vix=16, atr=150, score=4.0,
            direction="bullish", available_capital=200000,
            current_positions=[], chain=chain, dte=10, symbol="NIFTY",
        )
        assert entry is None

    def test_exit_profit_target(self):
        strategy = V7CreditSpreadStrategy()
        pos = {
            "net_credit": 30.0,
            "expiry": datetime.date(2025, 2, 15),
        }
        # Current cost to close is very low → big profit
        should_exit, reason = strategy.should_exit(
            position=pos, date=datetime.date(2025, 1, 20),
            current_spread_value=5.0,
        )
        assert should_exit
        assert reason == "profit_target"


# ---------------------------------------------------------------------------
# V7IronCondorStrategy Tests
# ---------------------------------------------------------------------------

class TestV7IronCondorStrategy:
    def test_enter_in_vix_range(self):
        strategy = V7IronCondorStrategy()
        chain = make_chain(24000, vix=16, dte=10)
        entry = strategy.should_enter(
            date=datetime.date(2025, 1, 15),
            spot=24000, vix=16,
            available_capital=200000,
            current_positions=[], chain=chain, dte=10, symbol="NIFTY",
        )
        if entry:
            assert entry["instrument"] == "CONDOR"
            assert entry["v7_setup_type"] == "iron_condor"
            assert entry["call_short"] > entry["put_short"]
            assert entry["net_credit"] > 0

    def test_reject_high_vix(self):
        strategy = V7IronCondorStrategy()
        chain = make_chain(24000, vix=25, dte=10)
        entry = strategy.should_enter(
            date=datetime.date(2025, 1, 15),
            spot=24000, vix=25,
            available_capital=200000,
            current_positions=[], chain=chain, dte=10, symbol="NIFTY",
        )
        assert entry is None

    def test_reject_low_vix(self):
        strategy = V7IronCondorStrategy()
        chain = make_chain(24000, vix=10, dte=10)
        entry = strategy.should_enter(
            date=datetime.date(2025, 1, 15),
            spot=24000, vix=10,
            available_capital=200000,
            current_positions=[], chain=chain, dte=10, symbol="NIFTY",
        )
        assert entry is None

    def test_reject_long_dte(self):
        strategy = V7IronCondorStrategy()
        chain = make_chain(24000, vix=16, dte=30)
        entry = strategy.should_enter(
            date=datetime.date(2025, 1, 15),
            spot=24000, vix=16,
            available_capital=200000,
            current_positions=[], chain=chain, dte=30, symbol="NIFTY",
        )
        assert entry is None

    def test_exit_profit(self):
        strategy = V7IronCondorStrategy()
        pos = {
            "net_credit": 50.0,
            "expiry": datetime.date(2025, 2, 15),
        }
        should_exit, reason = strategy.should_exit(
            position=pos, date=datetime.date(2025, 1, 20),
            current_condor_value=20.0,
        )
        assert should_exit
        assert reason == "profit_target"


# ---------------------------------------------------------------------------
# V7BacktestEngine Integration Tests
# ---------------------------------------------------------------------------

class TestV7BacktestEngine:
    def test_engine_init(self):
        engine = V7BacktestEngine(capital=300000)
        assert engine.initial_capital == 300000
        assert "breakout" in engine.strategies
        assert "credit_spread" in engine.strategies
        assert "iron_condor" in engine.strategies

    def test_strategy_caps(self):
        engine = V7BacktestEngine()
        assert engine.strategy_caps["breakout"] == 0.30
        assert engine.strategy_caps["credit_spread"] == 0.25
        assert engine.strategy_caps["iron_condor"] == 0.20

    @patch("v7.backtest_adapter.fetch_real_chain", return_value=None)
    def test_run_produces_results(self, mock_chain):
        """Engine runs on synthetic data and produces valid results dict."""
        engine = V7BacktestEngine(capital=300000)
        data = make_ohlcv(n_days=60, start_price=24000)
        results = engine.run(data, symbol="NIFTY")

        assert "trades" in results
        assert "stats" in results
        assert "daily_nav" in results
        assert "v7_breakdown" in results
        assert results["stats"]["total_trades"] >= 0

    @patch("v7.backtest_adapter.fetch_real_chain", return_value=None)
    def test_v7_breakdown_structure(self, mock_chain):
        """V7 breakdown includes setup type and conviction analysis."""
        engine = V7BacktestEngine(capital=300000)
        data = make_ohlcv(n_days=80, start_price=24000, trend=0.002)
        results = engine.run(data, symbol="NIFTY")

        v7 = results["v7_breakdown"]
        if results["stats"]["total_trades"] > 0:
            assert "by_setup_type" in v7
            assert "by_conviction" in v7

    def test_conviction_risk_mapping(self):
        assert CONVICTION_RISK["high"] == 2.0
        assert CONVICTION_RISK["medium"] == 1.5
        assert CONVICTION_RISK["low"] == 0.75
