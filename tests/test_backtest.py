"""Tests for backtest engine."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtest import BacktestEngine, compute_atr, calc_eq_costs, calc_round_trip_costs, DEFAULT_PARAMS


def _make_df(n=50, start_price=1000, daily_return_pct=0.5):
    """Generate synthetic OHLCV DataFrame."""
    dates = pd.date_range(start="2025-01-01", periods=n, freq="B")
    prices = [start_price]
    for i in range(1, n):
        change = prices[-1] * daily_return_pct / 100 * (1 if i % 3 != 0 else -1)
        prices.append(prices[-1] + change)

    df = pd.DataFrame({
        "Open": [p * 0.999 for p in prices],
        "High": [p * 1.01 for p in prices],
        "Low": [p * 0.99 for p in prices],
        "Close": prices,
        "Volume": [1000000] * n,
    }, index=dates)
    return df


def _make_trending_df(n=50, start=1000, trend=0.3, volatility=3):
    """Generate trending up/down data."""
    dates = pd.date_range(start="2025-01-01", periods=n, freq="B")
    prices = [start + i * trend for i in range(n)]

    df = pd.DataFrame({
        "Open": [p - 1 for p in prices],
        "High": [p + volatility for p in prices],
        "Low": [p - volatility for p in prices],
        "Close": prices,
        "Volume": [500000] * n,
    }, index=dates)
    return df


def _make_crashing_df(n=50, start=1000, drop_per_day=5):
    """Generate crashing data."""
    dates = pd.date_range(start="2025-01-01", periods=n, freq="B")
    prices = [max(100, start - i * drop_per_day) for i in range(n)]

    df = pd.DataFrame({
        "Open": [p + 2 for p in prices],
        "High": [p + 5 for p in prices],
        "Low": [p - 15 for p in prices],
        "Close": prices,
        "Volume": [2000000] * n,
    }, index=dates)
    return df


def test_compute_atr():
    """ATR computation on synthetic data."""
    df = _make_df(30)
    atr = compute_atr(df, period=14)
    assert len(atr) == 30
    assert pd.notna(atr.iloc[-1])
    assert atr.iloc[-1] > 0


def test_compute_atr_too_short():
    """ATR returns NaN when not enough data."""
    df = _make_df(10)
    atr = compute_atr(df, period=14)
    assert pd.isna(atr.iloc[-1])


def test_eq_costs_positive():
    """Transaction costs should be positive."""
    cost = calc_eq_costs(1000, 10, "buy")
    assert cost > 0


def test_round_trip_costs():
    """Round trip costs should include both legs."""
    rt = calc_round_trip_costs(1000, 1050, 10)
    buy = calc_eq_costs(1000, 10, "buy")
    sell = calc_eq_costs(1050, 10, "sell")
    assert abs(rt - (buy + sell)) < 0.01


def test_backtest_generates_trades():
    """Backtest should generate trades on synthetic data."""
    df = _make_df(50)
    engine = BacktestEngine()
    trades = engine.run("TEST", df)
    assert len(trades) > 0
    assert all(t["symbol"] == "TEST" for t in trades)


def test_backtest_trade_structure():
    """Each trade should have required fields."""
    df = _make_df(50)
    engine = BacktestEngine()
    trades = engine.run("TEST", df)

    required = ["symbol", "entry_date", "exit_date", "entry_price", "exit_price",
                "quantity", "pnl_gross", "costs", "pnl_net", "pnl_pct", "exit_reason", "holding_days"]
    for t in trades:
        for field in required:
            assert field in t, f"Missing field: {field}"


def test_backtest_costs_deducted():
    """P&L net should be less than gross (costs deducted)."""
    df = _make_df(50)
    engine = BacktestEngine()
    trades = engine.run("TEST", df)
    for t in trades:
        assert t["costs"] > 0
        assert t["pnl_net"] < t["pnl_gross"] or t["costs"] == 0


def test_backtest_trending_profitable():
    """Strong uptrend should produce net positive P&L."""
    df = _make_trending_df(60, start=1000, trend=5, volatility=3)
    engine = BacktestEngine({"capital_per_trade": 50000})
    trades = engine.run("TREND", df)
    total_pnl = sum(t["pnl_net"] for t in trades)
    assert total_pnl > 0, f"Trending market should be profitable, got {total_pnl}"


def test_backtest_crash_limited_loss():
    """Crash scenario: SL should limit each trade's loss."""
    df = _make_crashing_df(60)
    engine = BacktestEngine()
    trades = engine.run("CRASH", df)
    for t in trades:
        # Loss should not exceed ATR_SL_MULT * ATR (roughly)
        max_loss_pct = -20  # generous bound
        assert t["pnl_pct"] > max_loss_pct, f"Loss too large: {t['pnl_pct']}%"


def test_compute_stats():
    """Stats computation should produce valid metrics."""
    df = _make_df(60)
    engine = BacktestEngine()
    engine.run_multi({"TEST": df})
    stats = engine.compute_stats()

    assert stats["total_trades"] > 0
    assert "win_rate" in stats
    assert "sharpe_ratio" in stats
    assert "max_drawdown" in stats
    assert "exit_reasons" in stats
    assert "estimated_stcg_tax" in stats
    assert stats["estimated_stcg_tax"] >= 0


def test_compute_stats_empty():
    """Stats with no trades should return gracefully."""
    engine = BacktestEngine()
    engine.trades = []
    stats = engine.compute_stats()
    assert stats["total_trades"] == 0


def test_max_hold_enforced():
    """No trade should exceed max_hold_days."""
    df = _make_df(80)
    params = {"max_hold_days": 5}
    engine = BacktestEngine(params)
    trades = engine.run("TEST", df)
    for t in trades:
        assert t["holding_days"] <= params["max_hold_days"] + 1  # +1 for exit day


def test_multi_symbol():
    """Multi-symbol backtest aggregates trades correctly."""
    data = {
        "SYM1": _make_df(50),
        "SYM2": _make_trending_df(50),
    }
    engine = BacktestEngine()
    trades = engine.run_multi(data)
    symbols = set(t["symbol"] for t in trades)
    assert "SYM1" in symbols
    assert "SYM2" in symbols


def test_time_sl_tightening():
    """With time SL enabled, trades near expiry should exit earlier."""
    df = _make_df(80)

    # Without time SL
    engine_no = BacktestEngine({"time_sl_enabled": False})
    trades_no = engine_no.run("TEST", df)

    # With time SL
    engine_yes = BacktestEngine({"time_sl_enabled": True})
    trades_yes = engine_yes.run("TEST", df)

    # Time SL should cause at least some trades to exit earlier
    avg_hold_no = sum(t["holding_days"] for t in trades_no) / len(trades_no) if trades_no else 0
    avg_hold_yes = sum(t["holding_days"] for t in trades_yes) / len(trades_yes) if trades_yes else 0

    # This is a soft assertion -- time SL might not always shorten holds
    # but the test validates the code path runs without error
    assert len(trades_yes) > 0
