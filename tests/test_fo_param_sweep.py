"""Tests for fo_param_sweep — parameter grid search."""
import datetime
import json
import os
import sys
import unittest

import numpy as np
import pandas as pd


def _make_short_data(bars: int = 60) -> pd.DataFrame:
    """Create minimal OHLCV+VIX data for fast sweep tests."""
    dates = pd.bdate_range(end=datetime.date(2025, 6, 30), periods=bars)
    rng = np.random.RandomState(42)
    close = 23000 + rng.randn(bars).cumsum() * 50
    return pd.DataFrame({
        "Date": dates,
        "Open": close - rng.rand(bars) * 30,
        "High": close + rng.rand(bars) * 50,
        "Low": close - rng.rand(bars) * 50,
        "Close": close,
        "Volume": rng.randint(100000, 500000, bars),
        "vix": 14 + rng.randn(bars) * 2,
    }).set_index("Date")


class TestParamSweep(unittest.TestCase):

    def test_build_grid_futures(self):
        from fo_param_sweep import build_param_grid
        grid = build_param_grid("futures")
        assert len(grid) == 27  # 3x3x3
        assert all("score_threshold" in g for g in grid)
        assert all("target_atr_mult" in g for g in grid)
        assert all("sl_atr_mult" in g for g in grid)

    def test_build_grid_all_strategies(self):
        from fo_param_sweep import build_param_grid, SWEEP_GRIDS
        for strategy_name in SWEEP_GRIDS:
            grid = build_param_grid(strategy_name)
            assert len(grid) == 27, f"{strategy_name} grid should have 27 combos, got {len(grid)}"

    def test_sweep_single_combo(self):
        """Sweep with 1 combo should return results dict."""
        from fo_param_sweep import run_single_sweep
        data = _make_short_data(60)
        result = run_single_sweep(
            data=data,
            strategy_name="futures",
            params={"score_threshold": 3.5, "target_atr_mult": 1.5, "sl_atr_mult": 3.5},
            symbol="NIFTY",
            capital=1_000_000,
        )
        assert "params" in result
        assert "trades" in result
        assert "sharpe" in result
        assert "win_rate" in result
        assert "total_pnl" in result

    def test_sweep_strategy_returns_ranked(self):
        """Full sweep of one strategy returns results sorted by Sharpe."""
        from fo_param_sweep import sweep_strategy
        data = _make_short_data(60)
        results = sweep_strategy(
            data=data,
            strategy_name="futures",
            symbol="NIFTY",
            capital=1_000_000,
        )
        assert len(results) == 27
        # Should be sorted by Sharpe descending
        sharpes = [r["sharpe"] for r in results]
        assert sharpes == sorted(sharpes, reverse=True)


if __name__ == "__main__":
    unittest.main()
