import datetime
import pytest
import pandas as pd
import numpy as np


class TestFOBacktestEngine:
    def _make_spot_data(self, days=100):
        """Generate synthetic spot+VIX data for testing without yfinance."""
        dates = pd.bdate_range(start="2025-01-01", periods=days)
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.012, days)
        close = 23000 * np.cumprod(1 + returns)
        high = close * (1 + np.random.uniform(0.002, 0.015, days))
        low = close * (1 - np.random.uniform(0.002, 0.015, days))
        open_ = close * (1 + np.random.normal(0, 0.005, days))
        volume = np.random.randint(100000, 1000000, days).astype(float)

        return pd.DataFrame({
            "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume,
            "open": open_, "high": high, "low": low, "close": close, "volume": volume,
            "vix": np.random.uniform(12, 18, days),
        }, index=dates)

    def test_engine_runs_without_error(self):
        from fo_backtest import FOBacktestEngine
        data = self._make_spot_data(100)
        engine = FOBacktestEngine(capital=1000000)
        results = engine.run(data, symbol="NIFTY")
        assert "trades" in results
        assert "stats" in results
        assert "daily_nav" in results

    def test_engine_respects_capital_limits(self):
        from fo_backtest import FOBacktestEngine
        data = self._make_spot_data(100)
        engine = FOBacktestEngine(capital=1000000)
        results = engine.run(data, symbol="NIFTY")
        assert all(nav >= 0 for nav in results["daily_nav"].values())

    def test_engine_produces_trades(self):
        from fo_backtest import FOBacktestEngine
        data = self._make_spot_data(200)
        engine = FOBacktestEngine(capital=1000000)
        results = engine.run(data, symbol="NIFTY")
        assert len(results["trades"]) > 0

    def test_engine_stats_valid(self):
        from fo_backtest import FOBacktestEngine
        data = self._make_spot_data(200)
        engine = FOBacktestEngine(capital=1000000)
        results = engine.run(data, symbol="NIFTY")
        stats = results["stats"]
        assert "total_trades" in stats
        assert "win_rate" in stats
        assert "total_pnl" in stats
        assert "max_drawdown" in stats
        if stats["total_trades"] > 0:
            assert 0 <= stats["win_rate"] <= 100

    def test_engine_strategy_breakdown(self):
        from fo_backtest import FOBacktestEngine
        data = self._make_spot_data(200)
        engine = FOBacktestEngine(capital=1000000)
        results = engine.run(data, symbol="NIFTY")
        assert "per_strategy" in results["stats"]

    def test_engine_prefers_real_chain(self):
        """Engine should use real chain when available, fall back to synthetic."""
        import tempfile
        from fo_data import fetch_real_chain
        from fo_chain_collector import convert_kite_chain_to_df, save_chain

        kite_chain = [
            {"strikePrice": 23000,
             "CE": {"lastTradedPrice": 200.0, "bidPrice": 199.0, "askPrice": 201.0,
                    "openInterest": 100, "volume": 50, "lotSize": 75},
             "PE": {"lastTradedPrice": 180.0, "bidPrice": 179.0, "askPrice": 181.0,
                    "openInterest": 100, "volume": 50, "lotSize": 75}},
        ]
        df = convert_kite_chain_to_df(kite_chain, spot=23000.0, vix=14.0,
                                       expiry_str="2026-04-02", symbol="NIFTY")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_chain(df, "NIFTY", datetime.date(2026, 3, 10), output_dir=tmpdir)
            loaded = fetch_real_chain("NIFTY", datetime.date(2026, 3, 10), data_dir=tmpdir)
            assert loaded is not None
            assert len(loaded) == 2
            assert "premium" in loaded.columns

            missing = fetch_real_chain("NIFTY", datetime.date(2020, 1, 1), data_dir=tmpdir)
            assert missing is None
