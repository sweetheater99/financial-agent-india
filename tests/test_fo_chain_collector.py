"""Tests for fo_chain_collector — daily chain snapshot collection."""
import datetime
import os
import tempfile
import unittest

import pandas as pd


def _make_mock_kite_chain():
    """Mock Kite chain in SmartAPI format."""
    return [
        {
            "strikePrice": 23000,
            "CE": {
                "lastTradedPrice": 285.0,
                "bidPrice": 284.0,
                "askPrice": 286.0,
                "openInterest": 500000,
                "volume": 120000,
                "lotSize": 75,
            },
            "PE": {
                "lastTradedPrice": 270.0,
                "bidPrice": 269.0,
                "askPrice": 271.0,
                "openInterest": 450000,
                "volume": 110000,
                "lotSize": 75,
            },
        },
        {
            "strikePrice": 23050,
            "CE": {
                "lastTradedPrice": 260.0,
                "bidPrice": 259.0,
                "askPrice": 261.0,
                "openInterest": 300000,
                "volume": 80000,
                "lotSize": 75,
            },
            "PE": {
                "lastTradedPrice": 295.0,
                "bidPrice": 294.0,
                "askPrice": 296.0,
                "openInterest": 350000,
                "volume": 90000,
                "lotSize": 75,
            },
        },
    ]


class TestChainCollector(unittest.TestCase):

    def test_convert_kite_to_backtest_format(self):
        """Kite chain should convert to backtest DataFrame format."""
        from fo_chain_collector import convert_kite_chain_to_df
        kite_chain = _make_mock_kite_chain()
        df = convert_kite_chain_to_df(
            kite_chain, spot=23000.0, vix=14.0,
            expiry_str="2026-04-02", symbol="NIFTY",
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4  # 2 strikes x 2 types
        required_cols = ["strike", "option_type", "premium", "bid", "ask",
                         "oi", "volume", "iv", "delta", "gamma", "theta", "vega",
                         "expiry", "spot", "vix"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_convert_premiums_match_kite(self):
        """Premiums should come from Kite lastTradedPrice."""
        from fo_chain_collector import convert_kite_chain_to_df
        kite_chain = _make_mock_kite_chain()
        df = convert_kite_chain_to_df(
            kite_chain, spot=23000.0, vix=14.0,
            expiry_str="2026-04-02", symbol="NIFTY",
        )
        ce_23000 = df[(df["strike"] == 23000) & (df["option_type"] == "CE")]
        assert float(ce_23000.iloc[0]["premium"]) == 285.0
        assert float(ce_23000.iloc[0]["bid"]) == 284.0
        assert float(ce_23000.iloc[0]["ask"]) == 286.0

    def test_convert_computes_iv(self):
        """IV should be computed via BS inversion, not zero."""
        from fo_chain_collector import convert_kite_chain_to_df
        kite_chain = _make_mock_kite_chain()
        df = convert_kite_chain_to_df(
            kite_chain, spot=23000.0, vix=14.0,
            expiry_str="2026-04-02", symbol="NIFTY",
        )
        ce_23000 = df[(df["strike"] == 23000) & (df["option_type"] == "CE")]
        assert float(ce_23000.iloc[0]["iv"]) > 0.05, "IV should be computed, not zero"

    def test_save_and_load_parquet(self):
        """Chain should round-trip through parquet."""
        from fo_chain_collector import convert_kite_chain_to_df, save_chain, load_chain
        kite_chain = _make_mock_kite_chain()
        df = convert_kite_chain_to_df(
            kite_chain, spot=23000.0, vix=14.0,
            expiry_str="2026-04-02", symbol="NIFTY",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            save_chain(df, "NIFTY", datetime.date(2026, 3, 10), output_dir=tmpdir)
            loaded = load_chain("NIFTY", datetime.date(2026, 3, 10), data_dir=tmpdir)
            assert loaded is not None
            assert len(loaded) == len(df)
            assert list(loaded.columns) == list(df.columns)

    def test_load_missing_returns_none(self):
        """Loading a non-existent chain should return None."""
        from fo_chain_collector import load_chain
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_chain("NIFTY", datetime.date(2020, 1, 1), data_dir=tmpdir)
            assert result is None


if __name__ == "__main__":
    unittest.main()
