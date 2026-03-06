"""Tests for global_intel module."""
import pandas as pd


def _make_history_df(prices):
    return pd.DataFrame({"Close": prices}, index=pd.date_range("2026-03-04", periods=len(prices)))


class TestUSMarkets:
    def test_sp500_down_2pct(self):
        from global_intel import _compute_us_market_data
        data = _compute_us_market_data(
            sp500_hist=_make_history_df([5000, 4850]),
            nasdaq_hist=_make_history_df([18000, 17700]),
        )
        assert data["sp500_pct_change"] < -2.0

    def test_sp500_flat(self):
        from global_intel import _compute_us_market_data
        data = _compute_us_market_data(
            sp500_hist=_make_history_df([5000, 5010]),
            nasdaq_hist=_make_history_df([18000, 18050]),
        )
        assert abs(data["sp500_pct_change"]) < 1.0

    def test_empty_history(self):
        from global_intel import _compute_us_market_data
        data = _compute_us_market_data(sp500_hist=pd.DataFrame(), nasdaq_hist=pd.DataFrame())
        assert data["sp500_pct_change"] == 0.0


class TestGIFTNifty:
    def test_gap_down(self):
        from global_intel import _compute_gift_nifty_gap
        result = _compute_gift_nifty_gap(gift_ltp=24100, prev_nifty_close=24500)
        assert result["gift_nifty_gap_pct"] < -1.5

    def test_gap_up(self):
        from global_intel import _compute_gift_nifty_gap
        result = _compute_gift_nifty_gap(gift_ltp=24900, prev_nifty_close=24500)
        assert result["gift_nifty_gap_pct"] > 1.0

    def test_no_data(self):
        from global_intel import _compute_gift_nifty_gap
        result = _compute_gift_nifty_gap(gift_ltp=None, prev_nifty_close=None)
        assert result["gift_nifty_gap_pct"] == 0.0


class TestAsiaMarkets:
    def test_asia_pct_changes(self):
        from global_intel import _compute_asia_data
        data = _compute_asia_data(
            hang_seng_hist=_make_history_df([20000, 19600]),
            nikkei_hist=_make_history_df([40000, 39500]),
        )
        assert data["hang_seng_pct"] < -1.0
        assert data["nikkei_pct"] < -1.0


class TestHardGateLogic:
    def test_block_all_on_severe_crash(self):
        from global_intel import compute_hard_gate
        gate = compute_hard_gate(sp500_pct=-3.5, nasdaq_pct=-4.0, gift_gap_pct=-1.0, fii_net=0, dii_net=0, pcr=1.0)
        assert gate["action"] == "BLOCK_ALL"

    def test_block_bullish_on_us_crash(self):
        from global_intel import compute_hard_gate
        gate = compute_hard_gate(sp500_pct=-2.5, nasdaq_pct=-1.5, gift_gap_pct=0, fii_net=0, dii_net=0, pcr=1.0)
        assert gate["action"] == "BLOCK_BULLISH"

    def test_reduce_50_on_mild_red(self):
        from global_intel import compute_hard_gate
        gate = compute_hard_gate(sp500_pct=-1.5, nasdaq_pct=-1.0, gift_gap_pct=0, fii_net=0, dii_net=0, pcr=1.0)
        assert gate["action"] == "REDUCE_50"

    def test_reduce_50_on_gift_gap(self):
        from global_intel import compute_hard_gate
        gate = compute_hard_gate(sp500_pct=0.5, nasdaq_pct=0.3, gift_gap_pct=-1.8, fii_net=0, dii_net=0, pcr=1.0)
        assert gate["action"] == "REDUCE_50"

    def test_block_bullish_on_gift_crash(self):
        from global_intel import compute_hard_gate
        gate = compute_hard_gate(sp500_pct=0, nasdaq_pct=0, gift_gap_pct=-3.0, fii_net=0, dii_net=0, pcr=1.0)
        assert gate["action"] == "BLOCK_BULLISH"

    def test_fii_heavy_selling(self):
        from global_intel import compute_hard_gate
        gate = compute_hard_gate(sp500_pct=0.5, nasdaq_pct=0.5, gift_gap_pct=0, fii_net=-6000, dii_net=0, pcr=1.0)
        assert gate["action"] == "REDUCE_50"

    def test_fii_extreme_selling(self):
        from global_intel import compute_hard_gate
        gate = compute_hard_gate(sp500_pct=0, nasdaq_pct=0, gift_gap_pct=0, fii_net=-12000, dii_net=0, pcr=1.0)
        assert gate["action"] == "BLOCK_BULLISH"

    def test_fii_selling_moderated_by_dii(self):
        from global_intel import compute_hard_gate
        gate = compute_hard_gate(sp500_pct=0, nasdaq_pct=0, gift_gap_pct=0, fii_net=-6000, dii_net=5000, pcr=1.0)
        assert gate["action"] == "NONE"

    def test_pcr_euphoria_blocks(self):
        from global_intel import compute_hard_gate
        gate = compute_hard_gate(sp500_pct=1.0, nasdaq_pct=1.0, gift_gap_pct=0.5, fii_net=2000, dii_net=1000, pcr=0.4)
        assert gate["action"] == "BLOCK_BULLISH"

    def test_pcr_extreme_call_reduces(self):
        from global_intel import compute_hard_gate
        gate = compute_hard_gate(sp500_pct=0, nasdaq_pct=0, gift_gap_pct=0, fii_net=0, dii_net=0, pcr=0.65)
        assert gate["action"] == "REDUCE_25"

    def test_all_green_no_gate(self):
        from global_intel import compute_hard_gate
        gate = compute_hard_gate(sp500_pct=1.0, nasdaq_pct=0.8, gift_gap_pct=0.5, fii_net=2000, dii_net=1000, pcr=1.0)
        assert gate["action"] == "NONE"

    def test_nasdaq_it_crash(self):
        from global_intel import compute_hard_gate
        gate = compute_hard_gate(sp500_pct=-1.0, nasdaq_pct=-3.5, gift_gap_pct=0, fii_net=0, dii_net=0, pcr=1.0)
        assert gate["action"] == "BLOCK_IT_BULLISH"
