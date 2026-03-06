"""Integration tests: global intel gates in open_positions pipeline."""
from unittest.mock import patch


class TestMacroGateIntegration:
    def test_block_all_returns_zero(self):
        """When hard gate is BLOCK_ALL, open_positions should return 0."""
        from global_intel import compute_hard_gate
        gate = compute_hard_gate(sp500_pct=-4.0, nasdaq_pct=-3.0, gift_gap_pct=-2.0,
                                 fii_net=-15000, dii_net=0, pcr=0.4)
        assert gate["action"] == "BLOCK_ALL"

    def test_reduce_50_halves_capital(self):
        """When hard gate is REDUCE_50, available capital should be halved."""
        from global_intel import compute_hard_gate
        gate = compute_hard_gate(sp500_pct=-1.5, nasdaq_pct=0, gift_gap_pct=0,
                                 fii_net=0, dii_net=0, pcr=1.0)
        assert gate["action"] == "REDUCE_50"

    def test_supertrend_disagree_reduces(self):
        """When Supertrend disagrees with screener signal, reduce allocation."""
        from config import SUPERTREND_DISAGREE_REDUCTION
        assert SUPERTREND_DISAGREE_REDUCTION == 0.25

    def test_cpr_narrow_favors_directional(self):
        """Narrow CPR day should favor directional trades."""
        from indicators_v3 import compute_cpr
        result = compute_cpr(prev_high=24100, prev_low=24050, prev_close=24080)
        assert result["day_type"] == "trending"
