"""Tests for fo_calibrate — synthetic vs real chain comparison."""
import unittest


class TestCalibrate(unittest.TestCase):

    def test_compare_chains_returns_report(self):
        """compare_chains should return a dict with per-strike comparisons."""
        from fo_calibrate import compare_chains
        from fo_data import generate_synthetic_chain

        chain = generate_synthetic_chain(spot=23000.0, vix=14.0, dte=30, symbol="NIFTY")
        report = compare_chains(
            real_chain=chain, synthetic_chain=chain,
            spot=23000.0, symbol="NIFTY",
        )
        assert "comparisons" in report
        assert "condor_real_credit_pct" in report
        assert "condor_synth_credit_pct" in report
        assert len(report["comparisons"]) > 0

    def test_compare_identical_chains_small_gap(self):
        """Identical chains should show ~0% gap."""
        from fo_calibrate import compare_chains
        from fo_data import generate_synthetic_chain

        chain = generate_synthetic_chain(spot=23000.0, vix=14.0, dte=30, symbol="NIFTY")
        report = compare_chains(
            real_chain=chain, synthetic_chain=chain,
            spot=23000.0, symbol="NIFTY",
        )
        for comp in report["comparisons"]:
            assert abs(comp["diff_pct"]) < 0.01, f"Gap too large for identical chains: {comp}"


if __name__ == "__main__":
    unittest.main()
