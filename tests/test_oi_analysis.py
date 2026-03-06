"""Tests for OI analysis — PCR, max pain, OI buildup."""


def _make_chain(strikes_oi):
    return [
        {
            "strikePrice": s,
            "CE": {"openInterest": c_oi, "lastPrice": 100},
            "PE": {"openInterest": p_oi, "lastPrice": 100},
        }
        for s, c_oi, p_oi in strikes_oi
    ]


class TestPCR:
    def test_neutral_pcr(self):
        from oi_analysis import compute_pcr
        chain = _make_chain([(24000, 1000, 1000), (24500, 1000, 1000)])
        assert compute_pcr(chain) == 1.0

    def test_bearish_pcr(self):
        from oi_analysis import compute_pcr
        chain = _make_chain([(24000, 500, 1500), (24500, 500, 1500)])
        assert compute_pcr(chain) == 3.0

    def test_bullish_pcr(self):
        from oi_analysis import compute_pcr
        chain = _make_chain([(24000, 2000, 500), (24500, 2000, 500)])
        assert compute_pcr(chain) == 0.25

    def test_zero_call_oi(self):
        from oi_analysis import compute_pcr
        chain = _make_chain([(24000, 0, 1000)])
        assert compute_pcr(chain) == 1.0

    def test_empty_chain(self):
        from oi_analysis import compute_pcr
        assert compute_pcr([]) == 1.0


class TestMaxPain:
    def test_basic_max_pain(self):
        from oi_analysis import compute_max_pain
        chain = _make_chain([
            (23500, 100, 5000),
            (24000, 200, 3000),
            (24500, 5000, 200),
            (25000, 3000, 100),
        ])
        mp = compute_max_pain(chain, lot_size=75)
        assert 23500 <= mp <= 25000

    def test_single_strike(self):
        from oi_analysis import compute_max_pain
        chain = _make_chain([(24000, 1000, 1000)])
        assert compute_max_pain(chain) == 24000

    def test_empty_chain(self):
        from oi_analysis import compute_max_pain
        assert compute_max_pain([]) == 0


class TestOIBuildup:
    def test_top_oi_strikes(self):
        from oi_analysis import get_top_oi_strikes
        chain = _make_chain([
            (23500, 100, 5000),
            (24000, 200, 3000),
            (24500, 5000, 200),
            (25000, 3000, 100),
        ])
        result = get_top_oi_strikes(chain, top_n=2)
        assert len(result["call_resistance"]) == 2
        assert len(result["put_support"]) == 2
        assert result["call_resistance"][0]["strike"] == 24500
        assert result["put_support"][0]["strike"] == 23500
