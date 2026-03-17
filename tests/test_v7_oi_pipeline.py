"""Tests for V7 OI Pipeline — option chain OI analysis."""
import json
import pytest
from pathlib import Path
from v7.oi_pipeline import OIPipeline, OISnapshot


# ── Sample chain data (7 strikes around ATM=24200) ──────────────────


def _sample_chain() -> list[dict]:
    """Simulated Kite option chain for NIFTY around 24200."""
    return [
        {"strikePrice": 23900, "CE": {"openInterest": 80000, "lastTradedPrice": 350, "volume": 8000}, "PE": {"openInterest": 20000, "lastTradedPrice": 30, "volume": 2000}},
        {"strikePrice": 24000, "CE": {"openInterest": 100000, "lastTradedPrice": 260, "volume": 12000}, "PE": {"openInterest": 30000, "lastTradedPrice": 45, "volume": 3000}},
        {"strikePrice": 24100, "CE": {"openInterest": 120000, "lastTradedPrice": 180, "volume": 15000}, "PE": {"openInterest": 50000, "lastTradedPrice": 70, "volume": 5000}},
        {"strikePrice": 24200, "CE": {"openInterest": 150000, "lastTradedPrice": 120, "volume": 20000}, "PE": {"openInterest": 80000, "lastTradedPrice": 100, "volume": 8000}},
        {"strikePrice": 24300, "CE": {"openInterest": 130000, "lastTradedPrice": 75, "volume": 18000}, "PE": {"openInterest": 110000, "lastTradedPrice": 150, "volume": 10000}},
        {"strikePrice": 24400, "CE": {"openInterest": 90000, "lastTradedPrice": 40, "volume": 10000}, "PE": {"openInterest": 140000, "lastTradedPrice": 220, "volume": 12000}},
        {"strikePrice": 24500, "CE": {"openInterest": 60000, "lastTradedPrice": 20, "volume": 6000}, "PE": {"openInterest": 160000, "lastTradedPrice": 300, "volume": 14000}},
    ]


def _modified_chain() -> list[dict]:
    """Chain with changed OI values (simulating passage of time)."""
    return [
        {"strikePrice": 23900, "CE": {"openInterest": 85000, "lastTradedPrice": 340, "volume": 9000}, "PE": {"openInterest": 22000, "lastTradedPrice": 32, "volume": 2200}},
        {"strikePrice": 24000, "CE": {"openInterest": 110000, "lastTradedPrice": 250, "volume": 13000}, "PE": {"openInterest": 35000, "lastTradedPrice": 48, "volume": 3500}},
        {"strikePrice": 24100, "CE": {"openInterest": 125000, "lastTradedPrice": 175, "volume": 16000}, "PE": {"openInterest": 55000, "lastTradedPrice": 72, "volume": 5500}},
        {"strikePrice": 24200, "CE": {"openInterest": 160000, "lastTradedPrice": 115, "volume": 22000}, "PE": {"openInterest": 90000, "lastTradedPrice": 105, "volume": 9000}},
        {"strikePrice": 24300, "CE": {"openInterest": 135000, "lastTradedPrice": 70, "volume": 19000}, "PE": {"openInterest": 120000, "lastTradedPrice": 155, "volume": 11000}},
        {"strikePrice": 24400, "CE": {"openInterest": 95000, "lastTradedPrice": 38, "volume": 11000}, "PE": {"openInterest": 150000, "lastTradedPrice": 225, "volume": 13000}},
        {"strikePrice": 24500, "CE": {"openInterest": 65000, "lastTradedPrice": 18, "volume": 7000}, "PE": {"openInterest": 170000, "lastTradedPrice": 305, "volume": 15000}},
    ]


@pytest.fixture
def tmp_data_dir(tmp_path):
    return tmp_path / "v7"


@pytest.fixture
def pipeline(tmp_data_dir):
    return OIPipeline(data_dir=tmp_data_dir)


# ── Tests ────────────────────────────────────────────────────────────


def test_compute_pcr(pipeline):
    """PCR = sum(put_oi) / sum(call_oi)."""
    chain = _sample_chain()
    pcr = pipeline.compute_pcr(chain)

    total_call = 80000 + 100000 + 120000 + 150000 + 130000 + 90000 + 60000  # 730000
    total_put = 20000 + 30000 + 50000 + 80000 + 110000 + 140000 + 160000   # 590000
    expected = total_put / total_call

    assert pcr == pytest.approx(expected, rel=1e-6)
    assert pcr == pytest.approx(590000 / 730000, rel=1e-6)


def test_compute_pcr_zero_call(pipeline):
    """PCR should be 0 when there's no call OI."""
    chain = [{"strikePrice": 24200, "CE": {"openInterest": 0}, "PE": {"openInterest": 5000}}]
    assert pipeline.compute_pcr(chain) == 0.0


def test_compute_max_pain(pipeline):
    """Max pain should find the strike minimizing total writer loss."""
    chain = _sample_chain()
    max_pain = pipeline.compute_max_pain(chain)

    # Manually verify: compute total loss at each candidate strike
    # and confirm the pipeline picks the minimum
    min_loss = float("inf")
    expected_strike = 0

    for settlement_entry in chain:
        settlement = settlement_entry["strikePrice"]
        total_loss = 0
        for entry in chain:
            strike = entry["strikePrice"]
            ce_oi = entry["CE"]["openInterest"]
            pe_oi = entry["PE"]["openInterest"]
            total_loss += max(0, settlement - strike) * ce_oi
            total_loss += max(0, strike - settlement) * pe_oi
        if total_loss < min_loss:
            min_loss = total_loss
            expected_strike = settlement

    assert max_pain == expected_strike
    # With this data, max pain should be around 24200 (where large call OI caps upside
    # and put OI caps downside)
    assert 23900 <= max_pain <= 24500


def test_compute_max_pain_empty(pipeline):
    """Max pain should return 0 for empty chain."""
    assert pipeline.compute_max_pain([]) == 0.0


def test_snapshot_roundtrip(pipeline):
    """to_dict -> from_dict should preserve all fields."""
    chain = _sample_chain()
    snapshot = pipeline.process_chain("NIFTY", chain, expiry="27MAR2026")

    d = snapshot.to_dict()
    restored = OISnapshot.from_dict(d)

    assert restored.symbol == snapshot.symbol
    assert restored.expiry == snapshot.expiry
    assert restored.timestamp == snapshot.timestamp
    assert restored.total_call_oi == snapshot.total_call_oi
    assert restored.total_put_oi == snapshot.total_put_oi
    assert restored.pcr == pytest.approx(snapshot.pcr)
    assert restored.max_pain_strike == snapshot.max_pain_strike
    assert len(restored.strikes) == len(snapshot.strikes)
    assert restored.strikes == snapshot.strikes


def test_save_and_load(pipeline):
    """Save + load should persist snapshot data."""
    chain = _sample_chain()
    snapshot = pipeline.process_chain("NIFTY", chain)
    pipeline.save_snapshot(snapshot)

    # Verify file exists
    path = pipeline.snapshot_dir / "NIFTY.json"
    assert path.exists()

    data = json.loads(path.read_text())
    assert len(data) == 1
    assert data[0]["symbol"] == "NIFTY"
    assert data[0]["total_call_oi"] == 730000


def test_oi_changes(pipeline):
    """Create two snapshots with different OI, verify changes computed correctly."""
    # Save first snapshot
    snap1 = pipeline.process_chain("NIFTY", _sample_chain())
    pipeline.save_snapshot(snap1)

    # Save second snapshot
    snap2 = pipeline.process_chain("NIFTY", _modified_chain())
    pipeline.save_snapshot(snap2)

    changes = pipeline.compute_oi_changes("NIFTY")

    assert "pcr_change" in changes
    assert "max_pain_change" in changes
    assert "total_call_oi_change" in changes
    assert "total_put_oi_change" in changes
    assert "top_strike_changes" in changes

    # Call OI increased overall
    assert changes["total_call_oi_change"] > 0
    # Put OI also increased overall
    assert changes["total_put_oi_change"] > 0

    # Verify strike-level changes are present
    assert len(changes["top_strike_changes"]) > 0
    # Each change entry should have the right keys
    for sc in changes["top_strike_changes"]:
        assert "strike" in sc
        assert "call_oi_change" in sc
        assert "put_oi_change" in sc
        assert "abs_change" in sc


def test_oi_changes_no_previous(pipeline):
    """compute_oi_changes should return {} when no previous snapshot."""
    snap = pipeline.process_chain("NIFTY", _sample_chain())
    pipeline.save_snapshot(snap)

    changes = pipeline.compute_oi_changes("NIFTY")
    assert changes == {}


def test_oi_buildup(pipeline):
    """get_oi_buildup should return top 5 by absolute change."""
    snap1 = pipeline.process_chain("NIFTY", _sample_chain())
    pipeline.save_snapshot(snap1)

    snap2 = pipeline.process_chain("NIFTY", _modified_chain())
    pipeline.save_snapshot(snap2)

    buildup = pipeline.get_oi_buildup("NIFTY")

    assert len(buildup) <= 5
    assert len(buildup) > 0

    # Verify sorted by abs_change descending
    for i in range(len(buildup) - 1):
        assert buildup[i]["abs_change"] >= buildup[i + 1]["abs_change"]

    # The biggest change should be at 24200 (10000 call + 10000 put = 20000 abs)
    # or 24300 (5000 + 10000 = 15000), etc.
    # 24200: call +10000, put +10000 = 20000
    # 24400: call +5000, put +10000 = 15000
    # 24000: call +10000, put +5000 = 15000
    # So 24200 should be first
    assert buildup[0]["strike"] == 24200


def test_format_for_prompt(pipeline):
    """format_for_prompt should return pcr, max_pain, and oi_changes keys."""
    snap1 = pipeline.process_chain("NIFTY", _sample_chain())
    pipeline.save_snapshot(snap1)

    snap2 = pipeline.process_chain("NIFTY", _modified_chain())
    pipeline.save_snapshot(snap2)

    result = pipeline.format_for_prompt(["NIFTY", "BANKNIFTY"])

    # NIFTY should be present, BANKNIFTY should not (no data)
    assert "NIFTY" in result
    assert "BANKNIFTY" not in result

    nifty = result["NIFTY"]
    assert "pcr" in nifty
    assert "max_pain" in nifty
    assert "oi_changes" in nifty
    assert "top_buildup" in nifty
    assert "total_call_oi" in nifty
    assert "total_put_oi" in nifty

    assert nifty["pcr"] > 0
    assert nifty["max_pain"] > 0
    assert len(nifty["oi_changes"]) > 0


def test_process_chain_missing_ce_pe(pipeline):
    """Chain entries with missing CE or PE should not crash."""
    chain = [
        {"strikePrice": 24200, "CE": {"openInterest": 50000}, "PE": {}},
        {"strikePrice": 24300, "CE": {}, "PE": {"openInterest": 60000}},
        {"strikePrice": 24400},  # no CE or PE at all
    ]
    snapshot = pipeline.process_chain("NIFTY", chain)
    assert snapshot.total_call_oi == 50000
    assert snapshot.total_put_oi == 60000
    assert len(snapshot.strikes) == 3


def test_snapshot_keeps_last_20(pipeline):
    """save_snapshot should keep at most 20 snapshots."""
    chain = _sample_chain()
    for i in range(25):
        snap = pipeline.process_chain("NIFTY", chain)
        pipeline.save_snapshot(snap)

    path = pipeline.snapshot_dir / "NIFTY.json"
    data = json.loads(path.read_text())
    assert len(data) == 20


def test_load_previous(pipeline):
    """load_previous should return the second-to-last snapshot."""
    snap1 = pipeline.process_chain("NIFTY", _sample_chain())
    pipeline.save_snapshot(snap1)

    snap2 = pipeline.process_chain("NIFTY", _modified_chain())
    pipeline.save_snapshot(snap2)

    prev = pipeline.load_previous("NIFTY")
    assert prev is not None
    assert prev.total_call_oi == 730000  # from _sample_chain


def test_load_previous_no_data(pipeline):
    """load_previous should return None when no snapshots."""
    assert pipeline.load_previous("NIFTY") is None


def test_load_previous_single_snapshot(pipeline):
    """load_previous should return None with only one snapshot."""
    snap = pipeline.process_chain("NIFTY", _sample_chain())
    pipeline.save_snapshot(snap)
    assert pipeline.load_previous("NIFTY") is None
