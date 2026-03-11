# tests/test_v7_strike_selector.py
"""Tests for mechanical strike selection."""
import pytest
from v7.strike_selector import (
    select_directional_strike, select_spread_strikes,
    select_hedge_strike, passes_liquidity_filter,
)


def make_chain_entry(strike, ce_ltp, pe_ltp, ce_oi=1000, pe_oi=1000,
                     ce_delta=0.5, pe_delta=-0.5,
                     ce_bid_ask=1.0, pe_bid_ask=1.0):
    return {
        "strikePrice": strike,
        "CE": {
            "ltp": ce_ltp, "oi": ce_oi, "volume": 500,
            "delta": ce_delta, "bidPrice": ce_ltp - ce_bid_ask/2,
            "askPrice": ce_ltp + ce_bid_ask/2,
        },
        "PE": {
            "ltp": pe_ltp, "oi": pe_oi, "volume": 500,
            "delta": pe_delta, "bidPrice": pe_ltp - pe_bid_ask/2,
            "askPrice": pe_ltp + pe_bid_ask/2,
        },
    }


@pytest.fixture
def sample_chain():
    """Nifty-like chain around 24200."""
    return [
        make_chain_entry(24000, 250, 50, ce_delta=0.65, pe_delta=-0.35, ce_oi=50000, pe_oi=40000),
        make_chain_entry(24100, 180, 80, ce_delta=0.58, pe_delta=-0.42, ce_oi=45000, pe_oi=35000),
        make_chain_entry(24200, 120, 120, ce_delta=0.50, pe_delta=-0.50, ce_oi=60000, pe_oi=60000),
        make_chain_entry(24300, 75, 175, ce_delta=0.42, pe_delta=-0.58, ce_oi=55000, pe_oi=30000),
        make_chain_entry(24400, 40, 240, ce_delta=0.33, pe_delta=-0.67, ce_oi=70000, pe_oi=20000),
        make_chain_entry(24500, 20, 320, ce_delta=0.22, pe_delta=-0.78, ce_oi=80000, pe_oi=15000),
        make_chain_entry(24600, 10, 410, ce_delta=0.14, pe_delta=-0.86, ce_oi=40000, pe_oi=10000),
    ]


def test_select_directional_call(sample_chain):
    result = select_directional_strike(
        chain=sample_chain, direction="bullish", spot=24200,
        risk_budget=6000, lot_size=75, symbol="NIFTY",
    )
    assert result is not None
    assert result["option_type"] == "CE"
    assert 0.35 <= abs(result["delta"]) <= 0.55
    assert result["premium"] * 75 <= 6000


def test_select_directional_put(sample_chain):
    result = select_directional_strike(
        chain=sample_chain, direction="bearish", spot=24200,
        risk_budget=4500, lot_size=75, symbol="NIFTY",
    )
    assert result is not None
    assert result["option_type"] == "PE"


def test_select_directional_respects_budget(sample_chain):
    result = select_directional_strike(
        chain=sample_chain, direction="bullish", spot=24200,
        risk_budget=1000, lot_size=75, symbol="NIFTY",
    )
    if result:
        assert result["premium"] * 75 <= 1000


def test_passes_liquidity_filter():
    assert passes_liquidity_filter(oi=1000, volume=200, bid_ask_spread=1.5, symbol="NIFTY")
    assert not passes_liquidity_filter(oi=100, volume=200, bid_ask_spread=1.5, symbol="NIFTY")
    assert not passes_liquidity_filter(oi=1000, volume=200, bid_ask_spread=5.0, symbol="NIFTY")


def test_select_spread_strikes(sample_chain):
    result = select_spread_strikes(
        chain=sample_chain, direction="bearish", spot=24200,
        risk_budget=4500, lot_size=75, symbol="NIFTY",
    )
    if result:
        assert result["sell_strike"] > result["buy_strike"]
        assert result["max_loss"] <= 4500


def test_select_hedge_strike(sample_chain):
    result = select_hedge_strike(
        chain=sample_chain, direction="bullish", spot=24200,
        max_cost=500, lot_size=75,
    )
    if result:
        assert result["option_type"] == "PE"
        assert result["premium"] * 75 <= 500
