"""Tests for agent_with_options.py pure functions."""

import re
from datetime import date, timedelta
import calendar

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_with_options import get_nearest_expiry


def test_nearest_expiry_format():
    result = get_nearest_expiry()
    # Should match DDMONYYYY pattern like 27FEB2026
    assert re.match(r"^\d{2}[A-Z]{3}\d{4}$", result), f"Unexpected format: {result}"


def test_nearest_expiry_is_thursday():
    result = get_nearest_expiry()
    # Parse back to date
    d = _parse_expiry(result)
    assert d.weekday() == 3, f"Expected Thursday (3), got weekday {d.weekday()} for {result}"


def test_nearest_expiry_is_future():
    result = get_nearest_expiry()
    d = _parse_expiry(result)
    assert d >= date.today(), f"Expiry {result} is in the past"


def test_nearest_expiry_is_last_thursday():
    result = get_nearest_expiry()
    d = _parse_expiry(result)
    # Verify it's the last Thursday of its month
    last_day = calendar.monthrange(d.year, d.month)[1]
    last_date = date(d.year, d.month, last_day)
    while last_date.weekday() != 3:
        last_date -= timedelta(days=1)
    assert d == last_date, f"{result} ({d}) is not the last Thursday ({last_date})"


def _parse_expiry(expiry_str: str) -> date:
    """Parse DDMONYYYY format back to a date object."""
    from datetime import datetime
    return datetime.strptime(expiry_str, "%d%b%Y").date()


# ── select_spread_strikes tests ─────────────────────────────────────────────

from agent_with_options import select_spread_strikes


def _mock_chain_bullish():
    """Mock option chain for testing bull call spread.

    Premiums are set so that wider spreads produce R:R >= 1.5:
      Bull call: long 1300 CE @15, short 1360 CE @3 → debit=12, width=60, profit=48, R:R=4.0
      Bear put:  long 1320 PE @15, short 1260 PE @3 → debit=12, width=60, profit=48, R:R=4.0
    """
    return [
        {"strikePrice": 1260, "CE": {"openInterest": 1500, "impliedVolatility": 28, "delta": 0.8, "lastPrice": 50, "totalTradedVolume": 400}, "PE": {"openInterest": 800, "lastPrice": 3, "delta": -0.05, "impliedVolatility": 18, "totalTradedVolume": 200}},
        {"strikePrice": 1280, "CE": {"openInterest": 2000, "impliedVolatility": 25, "delta": 0.7, "lastPrice": 35, "totalTradedVolume": 500}, "PE": {"openInterest": 1200, "lastPrice": 6, "delta": -0.1, "impliedVolatility": 20, "totalTradedVolume": 300}},
        {"strikePrice": 1300, "CE": {"openInterest": 5000, "impliedVolatility": 22, "delta": 0.5, "lastPrice": 15, "totalTradedVolume": 800}, "PE": {"openInterest": 3000, "lastPrice": 10, "delta": -0.3, "impliedVolatility": 22, "totalTradedVolume": 600}},
        {"strikePrice": 1320, "CE": {"openInterest": 3000, "impliedVolatility": 20, "delta": 0.35, "lastPrice": 8, "totalTradedVolume": 600}, "PE": {"openInterest": 4000, "lastPrice": 15, "delta": -0.5, "impliedVolatility": 24, "totalTradedVolume": 700}},
        {"strikePrice": 1340, "CE": {"openInterest": 1500, "impliedVolatility": 19, "delta": 0.22, "lastPrice": 5, "totalTradedVolume": 400}, "PE": {"openInterest": 2000, "lastPrice": 30, "delta": -0.65, "impliedVolatility": 26, "totalTradedVolume": 500}},
        {"strikePrice": 1360, "CE": {"openInterest": 800, "impliedVolatility": 18, "delta": 0.12, "lastPrice": 3, "totalTradedVolume": 300}, "PE": {"openInterest": 1500, "lastPrice": 50, "delta": -0.8, "impliedVolatility": 28, "totalTradedVolume": 400}},
    ]


def test_select_spread_bull_call_basic():
    chain = _mock_chain_bullish()
    result = select_spread_strikes(chain, spot=1305, direction="bullish", atr=30, budget=10000, lot_size=250)
    assert result is not None
    assert result["long_strike"] <= result["short_strike"]
    assert result["long_option_type"] == "CE"
    assert result["net_debit"] > 0
    assert result["net_debit"] <= 10000
    assert result["max_profit"] > 0
    assert result["rr_ratio"] >= 1.5


def test_select_spread_bear_put_basic():
    chain = _mock_chain_bullish()
    result = select_spread_strikes(chain, spot=1305, direction="bearish", atr=30, budget=10000, lot_size=250)
    assert result is not None
    assert result["long_strike"] >= result["short_strike"]
    assert result["long_option_type"] == "PE"
    assert result["net_debit"] > 0
    assert result["net_debit"] <= 10000


def test_select_spread_no_liquidity():
    chain = [
        {"strikePrice": 1300, "CE": {"openInterest": 100, "impliedVolatility": 22, "delta": 0.5, "lastPrice": 40, "totalTradedVolume": 50}, "PE": {"openInterest": 100, "lastPrice": 20, "delta": -0.3, "impliedVolatility": 22, "totalTradedVolume": 50}},
        {"strikePrice": 1320, "CE": {"openInterest": 50, "impliedVolatility": 20, "delta": 0.35, "lastPrice": 28, "totalTradedVolume": 30}, "PE": {"openInterest": 50, "lastPrice": 35, "delta": -0.5, "impliedVolatility": 24, "totalTradedVolume": 30}},
    ]
    result = select_spread_strikes(chain, spot=1305, direction="bullish", atr=30, budget=10000, lot_size=250)
    assert result is None  # OI too low


def test_select_spread_budget_exceeded():
    chain = _mock_chain_bullish()
    # Very small budget -> can't afford any spread
    result = select_spread_strikes(chain, spot=1305, direction="bullish", atr=30, budget=500, lot_size=250)
    assert result is None  # Too expensive


def test_select_spread_returns_all_fields():
    chain = _mock_chain_bullish()
    result = select_spread_strikes(chain, spot=1305, direction="bullish", atr=30, budget=10000, lot_size=250)
    assert result is not None
    required = ["long_strike", "short_strike", "long_premium", "short_premium",
                "net_debit", "max_profit", "max_loss", "rr_ratio", "width",
                "long_option_type", "short_option_type"]
    for field in required:
        assert field in result, f"Missing field: {field}"


def test_select_spread_width_correct():
    chain = _mock_chain_bullish()
    result = select_spread_strikes(chain, spot=1305, direction="bullish", atr=30, budget=10000, lot_size=250)
    if result:
        assert result["width"] == abs(result["short_strike"] - result["long_strike"])


def test_select_spread_max_loss_equals_debit():
    chain = _mock_chain_bullish()
    result = select_spread_strikes(chain, spot=1305, direction="bullish", atr=30, budget=10000, lot_size=250)
    if result:
        assert result["max_loss"] == result["net_debit"]
