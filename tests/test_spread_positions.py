import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from datetime import date

def test_calc_spread_pnl_profit():
    from paper_trade import calc_spread_pnl
    # Entry: bought long at 40, sold short at 18. Net debit = 22 per unit.
    # Now: long worth 50, short worth 10. Net value = 40 per unit.
    # P&L = (40 - 22) * 75 = 1350
    pnl = calc_spread_pnl(40, 50, 18, 10, 75)
    assert pnl == (50 - 10 - (40 - 18)) * 75
    assert pnl > 0

def test_calc_spread_pnl_loss():
    from paper_trade import calc_spread_pnl
    # Entry: long at 40, short at 18. Net debit = 22.
    # Now: long worth 20, short worth 15. Net value = 5.
    # P&L = (5 - 22) * 75 = -1275
    pnl = calc_spread_pnl(40, 20, 18, 15, 75)
    assert pnl < 0

def test_calc_spread_pnl_breakeven():
    from paper_trade import calc_spread_pnl
    pnl = calc_spread_pnl(40, 40, 18, 18, 75)
    assert pnl == 0

def test_open_spread_position_returns_dict():
    from paper_trade import _open_spread_position, _empty_portfolio
    portfolio = _empty_portfolio()
    spread_data = {
        "long_strike": 22400,
        "short_strike": 22600,
        "long_premium": 180,
        "short_premium": 90,
        "net_debit": 6750,
        "max_profit": 8250,
        "max_loss": 6750,
        "rr_ratio": 1.22,
        "width": 200,
        "long_option_type": "CE",
        "short_option_type": "CE",
        "spot": 22500,
    }
    pos = _open_spread_position(
        portfolio, symbol="NIFTY", direction="bullish",
        spread_data=spread_data, allocation=7000,
        regime_at_entry="TRENDING_UP", iv_percentile_at_entry=45,
        expiry_str="26MAR2026"
    )
    assert pos is not None
    assert pos["instrument"] == "SPREAD"
    assert pos["spread_type"] == "debit"
    assert pos["spread_direction"] == "bullish"
    assert pos["strategy"] == "bull_call_spread"
    assert pos["status"] == "open"

def test_open_spread_position_bearish():
    from paper_trade import _open_spread_position, _empty_portfolio
    portfolio = _empty_portfolio()
    spread_data = {
        "long_strike": 22400,
        "short_strike": 22200,
        "long_premium": 170,
        "short_premium": 85,
        "net_debit": 6375,
        "max_profit": 8625,
        "max_loss": 6375,
        "rr_ratio": 1.35,
        "width": 200,
        "long_option_type": "PE",
        "short_option_type": "PE",
        "spot": 22300,
    }
    pos = _open_spread_position(
        portfolio, symbol="NIFTY", direction="bearish",
        spread_data=spread_data, allocation=7000,
        regime_at_entry="TRENDING_DOWN", iv_percentile_at_entry=60,
        expiry_str="26MAR2026"
    )
    assert pos is not None
    assert pos["strategy"] == "bear_put_spread"
    assert pos["long_leg"]["option_type"] == "PE"

def test_open_spread_position_has_slippage():
    from paper_trade import _open_spread_position, _empty_portfolio
    portfolio = _empty_portfolio()
    spread_data = {
        "long_strike": 22400, "short_strike": 22600,
        "long_premium": 180, "short_premium": 90,
        "net_debit": 6750, "max_profit": 8250, "max_loss": 6750,
        "rr_ratio": 1.22, "width": 200,
        "long_option_type": "CE", "short_option_type": "CE",
        "spot": 22500,
    }
    pos = _open_spread_position(
        portfolio, symbol="NIFTY", direction="bullish",
        spread_data=spread_data, allocation=7000,
        regime_at_entry="TRENDING_UP", iv_percentile_at_entry=45,
        expiry_str="26MAR2026"
    )
    # Long leg should cost MORE than raw premium (buy slippage)
    assert pos["long_leg"]["entry_premium"] > 180
    # Short leg should receive LESS than raw premium (sell slippage)
    assert pos["short_leg"]["entry_premium"] < 90

def test_open_spread_position_deducts_capital():
    from paper_trade import _open_spread_position, _empty_portfolio
    portfolio = _empty_portfolio()
    initial_available = portfolio["available_capital"]
    spread_data = {
        "long_strike": 22400, "short_strike": 22600,
        "long_premium": 180, "short_premium": 90,
        "net_debit": 6750, "max_profit": 8250, "max_loss": 6750,
        "rr_ratio": 1.22, "width": 200,
        "long_option_type": "CE", "short_option_type": "CE",
        "spot": 22500,
    }
    pos = _open_spread_position(
        portfolio, symbol="NIFTY", direction="bullish",
        spread_data=spread_data, allocation=7000,
        regime_at_entry="TRENDING_UP", iv_percentile_at_entry=45,
        expiry_str="26MAR2026"
    )
    assert portfolio["available_capital"] < initial_available
    assert len(portfolio["positions"]) == 1
