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


# --- Spread exit logic tests ---

def _make_debit_spread_position(direction="bullish", underlying=22500):
    """Helper to create a test spread position."""
    if direction == "bullish":
        return {
            "strategy": "bull_call_spread",
            "spread_type": "debit",
            "spread_direction": "bullish",
            "symbol": "NIFTY",
            "underlying_at_entry": underlying,
            "entry_date": "2026-03-01",
            "expiry": "26MAR2026",
            "quantity": 75,
            "long_leg": {"strike": 22400, "option_type": "CE", "entry_premium": 180},
            "short_leg": {"strike": 22600, "option_type": "CE", "entry_premium": 90},
            "net_debit": (180 - 90) * 75,  # 6750
            "max_profit": (200 - 90) * 75,  # 8250
            "max_loss": (180 - 90) * 75,  # 6750
            "spread_width": 200,
            "atr_at_entry": 100,
            "status": "open",
        }
    else:
        return {
            "strategy": "bear_put_spread",
            "spread_type": "debit",
            "spread_direction": "bearish",
            "symbol": "NIFTY",
            "underlying_at_entry": underlying,
            "entry_date": "2026-03-01",
            "expiry": "26MAR2026",
            "quantity": 75,
            "long_leg": {"strike": 22600, "option_type": "PE", "entry_premium": 170},
            "short_leg": {"strike": 22400, "option_type": "PE", "entry_premium": 85},
            "net_debit": (170 - 85) * 75,
            "max_profit": (200 - 85) * 75,
            "max_loss": (170 - 85) * 75,
            "spread_width": 200,
            "atr_at_entry": 100,
            "status": "open",
        }


def test_spread_exit_profit_cap():
    from paper_trade import check_spread_exit
    pos = _make_debit_spread_position("bullish")
    # max_profit = 8250. Need current_pnl >= 6600 (80%)
    # entry spread = 180-90 = 90. need (current_spread - 90)*75 >= 6600
    # current_spread >= 90 + 88 = 178. So long=200, short=22 -> spread=178
    exit_reason = check_spread_exit(pos, underlying_ltp=22650, today="2026-03-05",
                                     long_premium=200, short_premium=22)
    assert exit_reason == "spread_profit_cap"


def test_spread_exit_stoploss_bullish():
    from paper_trade import check_spread_exit
    pos = _make_debit_spread_position("bullish")
    # Underlying dropped below entry - 1.5*ATR = 22500 - 150 = 22350
    exit_reason = check_spread_exit(pos, underlying_ltp=22340, today="2026-03-03",
                                     long_premium=80, short_premium=60)
    assert exit_reason == "spread_stoploss"


def test_spread_exit_stoploss_bearish():
    from paper_trade import check_spread_exit
    pos = _make_debit_spread_position("bearish")
    # Underlying rose above entry + 1.5*ATR = 22500 + 150 = 22650
    exit_reason = check_spread_exit(pos, underlying_ltp=22660, today="2026-03-03",
                                     long_premium=80, short_premium=60)
    assert exit_reason == "spread_stoploss"


def test_spread_exit_time_exit():
    from paper_trade import check_spread_exit
    pos = _make_debit_spread_position("bullish")
    # 5+ trading days after entry (Mar 1 is Sun, Mar 2 is Mon... Mar 9 is Mon)
    # Trading days: Mar 2,3,4,5,6 = 5 trading days by Mar 6, then Mar 9 = 6
    exit_reason = check_spread_exit(pos, underlying_ltp=22510, today="2026-03-09",
                                     long_premium=150, short_premium=80)
    assert exit_reason == "spread_time_exit"


def test_spread_exit_expiry_safety():
    from paper_trade import check_spread_exit
    pos = _make_debit_spread_position("bullish")
    # Override entry_date so time_exit doesn't trigger first
    pos["entry_date"] = "2026-03-20"
    # Close to expiry (expiry is 26MAR2026, today is 23MAR = 3 days away)
    exit_reason = check_spread_exit(pos, underlying_ltp=22510, today="2026-03-23",
                                     long_premium=150, short_premium=80)
    assert exit_reason == "spread_expiry_safety"


def test_spread_no_exit():
    from paper_trade import check_spread_exit
    pos = _make_debit_spread_position("bullish")
    # Normal conditions -- no exit trigger
    exit_reason = check_spread_exit(pos, underlying_ltp=22520, today="2026-03-03",
                                     long_premium=185, short_premium=88)
    assert exit_reason is None


def test_estimate_premium_ce_itm():
    from paper_trade import estimate_premium_from_underlying
    pos = _make_debit_spread_position("bullish")
    # Underlying at 22700, long CE strike 22400 -> deep ITM
    premium = estimate_premium_from_underlying(pos, "long", 22700, "2026-03-05")
    assert premium >= 300  # at least intrinsic (22700 - 22400)


def test_estimate_premium_ce_otm():
    from paper_trade import estimate_premium_from_underlying
    pos = _make_debit_spread_position("bullish")
    # Underlying at 22300, long CE strike 22400 -> OTM
    premium = estimate_premium_from_underlying(pos, "long", 22300, "2026-03-05")
    assert premium >= 0.05  # minimum floor
    assert premium < 180  # less than entry premium (decaying OTM)


def test_estimate_premium_pe_itm():
    from paper_trade import estimate_premium_from_underlying
    pos = _make_debit_spread_position("bearish")
    # Underlying at 22300, long PE strike 22600 -> ITM
    premium = estimate_premium_from_underlying(pos, "long", 22300, "2026-03-05")
    assert premium >= 300  # at least intrinsic (22600 - 22300)
