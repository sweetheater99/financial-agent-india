"""Tests for V4 features: vol-adjusted sizing, time-based SL, option partial profit, weekly report."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime

import config


# ── Volatility-adjusted sizing ─────────────────────────────────────────────


def test_vol_sizing_normal_atr():
    """Stock with normal ATR% (~2%) should get ~1.0x multiplier."""
    ltp = 1000
    atr = 20  # 2% ATR
    atr_pct = (atr / ltp) * 100
    vol_mult = config.VOL_SIZING_BASELINE_ATR_PCT / atr_pct
    vol_mult = max(config.VOL_SIZING_MIN_MULT, min(config.VOL_SIZING_MAX_MULT, vol_mult))
    assert 0.95 <= vol_mult <= 1.05


def test_vol_sizing_high_volatility():
    """Stock with 4% ATR should get ~0.5x multiplier (smaller position)."""
    ltp = 1000
    atr = 40  # 4% ATR
    atr_pct = (atr / ltp) * 100
    vol_mult = config.VOL_SIZING_BASELINE_ATR_PCT / atr_pct
    vol_mult = max(config.VOL_SIZING_MIN_MULT, min(config.VOL_SIZING_MAX_MULT, vol_mult))
    assert 0.45 <= vol_mult <= 0.55


def test_vol_sizing_low_volatility():
    """Stock with 1% ATR should get ~1.5x multiplier (capped)."""
    ltp = 1000
    atr = 10  # 1% ATR
    atr_pct = (atr / ltp) * 100
    vol_mult = config.VOL_SIZING_BASELINE_ATR_PCT / atr_pct
    vol_mult = max(config.VOL_SIZING_MIN_MULT, min(config.VOL_SIZING_MAX_MULT, vol_mult))
    assert vol_mult == config.VOL_SIZING_MAX_MULT  # capped at 1.5


def test_vol_sizing_extreme_volatility():
    """Stock with 8% ATR should hit min multiplier floor."""
    ltp = 1000
    atr = 80  # 8% ATR
    atr_pct = (atr / ltp) * 100
    vol_mult = config.VOL_SIZING_BASELINE_ATR_PCT / atr_pct
    vol_mult = max(config.VOL_SIZING_MIN_MULT, min(config.VOL_SIZING_MAX_MULT, vol_mult))
    assert vol_mult == config.VOL_SIZING_MIN_MULT  # floored at 0.4


# ── Time-based SL tightening ──────────────────────────────────────────────


def test_time_sl_config_exists():
    """V4 time SL config constants exist."""
    assert hasattr(config, "TIME_SL_ENABLED")
    assert hasattr(config, "TIME_SL_HALF_LIFE_MULT")
    assert hasattr(config, "TIME_SL_THREE_QUARTER_MULT")
    assert config.TIME_SL_HALF_LIFE_MULT == 0.75
    assert config.TIME_SL_THREE_QUARTER_MULT == 0.5


def test_time_sl_progression():
    """Half-life mult should be less aggressive than three-quarter mult."""
    # 0.75 ATR at 50% > 0.5 ATR at 75% → tighter as time progresses
    assert config.TIME_SL_HALF_LIFE_MULT > config.TIME_SL_THREE_QUARTER_MULT


# ── Partial profit on options ──────────────────────────────────────────────


def test_opt_partial_config():
    """Option partial profit config exists and has sane values."""
    assert config.OPT_PARTIAL_PROFIT_PCT == 30.0
    assert config.OPT_PARTIAL_RATIO == 0.5


# ── Weekly performance report ──────────────────────────────────────────────


def test_weekly_report_no_trades():
    """Weekly report handles portfolio with no trades."""
    from paper_trade import generate_weekly_report

    portfolio = {
        "capital": 100000,
        "available_capital": 100000,
        "positions": [],
        "closed_trades": [],
        "stats": {"total_trades": 0, "winning_trades": 0, "losing_trades": 0,
                  "total_pnl": 0, "total_pnl_pct": 0, "best_trade": None, "worst_trade": None},
    }
    report = generate_weekly_report(portfolio)
    assert "Weekly Performance Report" in report
    assert "No trades closed this week" in report


def test_weekly_report_with_trades():
    """Weekly report correctly summarizes recent trades."""
    from paper_trade import generate_weekly_report

    today = datetime.now().strftime("%Y-%m-%d")
    portfolio = {
        "capital": 100000,
        "available_capital": 90000,
        "positions": [],
        "closed_trades": [
            {"symbol": "RELIANCE", "direction": "bullish", "instrument": "EQ",
             "entry_price": 2500, "exit_price": 2600, "quantity": 4,
             "pnl_gross": 400, "pnl": 380, "pnl_pct": 3.8,
             "entry_date": today, "exit_date": today, "exit_reason": "target"},
            {"symbol": "TCS", "direction": "bullish", "instrument": "EQ",
             "entry_price": 3500, "exit_price": 3400, "quantity": 3,
             "pnl_gross": -300, "pnl": -320, "pnl_pct": -3.0,
             "entry_date": today, "exit_date": today, "exit_reason": "stoploss"},
        ],
        "stats": {"total_trades": 2, "winning_trades": 1, "losing_trades": 1,
                  "total_pnl": 60, "total_pnl_pct": 0.06, "best_trade": None, "worst_trade": None},
    }
    report = generate_weekly_report(portfolio)
    assert "Week Summary" in report
    assert "RELIANCE" in report
    assert "TCS" in report
    assert "Per-Strategy" in report
    assert "Exit Reasons" in report


def test_weekly_report_per_strategy():
    """Weekly report breaks down by instrument type."""
    from paper_trade import generate_weekly_report

    today = datetime.now().strftime("%Y-%m-%d")
    portfolio = {
        "capital": 100000,
        "available_capital": 95000,
        "positions": [],
        "closed_trades": [
            {"symbol": "RELIANCE", "direction": "bullish", "instrument": "EQ",
             "entry_price": 2500, "exit_price": 2600, "quantity": 4,
             "pnl_gross": 400, "pnl": 380, "pnl_pct": 3.8,
             "entry_date": today, "exit_date": today, "exit_reason": "target"},
            {"symbol": "NIFTY", "direction": "bearish", "instrument": "OPT",
             "entry_price": 200, "exit_price": 100, "quantity": 75,
             "pnl_gross": -7500, "pnl": -7600, "pnl_pct": -50.0,
             "entry_date": today, "exit_date": today, "exit_reason": "stoploss"},
        ],
        "stats": {"total_trades": 2, "winning_trades": 1, "losing_trades": 1,
                  "total_pnl": -7220, "total_pnl_pct": -7.2, "best_trade": None, "worst_trade": None},
    }
    report = generate_weekly_report(portfolio)
    assert "EQ:" in report
    assert "OPT:" in report


# ── Config constants ───────────────────────────────────────────────────────


def test_sector_rotation_config():
    """Sector rotation config constants exist."""
    assert hasattr(config, "SECTOR_ROTATION_BOOST")
    assert hasattr(config, "SECTOR_ROTATION_CUT")
    assert config.SECTOR_ROTATION_BOOST == 0.15
    assert config.SECTOR_ROTATION_CUT == 0.15


def test_oi_unusual_config():
    """Unusual OI config constants exist."""
    assert hasattr(config, "OI_UNUSUAL_CAUTIONARY_CUT")
    assert config.OI_UNUSUAL_CAUTIONARY_CUT == 0.10
