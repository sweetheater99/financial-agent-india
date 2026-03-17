# v7/config_v7.py
"""V7-specific configuration.

Imports shared config from parent config.py for credentials, market hours,
holidays, etc. Defines V7-specific constants.
"""
from datetime import time


# ── Watchlist ──────────────────────────────────────────────────────────
WATCHLIST = [
    {"symbol": "NIFTY",      "type": "index", "lot_size": 75,  "token": "99926000", "exchange": "NSE"},
    {"symbol": "BANKNIFTY",  "type": "index", "lot_size": 30,  "token": "99926009", "exchange": "NSE"},
    {"symbol": "RELIANCE",   "type": "stock", "lot_size": 250, "token": "2885",     "exchange": "NSE"},
    {"symbol": "HDFCBANK",   "type": "stock", "lot_size": 550, "token": "1333",     "exchange": "NSE"},
    {"symbol": "ICICIBANK",  "type": "stock", "lot_size": 700, "token": "4963",     "exchange": "NSE"},
    {"symbol": "TCS",        "type": "stock", "lot_size": 175, "token": "11536",    "exchange": "NSE"},
    {"symbol": "TMPV",       "type": "stock", "lot_size": 575, "token": "884737",   "exchange": "NSE"},
    {"symbol": "BAJFINANCE", "type": "stock", "lot_size": 125, "token": "317",      "exchange": "NSE"},
    {"symbol": "SBIN",       "type": "stock", "lot_size": 750, "token": "3045",     "exchange": "NSE"},
    {"symbol": "INFY",       "type": "stock", "lot_size": 300, "token": "1594",     "exchange": "NSE"},
]

# ── Capital ────────────────────────────────────────────────────────────
CAPITAL = {
    "initial": 300_000,
    "cash_reserve_pct": 0.20,
    "margin_buffer_pct": 0.30,
}

# ── Phase Times (IST) ─────────────────────────────────────────────────
PHASE_TIMES = {
    "premarket_start": time(8, 45),
    "market_open": time(9, 15),
    "opening_read_end": time(9, 45),
    "active_start": time(9, 45),
    "checkin_1": time(10, 30),
    "checkin_2": time(13, 0),
    "wind_down_start": time(14, 30),
    "wind_down_end": time(15, 15),
    "post_close_end": time(15, 30),
    "eod_review": time(15, 33),
}

# ── Risk Limits ────────────────────────────────────────────────────────
RISK_LIMITS = {
    "max_daily_risk_pct": 4.0,
    "max_per_trade_risk_pct": 1.5,
    "max_trades_per_day": 4,
    "max_concurrent_positions": 4,
    "max_consecutive_sl_daily": 3,
    "survival_mode_threshold_pct": 5.0,
    "full_stop_threshold_pct": 8.0,
    "drawdown_reduce_pct": 3.0,
}

# ── Theta Engine Limits ────────────────────────────────────────────────
THETA_LIMITS = {
    "max_margin_pct": 0.40,
    "min_vix": 18.0,
    "max_vix": 25.0,
    "short_delta": 0.20,
    "wing_gap_nifty": 200,
    "profit_target_pct": 0.50,
    "close_by_day": "wednesday",
    "survival_delta": 0.15,
    "max_risk_pct": 3.0,
}

# ── State & Recovery ──────────────────────────────────────────────────
STATE_DIR = "data/v7"
RESTART_COOLDOWN_SECONDS = 300

# ── Brokerage (Zerodha) ───────────────────────────────────────────────
BROKERAGE = {
    "flat_per_order": 20.0,
    "min_trade_value": 2000.0,
    "opt_stt_sell_pct": 0.000625,
    "opt_exchange_pct": 0.000495,
    "opt_stamp_duty_pct": 0.00003,
    "gst_pct": 0.18,
    "slippage_pct": 0.015,
}

# ── Strike Selection ───────────────────────────────────────────────────
STRIKE_FILTERS = {
    "min_oi": 500,
    "min_volume": 100,
    "max_bid_ask_nifty": 2.0,
    "max_bid_ask_banknifty": 5.0,
    "max_bid_ask_stock": 3.0,
    "min_premium": 10.0,
    "directional_delta_range": (0.25, 0.50),
    "spread_sell_delta": 0.25,
    "hedge_delta": 0.10,
}

# ── Trailing Stop ──────────────────────────────────────────────────────
TRAILING = {
    "atr_period": 14,
    "atr_multiplier": 1.5,
    "breakeven_rr": 1.0,
}

# ── Carry Rules ────────────────────────────────────────────────────────
CARRY = {
    "min_profit_pct": 1.5,
    "max_vix": 25.0,
    "min_dte": 3,
    "max_hedge_cost": 500.0,
    "never_carry": ["expiry_day", "event_tomorrow", "vix_above_22"],
}

# ── Telegram ───────────────────────────────────────────────────────────
TELEGRAM = {
    "heartbeat_interval_min": 30,
    "alert_on": ["entry", "exit", "exception", "carry", "eod"],
}


def is_15min_boundary(t: time) -> bool:
    """Check if time is within 1 minute of a 15-min candle close."""
    return t.minute % 15 <= 1


def get_conviction_risk_pct(conviction: str) -> float:
    """Get risk % per trade for a given conviction level."""
    return {"high": 2.0, "medium": 1.5, "low": 0.75}.get(conviction, 1.5)
