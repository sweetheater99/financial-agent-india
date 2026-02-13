"""
Paper Trading Engine — tracks simulated positions based on F&O screener signals.

Runs the screener, opens score-weighted paper positions on top candidates,
monitors for +5% target / -3% stop-loss / 5-day expiry, and tracks cumulative P&L.

Usage:
    python paper_trade.py open        # run screener + open new positions
    python paper_trade.py monitor     # check exits on open positions
    python paper_trade.py status      # print portfolio dashboard
    python paper_trade.py close-all   # force close everything
"""

import argparse
import json
import logging
import math
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config
from connect import get_session
from agent_with_options import fetch_option_chain, get_nearest_expiry
from indicators import compute_atr
from screener import (
    run_screener, fetch_all_signals, score_signals,
    enrich_candidates, apply_indicator_adjustments,
    load_previous_snapshots, compute_persistence,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PORTFOLIO_DIR = Path(__file__).parent / "data" / "paper_trades"
PORTFOLIO_FILE = PORTFOLIO_DIR / "portfolio.json"

TOTAL_CAPITAL = 100_000
TARGET_PCT = 5.0       # +5% profit target
STOPLOSS_PCT = -3.0    # -3% stop-loss
MAX_HOLD_DAYS = 5      # force exit after 5 trading days
TOP_N = 5              # top candidates to trade
MIN_CAPITAL = 1_000    # minimum capital to open new positions

# Put option thresholds (leveraged instrument → wider bands)
PUT_TARGET_PCT = 50.0       # +50% premium gain
PUT_STOPLOSS_PCT = -40.0    # -40% premium loss
MIN_DTE_TO_OPEN = 3         # don't open puts with < 3 days to expiry
MIN_DTE_TO_HOLD = 2         # force exit if < 2 days to expiry

IST = timezone(timedelta(hours=5, minutes=30))

# --- Transaction Cost Rates (AngelOne) ---
BROKERAGE_FLAT = 20.0           # ₹20 per executed order

# Equity Delivery
EQ_STT_PCT = 0.001              # 0.1% on both buy and sell
EQ_EXCHANGE_PCT = 0.0000345     # 0.00345%
EQ_STAMP_DUTY_PCT = 0.00015    # 0.015% buy side only
EQ_SEBI_PCT = 0.000001          # 0.0001%

# Options (F&O)
OPT_STT_SELL_PCT = 0.000625    # 0.0625% sell side only
OPT_EXCHANGE_PCT = 0.000495    # 0.0495%
OPT_STAMP_DUTY_PCT = 0.00003   # 0.003% buy side only
OPT_SEBI_PCT = 0.000001        # 0.0001%

GST_PCT = 0.18                  # 18% on (brokerage + exchange charges)

# --- Slippage ---
EQ_SLIPPAGE_PCT = 0.001         # 0.1% for equity
OPT_SLIPPAGE_PCT = 0.015        # 1.5% for options

# --- Re-entry Cooldown ---
COOLDOWN_DAYS = 5

# --- Sector Concentration ---
MAX_POSITIONS_PER_SECTOR = 2
SYMBOL_SECTOR = {
    # Banking
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "KOTAKBANK": "Banking",
    "SBIN": "Banking", "AXISBANK": "Banking", "BANKBARODA": "Banking",
    "INDUSINDBK": "Banking", "PNB": "Banking", "FEDERALBNK": "Banking",
    "IDFCFIRSTB": "Banking", "AUBANK": "Banking", "BANDHANBNK": "Banking",
    # Financial Services
    "BAJFINANCE": "FinServices", "BAJAJFINSV": "FinServices", "HDFCLIFE": "FinServices",
    "SBILIFE": "FinServices", "ICICIPRULI": "FinServices", "MFSL": "FinServices",
    "CHOLAFIN": "FinServices", "MANAPPURAM": "FinServices", "MUTHOOTFIN": "FinServices",
    "PEL": "FinServices", "LICHSGFIN": "FinServices", "RECLTD": "FinServices",
    "PFC": "FinServices", "SHRIRAMFIN": "FinServices", "CANBK": "FinServices",
    # IT
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT",
    "TECHM": "IT", "LTIM": "IT", "PERSISTENT": "IT", "COFORGE": "IT",
    "MPHASIS": "IT", "TATAELXSI": "IT", "LTTS": "IT",
    # Auto
    "MARUTI": "Auto", "TATAMOTORS": "Auto", "M&M": "Auto",
    "BAJAJ-AUTO": "Auto", "HEROMOTOCO": "Auto", "EICHERMOT": "Auto",
    "ASHOKLEY": "Auto", "BHARATFORG": "Auto", "MOTHERSON": "Auto",
    "TVSMOTOR": "Auto", "BALKRISIND": "Auto", "MRF": "Auto",
    # Pharma
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma",
    "DIVISLAB": "Pharma", "APOLLOHOSP": "Pharma", "BIOCON": "Pharma",
    "LUPIN": "Pharma", "AUROPHARMA": "Pharma", "TORNTPHARM": "Pharma",
    "GRANULES": "Pharma",
    # Metals & Mining
    "TATASTEEL": "Metals", "HINDALCO": "Metals", "JSWSTEEL": "Metals",
    "VEDL": "Metals", "COALINDIA": "Metals", "NMDC": "Metals",
    "SAIL": "Metals", "NATIONALUM": "Metals", "JINDALSTEL": "Metals",
    # Energy
    "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy",
    "IOC": "Energy", "NTPC": "Energy", "POWERGRID": "Energy",
    "ADANIGREEN": "Energy", "TATAPOWER": "Energy", "GAIL": "Energy",
    # FMCG
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "DABUR": "FMCG", "MARICO": "FMCG",
    "GODREJCP": "FMCG", "COLPAL": "FMCG", "TATACONSUM": "FMCG",
    # Infra / Cement
    "ULTRACEMCO": "Infra", "AMBUJACEM": "Infra", "ACC": "Infra",
    "ADANIENT": "Infra", "ADANIPORTS": "Infra", "LT": "Infra",
    "DLF": "Infra", "GODREJPROP": "Infra",
    # Telecom
    "BHARTIARTL": "Telecom", "IDEA": "Telecom",
}

# --- Trailing Stop-Loss ---
TRAILING_STOP_PCT = 2.0         # -2% from peak for equity
OPT_TRAILING_STOP_PCT = 25.0    # -25% from peak premium for options

# --- ATR-Based Exits (equity only) ---
ATR_PERIOD = 14
ATR_TARGET_MULTIPLIER = 2.0     # target = entry + 2*ATR
ATR_STOPLOSS_MULTIPLIER = 1.5   # SL = entry - 1.5*ATR

# --- Option Theta Awareness ---
OPT_THETA_DTE_THRESHOLD = 5     # DTE cutoff
OPT_THETA_MIN_GAIN_PCT = 10.0   # minimum gain to hold through theta

# --- Market Regime ---
NIFTY_TOKEN = "99926000"         # Nifty 50 index token
VIX_TOKEN = "26017"              # India VIX token
NIFTY_SMA_PERIOD = 20
VIX_HIGH_THRESHOLD = 25.0        # reduce sizes above this
VIX_BEARISH_THRESHOLD = 20.0     # skip opens above this AND Nifty < SMA
REGIME_SIZE_REDUCTION = 0.5

# --- Partial Exits (equity only) ---
PARTIAL_EXIT_RATIO = 0.5

# --- Entry Quality Filters ---
RSI_BULLISH_MIN = 40   # skip bullish entries outside [40, 70]
RSI_BULLISH_MAX = 70
RSI_BEARISH_MIN = 30   # skip bearish entries outside [30, 60]
RSI_BEARISH_MAX = 60
VOLUME_MIN_RATIO = 1.0  # skip if volume_ratio < 1.0
SKIP_EARNINGS_SOON = True
SKIP_CONTRADICTING_NEWS = True

# --- Intraday Momentum Confirmation ---
INTRADAY_CONFIRM_ENABLED = True
INTRADAY_MIN_CANDLES = 3            # need at least 3 five-min candles (15 min after open)
INTRADAY_MIN_MOVE_PCT = 0.1        # minimum absolute move to count as confirmation (avoid noise)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logger():
    """Configure paper_trade logger with file + console handlers."""
    log = logging.getLogger("paper_trade")
    if log.handlers:
        return log
    log.setLevel(logging.DEBUG)

    log_dir = Path(__file__).parent / "data" / "paper_trades"
    log_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(log_dir / "paper_trade.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    log.addHandler(fh)
    log.addHandler(ch)
    return log


logger = _setup_logger()


# ---------------------------------------------------------------------------
# macOS Notifications
# ---------------------------------------------------------------------------

def _notify(title: str, message: str) -> None:
    """Send a macOS notification. Fails silently if osascript unavailable."""
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{message}" with title "{title}"'],
            capture_output=True, timeout=5,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Slippage Modeling
# ---------------------------------------------------------------------------

def apply_slippage(price: float, instrument: str, side: str) -> float:
    """
    Apply slippage to a price. Buys get worse (higher), sells get worse (lower).

    instrument: "EQ" or "OPT"
    side: "buy" or "sell"
    """
    pct = OPT_SLIPPAGE_PCT if instrument == "OPT" else EQ_SLIPPAGE_PCT
    if side == "buy":
        return round(price * (1 + pct), 2)
    else:
        return round(price * (1 - pct), 2)


# ---------------------------------------------------------------------------
# Re-entry Cooldown
# ---------------------------------------------------------------------------

def check_cooldown(symbol: str, closed_trades: list, today: str) -> bool:
    """
    Return True if symbol had a stoploss/trailing_stop exit within COOLDOWN_DAYS trading days.

    This blocks re-entry for recently stopped-out symbols.
    """
    for trade in reversed(closed_trades):
        if trade["symbol"] != symbol:
            continue
        if trade.get("exit_reason") not in ("stoploss", "trailing_stop"):
            continue
        exit_date = trade.get("exit_date", "")
        if not exit_date:
            continue
        days_since = _trading_days_between(exit_date, today)
        if days_since <= COOLDOWN_DAYS:
            return True
    return False


# ---------------------------------------------------------------------------
# Sector Concentration
# ---------------------------------------------------------------------------

def get_sector(symbol: str) -> str:
    """Return the sector for a symbol, or 'Other' if unknown."""
    return SYMBOL_SECTOR.get(symbol, "Other")


def check_sector_limit(symbol: str, positions: list) -> bool:
    """
    Return True if adding this symbol would exceed MAX_POSITIONS_PER_SECTOR.

    'Other' sector has no limit.
    """
    sector = get_sector(symbol)
    if sector == "Other":
        return False
    count = sum(1 for p in positions
                if p.get("status") == "open" and get_sector(p["symbol"]) == sector)
    return count >= MAX_POSITIONS_PER_SECTOR


# ---------------------------------------------------------------------------
# Portfolio Persistence
# ---------------------------------------------------------------------------

def _empty_portfolio() -> dict:
    return {
        "capital": TOTAL_CAPITAL,
        "available_capital": TOTAL_CAPITAL,
        "positions": [],
        "closed_trades": [],
        "stats": {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "total_costs": 0.0,
            "best_trade": None,
            "worst_trade": None,
        },
    }


def load_portfolio() -> dict:
    """Load portfolio from JSON, or create a fresh one."""
    if PORTFOLIO_FILE.exists():
        return json.loads(PORTFOLIO_FILE.read_text())
    return _empty_portfolio()


def save_portfolio(portfolio: dict) -> None:
    """Persist portfolio state to disk."""
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    PORTFOLIO_FILE.write_text(json.dumps(portfolio, indent=2, default=str))


# ---------------------------------------------------------------------------
# Token Resolution & Market Data
# ---------------------------------------------------------------------------

def resolve_token(smart_api, symbol: str) -> tuple[str, str] | None:
    """Resolve a symbol to (trading_symbol, token) via searchScrip. Prefers -EQ suffix."""
    try:
        resp = smart_api.searchScrip("NSE", symbol)
        if resp and resp.get("data"):
            for match in resp["data"]:
                if match.get("tradingsymbol", "").endswith("-EQ"):
                    return match["tradingsymbol"], match["symboltoken"]
            first = resp["data"][0]
            return first.get("tradingsymbol", symbol), first["symboltoken"]
    except Exception as e:
        logger.warning("searchScrip failed for %s: %s", symbol, e)
    return None


def get_ltp(smart_api, symbol: str, token: str) -> float | None:
    """Fetch current last traded price for a stock."""
    try:
        resp = smart_api.ltpData("NSE", symbol, token)
        if resp and resp.get("data"):
            ltp = resp["data"].get("ltp")
            if ltp is not None:
                return float(ltp)
    except Exception:
        pass

    # Fallback: getMarketData
    try:
        resp = smart_api.getMarketData({
            "mode": "LTP",
            "exchangeTokens": {"NSE": [token]},
        })
        if resp and resp.get("data") and resp["data"].get("fetched"):
            return float(resp["data"]["fetched"][0]["ltp"])
    except Exception as e:
        logger.warning("LTP fetch failed for %s: %s", symbol, e)
    return None


# ---------------------------------------------------------------------------
# Put Option Helpers
# ---------------------------------------------------------------------------

def get_ltp_nfo(smart_api, symbol: str, token: str) -> float | None:
    """Fetch current last traded price for an NFO contract (options/futures)."""
    try:
        resp = smart_api.ltpData("NFO", symbol, token)
        if resp and resp.get("data"):
            ltp = resp["data"].get("ltp")
            if ltp is not None:
                return float(ltp)
    except Exception:
        pass

    try:
        resp = smart_api.getMarketData({
            "mode": "LTP",
            "exchangeTokens": {"NFO": [token]},
        })
        if resp and resp.get("data") and resp["data"].get("fetched"):
            return float(resp["data"]["fetched"][0]["ltp"])
    except Exception as e:
        logger.warning("NFO LTP fetch failed for %s: %s", symbol, e)
    return None


def days_to_expiry(expiry_str: str) -> int:
    """Calendar days from today to expiry. expiry_str in 'DDMONYYYY' format (e.g. '27FEB2026')."""
    expiry_date = datetime.strptime(expiry_str, "%d%b%Y").date()
    today = datetime.now(IST).date()
    return (expiry_date - today).days


def select_put_strike(option_chain: list, spot_price: float) -> dict | None:
    """
    Pick ATM or 1-strike-OTM put from the option chain.

    Selects the strike ≤ spot_price closest to spot, filtering out zero-LTP strikes.
    Returns dict with strike details or None.
    """
    candidates = []
    for row in option_chain:
        strike = row.get("strikePrice", 0)
        pe = row.get("PE") or {}
        pe_ltp = pe.get("lastTradedPrice", 0) or 0
        if strike <= 0 or pe_ltp <= 0 or strike > spot_price:
            continue
        candidates.append({
            "strike": strike,
            "pe_ltp": float(pe_ltp),
            "pe_delta": float(pe.get("delta", 0) or 0),
            "pe_theta": float(pe.get("theta", 0) or 0),
            "pe_iv": float(pe.get("impliedVolatility", 0) or 0),
            "pe_oi": int(pe.get("openInterest", 0) or 0),
            "pe_gamma": float(pe.get("gamma", 0) or 0),
            "pe_vega": float(pe.get("vega", 0) or 0),
        })

    if not candidates:
        return None

    # Closest strike to spot (ATM or 1-strike OTM put)
    candidates.sort(key=lambda c: spot_price - c["strike"])
    return candidates[0]


def resolve_option_contract(smart_api, symbol: str, strike: float, expiry: str) -> dict | None:
    """
    Resolve a PE option contract via searchScrip.

    Returns {trading_symbol, token, lot_size} or None.
    """
    # Build query like "PERSISTENT 27FEB2026 PE 5400"
    strike_str = str(int(strike)) if strike == int(strike) else str(strike)
    query = f"{symbol}{expiry}P{strike_str}"

    try:
        resp = smart_api.searchScrip("NFO", query)
        if resp and resp.get("data"):
            for match in resp["data"]:
                tsym = match.get("tradingsymbol", "")
                if "PE" in tsym.upper() or tsym.upper().endswith(f"P{strike_str}"):
                    return {
                        "trading_symbol": tsym,
                        "token": match["symboltoken"],
                        "lot_size": int(match.get("lotsize", 1)),
                    }
            # Fallback: first result
            first = resp["data"][0]
            return {
                "trading_symbol": first.get("tradingsymbol", query),
                "token": first["symboltoken"],
                "lot_size": int(first.get("lotsize", 1)),
            }
    except Exception as e:
        logger.warning("searchScrip NFO failed for %s: %s", query, e)
    return None


def _open_put_position(smart_api, symbol: str, eq_token: str, spot_price: float,
                       allocation: float, alloc: dict) -> dict | None:
    """
    Open a put option position for a bearish candidate.

    Returns position dict or None if skipped.
    """
    expiry = get_nearest_expiry()
    if not expiry:
        logger.info("SKIP %s: could not determine option expiry", symbol)
        return None

    dte = days_to_expiry(expiry)
    if dte < MIN_DTE_TO_OPEN:
        logger.info("SKIP %s: expiry %s only %dd away (min %dd)", symbol, expiry, dte, MIN_DTE_TO_OPEN)
        return None

    # Fetch option chain
    option_chain = fetch_option_chain(smart_api, symbol, eq_token)
    time.sleep(config.API_DELAY)
    if not option_chain:
        logger.info("SKIP %s: option chain fetch failed", symbol)
        return None

    # Select strike
    strike_info = select_put_strike(option_chain, spot_price)
    if not strike_info:
        logger.info("SKIP %s: no suitable put strike found", symbol)
        return None

    strike = strike_info["strike"]
    premium = strike_info["pe_ltp"]

    # Resolve NFO contract for token + lot size
    contract = resolve_option_contract(smart_api, symbol, strike, expiry)
    time.sleep(config.API_DELAY)
    if not contract:
        logger.info("SKIP %s: could not resolve PE contract", symbol)
        return None

    lot_size = contract["lot_size"]
    # Size: how many lots can we afford?
    cost_per_lot = premium * lot_size
    if cost_per_lot <= 0:
        logger.info("SKIP %s: zero premium cost", symbol)
        return None

    num_lots = max(1, math.floor(allocation / cost_per_lot))
    actual_allocated = round(num_lots * cost_per_lot, 2)

    # Verify live premium from NFO
    live_premium = get_ltp_nfo(smart_api, contract["trading_symbol"], contract["token"])
    time.sleep(config.API_DELAY)
    if live_premium and live_premium > 0:
        premium = live_premium
        actual_allocated = round(num_lots * premium * lot_size, 2)

    raw_premium = premium
    entry_premium = apply_slippage(premium, "OPT", "buy")
    actual_allocated = round(num_lots * entry_premium * lot_size, 2)

    today = _today_ist()
    max_hold_date = _add_trading_days(today, MAX_HOLD_DAYS)

    position = {
        "symbol": symbol,
        "token": contract["token"],
        "direction": "bearish",
        "instrument": "OPT",
        "option_type": "PE",
        "strike": strike,
        "expiry": expiry,
        "contract_symbol": contract["trading_symbol"],
        "lot_size": lot_size,
        "num_lots": num_lots,
        "entry_price": entry_premium,
        "ltp_at_entry": raw_premium,
        "quantity": num_lots * lot_size,
        "allocated": actual_allocated,
        "underlying_price_at_entry": spot_price,
        "greeks_at_entry": {
            "delta": strike_info["pe_delta"],
            "theta": strike_info["pe_theta"],
            "iv": strike_info["pe_iv"],
            "gamma": strike_info["pe_gamma"],
            "vega": strike_info["pe_vega"],
        },
        "peak_premium": raw_premium,
        "score": alloc["score"],
        "categories": alloc.get("categories", []),
        "entry_date": today,
        "max_hold_date": max_hold_date,
        "status": "open",
    }

    logger.info("OPENED %s PE%d x%dlot (%dqty) @ ₹%.2f (slipped from ₹%.2f, ₹%.0f, score=%.1f, DTE:%d)",
                symbol, int(strike), num_lots, num_lots * lot_size,
                entry_premium, raw_premium, actual_allocated, alloc['score'], dte)

    return position


# ---------------------------------------------------------------------------
# Trading Day Helpers
# ---------------------------------------------------------------------------

def _today_ist() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d")


def _trading_days_between(start_date: str, end_date: str) -> int:
    """Count trading days (weekdays) between two dates, inclusive of both."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    if end < start:
        return 0
    count = 0
    current = start
    while current <= end:
        if current.weekday() < 5:
            count += 1
        current += timedelta(days=1)
    return count


def _add_trading_days(start_date: str, n_days: int) -> str:
    """Add n trading days (weekdays) to a date string."""
    current = datetime.strptime(start_date, "%Y-%m-%d")
    added = 0
    while added < n_days:
        current += timedelta(days=1)
        if current.weekday() < 5:
            added += 1
    return current.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Position Sizing
# ---------------------------------------------------------------------------

def compute_allocations(candidates: list[dict], available_capital: float) -> list[dict]:
    """
    Score-weight capital allocation across candidates.

    Returns list of dicts with 'symbol', 'weight', 'allocation' added.
    Candidates must have 'score' field.
    """
    total_score = sum(c["score"] for c in candidates)
    if total_score <= 0:
        return []

    result = []
    for c in candidates:
        weight = c["score"] / total_score
        allocation = weight * available_capital
        result.append({
            **c,
            "weight": round(weight, 4),
            "allocation": round(allocation, 2),
        })
    return result


# ---------------------------------------------------------------------------
# ATR Computation — imported from indicators.py
# compute_atr is re-exported via `from indicators import compute_atr` above
# ---------------------------------------------------------------------------

def fetch_daily_candles(smart_api, token: str, days: int = 25) -> list | None:
    """Fetch daily candles via getCandleData. Returns raw candle list or None."""
    try:
        from_date = (datetime.now(IST) - timedelta(days=days + 10)).strftime("%Y-%m-%d %H:%M")
        to_date = datetime.now(IST).strftime("%Y-%m-%d %H:%M")
        params = {
            "exchange": "NSE",
            "symboltoken": token,
            "interval": "ONE_DAY",
            "fromdate": from_date,
            "todate": to_date,
        }
        resp = smart_api.getCandleData(params)
        if resp and resp.get("data"):
            return resp["data"]
    except Exception as e:
        logger.debug("getCandleData failed for token %s: %s", token, e)
    return None


def fetch_intraday_candles(smart_api, token: str) -> list | None:
    """Fetch FIVE_MINUTE candles for today's session."""
    try:
        today = datetime.now(IST).strftime("%Y-%m-%d")
        params = {
            "exchange": "NSE",
            "symboltoken": token,
            "interval": "FIVE_MINUTE",
            "fromdate": f"{today} 09:15",
            "todate": f"{today} 15:30",
        }
        resp = smart_api.getCandleData(params)
        if resp and resp.get("data"):
            return resp["data"]
    except Exception as e:
        logger.debug("Intraday candle fetch failed for token %s: %s", token, e)
    return None


def check_intraday_momentum(candles: list, direction: str) -> tuple[bool, str]:
    """Check if intraday price action confirms signal direction.

    Returns (confirmed, reason_str).
    """
    if not candles or len(candles) < INTRADAY_MIN_CANDLES:
        return True, "insufficient_candles"  # pass through if not enough data

    session_open = float(candles[0][1])  # open of first 5-min candle
    latest_close = float(candles[-1][4])  # close of latest candle

    if session_open <= 0:
        return True, "invalid_open"

    intraday_move_pct = (latest_close - session_open) / session_open * 100

    if abs(intraday_move_pct) < INTRADAY_MIN_MOVE_PCT:
        return True, f"flat ({intraday_move_pct:+.2f}%)"  # no clear direction

    if direction == "bullish":
        confirmed = intraday_move_pct > 0
    else:
        confirmed = intraday_move_pct < 0

    status = "confirmed" if confirmed else "contradicted"
    return confirmed, f"{status} ({intraday_move_pct:+.2f}%)"


# ---------------------------------------------------------------------------
# Market Regime
# ---------------------------------------------------------------------------

def classify_regime(nifty_ltp: float | None, nifty_sma: float | None,
                    vix: float | None) -> str:
    """
    Classify market regime based on Nifty position vs SMA and VIX level.

    Returns "normal", "caution", or "skip".
    """
    if nifty_ltp is None or nifty_sma is None or vix is None:
        return "normal"  # missing data → default to normal

    if vix > VIX_BEARISH_THRESHOLD and nifty_ltp < nifty_sma:
        return "skip"
    if vix > VIX_HIGH_THRESHOLD:
        return "caution"
    return "normal"


def fetch_market_regime(smart_api) -> dict:
    """
    Fetch Nifty LTP, 20-day SMA, and VIX to classify market regime.

    Returns {"regime": str, "detail": str, "nifty_ltp", "nifty_sma", "vix"}.
    """
    nifty_ltp = None
    nifty_sma = None
    vix = None

    try:
        resp = smart_api.ltpData("NSE", "Nifty 50", NIFTY_TOKEN)
        if resp and resp.get("data"):
            nifty_ltp = float(resp["data"]["ltp"])
        time.sleep(config.API_DELAY)
    except Exception as e:
        logger.debug("Nifty LTP fetch failed: %s", e)

    try:
        candles = fetch_daily_candles(smart_api, NIFTY_TOKEN, days=NIFTY_SMA_PERIOD + 10)
        if candles and len(candles) >= NIFTY_SMA_PERIOD:
            closes = [float(c[4]) for c in candles[-NIFTY_SMA_PERIOD:]]
            nifty_sma = round(sum(closes) / len(closes), 2)
        time.sleep(config.API_DELAY)
    except Exception as e:
        logger.debug("Nifty SMA calc failed: %s", e)

    try:
        resp = smart_api.ltpData("NSE", "India VIX", VIX_TOKEN)
        if resp and resp.get("data"):
            vix = float(resp["data"]["ltp"])
        time.sleep(config.API_DELAY)
    except Exception as e:
        logger.debug("VIX fetch failed: %s", e)

    regime = classify_regime(nifty_ltp, nifty_sma, vix)
    detail = f"Nifty={nifty_ltp}, SMA20={nifty_sma}, VIX={vix} → {regime}"
    logger.info("Market regime: %s", detail)

    return {
        "regime": regime,
        "detail": detail,
        "nifty_ltp": nifty_ltp,
        "nifty_sma": nifty_sma,
        "vix": vix,
    }


# ---------------------------------------------------------------------------
# Performance Analytics
# ---------------------------------------------------------------------------

def calc_performance_analytics(closed_trades: list, capital: float) -> dict | None:
    """
    Calculate performance metrics from closed trades.

    Returns dict with sharpe_ratio, max_drawdown, max_drawdown_pct,
    profit_factor, expectancy, avg_holding_days, or None if < 3 trades.
    """
    if len(closed_trades) < 3:
        return None

    pnls = [t["pnl"] for t in closed_trades]
    pnl_pcts = [t["pnl_pct"] for t in closed_trades]

    # Sharpe ratio (annualized)
    mean_ret = sum(pnl_pcts) / len(pnl_pcts)
    variance = sum((r - mean_ret) ** 2 for r in pnl_pcts) / len(pnl_pcts)
    std_ret = math.sqrt(variance) if variance > 0 else 0
    sharpe = round((mean_ret / std_ret) * math.sqrt(252), 2) if std_ret > 0 else 0.0

    # Max drawdown (peak-to-trough in cumulative P&L)
    cumulative = 0.0
    peak_cum = 0.0
    max_dd = 0.0
    for p in pnls:
        cumulative += p
        if cumulative > peak_cum:
            peak_cum = cumulative
        dd = peak_cum - cumulative
        if dd > max_dd:
            max_dd = dd
    max_dd_pct = round(max_dd / capital * 100, 2) if capital > 0 else 0.0

    # Profit factor
    wins_sum = sum(p for p in pnls if p > 0)
    losses_sum = abs(sum(p for p in pnls if p < 0))
    profit_factor = round(wins_sum / losses_sum, 2) if losses_sum > 0 else float("inf")

    # Expectancy
    wins = [p for p in pnls if p >= 0]
    losses = [p for p in pnls if p < 0]
    win_rate = len(wins) / len(pnls) if pnls else 0
    loss_rate = 1 - win_rate
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0
    expectancy = round(avg_win * win_rate - avg_loss * loss_rate, 2)

    # Average holding period (trading days)
    holding_days = []
    for t in closed_trades:
        entry = t.get("entry_date", "")
        exit_d = t.get("exit_date", "")
        if entry and exit_d:
            holding_days.append(_trading_days_between(entry, exit_d))
    avg_holding = round(sum(holding_days) / len(holding_days), 1) if holding_days else 0

    return {
        "sharpe_ratio": sharpe,
        "max_drawdown": round(max_dd, 2),
        "max_drawdown_pct": max_dd_pct,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_holding_days": avg_holding,
    }


# ---------------------------------------------------------------------------
# Open Positions
# ---------------------------------------------------------------------------

def open_positions(smart_api, portfolio: dict, candidates: list[dict]) -> int:
    """
    Open paper positions for top candidates. Returns number of positions opened.

    Skips candidates already in open positions or if capital is insufficient.
    """
    if portfolio["available_capital"] < MIN_CAPITAL:
        logger.info("Insufficient capital (₹%.0f < ₹%.0f). Skipping.",
                    portfolio['available_capital'], MIN_CAPITAL)
        return 0

    # --- Market Regime Filter ---
    regime_info = fetch_market_regime(smart_api)
    regime = regime_info["regime"]
    if regime == "skip":
        logger.warning("Market regime: SKIP — %s", regime_info["detail"])
        return 0

    available_capital = portfolio["available_capital"]
    if regime == "caution":
        available_capital = round(available_capital * REGIME_SIZE_REDUCTION, 2)
        logger.info("Market regime: CAUTION — reducing available capital to ₹%.0f", available_capital)

    # Filter out stocks already held
    open_symbols = {p["symbol"] for p in portfolio["positions"] if p["status"] == "open"}
    eligible = [c for c in candidates if c["symbol"] not in open_symbols]

    if not eligible:
        logger.info("No new eligible candidates (all already in portfolio).")
        return 0

    # Filter cooldown and sector limits
    today = _today_ist()
    filtered = []
    for c in eligible:
        if check_cooldown(c["symbol"], portfolio["closed_trades"], today):
            logger.info("SKIP %s: cooldown active (recent stoploss)", c["symbol"])
            continue
        if check_sector_limit(c["symbol"], portfolio["positions"]):
            logger.info("SKIP %s: sector limit reached (%s)", c["symbol"], get_sector(c["symbol"]))
            continue
        filtered.append(c)

    if not filtered:
        logger.info("No candidates after cooldown/sector filters.")
        return 0

    # --- Entry Quality Filters (RSI, volume, news) ---
    quality_filtered = []
    for c in filtered:
        symbol = c["symbol"]
        direction = c["direction"]
        rsi = c.get("rsi")
        vol_ratio = c.get("volume_ratio")
        news = c.get("news_sentiment")

        # RSI filter
        if rsi is not None:
            if direction == "bullish" and not (RSI_BULLISH_MIN <= rsi <= RSI_BULLISH_MAX):
                logger.info("SKIP %s: RSI=%.1f outside bullish range [%d,%d]",
                            symbol, rsi, RSI_BULLISH_MIN, RSI_BULLISH_MAX)
                continue
            if direction == "bearish" and not (RSI_BEARISH_MIN <= rsi <= RSI_BEARISH_MAX):
                logger.info("SKIP %s: RSI=%.1f outside bearish range [%d,%d]",
                            symbol, rsi, RSI_BEARISH_MIN, RSI_BEARISH_MAX)
                continue

        # Volume filter
        if vol_ratio is not None and vol_ratio < VOLUME_MIN_RATIO:
            logger.info("SKIP %s: volume ratio %.2f < %.1f minimum",
                        symbol, vol_ratio, VOLUME_MIN_RATIO)
            continue

        # News / earnings filter
        if news is not None:
            if SKIP_EARNINGS_SOON and news.get("has_earnings_soon"):
                logger.info("SKIP %s: earnings event imminent (gap risk)", symbol)
                continue
            if SKIP_CONTRADICTING_NEWS:
                sentiment = news.get("sentiment")
                if direction == "bullish" and sentiment == "negative":
                    logger.info("SKIP %s: bullish signal contradicts negative news", symbol)
                    continue
                if direction == "bearish" and sentiment == "positive":
                    logger.info("SKIP %s: bearish signal contradicts positive news", symbol)
                    continue

        quality_filtered.append(c)

    if not quality_filtered:
        logger.info("No candidates after quality filters (RSI/volume/news).")
        return 0

    top = quality_filtered[:TOP_N]
    allocations = compute_allocations(top, available_capital)
    opened = 0

    for alloc in allocations:
        symbol = alloc["symbol"]
        direction = alloc["direction"]

        # Resolve equity token + LTP (needed for both paths)
        resolved = resolve_token(smart_api, symbol)
        if not resolved:
            logger.info("SKIP %s: token resolution failed", symbol)
            continue
        trading_symbol, token = resolved
        time.sleep(config.API_DELAY)

        ltp = get_ltp(smart_api, trading_symbol, token)
        if ltp is None or ltp <= 0:
            logger.info("SKIP %s: could not get LTP", symbol)
            continue
        time.sleep(config.API_DELAY)

        allocation = alloc["allocation"]

        # Intraday momentum confirmation
        if INTRADAY_CONFIRM_ENABLED:
            is_open, _ = config.is_market_open()
            if is_open:
                intraday_candles = fetch_intraday_candles(smart_api, token)
                time.sleep(config.API_DELAY)
                confirmed, momentum_detail = check_intraday_momentum(intraday_candles, direction)
                if not confirmed:
                    logger.info("SKIP %s: intraday momentum %s", symbol, momentum_detail)
                    continue
                logger.info("  %s: intraday momentum %s", symbol, momentum_detail)

        if direction == "bearish":
            # --- Put option path ---
            position = _open_put_position(smart_api, symbol, token, ltp, allocation, alloc)
            if position is None:
                continue
            position["market_regime"] = regime
        else:
            # --- Equity buy path ---
            entry_price = apply_slippage(ltp, "EQ", "buy")
            quantity = math.floor(allocation / entry_price)
            if quantity < 1:
                logger.info("SKIP %s: allocation ₹%.0f < 1 share at ₹%.2f", symbol, allocation, entry_price)
                continue

            actual_allocated = round(quantity * entry_price, 2)

            # ATR-based exit thresholds
            atr_at_entry = None
            candles = fetch_daily_candles(smart_api, token)
            time.sleep(config.API_DELAY)
            if candles:
                atr_at_entry = compute_atr(candles, ATR_PERIOD)

            if atr_at_entry is not None:
                target_price = round(entry_price + ATR_TARGET_MULTIPLIER * atr_at_entry, 2)
                stoploss_price = round(entry_price - ATR_STOPLOSS_MULTIPLIER * atr_at_entry, 2)
                logger.debug("%s ATR=%.2f → target=%.2f, SL=%.2f", symbol, atr_at_entry, target_price, stoploss_price)
            else:
                target_price = round(entry_price * (1 + TARGET_PCT / 100), 2)
                stoploss_price = round(entry_price * (1 + STOPLOSS_PCT / 100), 2)

            max_hold_date = _add_trading_days(today, MAX_HOLD_DAYS)

            position = {
                "symbol": symbol,
                "token": token,
                "direction": direction,
                "instrument": "EQ",
                "entry_price": entry_price,
                "ltp_at_entry": ltp,
                "quantity": quantity,
                "allocated": actual_allocated,
                "score": alloc["score"],
                "categories": alloc.get("categories", []),
                "entry_date": today,
                "target_price": target_price,
                "stoploss_price": stoploss_price,
                "atr_at_entry": atr_at_entry,
                "peak_price": ltp,
                "market_regime": regime,
                "max_hold_date": max_hold_date,
                "status": "open",
            }

            logger.info("OPENED %s BUY x%d @ ₹%.2f (slipped from ₹%.2f, ₹%.0f, score=%.1f)",
                        symbol, quantity, entry_price, ltp, actual_allocated, alloc['score'])

        portfolio["positions"].append(position)
        portfolio["available_capital"] = round(
            portfolio["available_capital"] - position["allocated"], 2)
        opened += 1

    return opened


# ---------------------------------------------------------------------------
# Exit / Close Logic
# ---------------------------------------------------------------------------

def calc_pnl_pct(entry_price: float, current_price: float, direction: str) -> float:
    """Calculate unrealized P&L percentage."""
    if direction == "bullish":
        return (current_price - entry_price) / entry_price * 100
    else:
        return (entry_price - current_price) / entry_price * 100


def calc_transaction_costs(instrument: str, side: str, price: float, quantity: int) -> dict:
    """
    Calculate transaction costs for one leg (buy or sell).

    instrument: "EQ" or "OPT"
    side: "buy" or "sell"
    Returns: {"brokerage", "stt", "exchange", "stamp_duty", "sebi", "gst", "total"}
    """
    turnover = price * quantity

    if instrument == "OPT":
        brokerage = BROKERAGE_FLAT
        stt = turnover * OPT_STT_SELL_PCT if side == "sell" else 0.0
        exchange = turnover * OPT_EXCHANGE_PCT
        stamp_duty = turnover * OPT_STAMP_DUTY_PCT if side == "buy" else 0.0
        sebi = turnover * OPT_SEBI_PCT
    else:
        # Equity delivery: brokerage is min(₹20, 0.03% of turnover)
        brokerage = min(BROKERAGE_FLAT, turnover * 0.0003)
        stt = turnover * EQ_STT_PCT
        exchange = turnover * EQ_EXCHANGE_PCT
        stamp_duty = turnover * EQ_STAMP_DUTY_PCT if side == "buy" else 0.0
        sebi = turnover * EQ_SEBI_PCT

    gst = (brokerage + exchange) * GST_PCT

    total = brokerage + stt + exchange + stamp_duty + sebi + gst
    return {
        "brokerage": round(brokerage, 4),
        "stt": round(stt, 4),
        "exchange": round(exchange, 4),
        "stamp_duty": round(stamp_duty, 4),
        "sebi": round(sebi, 4),
        "gst": round(gst, 4),
        "total": round(total, 4),
    }


def calc_round_trip_costs(instrument: str, entry_price: float, exit_price: float,
                          quantity: int) -> dict:
    """
    Calculate total round-trip transaction costs (buy + sell).

    Returns combined breakdown with entry_total, exit_total, and total.
    """
    entry_costs = calc_transaction_costs(instrument, "buy", entry_price, quantity)
    exit_costs = calc_transaction_costs(instrument, "sell", exit_price, quantity)

    return {
        "entry_total": entry_costs["total"],
        "exit_total": exit_costs["total"],
        "total": round(entry_costs["total"] + exit_costs["total"], 4),
        "entry": entry_costs,
        "exit": exit_costs,
    }


def close_position(portfolio: dict, pos: dict, exit_price: float, reason: str,
                   close_qty: int | None = None) -> dict:
    """
    Close a position (fully or partially): move to closed_trades, update stats, free capital.

    close_qty: if set, only close this many units (partial exit). Position stays open with reduced qty.
    Returns the closed trade record.
    """
    direction = pos["direction"]
    entry_price = pos["entry_price"]
    quantity = pos["quantity"]
    abs_qty = abs(quantity)

    # Partial vs full close
    if close_qty is not None:
        exit_qty = close_qty
    else:
        exit_qty = abs_qty

    # Option positions: we bought the put, so P&L = premium change (bullish calc)
    is_option = pos.get("instrument") == "OPT"
    pnl_direction = "bullish" if is_option else direction

    if pnl_direction == "bullish":
        pnl_gross = round((exit_price - entry_price) * exit_qty, 2)
    else:
        pnl_gross = round((entry_price - exit_price) * exit_qty, 2)

    instrument = pos.get("instrument", "EQ")
    costs = calc_round_trip_costs(instrument, entry_price, exit_price, exit_qty)
    pnl = round(pnl_gross - costs["total"], 2)
    pnl_pct = round(pnl / (entry_price * exit_qty) * 100, 2)

    closed = {
        "symbol": pos["symbol"],
        "direction": direction,
        "instrument": instrument,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "quantity": exit_qty if quantity >= 0 else -exit_qty,
        "pnl_gross": pnl_gross,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "transaction_costs": round(costs["total"], 2),
        "entry_date": pos["entry_date"],
        "exit_date": _today_ist(),
        "exit_reason": reason,
    }

    portfolio["closed_trades"].append(closed)

    # macOS notification
    emoji = "+" if pnl >= 0 else ""
    reason_label = {"target": "Target hit", "stoploss": "Stop-loss hit",
                    "trailing_stop": "Trailing stop hit", "theta_decay": "Theta decay exit",
                    "partial_target": "Partial target", "expiry": "Max hold expired",
                    "manual": "Manual close"}.get(reason, reason)
    _notify("Paper Trade Exit",
            f"{pos['symbol']} {reason_label}: {pnl_pct:+.1f}% (₹{emoji}{pnl:,.0f})")

    if close_qty is not None:
        # Partial close: reduce position, free proportional capital
        ratio = exit_qty / abs_qty
        freed = round(pos["allocated"] * ratio, 2)
        pos["allocated"] = round(pos["allocated"] - freed, 2)
        if quantity >= 0:
            pos["quantity"] = quantity - exit_qty
        else:
            pos["quantity"] = quantity + exit_qty
        portfolio["available_capital"] = round(portfolio["available_capital"] + freed, 2)
    else:
        # Full close
        pos["status"] = "closed"
        portfolio["available_capital"] = round(portfolio["available_capital"] + pos["allocated"], 2)

    # Update cumulative stats
    stats = portfolio["stats"]
    stats["total_trades"] += 1
    stats["total_pnl"] = round(stats["total_pnl"] + pnl, 2)
    stats["total_pnl_pct"] = round(stats["total_pnl"] / portfolio["capital"] * 100, 2)
    stats["total_costs"] = round(stats.get("total_costs", 0) + costs["total"], 2)

    if pnl >= 0:
        stats["winning_trades"] += 1
    else:
        stats["losing_trades"] += 1

    if stats["best_trade"] is None or pnl_pct > stats["best_trade"]["pnl_pct"]:
        stats["best_trade"] = {"symbol": pos["symbol"], "pnl_pct": round(pnl_pct, 2)}
    if stats["worst_trade"] is None or pnl_pct < stats["worst_trade"]["pnl_pct"]:
        stats["worst_trade"] = {"symbol": pos["symbol"], "pnl_pct": round(pnl_pct, 2)}

    return closed


def monitor_positions(smart_api, portfolio: dict) -> int:
    """
    Check each open position for exit conditions. Returns number of exits.
    """
    open_pos = [p for p in portfolio["positions"] if p["status"] == "open"]
    if not open_pos:
        logger.info("No open positions to monitor.")
        return 0

    today = _today_ist()
    exits = 0

    for pos in open_pos:
        symbol = pos["symbol"]
        is_option = pos.get("instrument") == "OPT"

        # Check max hold expiry first (doesn't need LTP)
        expired = today > pos["max_hold_date"]

        # Check DTE-based forced exit for options
        dte_expired = False
        dte = None
        if is_option:
            dte = days_to_expiry(pos["expiry"])
            if dte < MIN_DTE_TO_HOLD:
                dte_expired = True

        # Fetch current LTP (NFO for options, NSE for equity)
        if is_option:
            ltp = get_ltp_nfo(smart_api, pos.get("contract_symbol", symbol), pos["token"])
        else:
            ltp = get_ltp(smart_api, symbol, pos["token"])
            if ltp is None:
                ltp = get_ltp(smart_api, f"{symbol}-EQ", pos["token"])
        time.sleep(config.API_DELAY)

        if ltp is None:
            logger.warning("%s: could not fetch LTP, skipping", symbol)
            continue

        # --- Trailing stop: update peak ---
        if is_option:
            peak_key = "peak_premium"
            peak = pos.get(peak_key, pos["entry_price"])
            if ltp > peak:
                pos[peak_key] = ltp
                peak = ltp
        else:
            peak_key = "peak_price"
            peak = pos.get(peak_key, pos.get("ltp_at_entry", pos["entry_price"]))
            if pos["direction"] == "bullish":
                if ltp > peak:
                    pos[peak_key] = ltp
                    peak = ltp
            else:
                if ltp < peak:
                    pos[peak_key] = ltp
                    peak = ltp

        # P&L: options use "bullish" direction (bought the put)
        pnl_direction = "bullish" if is_option else pos["direction"]
        pnl_pct = calc_pnl_pct(pos["entry_price"], ltp, pnl_direction)

        # --- Determine exit reason ---
        reason = None

        if is_option:
            # Theta awareness: low DTE + low gain → early exit
            if dte is not None and dte < OPT_THETA_DTE_THRESHOLD and pnl_pct < OPT_THETA_MIN_GAIN_PCT:
                if not dte_expired:  # dte_expired (DTE < 2) handled separately
                    reason = "theta_decay"

            if reason is None:
                # Option trailing stop
                trailing_stop_pct = OPT_TRAILING_STOP_PCT
                trailing_sl = peak * (1 - trailing_stop_pct / 100)
                fixed_sl = pos["entry_price"] * (1 + PUT_STOPLOSS_PCT / 100)
                effective_sl = max(trailing_sl, fixed_sl)

                if pnl_pct >= PUT_TARGET_PCT:
                    reason = "target"
                elif ltp <= effective_sl:
                    reason = "trailing_stop" if trailing_sl > fixed_sl else "stoploss"
                elif dte_expired:
                    reason = "expiry"
                elif expired:
                    reason = "expiry"
        else:
            # Equity: ATR-based or percentage-based exits
            atr_at_entry = pos.get("atr_at_entry")

            if atr_at_entry is not None:
                # ATR-based: price thresholds
                if pos["direction"] == "bullish":
                    if ltp >= pos["target_price"]:
                        reason = "target"
                    else:
                        # Trailing stop
                        trailing_sl = peak * (1 - TRAILING_STOP_PCT / 100)
                        fixed_sl = pos["stoploss_price"]
                        effective_sl = max(trailing_sl, fixed_sl)
                        if ltp <= effective_sl:
                            reason = "trailing_stop" if trailing_sl > fixed_sl else "stoploss"
                else:
                    if ltp <= pos["target_price"]:
                        reason = "target"
                    else:
                        trailing_sl = peak * (1 + TRAILING_STOP_PCT / 100)
                        fixed_sl = pos["stoploss_price"]
                        effective_sl = min(trailing_sl, fixed_sl)
                        if ltp >= effective_sl:
                            reason = "trailing_stop" if trailing_sl < fixed_sl else "stoploss"
            else:
                # Legacy percentage-based
                target_pct = TARGET_PCT
                stoploss_pct = STOPLOSS_PCT

                if pos["direction"] == "bullish":
                    # Trailing stop for bullish equity
                    trailing_sl = peak * (1 - TRAILING_STOP_PCT / 100)
                    fixed_sl = pos.get("stoploss_price", pos["entry_price"] * (1 + stoploss_pct / 100))
                    effective_sl = max(trailing_sl, fixed_sl)

                    if pnl_pct >= target_pct:
                        reason = "target"
                    elif ltp <= effective_sl:
                        reason = "trailing_stop" if trailing_sl > fixed_sl else "stoploss"
                else:
                    trailing_sl = peak * (1 + TRAILING_STOP_PCT / 100)
                    fixed_sl = pos.get("stoploss_price", pos["entry_price"] * (1 - stoploss_pct / 100))
                    effective_sl = min(trailing_sl, fixed_sl)

                    if pnl_pct >= target_pct:
                        reason = "target"
                    elif ltp >= effective_sl:
                        reason = "trailing_stop" if trailing_sl < fixed_sl else "stoploss"

            if reason is None and expired:
                reason = "expiry"

        if reason is None:
            # HOLD
            day_num = _trading_days_between(pos["entry_date"], today)
            max_days = _trading_days_between(pos["entry_date"], pos["max_hold_date"])
            if is_option:
                logger.info("%s PE%d: Prem ₹%.2f  P&L: %+.1f%%  DTE:%s  Day %d/%d  [HOLD]",
                            symbol, int(pos['strike']), ltp, pnl_pct, dte, day_num, max_days)
            else:
                logger.info("%s: LTP ₹%.2f  P&L: %+.1f%%  Day %d/%d  [HOLD]",
                            symbol, ltp, pnl_pct, day_num, max_days)
            continue

        # --- Partial exit for equity target ---
        if reason == "target" and not is_option and not pos.get("partial_exit_done"):
            partial_qty = abs(pos["quantity"]) // 2
            if partial_qty >= 1:
                exit_price = apply_slippage(ltp, "EQ", "sell")
                close_position(portfolio, pos, exit_price, "partial_target", close_qty=partial_qty)
                pos["partial_exit_done"] = True
                logger.info("%s: PARTIAL EXIT (%d/%d) @ ₹%.2f  [remaining holds]",
                            symbol, partial_qty, abs(pos["quantity"]) + partial_qty, exit_price)
                continue

        # Apply slippage on exit
        instrument = pos.get("instrument", "EQ")
        exit_price = apply_slippage(ltp, instrument, "sell")

        closed = close_position(portfolio, pos, exit_price, reason)
        tag = reason.upper().replace("_", " ")
        if is_option:
            logger.info("%s PE%d: EXIT (%s) @ ₹%.2f  P&L: %+.1f%% (₹%+.0f)",
                        symbol, int(pos['strike']), tag, exit_price, closed['pnl_pct'], closed['pnl'])
        else:
            logger.info("%s: EXIT (%s) @ ₹%.2f  P&L: %+.1f%% (₹%+.0f)",
                        symbol, tag, exit_price, closed['pnl_pct'], closed['pnl'])
        exits += 1

    # Clean up closed positions from the open list
    portfolio["positions"] = [p for p in portfolio["positions"] if p["status"] == "open"]

    return exits


def close_all_positions(smart_api, portfolio: dict) -> int:
    """Force-close all open positions at current LTP."""
    open_pos = [p for p in portfolio["positions"] if p["status"] == "open"]
    if not open_pos:
        logger.info("No open positions to close.")
        return 0

    exits = 0
    for pos in open_pos:
        is_option = pos.get("instrument") == "OPT"
        if is_option:
            ltp = get_ltp_nfo(smart_api, pos.get("contract_symbol", pos["symbol"]), pos["token"])
        else:
            ltp = get_ltp(smart_api, pos["symbol"], pos["token"])
            if ltp is None:
                ltp = get_ltp(smart_api, f"{pos['symbol']}-EQ", pos["token"])
        time.sleep(config.API_DELAY)

        if ltp is None:
            logger.warning("%s: could not fetch LTP, using entry price", pos['symbol'])
            ltp = pos["entry_price"]

        instrument = pos.get("instrument", "EQ")
        exit_price = apply_slippage(ltp, instrument, "sell")
        closed = close_position(portfolio, pos, exit_price, "manual")
        logger.info("%s: CLOSED @ ₹%.2f  P&L: %+.1f%% (₹%+.0f)",
                    pos['symbol'], exit_price, closed['pnl_pct'], closed['pnl'])
        exits += 1

    portfolio["positions"] = [p for p in portfolio["positions"] if p["status"] == "open"]
    return exits


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def print_portfolio_status(portfolio: dict, smart_api=None) -> None:
    """Formatted terminal dashboard."""
    now = datetime.now(IST)
    today = now.strftime("%Y-%m-%d")
    border = "═" * 59
    sep = "─" * 57

    open_pos = [p for p in portfolio["positions"] if p["status"] == "open"]
    deployed = sum(p["allocated"] for p in open_pos)

    print(f"\n{border}")
    print(f"  PAPER TRADING DASHBOARD")
    print(f"  {now.strftime('%d %b %Y, %H:%M')}")
    print(border)
    print()
    print(f"  Capital: ₹{portfolio['capital']:,.0f}  |  "
          f"Deployed: ₹{deployed:,.0f}  |  "
          f"Available: ₹{portfolio['available_capital']:,.0f}")

    # Open positions
    print(f"\n  OPEN POSITIONS ({len(open_pos)})")
    print(f"  {sep}")

    if open_pos:
        for pos in open_pos:
            is_option = pos.get("instrument") == "OPT"
            day_num = _trading_days_between(pos["entry_date"], today)
            max_days = _trading_days_between(pos["entry_date"], pos["max_hold_date"])

            # Try to get live LTP if smart_api is available
            ltp = None
            if smart_api:
                if is_option:
                    ltp = get_ltp_nfo(smart_api, pos.get("contract_symbol", pos["symbol"]), pos["token"])
                else:
                    ltp = get_ltp(smart_api, pos["symbol"], pos["token"])
                    if ltp is None:
                        ltp = get_ltp(smart_api, f"{pos['symbol']}-EQ", pos["token"])
                time.sleep(config.API_DELAY)

            if is_option:
                dte = days_to_expiry(pos["expiry"])
                tag = f"PE{int(pos['strike'])}"
                if ltp:
                    pnl_pct = calc_pnl_pct(pos["entry_price"], ltp, "bullish")
                    print(f"  {pos['symbol']:<13}{tag}  "
                          f"Prem: ₹{pos['entry_price']:,.2f} → ₹{ltp:,.2f}  "
                          f"P&L: {pnl_pct:+.1f}%  "
                          f"DTE:{dte}  Day {day_num}/{max_days}")
                else:
                    print(f"  {pos['symbol']:<13}{tag}  "
                          f"Prem: ₹{pos['entry_price']:,.2f} → ---  "
                          f"DTE:{dte}  Day {day_num}/{max_days}")
            else:
                dir_tag = "BULL" if pos["direction"] == "bullish" else "BEAR"
                if ltp:
                    pnl_pct = calc_pnl_pct(pos["entry_price"], ltp, pos["direction"])
                    print(f"  {pos['symbol']:<13}{dir_tag}  "
                          f"Entry: ₹{pos['entry_price']:,.2f}  "
                          f"LTP: ₹{ltp:,.2f}  "
                          f"P&L: {pnl_pct:+.1f}%  "
                          f"Day {day_num}/{max_days}")
                else:
                    print(f"  {pos['symbol']:<13}{dir_tag}  "
                          f"Entry: ₹{pos['entry_price']:,.2f}  "
                          f"LTP: ---  "
                          f"Day {day_num}/{max_days}")
    else:
        print("  (none)")

    # Recent closed trades
    closed = portfolio["closed_trades"]
    recent = closed[-5:] if closed else []
    print(f"\n  RECENT CLOSED (last {len(recent)})")
    print(f"  {sep}")

    if recent:
        for t in reversed(recent):
            dir_tag = "BULL" if t["direction"] == "bullish" else "BEAR"
            pnl_sign = "+" if t["pnl"] >= 0 else ""
            dates = f"({t['entry_date'][5:]}–{t['exit_date'][5:]})"
            print(f"  {t['symbol']:<13}{dir_tag}  "
                  f"{t['pnl_pct']:+.1f}%  "
                  f"₹{pnl_sign}{t['pnl']:,.0f}  "
                  f"{t['exit_reason']:<10} {dates}")
    else:
        print("  (none)")

    # Cumulative stats
    stats = portfolio["stats"]
    print(f"\n  CUMULATIVE STATS")
    print(f"  {sep}")

    total = stats["total_trades"]
    if total > 0:
        win_rate = stats["winning_trades"] / total * 100
        print(f"  Total Trades: {total}  |  "
              f"Win Rate: {win_rate:.1f}%  ({stats['winning_trades']}/{total})")
        print(f"  Total P&L: ₹{stats['total_pnl']:+,.0f}  "
              f"({stats['total_pnl_pct']:+.2f}% on capital)")

        total_costs = stats.get("total_costs", 0)
        if total_costs > 0:
            brok_sum = stt_sum = other_sum = 0.0
            for t in closed:
                if not t.get("transaction_costs"):
                    continue
                inst = t.get("instrument", "EQ")
                rt = calc_round_trip_costs(inst, t["entry_price"], t["exit_price"], abs(t["quantity"]))
                brok_sum += rt["entry"]["brokerage"] + rt["exit"]["brokerage"]
                stt_sum += rt["entry"]["stt"] + rt["exit"]["stt"]
                other_sum += (rt["entry"]["exchange"] + rt["exit"]["exchange"]
                              + rt["entry"]["stamp_duty"] + rt["exit"]["stamp_duty"]
                              + rt["entry"]["sebi"] + rt["exit"]["sebi"]
                              + rt["entry"]["gst"] + rt["exit"]["gst"])
            print(f"  Total Costs: ₹{total_costs:,.0f}  "
                  f"(brokerage ₹{brok_sum:,.0f}, STT ₹{stt_sum:,.0f}, other ₹{other_sum:,.0f})")

        # Average win/loss
        wins = [t for t in closed if t["pnl"] >= 0]
        losses = [t for t in closed if t["pnl"] < 0]
        avg_win = sum(t["pnl_pct"] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0
        print(f"  Avg Win: {avg_win:+.1f}%  |  Avg Loss: {avg_loss:+.1f}%")

        if stats["best_trade"]:
            print(f"  Best: {stats['best_trade']['symbol']} ({stats['best_trade']['pnl_pct']:+.1f}%)  |  "
                  f"Worst: {stats['worst_trade']['symbol']} ({stats['worst_trade']['pnl_pct']:+.1f}%)")
    else:
        print("  No trades yet.")

    # Performance analytics
    analytics = calc_performance_analytics(closed, portfolio["capital"])
    if analytics:
        print(f"\n  PERFORMANCE ANALYTICS")
        print(f"  {sep}")
        pf_display = f"{analytics['profit_factor']:.2f}" if analytics['profit_factor'] != float('inf') else "∞"
        print(f"  Sharpe Ratio: {analytics['sharpe_ratio']:.2f}  |  "
              f"Profit Factor: {pf_display}")
        print(f"  Max Drawdown: ₹{analytics['max_drawdown']:,.0f} ({analytics['max_drawdown_pct']:.2f}%)")
        print(f"  Expectancy: ₹{analytics['expectancy']:,.0f}/trade  |  "
              f"Avg Holding: {analytics['avg_holding_days']:.1f} days")

    print(f"\n{border}\n")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_paper_trade(smart_api, mode: str) -> None:
    """
    Main orchestrator.
      mode="open"      — run screener + open new positions
      mode="monitor"   — check exits on open positions
      mode="status"    — print dashboard
      mode="close-all" — force close everything
    """
    portfolio = load_portfolio()

    if mode == "open":
        logger.info("\n=== PAPER TRADE: Opening new positions ===\n")

        is_open, status_msg = config.is_market_open()
        if not is_open:
            logger.info("Note: %s", status_msg)
            logger.info("Opening anyway (LTP will be previous close).\n")

        logger.info("Running screener (raw mode)...")
        result = run_screener(smart_api, top_n=TOP_N, raw_only=True)

        if "error" in result:
            logger.info("Screener error: %s", result['error'])
            return

        candidates = result.get("candidates", [])
        if not candidates:
            logger.info("No candidates from screener.")
            return

        # Enrich top candidates with RSI, volume, and indicator adjustments
        top_buffer = candidates[:TOP_N * 2]  # extra buffer for filters
        logger.info("\nEnriching top %d candidates with indicators...", len(top_buffer))
        enrich_candidates(smart_api, top_buffer)

        # Signal persistence (multi-day confirmation)
        prev_snapshots = load_previous_snapshots()
        if prev_snapshots:
            compute_persistence(top_buffer, prev_snapshots)
            logger.info("Persistence: checked %d previous snapshot(s)", len(prev_snapshots))

        apply_indicator_adjustments(top_buffer)

        # News sentiment (if fetch_news_sentiment is available)
        try:
            from screener import fetch_news_sentiment
            logger.info("Fetching news sentiment...")
            fetch_news_sentiment(top_buffer)
        except ImportError:
            pass

        logger.info("\nTop %d candidates for paper trading:", min(TOP_N, len(top_buffer)))
        for i, c in enumerate(top_buffer[:TOP_N], 1):
            dir_tag = "BULL" if c["direction"] == "bullish" else "BEAR"
            extras = []
            if c.get("rsi") is not None:
                extras.append(f"RSI={c['rsi']:.0f}")
            if c.get("volume_ratio") is not None:
                extras.append(f"Vol={c['volume_ratio']:.1f}x")
            extra_str = f"  ({', '.join(extras)})" if extras else ""
            logger.info("  %d. %-15s [%s]  Score: %.1f%s", i, c['symbol'], dir_tag, c['score'], extra_str)

        logger.info("\nOpening positions (available capital: ₹%.0f)...", portfolio['available_capital'])
        opened = open_positions(smart_api, portfolio, top_buffer)
        logger.info("\n  Opened %d position(s).", opened)

        save_portfolio(portfolio)
        print_portfolio_status(portfolio)

    elif mode == "monitor":
        logger.info("\n=== PAPER TRADE: Monitoring positions ===\n")

        exits = monitor_positions(smart_api, portfolio)
        if exits:
            logger.info("\n  %d position(s) closed.", exits)
        save_portfolio(portfolio)

        # Print brief status
        open_count = len([p for p in portfolio["positions"] if p["status"] == "open"])
        logger.info("Open: %d  |  Total P&L: ₹%+.0f", open_count, portfolio['stats']['total_pnl'])

    elif mode == "status":
        print_portfolio_status(portfolio, smart_api)

    elif mode == "close-all":
        logger.info("\n=== PAPER TRADE: Force closing all positions ===\n")
        exits = close_all_positions(smart_api, portfolio)
        logger.info("\n  Closed %d position(s).", exits)
        save_portfolio(portfolio)
        print_portfolio_status(portfolio)

    else:
        logger.error("Unknown mode: %s", mode)
        logger.info("Usage: python paper_trade.py [open|monitor|status|close-all]")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Paper trading engine for F&O screener signals")
    parser.add_argument("mode", choices=["open", "monitor", "status", "close-all"],
                        help="open: run screener + open positions, "
                             "monitor: check exits, "
                             "status: print dashboard, "
                             "close-all: force close everything")
    args = parser.parse_args()

    logger.info("Connecting to AngelOne SmartAPI...")
    smart_api = get_session()

    run_paper_trade(smart_api, args.mode)


if __name__ == "__main__":
    main()
