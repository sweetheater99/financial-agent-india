"""
Paper Trading Engine — tracks simulated positions based on F&O screener signals.

Runs the screener, opens score-weighted paper positions on top candidates,
monitors for ATR-based target / stop-loss / 7-day expiry, and tracks cumulative P&L.

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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import config
from connect import get_session, refresh_session
from agent_with_options import fetch_option_chain, get_nearest_expiry
from indicators import compute_atr
from regime import classify_regime as classify_regime_v2, vix_to_iv_percentile
from risk_manager import RiskManager
from risk_guardrails import (
    check_drawdown_circuit_breaker,
    check_daily_loss_limit,
    check_correlation_guard,
    check_portfolio_heat,
    check_event_calendar,
    run_all_guards,
)
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
MAX_HOLD_DAYS = 15     # force exit after 15 trading days — tuned via param sweep
TOP_N = 5              # top candidates to trade
MIN_CAPITAL = 1_000    # minimum capital to open new positions

# Put option thresholds (leveraged instrument → wider bands)
PUT_TARGET_PCT = 50.0       # +50% premium gain
PUT_STOPLOSS_PCT = -40.0    # -40% premium loss
MIN_DTE_TO_OPEN = 3         # don't open puts with < 3 days to expiry
MIN_DTE_TO_HOLD = 2         # force exit if < 2 days to expiry

# Futures trading (V8 — seasoned F&O trader rules)
FUT_MARGIN_PCT = 0.15           # ~15% SPAN+exposure margin for stock futures
FUT_STT_PCT = 0.0005            # 0.05% sell side only (Apr 2026 rate)
FUT_EXCHANGE_PCT = 0.0019 / 100
FUT_STAMP_DUTY_PCT = 0.002 / 100
FUT_SEBI_PCT = 0.000001
FUT_SLIPPAGE_PCT = 0.0005
FUT_TARGET_ATR_MULT = 1.5      # tight target — V8 param sweep
FUT_SL_ATR_MULT = 3.5          # wide SL — V8 param sweep
FUT_RISK_PER_TRADE_PCT = 2.0   # max 2% of capital at risk per futures trade
FUT_MIN_SCORE = 4.5            # higher threshold for futures (leveraged = need more conviction)
MAX_FUT_POSITIONS = 3           # max concurrent futures positions (leveraged — fewer is safer)
FUT_OI_CONFIRMATION = True      # require OI buildup alignment for futures entry

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
ATR_TRAILING_MULTIPLIER = 2.5       # trailing SL = peak ± 2.5*ATR (equity) — tuned via param sweep
LEGACY_TRAILING_STOP_PCT = 2.0      # fallback for positions without ATR
OPT_TRAILING_STOP_PCT = 25.0        # -25% from peak premium for options

# --- ATR-Based Exits (equity only) ---
ATR_PERIOD = 14
ATR_TARGET_MULTIPLIER = 1.5     # target = entry + 1.5*ATR — V8 param sweep (tight target, wide SL)
ATR_STOPLOSS_MULTIPLIER = 3.5   # SL = entry - 3.5*ATR — V8 param sweep (wide SL reduces premature exits)
MAX_EQUITY_POSITIONS = 5        # V8: max concurrent equity positions (portfolio risk control)

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

INDEX_CONFIGS = [
    {"name": "NIFTY", "token": NIFTY_TOKEN, "lot_size": 75,
     "otm_points": config.WEEKLY_THETA_OTM_POINTS_NIFTY, "vix_mult": 1.0},
    {"name": "BANKNIFTY", "token": config.BANKNIFTY_TOKEN, "lot_size": config.BANKNIFTY_LOT_SIZE,
     "otm_points": config.WEEKLY_THETA_OTM_POINTS_BANKNIFTY, "vix_mult": config.BANKNIFTY_VIX_MULTIPLIER},
]

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
# Session Error Detection
# ---------------------------------------------------------------------------

class SessionExpiredError(Exception):
    """Raised when an API call fails due to expired/invalid session."""
    pass


def _is_session_error(resp) -> bool:
    """Check if an API response indicates session expiry."""
    if not isinstance(resp, dict):
        return False
    code = resp.get("errorcode", "")
    if code == "AB1010":
        return True
    msg = resp.get("message", "")
    return "Invalid Session" in msg or "Session is Expired" in msg


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


def _send_single_telegram(text: str) -> None:
    """Send a single message via Telegram bot. Fails silently."""
    import os
    import urllib.request
    import urllib.parse
    token = os.environ.get("DEAL_BOT_TOKEN", "")
    forum_chat_id = os.environ.get("TELEGRAM_FORUM_CHAT_ID", "")
    topic_id = os.environ.get("TELEGRAM_TOPIC_STOCKS", "")
    chat_id = os.environ.get("DEAL_BOT_CHAT_ID", "")
    if not token or not (forum_chat_id or chat_id):
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": forum_chat_id or chat_id,
            "text": text,
            "parse_mode": "HTML",
        }
        if forum_chat_id and topic_id:
            payload["message_thread_id"] = topic_id
        data = urllib.parse.urlencode(payload).encode()
        req = urllib.request.Request(url, data=data)
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


def _telegram_send(text: str, **kwargs) -> None:
    """Send a message via Telegram, splitting if over 4000 chars."""
    if len(text) <= 4000:
        _send_single_telegram(text, **kwargs)
        return

    # Split into chunks at newline boundaries
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= 4000:
            chunks.append(remaining)
            break
        # Find last newline before the 4000-char limit
        split_at = remaining.rfind("\n", 0, 4000)
        if split_at <= 0:
            # No newline found — hard split at 4000
            split_at = 4000
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip("\n")

    total = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        header = f"({i}/{total}) " if total > 1 else ""
        _send_single_telegram(header + chunk, **kwargs)


def _telegram_notify_entry(positions_opened: list[dict]) -> None:
    """Send Telegram alert for new position entries."""
    if not positions_opened:
        return
    lines = ["<b>📈 Paper Trade: New Positions</b>\n"]
    for p in positions_opened:
        direction = "🟢 BULL" if p["direction"] == "bullish" else "🔴 BEAR"
        lines.append(
            f"<b>{p['symbol']}</b> {direction}\n"
            f"  Entry: ₹{p['entry_price']:,.2f}  |  Qty: {p['quantity']}\n"
            f"  Target: ₹{p['target_price']:,.2f} ({TARGET_PCT:+.0f}%)\n"
            f"  SL: ₹{p['stoploss_price']:,.2f} ({STOPLOSS_PCT:.0f}%)\n"
            f"  Allocated: ₹{p['allocated']:,.0f}"
        )
    _telegram_send("\n".join(lines))


def _telegram_notify_exit(pos: dict, reason: str, pnl: float, pnl_pct: float,
                          exit_price: float = 0, portfolio: dict | None = None) -> None:
    """Send Telegram alert for position exits."""
    emoji = "✅" if pnl >= 0 else "❌"
    reason_label = {
        "target": "Target hit", "stoploss": "Stop-loss hit",
        "trailing_stop": "Trailing stop hit", "theta_decay": "Theta decay",
        "partial_target": "Partial target", "expiry": "Max hold expired",
        "manual": "Manual close",
    }.get(reason, reason)
    # Holding period
    try:
        entry_dt = datetime.strptime(pos["entry_date"], "%Y-%m-%d")
        today_dt = datetime.strptime(_today_ist(), "%Y-%m-%d")
        held_days = (today_dt - entry_dt).days
    except Exception:
        held_days = 0
    instr = pos.get("instrument", "EQ")
    instr_suffix = f" ({instr})" if instr != "EQ" else ""
    lines = [
        f"{emoji} <b>Paper Trade Exit: {pos['symbol']}{instr_suffix}</b>",
        f"  Reason: {reason_label}",
        f"  P&amp;L: {pnl_pct:+.1f}% (₹{pnl:+,.0f})",
        f"  Entry: ₹{pos['entry_price']:,.2f} → Exit: ₹{exit_price:,.2f}",
        f"  Held: {held_days} day{'s' if held_days != 1 else ''}",
    ]
    if portfolio:
        stats = portfolio.get("stats", {})
        total_trades = stats.get("total_trades", 0)
        wins = stats.get("winning_trades", 0)
        losses = stats.get("losing_trades", 0)
        realized = stats.get("total_pnl", 0)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        capital = portfolio.get("capital", 100000)
        # Unrealized P&L from open positions (peak_price is updated each monitor cycle)
        open_pos = [p for p in portfolio.get("positions", []) if p.get("status") == "open"]
        unrealized = round(sum(
            (p.get("peak_price", p["entry_price"]) - p["entry_price"]) * abs(p["quantity"])
            if p["direction"] == "bullish"
            else (p["entry_price"] - p.get("peak_price", p["entry_price"])) * abs(p["quantity"])
            for p in open_pos
        ), 0)
        total_pnl = realized + unrealized
        total_pct = total_pnl / capital * 100
        lines.append("")
        lines.append(f"  <b>Realized:</b> ₹{realized:+,.0f}  |  <b>Unrealized:</b> ~₹{unrealized:+,.0f}")
        lines.append(f"  <b>Total:</b> {total_pct:+.1f}% on capital  |  <b>Open:</b> {len(open_pos)}")
        lines.append(f"  <b>Win Rate:</b> {win_rate:.0f}% ({wins}W / {losses}L)")
    _telegram_send("\n".join(lines))


def _telegram_daily_summary(portfolio: dict) -> None:
    """Send daily portfolio summary via Telegram, including V3 global macro context."""
    open_pos = [p for p in portfolio["positions"] if p["status"] == "open"]
    stats = portfolio["stats"]
    realized = stats["total_pnl"]

    lines = ["<b>📊 Paper Trade Daily Summary</b>\n"]
    lines.append(f"Capital: ₹{portfolio['capital']:,.0f}  |  Available: ₹{portfolio['available_capital']:,.0f}")
    lines.append(f"Trades: {stats['total_trades']}  |  Win: {stats['winning_trades']}  |  Loss: {stats['losing_trades']}")

    if stats["total_trades"] > 0:
        wr = stats["winning_trades"] / stats["total_trades"] * 100
        lines.append(f"Win Rate: {wr:.0f}%  |  Realized P&amp;L: ₹{realized:+,.0f}")

    # V3: Global macro context
    try:
        from global_intel import fetch_macro_context
        macro = fetch_macro_context(prev_nifty_close=0, pcr=1.0)
        if macro:
            lines.append(f"\n<b>Global Cues:</b>")
            lines.append(f"S&amp;P {macro.get('sp500_pct_change', 0):+.1f}% | "
                         f"Nasdaq {macro.get('nasdaq_pct_change', 0):+.1f}% | "
                         f"GIFT gap {macro.get('gift_nifty_gap_pct', 0):+.1f}% | "
                         f"FII {macro.get('fii_net_crores', 0):+.0f}cr")
            if macro.get("hard_gate") != "NONE":
                lines.append(f"⚠️ Gate: {macro['hard_gate']} — {macro['hard_gate_reason']}")
    except Exception:
        pass

    try:
        from oi_analysis import compute_pcr
        from indicators_v3 import compute_supertrend
        # PCR and Supertrend already cached from open_positions, just log what we have
    except Exception:
        pass

    try:
        from x_intel import fetch_x_sentiment
        x_sentiment = fetch_x_sentiment()
        if x_sentiment:
            themes = ", ".join(x_sentiment.get("key_themes", [])[:3])
            lines.append(f"X: {x_sentiment.get('sentiment', '?')} ({themes})")
    except Exception:
        pass

    try:
        from breadth import compute_breadth
        breadth = compute_breadth()
        if breadth:
            lines.append(f"Breadth: {breadth['advancing']}/{breadth['total']} above 20EMA ({breadth['breadth_pct']:.0f}%) — {breadth['signal']}")
    except Exception:
        pass

    try:
        from sector_rotation import get_sector_rotation
        rotation = get_sector_rotation()
        if rotation and rotation.get("rotation_detected"):
            lines.append(f"Rotation: ↑{','.join(rotation['strengthening'][:3])} ↓{','.join(rotation['weakening'][:3])}")
    except Exception:
        pass

    if open_pos:
        lines.append(f"\n<b>Open Positions ({len(open_pos)}):</b>")
        for p in open_pos:
            ltp = p.get("ltp_at_entry", p["entry_price"])
            entry_val = p["entry_price"] * abs(p["quantity"])
            lines.append(f"  <b>{p['symbol']}</b> — Entry: ₹{p['entry_price']:,.2f}  Qty: {p['quantity']}")
    else:
        lines.append("\nNo open positions.")

    if stats.get("best_trade"):
        lines.append(f"\nBest: {stats['best_trade']['symbol']} ({stats['best_trade']['pnl_pct']:+.1f}%)")
    if stats.get("worst_trade"):
        lines.append(f"Worst: {stats['worst_trade']['symbol']} ({stats['worst_trade']['pnl_pct']:+.1f}%)")

    _telegram_send("\n".join(lines))


# ---------------------------------------------------------------------------
# Slippage Modeling
# ---------------------------------------------------------------------------

def apply_slippage(price: float, instrument: str, side: str) -> float:
    """
    Apply slippage to a price. Buys get worse (higher), sells get worse (lower).

    instrument: "EQ", "OPT", or "FUT"
    side: "buy" or "sell"
    """
    if instrument == "OPT":
        pct = OPT_SLIPPAGE_PCT
    elif instrument == "FUT":
        pct = FUT_SLIPPAGE_PCT
    else:
        pct = EQ_SLIPPAGE_PCT
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
    """Load portfolio from JSON, or create a fresh one.

    Falls back to .bak if main file is corrupt; returns empty portfolio if both fail.
    """
    bak = PORTFOLIO_FILE.with_suffix(".json.bak")
    for path in (PORTFOLIO_FILE, bak):
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt portfolio file: %s — trying next", path)
    return _empty_portfolio()


def save_portfolio(portfolio: dict) -> None:
    """Persist portfolio state to disk with backup + atomic write."""
    import shutil

    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)

    # Backup existing file before overwriting
    if PORTFOLIO_FILE.exists():
        shutil.copy2(PORTFOLIO_FILE, PORTFOLIO_FILE.with_suffix(".json.bak"))

    # Atomic write: write to .tmp, then rename
    tmp = PORTFOLIO_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(portfolio, indent=2, default=str))
    tmp.replace(PORTFOLIO_FILE)


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
    """Fetch current last traded price for a stock.

    Raises SessionExpiredError if the API session has expired.
    """
    try:
        resp = smart_api.ltpData("NSE", symbol, token)
        if _is_session_error(resp):
            raise SessionExpiredError(symbol)
        if resp and resp.get("data"):
            ltp = resp["data"].get("ltp")
            if ltp is not None:
                return float(ltp)
    except SessionExpiredError:
        raise
    except Exception:
        pass

    # Fallback: getMarketData
    try:
        resp = smart_api.getMarketData(mode="LTP", exchangeTokens={"NSE": [token]})
        if _is_session_error(resp):
            raise SessionExpiredError(symbol)
        if resp and resp.get("data") and resp["data"].get("fetched"):
            return float(resp["data"]["fetched"][0]["ltp"])
    except SessionExpiredError:
        raise
    except Exception as e:
        logger.warning("LTP fetch failed for %s: %s", symbol, e)
    return None


# ---------------------------------------------------------------------------
# Put Option Helpers
# ---------------------------------------------------------------------------

def get_ltp_nfo(smart_api, symbol: str, token: str) -> float | None:
    """Fetch current last traded price for an NFO contract (options/futures).

    Raises SessionExpiredError if the API session has expired.
    """
    try:
        resp = smart_api.ltpData("NFO", symbol, token)
        if _is_session_error(resp):
            raise SessionExpiredError(symbol)
        if resp and resp.get("data"):
            ltp = resp["data"].get("ltp")
            if ltp is not None:
                return float(ltp)
    except SessionExpiredError:
        raise
    except Exception:
        pass

    try:
        resp = smart_api.getMarketData(mode="LTP", exchangeTokens={"NFO": [token]})
        if _is_session_error(resp):
            raise SessionExpiredError(symbol)
        if resp and resp.get("data") and resp["data"].get("fetched"):
            return float(resp["data"]["fetched"][0]["ltp"])
    except SessionExpiredError:
        raise
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


def resolve_futures_contract(smart_api, symbol: str) -> dict | None:
    """Resolve a stock futures contract via searchScrip.

    Searches for the nearest month futures contract.
    Returns {trading_symbol, token, lot_size, expiry} or None.
    """
    # Search for FUT contracts
    query = f"{symbol}FUT"
    try:
        resp = smart_api.searchScrip("NFO", query)
        if resp and resp.get("data"):
            # Filter for FUT contracts, pick nearest expiry
            fut_matches = []
            for match in resp["data"]:
                tsym = match.get("tradingsymbol", "")
                if "FUT" in tsym.upper():
                    fut_matches.append({
                        "trading_symbol": tsym,
                        "token": match["symboltoken"],
                        "lot_size": int(match.get("lotsize", 1)),
                        "expiry": match.get("expiry", ""),
                    })
            if fut_matches:
                # Sort by expiry, pick nearest
                fut_matches.sort(key=lambda x: x["expiry"])
                return fut_matches[0]
    except Exception as e:
        logger.warning("searchScrip NFO FUT failed for %s: %s", symbol, e)
    return None


def _calc_fut_costs(entry_price: float, exit_price: float, quantity: int) -> float:
    """Calculate futures round-trip transaction costs (Apr 2026 rates)."""
    entry_turnover = entry_price * quantity
    exit_turnover = exit_price * quantity

    # Buy side
    buy_brokerage = min(BROKERAGE_FLAT, entry_turnover * 0.0003)
    buy_stamp = entry_turnover * FUT_STAMP_DUTY_PCT
    buy_exchange = entry_turnover * FUT_EXCHANGE_PCT
    buy_sebi = entry_turnover * FUT_SEBI_PCT

    # Sell side (STT on sell only)
    sell_brokerage = min(BROKERAGE_FLAT, exit_turnover * 0.0003)
    sell_stt = exit_turnover * FUT_STT_PCT
    sell_exchange = exit_turnover * FUT_EXCHANGE_PCT
    sell_sebi = exit_turnover * FUT_SEBI_PCT

    total_brokerage = buy_brokerage + sell_brokerage
    total_exchange = buy_exchange + sell_exchange
    gst = (total_brokerage + total_exchange) * GST_PCT

    return round(total_brokerage + sell_stt + buy_stamp + total_exchange +
                 buy_sebi + sell_sebi + gst, 2)


def _open_futures_position(smart_api, symbol: str, eq_token: str, spot_price: float,
                           allocation: float, alloc: dict, atr: float,
                           direction: str = "bullish") -> dict | None:
    """Open a stock futures position.

    Seasoned F&O trader rules applied:
    - Risk-based sizing: max loss (SL distance × qty) ≤ 2% of total capital
    - OI confirmation required: Long Buildup for longs, Short Buildup for shorts
    - Higher score threshold (4.5) for leveraged instruments
    - ATR-based target/SL from V8 param sweep
    """
    # 1. Resolve futures contract
    contract = resolve_futures_contract(smart_api, symbol)
    time.sleep(config.API_DELAY)
    if not contract:
        logger.info("SKIP FUT %s: could not resolve futures contract", symbol)
        return None

    lot_size = contract["lot_size"]
    logger.info("  %s FUT: %s (lot=%d, expiry=%s)",
                symbol, contract["trading_symbol"], lot_size, contract.get("expiry", "?"))

    # 2. Get futures LTP
    fut_ltp = get_ltp_nfo(smart_api, contract["trading_symbol"], contract["token"])
    time.sleep(config.API_DELAY)
    if not fut_ltp or fut_ltp <= 0:
        logger.info("SKIP FUT %s: no LTP available", symbol)
        return None

    # 3. Risk-based position sizing
    # Max loss per trade = FUT_RISK_PER_TRADE_PCT% of total capital
    max_loss = TOTAL_CAPITAL * (FUT_RISK_PER_TRADE_PCT / 100)
    sl_distance = FUT_SL_ATR_MULT * atr

    if sl_distance <= 0:
        logger.info("SKIP FUT %s: zero ATR/SL distance", symbol)
        return None

    # Lots based on risk
    risk_qty = int(max_loss / sl_distance)
    risk_lots = max(1, risk_qty // lot_size)

    # Also check margin requirement doesn't exceed allocation
    margin_per_lot = fut_ltp * lot_size * FUT_MARGIN_PCT
    margin_lots = max(1, int(allocation / margin_per_lot))

    num_lots = min(risk_lots, margin_lots)
    quantity = num_lots * lot_size
    margin_required = round(fut_ltp * quantity * FUT_MARGIN_PCT, 2)

    # Apply slippage
    if direction == "bullish":
        entry_price = round(fut_ltp * (1 + FUT_SLIPPAGE_PCT), 2)
        target_price = round(entry_price + FUT_TARGET_ATR_MULT * atr, 2)
        stoploss_price = round(entry_price - FUT_SL_ATR_MULT * atr, 2)
    else:
        entry_price = round(fut_ltp * (1 - FUT_SLIPPAGE_PCT), 2)
        target_price = round(entry_price - FUT_TARGET_ATR_MULT * atr, 2)
        stoploss_price = round(entry_price + FUT_SL_ATR_MULT * atr, 2)

    today = _today_ist()
    max_hold_date = _add_trading_days(today, MAX_HOLD_DAYS)

    position = {
        "symbol": symbol,
        "token": contract["token"],
        "direction": direction,
        "instrument": "FUT",
        "contract_symbol": contract["trading_symbol"],
        "lot_size": lot_size,
        "num_lots": num_lots,
        "entry_price": entry_price,
        "ltp_at_entry": fut_ltp,
        "quantity": quantity,
        "allocated": margin_required,
        "margin_required": margin_required,
        "underlying_price_at_entry": spot_price,
        "atr_at_entry": round(atr, 2),
        "target_price": target_price,
        "stoploss_price": stoploss_price,
        "peak_price": entry_price,
        "score": alloc["score"],
        "categories": alloc.get("categories", []),
        "entry_date": today,
        "max_hold_date": max_hold_date,
        "expiry": contract.get("expiry", ""),
        "status": "open",
    }

    logger.info("OPENED FUT %s %s x%dlot (%dqty) @ ₹%.2f | T=₹%.2f SL=₹%.2f | Margin=₹%.0f | ATR=%.2f",
                direction.upper(), symbol, num_lots, quantity,
                entry_price, target_price, stoploss_price, margin_required, atr)

    return position


def _open_put_position(smart_api, symbol: str, eq_token: str, spot_price: float,
                       allocation: float, alloc: dict) -> dict | None:
    """
    Open a put option position for a bearish candidate.

    Returns position dict or None if skipped.
    """
    expiry = get_nearest_expiry(min_dte=MIN_DTE_TO_OPEN)
    if not expiry:
        logger.info("SKIP %s: could not determine option expiry with min %dd DTE", symbol, MIN_DTE_TO_OPEN)
        return None

    dte = days_to_expiry(expiry)
    logger.info("  %s: using expiry %s (%dd DTE)", symbol, expiry, dte)

    # Fetch option chain (retry once on failure)
    option_chain = fetch_option_chain(smart_api, symbol, eq_token)
    time.sleep(config.API_DELAY)
    if not option_chain:
        logger.info("  %s: option chain fetch failed, retrying...", symbol)
        smart_api = refresh_session()
        time.sleep(config.API_DELAY)
        option_chain = fetch_option_chain(smart_api, symbol, eq_token)
        time.sleep(config.API_DELAY)
    if not option_chain:
        logger.info("SKIP %s: option chain fetch failed after retry", symbol)
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
# Spread Type Routing
# ---------------------------------------------------------------------------

def _select_spread_type(direction: str, iv_percentile: int, vix: float | None) -> str | None:
    """Route between debit and credit spread based on IV environment.

    Returns 'debit', 'credit', or None if neither is viable.
    """
    can_debit = iv_percentile <= config.DEBIT_SPREAD_MAX_IV_PERCENTILE
    can_credit = (
        iv_percentile >= config.CREDIT_SPREAD_MIN_IV_PERCENTILE
        and vix is not None
        and config.CREDIT_SPREAD_MIN_VIX <= vix <= config.CREDIT_SPREAD_MAX_VIX
    )

    if can_credit and not can_debit:
        return "credit"
    if can_debit and not can_credit:
        return "debit"
    if can_debit and can_credit:
        return "debit"  # prefer debit in overlap zone (50-70%)
    return None


# ---------------------------------------------------------------------------
# Spread Position Helpers
# ---------------------------------------------------------------------------

def calc_spread_pnl(entry_long_premium, current_long_premium,
                    entry_short_premium, current_short_premium, quantity):
    """Calculate current P&L for a spread position.

    For debit spreads: paid (long - short) at entry, now worth (long - short) current.
    P&L = (current_spread_value - entry_spread_value) * quantity

    For credit spreads: received (short - long) at entry.
    P&L = entry_credit - current_cost_to_close
    """
    entry_spread = entry_long_premium - entry_short_premium
    current_spread = current_long_premium - current_short_premium
    return (current_spread - entry_spread) * quantity


def estimate_premium_from_underlying(pos, leg, underlying_ltp, today):
    """Estimate current option premium based on underlying move.

    Simple model: intrinsic value + decayed time value.
    For paper trading approximation only.

    Args:
        pos: spread position dict
        leg: "long" or "short"
        underlying_ltp: current underlying price
        today: date string "YYYY-MM-DD" or date object

    Returns:
        estimated premium (float, min 0.05)
    """
    from datetime import date as _date

    leg_data = pos["long_leg"] if leg == "long" else pos["short_leg"]
    strike = leg_data["strike"]
    entry_premium = leg_data["entry_premium"]
    opt_type = leg_data["option_type"]

    # Intrinsic value
    if opt_type == "CE":
        intrinsic = max(0, underlying_ltp - strike)
    else:  # PE
        intrinsic = max(0, strike - underlying_ltp)

    # Original intrinsic at entry
    underlying_at_entry = pos["underlying_at_entry"]
    if opt_type == "CE":
        entry_intrinsic = max(0, underlying_at_entry - strike)
    else:
        entry_intrinsic = max(0, strike - underlying_at_entry)

    # Time value at entry
    time_value_at_entry = max(0, entry_premium - entry_intrinsic)

    # Decay time value (simple linear decay)
    entry_date = datetime.strptime(pos["entry_date"], "%Y-%m-%d").date()
    if isinstance(today, str):
        today_date = datetime.strptime(today, "%Y-%m-%d").date()
    else:
        today_date = today
    days_held = (today_date - entry_date).days

    # Rough total DTE at entry (assume 30 if not available)
    total_dte = 30
    remaining_fraction = max(0, (total_dte - days_held) / total_dte)
    time_value_now = time_value_at_entry * remaining_fraction * 0.8  # 0.8 for non-linear decay

    return max(0.05, intrinsic + time_value_now)


def check_spread_exit(pos, underlying_ltp, today, long_premium, short_premium):
    """Check if a spread position should be exited.

    Returns exit reason string or None if no exit needed.

    Exit conditions for CREDIT spreads (checked first):
    1. Target: P&L >= 50% of net credit received
    2. Stoploss: loss exceeds 2x credit received
    (then falls through to time/expiry checks)

    Exit conditions for DEBIT spreads (checked in order):
    1. Profit cap: P&L >= 80% of max_profit
    2. Underlying target: moved 2.0x ATR in favorable direction
    3. Stoploss: moved 1.5x ATR in adverse direction
    4. Time exit: held >= SPREAD_TIME_EXIT_DAYS trading days
    5. Expiry safety: DTE <= 5 calendar days
    """
    spread_type = pos.get("spread_type")

    # ── Credit spread exits ──────────────────────────────────────────────
    if spread_type == "credit":
        net_credit = pos.get("net_credit", 0)
        cost_to_close = abs(short_premium - long_premium) * pos.get("quantity", 75)
        current_pnl = net_credit - cost_to_close

        # Target: 50% of credit received
        if net_credit > 0 and current_pnl >= net_credit * config.CREDIT_SPREAD_TARGET_PCT:
            return "credit_target"

        # Stoploss: loss exceeds 2x credit
        if net_credit > 0 and current_pnl <= -(net_credit * config.CREDIT_SPREAD_SL_MULTIPLIER):
            return "credit_stoploss"

        # Fall through to time/expiry checks below

    # ── Debit spread exits ───────────────────────────────────────────────
    elif spread_type == "debit":
        direction = pos["spread_direction"]
        underlying_at_entry = pos["underlying_at_entry"]
        atr = pos.get("atr_at_entry", pos["spread_width"] / 3)
        quantity = pos.get("quantity", 75)

        # 1. Profit cap
        current_pnl = calc_spread_pnl(
            pos["long_leg"]["entry_premium"], long_premium,
            pos["short_leg"]["entry_premium"], short_premium,
            quantity
        )
        profit_cap = pos["max_profit"] * config.SPREAD_PROFIT_CAP_PCT
        if current_pnl >= profit_cap:
            return "spread_profit_cap"

        # 2. Underlying target
        target_move = atr * config.SPREAD_TARGET_ATR_MULT
        if direction == "bullish" and underlying_ltp >= underlying_at_entry + target_move:
            return "spread_target"
        if direction == "bearish" and underlying_ltp <= underlying_at_entry - target_move:
            return "spread_target"

        # 3. Stoploss
        sl_move = atr * config.SPREAD_SL_ATR_MULT
        if direction == "bullish" and underlying_ltp <= underlying_at_entry - sl_move:
            return "spread_stoploss"
        if direction == "bearish" and underlying_ltp >= underlying_at_entry + sl_move:
            return "spread_stoploss"
    else:
        return None

    # ── Shared time/expiry exits (both debit and credit) ─────────────────
    # 4. Time exit
    time_exit_days = (config.CREDIT_SPREAD_TIME_EXIT_DAYS
                      if spread_type == "credit"
                      else config.SPREAD_TIME_EXIT_DAYS)
    trading_days = _trading_days_between(pos["entry_date"], today)
    if trading_days >= time_exit_days:
        return "spread_time_exit"

    # 5. Expiry safety (DTE <= 5 calendar days)
    if isinstance(today, str):
        today_date = datetime.strptime(today, "%Y-%m-%d").date()
    else:
        today_date = today

    # Parse expiry "DDMONYYYY" format (e.g. "26MAR2026")
    expiry_str = pos.get("expiry", "")
    try:
        expiry_date = datetime.strptime(expiry_str, "%d%b%Y").date()
    except ValueError:
        try:
            expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        except ValueError:
            expiry_date = None

    if expiry_date is not None:
        dte = (expiry_date - today_date).days
        if dte <= 5:
            return "spread_expiry_safety"

    return None


def check_condor_exit(pos, call_short_prem, call_long_prem, put_short_prem, put_long_prem,
                      vix=None, prev_vix=None):
    """Check exit conditions for an iron condor position.

    Returns exit reason string or None.

    Exit conditions checked in order:
    1. VIX spike > 20% in 1 day → immediate close
    2. Stoploss: loss exceeds 2x credit received
    3. Target: P&L >= 50% of net credit received
    4. Time exit: past max_hold_date
    """
    net_credit = pos["net_credit"]
    quantity = pos["quantity"]
    cost_to_close = (call_short_prem - call_long_prem + put_short_prem - put_long_prem) * quantity
    current_pnl = net_credit - cost_to_close

    # VIX spike > 20%
    if vix is not None and prev_vix is not None and prev_vix > 0:
        vix_change = (vix - prev_vix) / prev_vix
        if vix_change > 0.20:
            return "condor_vix_spike"

    # Stoploss: loss exceeds 2x credit
    if current_pnl <= -(net_credit * config.CONDOR_SL_MULTIPLIER):
        return "condor_stoploss"

    # Target: 50% of credit
    if current_pnl >= net_credit * config.CONDOR_TARGET_PCT:
        return "condor_target"

    # Time exit
    today = _today_ist()
    max_hold = pos.get("max_hold_date", pos.get("expiry", ""))
    if max_hold and today >= max_hold:
        return "condor_time_exit"

    return None


def _open_iron_condor(portfolio, condor_data, allocation, regime_at_entry, iv_pct, expiry_str):
    """Open iron condor position. 4-leg position stored as single entry.

    Args:
        portfolio: portfolio dict
        condor_data: dict from select_iron_condor_strikes()
        allocation: amount allocated from capital
        regime_at_entry: current market regime string
        iv_pct: IV percentile at entry
        expiry_str: expiry date string in "DDMONYYYY" format

    Returns:
        position dict
    """
    today = _today_ist()
    slippage = config.PAPER_TRADE_SLIPPAGE_PCT
    lot_size = condor_data.get("lot_size", 75)

    # Apply slippage: sell legs get worse (lower premium), buy legs get worse (higher premium)
    cs_prem = round(condor_data["call_short_premium"] * (1 - slippage), 2)
    cl_prem = round(condor_data["call_long_premium"] * (1 + slippage), 2)
    ps_prem = round(condor_data["put_short_premium"] * (1 - slippage), 2)
    pl_prem = round(condor_data["put_long_premium"] * (1 + slippage), 2)

    net_credit = round((cs_prem + ps_prem - cl_prem - pl_prem) * lot_size, 2)
    width = condor_data["width"]
    max_loss = round(width * lot_size - net_credit, 2)

    # Max hold date = CONDOR_TIME_EXIT_DAYS trading days before expiry
    try:
        expiry_date = datetime.strptime(expiry_str, "%d%b%Y").date()
        from config import add_trading_days as _config_add_td
        max_hold_date = _config_add_td(expiry_date, -config.CONDOR_TIME_EXIT_DAYS)
        max_hold_str = max_hold_date.strftime("%Y-%m-%d")
    except Exception:
        max_hold_str = _add_trading_days(today, 15)

    position = {
        "id": f"condor_{today}_NIFTY",
        "strategy": "iron_condor",
        "symbol": "NIFTY",
        "instrument": "CONDOR",
        "entry_date": today,
        "expiry": expiry_str,
        "quantity": lot_size,
        "direction": "neutral",
        "entry_price": 0,  # condor uses net_credit instead
        "call_short": {
            "strike": condor_data["call_short_strike"],
            "entry_premium": cs_prem,
            "option_type": "CE",
        },
        "call_long": {
            "strike": condor_data["call_long_strike"],
            "entry_premium": cl_prem,
            "option_type": "CE",
        },
        "put_short": {
            "strike": condor_data["put_short_strike"],
            "entry_premium": ps_prem,
            "option_type": "PE",
        },
        "put_long": {
            "strike": condor_data["put_long_strike"],
            "entry_premium": pl_prem,
            "option_type": "PE",
        },
        "net_credit": net_credit,
        "max_profit": net_credit,
        "max_loss": max_loss,
        "allocated": allocation,
        "max_hold_date": max_hold_str,
        "underlying_at_entry": condor_data.get("spot", 0),
        "vix_at_entry": condor_data.get("vix_at_entry", 0),
        "status": "open",
        "regime_at_entry": regime_at_entry,
        "iv_percentile_at_entry": iv_pct,
    }

    portfolio["positions"].append(position)
    portfolio["available_capital"] = round(portfolio["available_capital"] - allocation, 2)
    return position


# ---------------------------------------------------------------------------
# Momentum Options
# ---------------------------------------------------------------------------

def check_nifty_breakout(candles, period=20):
    """Check if Nifty broke 20-day high or low.

    Args:
        candles: list of candle lists [date, open, high, low, close, volume]
        period: lookback period (default 20)

    Returns:
        "bullish" (broke high), "bearish" (broke low), or None.
    """
    if not candles or len(candles) < period + 1:
        return None

    lookback = candles[-(period + 1):-1]  # previous N days (excluding today)
    today = candles[-1]

    high_n = max(c[2] for c in lookback)
    low_n = min(c[3] for c in lookback)

    today_close = today[4]

    if today_close > high_n:
        return "bullish"
    if today_close < low_n:
        return "bearish"
    return None


def check_momentum_exit(pos, current_premium, today):
    """Check momentum option exit conditions.

    Args:
        pos: momentum position dict
        current_premium: current option premium
        today: date string "YYYY-MM-DD"

    Returns:
        exit reason string or None.
    """
    entry = pos["entry_price"]
    if entry <= 0:
        return None

    # Target hit
    if current_premium >= pos["target_price"]:
        return "target"

    # Stoploss hit
    if current_premium <= pos["stoploss_price"]:
        return "stoploss"

    # Time exit: past max_hold_date with < 15% profit
    pnl_pct = (current_premium - entry) / entry * 100
    if today >= pos["max_hold_date"] and pnl_pct < 15:
        return "time_exit"

    # Trailing stop from peak (only after 30% gain)
    peak = pos.get("peak_premium", entry)
    if peak > entry * 1.3:
        trail_sl = peak * 0.75  # 25% trailing stop from peak
        if current_premium <= trail_sl:
            return "trailing_stop"

    return None


def _open_momentum_position(portfolio, direction, option_data, allocation, regime, iv_pct, expiry_str):
    """Open a momentum options position (naked ATM call or put on Nifty).

    Args:
        portfolio: portfolio dict
        direction: "bullish" or "bearish"
        option_data: dict with "premium", "strike", optionally "lot_size"
        allocation: amount allocated from capital
        regime: current market regime
        iv_pct: IV percentile at entry
        expiry_str: expiry date string in "DDMONYYYY" format

    Returns:
        position dict
    """
    today = _today_ist()
    slippage = config.PAPER_TRADE_SLIPPAGE_PCT

    entry_premium = round(option_data["premium"] * (1 + slippage), 2)
    quantity = option_data.get("lot_size", 75)

    position = {
        "id": f"momentum_{today}_NIFTY_{direction}",
        "strategy": "momentum",
        "symbol": "NIFTY",
        "instrument": "MOMENTUM",
        "direction": direction,
        "entry_date": today,
        "expiry": expiry_str,
        "option_type": "CE" if direction == "bullish" else "PE",
        "strike": option_data["strike"],
        "entry_price": entry_premium,
        "quantity": quantity,
        "allocated": allocation,
        "target_price": round(entry_premium * (1 + config.MOMENTUM_TARGET_PCT), 2),
        "stoploss_price": round(entry_premium * (1 - config.MOMENTUM_SL_PCT), 2),
        "max_hold_date": _add_trading_days(today, config.MOMENTUM_TIME_EXIT_DAYS),
        "peak_premium": entry_premium,
        "status": "open",
        "regime_at_entry": regime,
        "iv_percentile_at_entry": iv_pct,
    }

    portfolio["positions"].append(position)
    portfolio["available_capital"] = round(portfolio["available_capital"] - allocation, 2)
    return position


# ---------------------------------------------------------------------------
# Earnings Calendar & Macro Event Checking
# ---------------------------------------------------------------------------

MACRO_EVENTS_2026 = [
    "2026-02-01",  # Union Budget
    "2026-04-09",  # RBI MPC
    "2026-06-04",  # RBI MPC
    "2026-08-06",  # RBI MPC
    "2026-10-08",  # RBI MPC
    "2026-12-03",  # RBI MPC
    # US Fed dates
    "2026-01-29",  # FOMC
    "2026-03-19",  # FOMC
    "2026-05-07",  # FOMC
    "2026-06-18",  # FOMC
    "2026-07-30",  # FOMC
    "2026-09-17",  # FOMC
    "2026-11-05",  # FOMC
    "2026-12-17",  # FOMC
    # Nifty expiry weeks (last Thursday of month)
    "2026-01-29",
    "2026-02-26",
    "2026-03-26",
    "2026-04-30",
    "2026-05-28",
    "2026-06-25",
    "2026-07-30",
    "2026-08-27",
    "2026-09-24",
    "2026-10-29",
    "2026-11-26",
    "2026-12-31",
]


def is_near_earnings(symbol: str, today: str = None, blackout_days: int = 2) -> bool:
    """Check if symbol has earnings within blackout_days.

    Reads from data/earnings_calendar.json.
    """
    if today is None:
        today = _today_ist()

    cal_path = Path(__file__).parent / "data" / "earnings_calendar.json"
    if not cal_path.exists():
        return False

    try:
        cal = json.loads(cal_path.read_text())
        dates = cal.get(symbol, [])
        for d in dates:
            days_until = (datetime.strptime(d, "%Y-%m-%d") - datetime.strptime(today, "%Y-%m-%d")).days
            if 0 <= days_until <= blackout_days:
                return True
    except Exception:
        pass
    return False


def is_near_macro_event(today: str = None, blackout_days: int = 2) -> bool:
    """Check if a macro event is within blackout_days."""
    if today is None:
        today = _today_ist()

    today_dt = datetime.strptime(today, "%Y-%m-%d")
    for event_date in MACRO_EVENTS_2026:
        event_dt = datetime.strptime(event_date, "%Y-%m-%d")
        days_until = (event_dt - today_dt).days
        if 0 <= days_until <= blackout_days:
            return True
    return False


def _open_spread_position(portfolio, symbol, direction, spread_data, allocation,
                          regime_at_entry, iv_percentile_at_entry, expiry_str):
    """Open a vertical spread position (debit or credit).

    This is for paper trading -- no actual orders placed. Records both legs
    in a single position entry in the portfolio.

    Args:
        portfolio: portfolio dict
        symbol: underlying symbol (e.g. "NIFTY" or "RELIANCE")
        direction: "bullish" or "bearish"
        spread_data: dict from select_spread_strikes() with keys:
            long_strike, short_strike, long_premium, short_premium,
            net_debit, max_profit, max_loss, rr_ratio, width,
            long_option_type, short_option_type
        allocation: amount allocated from capital
        regime_at_entry: current market regime string
        iv_percentile_at_entry: IV percentile at entry time
        expiry_str: expiry date string in "DDMONYYYY" format

    Returns:
        position dict if opened, None if failed
    """
    try:
        slippage_pct = config.PAPER_TRADE_SLIPPAGE_PCT

        # Determine strategy name
        if direction == "bullish":
            strategy_name = "bull_call_spread"
        else:
            strategy_name = "bear_put_spread"

        # Apply slippage to premiums
        long_premium_raw = spread_data["long_premium"]
        short_premium_raw = spread_data["short_premium"]

        long_premium_with_slippage = round(long_premium_raw * (1 + slippage_pct), 2)
        short_premium_with_slippage = round(short_premium_raw * (1 - slippage_pct), 2)

        # Lot size from spread_data or default to 75 (NIFTY lot size)
        lot_size = spread_data.get("lot_size", 75)

        # Net debit after slippage
        net_debit_with_slippage = round(
            (long_premium_with_slippage - short_premium_with_slippage) * lot_size, 2
        )

        # Max profit = (width - net_debit_per_unit) * quantity
        net_debit_per_unit = long_premium_with_slippage - short_premium_with_slippage
        max_profit = round((spread_data["width"] - net_debit_per_unit) * lot_size, 2)

        # Risk-reward ratio
        if net_debit_with_slippage > 0:
            rr_ratio = round(max_profit / net_debit_with_slippage, 2)
        else:
            rr_ratio = spread_data.get("rr_ratio", 0)

        today = _today_ist()

        position = {
            "id": f"spread_{today}_{symbol}_{strategy_name}",
            "strategy": strategy_name,
            "spread_type": "debit",
            "spread_direction": direction,
            "symbol": symbol,
            "instrument": "SPREAD",
            "underlying_at_entry": spread_data.get("spot", 0),
            "entry_date": today,
            "expiry": expiry_str,
            "quantity": lot_size,
            "long_leg": {
                "strike": spread_data["long_strike"],
                "option_type": spread_data["long_option_type"],
                "entry_premium": long_premium_with_slippage,
            },
            "short_leg": {
                "strike": spread_data["short_strike"],
                "option_type": spread_data["short_option_type"],
                "entry_premium": short_premium_with_slippage,
            },
            "net_debit": net_debit_with_slippage,
            "net_credit": 0,
            "max_profit": max_profit,
            "max_loss": net_debit_with_slippage,
            "spread_width": spread_data["width"],
            "rr_ratio": rr_ratio,
            "allocated": allocation,
            "status": "open",
            "regime_at_entry": regime_at_entry,
            "iv_percentile_at_entry": iv_percentile_at_entry,
            "peak_pnl": 0,
        }

        # Update portfolio
        portfolio["positions"].append(position)
        portfolio["available_capital"] = round(
            portfolio["available_capital"] - allocation, 2
        )

        logger.info(
            "OPENED %s %s %s/%s @ net_debit=%.2f (alloc=%.0f, regime=%s, IV%%=%.0f)",
            symbol, strategy_name,
            spread_data["long_strike"], spread_data["short_strike"],
            net_debit_with_slippage, allocation, regime_at_entry,
            iv_percentile_at_entry,
        )

        return position

    except Exception as e:
        logger.error("Failed to open spread position for %s: %s", symbol, e)
        return None


# ---------------------------------------------------------------------------
# Trading Day Helpers
# ---------------------------------------------------------------------------

def _today_ist() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d")


def _compute_weekly_pnl(portfolio: dict) -> float:
    """Compute realized P&L from trades closed this week (Mon-Sun)."""
    today = datetime.now(IST).date()
    # Start of current week (Monday)
    week_start = today - timedelta(days=today.weekday())
    week_start_str = week_start.strftime("%Y-%m-%d")

    weekly_realized = 0.0
    for trade in portfolio.get("closed_trades", []):
        exit_date = trade.get("exit_date", "")
        if exit_date >= week_start_str:
            weekly_realized += trade.get("pnl", 0)

    # Add unrealized P&L from open positions
    weekly_unrealized = 0.0
    for pos in portfolio.get("positions", []):
        if pos.get("status") != "open":
            continue
        qty = abs(pos["quantity"])
        peak = pos.get("peak_price", pos.get("ltp_at_entry", pos["entry_price"]))
        if pos["direction"] == "bullish":
            weekly_unrealized += (peak - pos["entry_price"]) * qty
        else:
            weekly_unrealized += (pos["entry_price"] - peak) * qty

    return round(weekly_realized + weekly_unrealized, 2)


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

def open_positions(smart_api, portfolio: dict, candidates: list[dict],
                   yt_market: dict | None = None) -> int:
    """
    Open paper positions for top candidates. Returns number of positions opened.

    Skips candidates already in open positions or if capital is insufficient.
    """
    if portfolio["available_capital"] < MIN_CAPITAL:
        logger.info("Insufficient capital (₹%.0f < ₹%.0f). Skipping.",
                    portfolio['available_capital'], MIN_CAPITAL)
        return 0

    # --- Risk Manager: circuit breakers ---
    rm = RiskManager(initial_capital=TOTAL_CAPITAL)
    total_pnl = portfolio.get("stats", {}).get("total_pnl", 0)
    rm._total_pnl = total_pnl
    rm.current_capital = TOTAL_CAPITAL + total_pnl

    if rm.check_total_stop():
        logger.warning("FULL STOP: Total drawdown exceeds %.0f%%. No new positions.",
                        config.DRAWDOWN_TOTAL_STOP * 100)
        return 0

    # --- Risk Guardrails: drawdown circuit breaker ---
    dd_blocked, dd_reason, dd_mult = check_drawdown_circuit_breaker(portfolio)
    if dd_blocked:
        logger.warning("GUARDRAIL BLOCK: %s", dd_reason)
        _telegram_send(f"<b>Risk Guardrail</b>\n{dd_reason}")
        return 0

    # --- Risk Guardrails: daily loss limit ---
    dl_blocked, dl_reason = check_daily_loss_limit(portfolio)
    if dl_blocked:
        logger.warning("GUARDRAIL BLOCK: %s", dl_reason)
        _telegram_send(f"<b>Risk Guardrail</b>\n{dl_reason}")
        return 0

    # --- Smart Monitor: circuit breaker (3% daily realized loss) ---
    try:
        from smart_monitor import check_circuit_breaker, _load_state, _save_state
        sm_state = _load_state()
        if check_circuit_breaker(portfolio, sm_state):
            _save_state(sm_state)
            msg = (f"<b>⚡ Circuit Breaker Active</b>\n"
                   f"Daily realized loss: ₹{sm_state['daily_realized_loss']:,.0f} "
                   f"(>{config.DAILY_LOSS_CIRCUIT_BREAKER_PCT}% of capital)\n"
                   f"No new entries until tomorrow.")
            logger.warning("CIRCUIT BREAKER: blocking new entries")
            _telegram_send(msg)
            return 0
        _save_state(sm_state)
    except Exception as e:
        logger.debug("Smart monitor circuit breaker check failed: %s", e)

    size_mult = rm.get_size_multiplier()
    risk_state = rm.get_risk_state()

    # Apply drawdown allocation multiplier (weekly DD reduction)
    if dd_mult < 1.0:
        logger.info("Risk guardrail: drawdown reduction → allocation multiplier %.2f", dd_mult)

    # --- V2 Market Regime Detection ---
    nifty_candles = fetch_daily_candles(smart_api, NIFTY_TOKEN, days=50)
    vix_ltp = get_ltp(smart_api, "India VIX", VIX_TOKEN)
    time.sleep(config.API_DELAY)

    # Fallback: fetch VIX from yfinance if SmartAPI token is stale
    if vix_ltp is None:
        try:
            import yfinance as yf
            vix_ticker = yf.Ticker("^INDIAVIX")
            vix_hist = vix_ticker.history(period="1d")
            if not vix_hist.empty:
                vix_ltp = float(vix_hist["Close"].iloc[-1])
                logger.info("VIX from yfinance fallback: %.2f", vix_ltp)
        except Exception as e:
            logger.warning("VIX yfinance fallback failed: %s", e)

    regime_result = None
    iv_percentile = 50  # default

    if nifty_candles and vix_ltp:
        nifty_df = pd.DataFrame(nifty_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        regime_result = classify_regime_v2(nifty_df, vix_ltp, risk_state=risk_state)
        iv_percentile = regime_result.get("iv_percentile", 50)
        logger.info("Regime: %s (confidence=%.2f) — %s",
                    regime_result["regime"], regime_result["confidence"], regime_result["reason"])
    else:
        regime_result = {"regime": "UNCERTAIN", "confidence": 0.3, "reason": "data unavailable", "iv_percentile": 50}
        logger.info("Regime: UNCERTAIN (data unavailable)")

    regime = regime_result["regime"]

    # CASH regime → skip all new positions
    if regime == "CASH":
        logger.warning("CASH regime active — skipping all new positions. Reason: %s",
                        regime_result["reason"])
        return 0

    available_capital = portfolio["available_capital"]

    # V9: PR Sundar's 40% cash reserve — never deploy more than 60% of total capital
    deployed_capital = TOTAL_CAPITAL - portfolio["available_capital"]
    max_deployable = TOTAL_CAPITAL * (1 - config.CASH_RESERVE_PCT)
    if deployed_capital >= max_deployable:
        logger.info("CASH RESERVE: Already deployed ₹%.0f / ₹%.0f max (%.0f%% reserve enforced). No new positions.",
                    deployed_capital, max_deployable, config.CASH_RESERVE_PCT * 100)
        return 0
    # Cap available capital to stay within reserve
    available_capital = min(available_capital, round(max_deployable - deployed_capital, 2))

    # V9: Weekly portfolio stop — if weekly realized+unrealized loss > 4%, stop all entries
    weekly_pnl = _compute_weekly_pnl(portfolio)
    weekly_pnl_pct = weekly_pnl / TOTAL_CAPITAL * 100
    if weekly_pnl_pct <= -(config.WEEKLY_PORTFOLIO_SL_PCT * 100):
        logger.warning("WEEKLY PORTFOLIO SL: Weekly P&L %.1f%% exceeds -%.0f%% limit. No new positions.",
                       weekly_pnl_pct, config.WEEKLY_PORTFOLIO_SL_PCT * 100)
        _telegram_send(f"<b>Weekly Portfolio SL Hit</b>\nWeekly P&amp;L: {weekly_pnl_pct:+.1f}% (₹{weekly_pnl:+,.0f})\nNo new entries until next week.")
        return 0

    # Apply size multiplier from risk manager (weekly drawdown)
    if size_mult < 1.0:
        available_capital = round(available_capital * size_mult, 2)
        logger.info("Risk manager: weekly drawdown → reducing capital to ₹%.0f (%.0f%%)",
                    available_capital, size_mult * 100)

    # VOLATILE regime → reduce capital by 50%
    if regime == "VOLATILE":
        available_capital = round(available_capital * REGIME_SIZE_REDUCTION, 2)
        logger.info("VOLATILE regime — reducing available capital to ₹%.0f", available_capital)

    # --- V3 Global Macro Intelligence ---
    pcr = 1.0  # default neutral
    max_pain = 0
    try:
        from oi_analysis import compute_pcr, compute_max_pain
        chain = fetch_option_chain(smart_api, "NIFTY", NIFTY_TOKEN)
        time.sleep(config.API_DELAY)
        if chain:
            pcr = compute_pcr(chain)
            max_pain = compute_max_pain(chain)
            logger.info("PCR: %.2f | Max Pain: %.0f", pcr, max_pain)
    except Exception as e:
        logger.debug("OI analysis failed: %s", e)

    # Fetch macro context (US, GIFT, Asia, FII/DII)
    macro = None
    try:
        from global_intel import fetch_macro_context
        prev_close = nifty_candles[-1][4] if nifty_candles else 0
        macro = fetch_macro_context(prev_nifty_close=prev_close, pcr=pcr)
        logger.info("Global: S&P %+.1f%% | Nasdaq %+.1f%% | GIFT gap %+.1f%% | FII %+.0fcr",
                    macro.get("sp500_pct_change", 0), macro.get("nasdaq_pct_change", 0),
                    macro.get("gift_nifty_gap_pct", 0), macro.get("fii_net_crores", 0))
    except Exception as e:
        logger.debug("Global intel failed: %s", e)

    # Apply macro hard gates
    if macro:
        gate = macro.get("hard_gate", "NONE")
        gate_reason = macro.get("hard_gate_reason", "")

        if gate == "BLOCK_ALL":
            logger.warning("MACRO BLOCK ALL: %s — skipping all new positions", gate_reason)
            return 0
        elif gate == "BLOCK_BULLISH":
            logger.warning("MACRO BLOCK BULLISH: %s", gate_reason)
            # Filter out bullish candidates in the loop below
        elif gate == "REDUCE_50":
            available_capital = round(available_capital * 0.5, 2)
            logger.info("MACRO REDUCE 50%%: %s → capital ₹%.0f", gate_reason, available_capital)
        elif gate == "REDUCE_25":
            available_capital = round(available_capital * 0.75, 2)
            logger.info("MACRO REDUCE 25%%: %s → capital ₹%.0f", gate_reason, available_capital)

    # --- V3 Supertrend + CPR ---
    supertrend_signal = "unknown"
    cpr_day_type = "normal"
    try:
        from indicators_v3 import compute_supertrend, compute_cpr
        if nifty_candles and len(nifty_candles) >= 15:
            nifty_df_st = pd.DataFrame(nifty_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            supertrend_signal = compute_supertrend(nifty_df_st)
            logger.info("Supertrend: %s", supertrend_signal)

            prev_candle = nifty_candles[-2] if len(nifty_candles) >= 2 else nifty_candles[-1]
            cpr_result = compute_cpr(prev_high=prev_candle[2], prev_low=prev_candle[3], prev_close=prev_candle[4])
            cpr_day_type = cpr_result["day_type"]
            logger.info("CPR: %s (width=%.3f%%, pivot=%.0f)", cpr_day_type, cpr_result["cpr_width_pct"], cpr_result["pivot"])
    except Exception as e:
        logger.debug("Supertrend/CPR failed: %s", e)

    # --- X/Twitter sentiment (soft signal) ---
    x_sentiment = None
    try:
        from x_intel import fetch_x_sentiment
        x_sentiment = fetch_x_sentiment()
        if x_sentiment:
            logger.info("X sentiment: %s (confidence=%s, themes=%s)",
                        x_sentiment.get("sentiment"), x_sentiment.get("confidence"),
                        x_sentiment.get("key_themes", []))
    except Exception as e:
        logger.debug("X intel unavailable: %s", e)

    # --- V3 Market Breadth ---
    breadth = None
    try:
        from breadth import compute_breadth
        breadth = compute_breadth()
        if breadth:
            logger.info("Breadth: %d/%d advancing (%.0f%%) — %s",
                        breadth["advancing"], breadth["total"],
                        breadth["breadth_pct"], breadth["signal"])
            # Divergent breadth + bullish regime = warning, reduce capital
            if breadth["signal"] == "divergent" and regime in ("TRENDING_UP",):
                available_capital = round(available_capital * 0.75, 2)
                logger.warning("BREADTH DIVERGENCE: market rising but breadth weak — reducing capital 25%%")
    except Exception as e:
        logger.debug("Breadth unavailable: %s", e)

    # --- V4 Sector Rotation ---
    sector_rotation = None
    try:
        from sector_rotation import get_sector_rotation
        sector_rotation = get_sector_rotation()
        if sector_rotation and sector_rotation.get("rotation_detected"):
            logger.info("SECTOR ROTATION: strengthening=%s, weakening=%s",
                        sector_rotation["strengthening"], sector_rotation["weakening"])
    except Exception as e:
        logger.debug("Sector rotation unavailable: %s", e)

    # --- V4 Unusual OI (per-stock, applied in allocation loop) ---
    # Will be checked per-candidate below using existing option chain data

    # V8: Max concurrent equity position check
    open_equity = [p for p in portfolio["positions"]
                   if p["status"] == "open" and p.get("instrument", "EQ") == "EQ"]
    if len(open_equity) >= MAX_EQUITY_POSITIONS:
        logger.info("MAX EQUITY POSITIONS (%d) reached — skipping new equity entries.", MAX_EQUITY_POSITIONS)
        # Still allow options/spreads, filter out equity-only candidates
        candidates = [c for c in candidates if c.get("instrument_hint") in ("PUT", "SPREAD", "CONDOR")]
        if not candidates:
            return 0

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
        if c["direction"] == "bullish" and check_sector_limit(c["symbol"], portfolio["positions"]):
            logger.info("SKIP %s: sector limit reached (%s)", c["symbol"], get_sector(c["symbol"]))
            continue
        filtered.append(c)

    if not filtered:
        logger.info("No candidates after cooldown/sector filters.")
        return 0

    # --- Entry Quality Filters (regime, RSI, volume, news) ---
    quality_filtered = []
    for c in filtered:
        symbol = c["symbol"]
        direction = c["direction"]

        # V2 regime filter
        if regime == "TRENDING_DOWN" and direction == "bullish":
            logger.info("SKIP %s: bullish entry blocked in TRENDING_DOWN regime", symbol)
            continue
        if regime == "TRENDING_UP" and direction == "bearish":
            logger.info("SKIP %s: bearish entry blocked in TRENDING_UP regime", symbol)
            continue
        if regime == "SIDEWAYS" and c.get("instrument_hint") != "SPREAD":
            # In SIDEWAYS, allow spreads but skip directional equity
            logger.info("SKIP %s: directional equity blocked in SIDEWAYS regime (spreads only)", symbol)
            continue
        if regime == "UNCERTAIN" and direction == "bearish":
            # UNCERTAIN → conservative, equity only (bullish)
            logger.info("SKIP %s: bearish entry blocked in UNCERTAIN regime (conservative)", symbol)
            continue
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

        # V9: PCR contrarian filter (Nitin Murarka)
        if config.PCR_FILTER_ENABLED and pcr > 0:
            if direction == "bullish" and pcr < config.PCR_BEARISH_CONTRARIAN:
                # PCR < 0.7 = too many calls, market likely topping → skip bullish
                logger.info("SKIP %s: PCR %.2f < %.1f (bearish extreme — skip bullish)", symbol, pcr, config.PCR_BEARISH_CONTRARIAN)
                continue
            if direction == "bearish" and pcr > config.PCR_BULLISH_CONTRARIAN:
                # PCR > 1.3 = too many puts, market likely bottoming → skip bearish
                logger.info("SKIP %s: PCR %.2f > %.1f (bullish extreme — skip bearish)", symbol, pcr, config.PCR_BULLISH_CONTRARIAN)
                continue

        # V9: Max pain gravity near expiry
        if config.MAX_PAIN_FILTER_ENABLED and max_pain > 0 and nifty_candles:
            nifty_ltp_current = float(nifty_candles[-1][4])
            distance_from_mp = abs(nifty_ltp_current - max_pain) / nifty_ltp_current * 100
            if distance_from_mp > config.MAX_PAIN_PROXIMITY_PCT:
                # Price is far from max pain — near expiry, price tends to gravitate toward it
                if direction == "bullish" and nifty_ltp_current > max_pain:
                    # Bullish but price is above max pain — gravity will pull down
                    c["allocation_multiplier"] = c.get("allocation_multiplier", 1.0) * 0.75
                    logger.info("%s: Price %.0f > max pain %.0f by %.1f%% — reducing bullish allocation 25%%",
                               symbol, nifty_ltp_current, max_pain, distance_from_mp)
                elif direction == "bearish" and nifty_ltp_current < max_pain:
                    # Bearish but price is below max pain — gravity will pull up
                    c["allocation_multiplier"] = c.get("allocation_multiplier", 1.0) * 0.75
                    logger.info("%s: Price %.0f < max pain %.0f by %.1f%% — reducing bearish allocation 25%%",
                               symbol, nifty_ltp_current, max_pain, distance_from_mp)

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

        # V3: Macro BLOCK_BULLISH gate
        if macro and macro.get("hard_gate") == "BLOCK_BULLISH" and direction == "bullish":
            logger.info("SKIP %s: bullish blocked by macro gate (%s)", symbol, macro.get("hard_gate_reason", ""))
            continue

        # V3: Nasdaq IT crash → block IT sector bullish
        if macro and macro.get("hard_gate") == "BLOCK_IT_BULLISH" and direction == "bullish":
            sector = SYMBOL_SECTOR.get(symbol, "")
            if sector == "IT":
                logger.info("SKIP %s: IT bullish blocked by Nasdaq crash", symbol)
                continue

        # V3: Supertrend disagreement → reduce allocation
        if supertrend_signal != "unknown":
            if direction == "bullish" and supertrend_signal == "sell":
                c["allocation_multiplier"] = c.get("allocation_multiplier", 1.0) * (1 - config.SUPERTREND_DISAGREE_REDUCTION)
                logger.info("%s: Supertrend disagrees (sell vs bullish), reducing allocation 25%%", symbol)
            elif direction == "bearish" and supertrend_signal == "buy":
                c["allocation_multiplier"] = c.get("allocation_multiplier", 1.0) * (1 - config.SUPERTREND_DISAGREE_REDUCTION)
                logger.info("%s: Supertrend disagrees (buy vs bearish), reducing allocation 25%%", symbol)

        # V3: X/Twitter contradiction filter (soft)
        if x_sentiment and x_sentiment.get("sentiment") == "crisis":
            logger.warning("X sentiment: CRISIS detected — %s", x_sentiment.get("key_themes", []))

        # YouTube stock intel gate (Mode B) — only as contradiction filter
        try:
            from youtube_intel import fetch_stock_intel
            yt_stock = fetch_stock_intel(symbol)
            if yt_stock and yt_stock.get("sentiment"):
                if (direction == "bullish" and yt_stock["sentiment"] == "bearish" and
                    yt_stock.get("confidence") == "high"):
                    logger.info("SKIP %s: YouTube strongly bearish (contradicts bullish signal)", symbol)
                    continue
                if (direction == "bearish" and yt_stock["sentiment"] == "bullish" and
                    yt_stock.get("confidence") == "high"):
                    logger.info("SKIP %s: YouTube strongly bullish (contradicts bearish signal)", symbol)
                    continue
        except Exception:
            pass  # YouTube intel is optional

        # Earnings blackout check
        if is_near_earnings(symbol, today):
            logger.info("SKIP %s: earnings within 48 hours (blackout)", symbol)
            continue

        # Multi-timeframe confirmation
        try:
            from multi_timeframe import check_multi_timeframe
            aligned, mtf_reason, mtf_confidence = check_multi_timeframe(symbol, direction)
            if not aligned:
                logger.info("SKIP %s: weekly trend disagrees (%s)", symbol, mtf_reason)
                continue
        except Exception:
            pass  # optional signal, don't block on failure

        quality_filtered.append(c)

    if not quality_filtered:
        logger.info("No candidates after quality filters (RSI/volume/news).")
        # Send Telegram summary even when no candidates
        try:
            from claude_intel import generate_cron_summary
            skip_reasons = [f"{c['symbol']}({c.get('skip_reason', '?')})" for c in candidates[:3]]
            summary = generate_cron_summary("open", portfolio, regime=regime, vix=vix_ltp, skipped_reasons=skip_reasons)
            _telegram_send(summary)
        except Exception:
            pass
        return 0

    # --- Claude Intelligence: evaluate candidates before trading ---
    try:
        from claude_intel import evaluate_candidates
        nifty_ltp_current = get_ltp(smart_api, "Nifty 50", NIFTY_TOKEN) if smart_api else None
        claude_approved = evaluate_candidates(
            quality_filtered,
            regime=regime,
            regime_confidence=regime_result.get("confidence", 0.5),
            vix=vix_ltp,
            macro=macro,
            pcr=pcr,
            max_pain=max_pain,
            portfolio=portfolio,
            nifty_ltp=nifty_ltp_current,
        )
        if not claude_approved:
            logger.info("Claude intel: no candidates approved.")
            try:
                from claude_intel import generate_cron_summary
                summary = generate_cron_summary("open", portfolio, regime=regime, vix=vix_ltp,
                                                skipped_reasons=["Claude rejected all"])
                _telegram_send(summary)
            except Exception:
                pass
            return 0
        quality_filtered = claude_approved
    except Exception as e:
        logger.debug("Claude intel unavailable: %s — using rule-based candidates", e)

    top = quality_filtered[:TOP_N]
    allocations = compute_allocations(top, available_capital)
    opened = 0

    for alloc in allocations:
        symbol = alloc["symbol"]
        direction = alloc["direction"]

        # Risk manager: check position limits before opening
        sector = SYMBOL_SECTOR.get(symbol, "Other")
        allocation = alloc["allocation"]

        # V4: Apply allocation_multiplier from Supertrend disagreement
        allocation = round(allocation * alloc.get("allocation_multiplier", 1.0), 2)

        # V10: Apply Claude intelligence allocation adjustment
        claude_adj = alloc.get("claude_allocation_adj", 1.0)
        if claude_adj != 1.0:
            allocation = round(allocation * claude_adj, 2)
            logger.info("  %s: Claude adj %.1fx → ₹%.0f", symbol, claude_adj, allocation)

        # V4: Sector rotation adjustment
        if sector_rotation and sector != "Other":
            if sector in sector_rotation.get("strengthening", []):
                allocation = round(allocation * (1 + config.SECTOR_ROTATION_BOOST), 2)
                logger.info("  %s: sector %s strengthening → +%.0f%% allocation", symbol, sector, config.SECTOR_ROTATION_BOOST * 100)
            elif sector in sector_rotation.get("weakening", []):
                allocation = round(allocation * (1 - config.SECTOR_ROTATION_CUT), 2)
                logger.info("  %s: sector %s weakening → -%.0f%% allocation", symbol, sector, config.SECTOR_ROTATION_CUT * 100)
        open_pos_for_rm = [
            {"strategy": p.get("instrument", "EQ"), "sector": SYMBOL_SECTOR.get(p["symbol"], "Other"),
             "max_loss": p.get("allocated", 0)}
            for p in portfolio["positions"] if p["status"] == "open"
        ]
        allowed, rm_reason = rm.can_open_position(open_pos_for_rm, new_sector=sector, new_max_loss=allocation)
        if not allowed:
            logger.info("SKIP %s: %s", symbol, rm_reason)
            continue

        # --- Risk Guardrails: per-candidate checks ---
        # Correlation guard (sector + direction concentration)
        corr_blocked, corr_reason, corr_mult = check_correlation_guard(
            portfolio, symbol, direction, SYMBOL_SECTOR)
        if corr_blocked:
            logger.info("SKIP %s: [Correlation] %s", symbol, corr_reason)
            continue
        if corr_mult < 1.0:
            allocation = round(allocation * corr_mult, 2)
            logger.info("  %s: %s", symbol, corr_reason)

        # Apply drawdown allocation multiplier from portfolio-level check
        if dd_mult < 1.0:
            allocation = round(allocation * dd_mult, 2)

        # Portfolio heat check
        entry_est = allocation  # rough estimate before we know exact entry
        risk_pct = None  # will use default 3%
        heat_blocked, heat_reason = check_portfolio_heat(portfolio, allocation, risk_pct)
        if heat_blocked:
            logger.info("SKIP %s: [Heat] %s", symbol, heat_reason)
            _telegram_send(f"<b>Risk Guardrail</b>\nSkipped {symbol}: {heat_reason}")
            continue

        # Event calendar check
        evt_blocked, evt_reason, evt_mult = check_event_calendar(symbol)
        if evt_blocked:
            logger.info("SKIP %s: [Event] %s", symbol, evt_reason)
            _telegram_send(f"<b>Risk Guardrail</b>\nSkipped {symbol}: {evt_reason}")
            continue
        if evt_mult < 1.0:
            allocation = round(allocation * evt_mult, 2)
            logger.info("  %s: %s", symbol, evt_reason)

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

        # V4: Unusual OI activity check
        try:
            from oi_unusual import detect_unusual_oi, get_oi_signal_for_trade
            stock_chain = fetch_option_chain(smart_api, symbol, token)
            time.sleep(config.API_DELAY)
            if stock_chain:
                unusual_oi = detect_unusual_oi(stock_chain, symbol)
                if unusual_oi.get("has_unusual_activity"):
                    oi_signal = get_oi_signal_for_trade(unusual_oi, direction, ltp)
                    if oi_signal["signal"] == "cautionary":
                        allocation = round(allocation * (1 - config.OI_UNUSUAL_CAUTIONARY_CUT), 2)
                        logger.info("  %s: unusual OI cautionary (%s) → -%.0f%% allocation",
                                    symbol, oi_signal["reason"], config.OI_UNUSUAL_CAUTIONARY_CUT * 100)
                    elif oi_signal["signal"] == "confirming":
                        logger.info("  %s: unusual OI confirms %s (%s)", symbol, direction, oi_signal["reason"])
        except Exception as e:
            logger.debug("Unusual OI check failed for %s: %s", symbol, e)

        # V3: F&O ban check for options/spreads
        if direction == "bearish":
            try:
                from fo_ban import is_in_fo_ban
                if is_in_fo_ban(symbol):
                    logger.info("SKIP %s: F&O ban active — no new F&O positions", symbol)
                    continue
            except Exception:
                pass  # fail-open if ban check unavailable

        # --- V8: Futures path (high-conviction + OI confirmation) ---
        # Check if this candidate qualifies for futures trading
        open_fut = [p for p in portfolio["positions"]
                    if p["status"] == "open" and p.get("instrument") == "FUT"]
        fut_eligible = (
            alloc["score"] >= FUT_MIN_SCORE
            and len(open_fut) < MAX_FUT_POSITIONS
            and regime not in ("CASH", "SIDEWAYS", "UNCERTAIN")
            and vix_ltp is not None and vix_ltp < VIX_HIGH_THRESHOLD
        )

        # OI confirmation check: require matching buildup pattern
        if fut_eligible and FUT_OI_CONFIRMATION:
            cats = alloc.get("categories", [])
            if direction == "bullish":
                fut_eligible = "LongBuildUp" in cats or "PercOIGainers" in cats
            elif direction == "bearish":
                fut_eligible = "ShortBuildUp" in cats or "PercOILosers" in cats

        # F&O ban check for futures
        if fut_eligible:
            try:
                from fo_ban import is_in_fo_ban
                if is_in_fo_ban(symbol):
                    logger.info("SKIP FUT %s: F&O ban active", symbol)
                    fut_eligible = False
            except Exception:
                pass

        if fut_eligible:
            # Fetch ATR for futures sizing
            candles = fetch_daily_candles(smart_api, token)
            time.sleep(config.API_DELAY)
            atr_for_fut = compute_atr(candles, ATR_PERIOD) if candles else None

            if atr_for_fut and atr_for_fut > 0:
                fut_alloc = min(allocation, TOTAL_CAPITAL * 0.20)  # max 20% of capital per futures trade
                fut_pos = _open_futures_position(
                    smart_api, symbol, token, ltp, fut_alloc, alloc,
                    atr_for_fut, direction=direction
                )
                if fut_pos:
                    fut_pos["market_regime"] = regime
                    fut_pos["iv_percentile"] = iv_percentile
                    if alloc.get("claude_reasoning"):
                        fut_pos["claude_reasoning"] = alloc["claude_reasoning"]
                        fut_pos["claude_conviction"] = alloc.get("claude_conviction", "")
                    position = fut_pos
                    # Skip equity/options for this symbol since we're in futures
                    portfolio["positions"].append(position)
                    portfolio["available_capital"] = round(
                        portfolio["available_capital"] - position["allocated"], 2)
                    opened += 1
                    logger.info("FUT position opened for %s — skipping equity/options path", symbol)

                    # Journal + live execution (same as equity path below)
                    if config.LIVE_MODE:
                        try:
                            from live_execution import execute_live_entry
                            execute_live_entry(smart_api, position)
                        except Exception as e:
                            logger.error("Live FUT entry failed for %s: %s", symbol, e)
                    try:
                        from journal import record_entry
                        record_entry(
                            symbol=symbol, direction=direction,
                            instrument="FUT",
                            entry_price=position["entry_price"],
                            entry_date=today,
                            signals={"categories": alloc.get("categories", [])},
                            regime=regime,
                            macro_gate=macro.get("hard_gate", "NONE") if macro else "",
                            supertrend=supertrend_signal,
                            pcr=pcr,
                            breadth=breadth.get("signal", "") if breadth else "",
                            sector_rotation=(sector_rotation.get("top_sector", "") if sector_rotation else ""),
                        )
                    except Exception:
                        pass
                    continue  # next candidate

        if direction == "bearish":
            # --- Bearish path: spread if low IV, else put ---
            if iv_percentile < 70:
                # Try spread first (cheaper in low IV)
                try:
                    spread_pos = _open_spread_position(
                        portfolio, symbol, direction, None, allocation,
                        regime, iv_percentile, ""
                    )
                except Exception:
                    spread_pos = None

                if spread_pos is not None:
                    position = spread_pos
                else:
                    # Fallback to put option
                    position = _open_put_position(smart_api, symbol, token, ltp, allocation, alloc)
                    if position is None:
                        continue
            else:
                position = _open_put_position(smart_api, symbol, token, ltp, allocation, alloc)
                if position is None:
                    continue
            position["market_regime"] = regime
            position["iv_percentile"] = iv_percentile
            if alloc.get("claude_reasoning"):
                position["claude_reasoning"] = alloc["claude_reasoning"]
                position["claude_conviction"] = alloc.get("claude_conviction", "")
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

            # V4: Volatility-adjusted position sizing
            if config.VOL_SIZING_ENABLED and atr_at_entry is not None and ltp > 0:
                atr_pct = (atr_at_entry / ltp) * 100
                if atr_pct > 0:
                    vol_mult = config.VOL_SIZING_BASELINE_ATR_PCT / atr_pct
                    vol_mult = max(config.VOL_SIZING_MIN_MULT, min(config.VOL_SIZING_MAX_MULT, vol_mult))
                    old_alloc = allocation
                    allocation = round(allocation * vol_mult, 2)
                    logger.info("  %s: ATR%%=%.2f → vol_mult=%.2f (alloc ₹%.0f → ₹%.0f)",
                                symbol, atr_pct, vol_mult, old_alloc, allocation)

            if atr_at_entry is not None:
                target_price = round(entry_price + ATR_TARGET_MULTIPLIER * atr_at_entry, 2)
                stoploss_price = round(entry_price - ATR_STOPLOSS_MULTIPLIER * atr_at_entry, 2)
                logger.debug("%s ATR=%.2f → target=%.2f, SL=%.2f", symbol, atr_at_entry, target_price, stoploss_price)
            else:
                target_price = round(entry_price * (1 + TARGET_PCT / 100), 2)
                stoploss_price = round(entry_price * (1 + STOPLOSS_PCT / 100), 2)

            max_hold_date = _add_trading_days(today, MAX_HOLD_DAYS)

            # Recompute quantity with volatility-adjusted allocation
            quantity = math.floor(allocation / entry_price)
            if quantity < 1:
                logger.info("SKIP %s: vol-adjusted allocation ₹%.0f < 1 share at ₹%.2f", symbol, allocation, entry_price)
                continue
            actual_allocated = round(quantity * entry_price, 2)

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
                "iv_percentile": iv_percentile,
                "max_hold_date": max_hold_date,
                "status": "open",
            }
            # Store Claude intel from batch eval
            if alloc.get("claude_reasoning"):
                position["claude_reasoning"] = alloc["claude_reasoning"]
                position["claude_conviction"] = alloc.get("claude_conviction", "")

            logger.info("OPENED %s BUY x%d @ ₹%.2f (slipped from ₹%.2f, ₹%.0f, score=%.1f)",
                        symbol, quantity, entry_price, ltp, actual_allocated, alloc['score'])

        portfolio["positions"].append(position)
        portfolio["available_capital"] = round(
            portfolio["available_capital"] - position["allocated"], 2)
        opened += 1

        # Live execution bridge: place real order if LIVE_MODE is on
        if config.LIVE_MODE:
            try:
                from live_execution import execute_live_entry
                live_result = execute_live_entry(smart_api, position)
                if live_result and live_result.get("status") == "filled":
                    logger.info("Live order filled for %s: order_id=%s, fill=%.2f, slippage=%s",
                                symbol, live_result["order_id"],
                                live_result["fill_price"], live_result.get("slippage"))
            except Exception as e:
                logger.error("Live execution entry failed for %s: %s", symbol, e)

        # Journal: record entry thesis
        try:
            from journal import record_entry
            record_entry(
                symbol=symbol, direction=direction,
                instrument=position.get("instrument", "EQ"),
                entry_price=position["entry_price"],
                entry_date=today,
                signals={"categories": alloc.get("categories", [])},
                regime=regime,
                macro_gate=macro.get("hard_gate", "NONE") if macro else "",
                supertrend=supertrend_signal,
                pcr=pcr,
                breadth=breadth.get("signal", "") if breadth else "",
                sector_rotation=(sector_rotation.get("top_sector", "") if sector_rotation else ""),
                x_sentiment=x_sentiment.get("sentiment", "") if x_sentiment else "",
                score=alloc.get("score", 0),
                allocation=position["allocated"],
            )
        except Exception:
            pass  # journal is optional

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

    instrument: "EQ", "OPT", or "FUT"
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
    elif instrument == "FUT":
        brokerage = min(BROKERAGE_FLAT, turnover * 0.0003)
        stt = turnover * FUT_STT_PCT if side == "sell" else 0.0
        exchange = turnover * FUT_EXCHANGE_PCT
        stamp_duty = turnover * FUT_STAMP_DUTY_PCT if side == "buy" else 0.0
        sebi = turnover * FUT_SEBI_PCT
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
                   close_qty: int | None = None, smart_api=None) -> dict:
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

    # Notifications (after stats update so portfolio summary is accurate)
    emoji = "+" if pnl >= 0 else ""
    reason_label = {"target": "Target hit", "stoploss": "Stop-loss hit",
                    "trailing_stop": "Trailing stop hit", "theta_decay": "Theta decay exit",
                    "partial_target": "Partial target", "expiry": "Max hold expired",
                    "manual": "Manual close"}.get(reason, reason)
    _notify("Paper Trade Exit",
            f"{pos['symbol']} {reason_label}: {pnl_pct:+.1f}% (₹{emoji}{pnl:,.0f})")
    # Rich Telegram exit notification with Claude reasoning
    try:
        from claude_intel import format_exit_telegram
        exit_msg = format_exit_telegram(
            pos, reason, pnl, pnl_pct, exit_price,
            pos.get("_claude_exit_reasoning", ""),
            portfolio,
        )
        _telegram_send(exit_msg)
    except Exception:
        _telegram_notify_exit(pos, reason, pnl, pnl_pct, exit_price, portfolio)

    # Live execution bridge: place real exit order if LIVE_MODE is on
    if config.LIVE_MODE and smart_api is not None:
        try:
            from live_execution import execute_live_exit
            live_result = execute_live_exit(smart_api, pos, exit_price, reason)
            if live_result and live_result.get("status") == "filled":
                closed["live_exit_order_id"] = live_result["order_id"]
                closed["live_exit_fill_price"] = live_result["fill_price"]
                closed["live_exit_slippage"] = live_result.get("slippage")
        except Exception as e:
            logger.error("Live execution exit failed for %s: %s", pos["symbol"], e)

    # Journal: record exit review
    try:
        from journal import record_exit
        holding = _trading_days_between(pos["entry_date"], _today_ist())
        record_exit(
            symbol=pos["symbol"], entry_date=pos["entry_date"],
            direction=direction, exit_price=exit_price,
            exit_date=_today_ist(), exit_reason=reason,
            pnl=pnl, pnl_pct=pnl_pct, costs=costs["total"],
            holding_days=holding,
        )
    except Exception:
        pass

    # Smart Monitor: Claude post-trade review → extract lesson
    try:
        from smart_monitor import record_trade_lesson
        record_trade_lesson(pos, closed, portfolio)
    except Exception as e:
        logger.debug("Trade lesson recording failed: %s", e)

    # Clean up closed symbols from monitor state
    try:
        from smart_monitor import cleanup_closed_positions, _load_state, _save_state
        state = _load_state()
        cleanup_closed_positions(portfolio, state)
        _save_state(state)
    except Exception as e:
        logger.debug("State cleanup failed: %s", e)

    return closed


def detect_gap_through_sl(pos: dict, current_ltp: float) -> bool:
    """Detect if a position has gapped through its stoploss.

    Returns True if current LTP is beyond the stoploss in the wrong direction.
    """
    sl = pos.get("stoploss_price")
    if sl is None:
        return False

    if pos["direction"] == "bullish":
        return current_ltp < sl
    else:
        return current_ltp > sl


def _is_friday_afternoon(weekday: int = None, hour: int = None, minute: int = None) -> bool:
    """Check if it's Friday afternoon (2 PM+). Used for weekend risk reduction.

    Args can be overridden for testing; defaults to current IST time.
    """
    if weekday is None:
        now = datetime.now(IST)
        weekday = now.weekday()
        hour = now.hour
        minute = now.minute
    return weekday == 4 and hour >= 14


def monitor_positions(smart_api, portfolio: dict) -> tuple:
    """
    Check each open position for exit conditions. Returns number of exits.
    """
    open_pos = [p for p in portfolio["positions"] if p["status"] == "open"]
    if not open_pos:
        logger.info("No open positions to monitor.")
        return 0, 0

    today = _today_ist()
    exits = 0
    unrealized_pnl = 0.0
    session_refreshed = False

    for pos in open_pos:
        symbol = pos["symbol"]
        is_option = pos.get("instrument") == "OPT"
        is_futures = pos.get("instrument") == "FUT"
        is_spread = pos.get("instrument") == "SPREAD"

        if is_spread:
            # --- Spread position monitoring ---
            underlying_token = pos.get("token", "")
            underlying_ltp = get_ltp(smart_api, symbol, underlying_token) if underlying_token else None
            time.sleep(config.API_DELAY)

            if underlying_ltp is None:
                logger.warning("%s spread: could not fetch underlying LTP, skipping", symbol)
                continue

            long_premium = estimate_premium_from_underlying(pos, "long", underlying_ltp, today)
            short_premium = estimate_premium_from_underlying(pos, "short", underlying_ltp, today)

            exit_reason = check_spread_exit(pos, underlying_ltp=underlying_ltp, today=today,
                                            long_premium=long_premium, short_premium=short_premium)

            if exit_reason:
                spread_pnl = calc_spread_pnl(
                    pos["long_leg"]["entry_premium"], long_premium,
                    pos["short_leg"]["entry_premium"], short_premium,
                    pos["quantity"],
                )
                exit_price = underlying_ltp  # track underlying for reference
                pnl_pct = (spread_pnl / pos["allocated"]) * 100 if pos["allocated"] > 0 else 0

                # Claude exit eval for spread
                try:
                    from claude_intel import evaluate_exit
                    should_exit, exit_reasoning = evaluate_exit(
                        pos, exit_reason, underlying_ltp, spread_pnl, pnl_pct, portfolio)
                    if not should_exit:
                        logger.info("CLAUDE HOLD %s spread: overriding %s — %s", symbol, exit_reason, exit_reasoning)
                        unrealized_pnl += spread_pnl
                        continue
                    pos["_claude_exit_reasoning"] = exit_reasoning
                except Exception as e:
                    logger.debug("Claude exit eval failed for spread %s: %s", symbol, e)

                logger.info("EXIT %s spread (%s): P&L ₹%.0f (%.1f%%), reason=%s",
                            symbol, pos.get("strategy", ""), spread_pnl, pnl_pct, exit_reason)
                close_position(portfolio, pos, exit_price, exit_reason, smart_api=smart_api)
                exits += 1
            else:
                # HOLD — log spread status
                spread_pnl = calc_spread_pnl(
                    pos["long_leg"]["entry_premium"], long_premium,
                    pos["short_leg"]["entry_premium"], short_premium,
                    pos["quantity"],
                )
                pnl_pct = (spread_pnl / pos["allocated"]) * 100 if pos["allocated"] > 0 else 0
                unrealized_pnl += spread_pnl
                logger.info("%s spread: underlying ₹%.2f  P&L: ₹%.0f (%.1f%%)  [HOLD]",
                            symbol, underlying_ltp, spread_pnl, pnl_pct)
            continue  # skip the EQ/OPT logic below

        # --- Iron Condor monitoring ---
        is_condor = pos.get("instrument") == "CONDOR"
        if is_condor:
            underlying_ltp = get_ltp(smart_api, "Nifty 50", NIFTY_TOKEN)
            time.sleep(config.API_DELAY)
            if underlying_ltp is None:
                logger.warning("NIFTY condor: could not fetch LTP, skipping")
                continue

            # Estimate premiums for all 4 legs
            call_short_prem = estimate_premium_from_underlying(
                {"long_leg": pos["call_short"], "underlying_at_entry": pos.get("underlying_at_entry", 0),
                 "expiry": pos["expiry"]},
                "long", underlying_ltp, today)
            call_long_prem = estimate_premium_from_underlying(
                {"long_leg": pos["call_long"], "underlying_at_entry": pos.get("underlying_at_entry", 0),
                 "expiry": pos["expiry"]},
                "long", underlying_ltp, today)
            put_short_prem = estimate_premium_from_underlying(
                {"long_leg": pos["put_short"], "underlying_at_entry": pos.get("underlying_at_entry", 0),
                 "expiry": pos["expiry"]},
                "long", underlying_ltp, today)
            put_long_prem = estimate_premium_from_underlying(
                {"long_leg": pos["put_long"], "underlying_at_entry": pos.get("underlying_at_entry", 0),
                 "expiry": pos["expiry"]},
                "long", underlying_ltp, today)

            # Get VIX for spike detection
            current_vix = get_ltp(smart_api, "India VIX", VIX_TOKEN)
            time.sleep(config.API_DELAY)
            prev_vix = pos.get("vix_at_entry", current_vix)

            exit_reason = check_condor_exit(
                pos, call_short_prem, call_long_prem, put_short_prem, put_long_prem,
                vix=current_vix, prev_vix=prev_vix,
            )

            # Calculate condor P&L
            cost_to_close = (call_short_prem - call_long_prem + put_short_prem - put_long_prem) * pos["quantity"]
            condor_pnl = pos["net_credit"] - cost_to_close
            pnl_pct = (condor_pnl / pos["allocated"]) * 100 if pos["allocated"] > 0 else 0

            # V9: Monthly condor SL cap — if monthly loss exceeds cap, force exit
            if (exit_reason is None and pos.get("strategy") == "monthly_condor"
                    and condor_pnl < 0):
                monthly_sl_cap = pos.get("monthly_sl_cap", TOTAL_CAPITAL * 0.04)
                if abs(condor_pnl) >= monthly_sl_cap:
                    exit_reason = "monthly_sl_cap"
                    logger.warning("MONTHLY CONDOR SL CAP: loss ₹%.0f >= cap ₹%.0f", abs(condor_pnl), monthly_sl_cap)

            if exit_reason:
                # Claude exit eval for condor
                try:
                    from claude_intel import evaluate_exit
                    should_exit, exit_reasoning = evaluate_exit(
                        pos, exit_reason, underlying_ltp, condor_pnl, pnl_pct, portfolio, vix=current_vix)
                    if not should_exit:
                        logger.info("CLAUDE HOLD %s condor: overriding %s — %s",
                                    pos.get("index_name", "NIFTY"), exit_reason, exit_reasoning)
                        unrealized_pnl += condor_pnl
                        continue
                    pos["_claude_exit_reasoning"] = exit_reasoning
                except Exception as e:
                    logger.debug("Claude exit eval failed for condor: %s", e)

                logger.info("EXIT %s condor: P&L ₹%.0f (%.1f%%), reason=%s",
                            pos.get("index_name", "NIFTY"), condor_pnl, pnl_pct, exit_reason)
                close_position(portfolio, pos, underlying_ltp, exit_reason, smart_api=smart_api)
                exits += 1
            else:
                # V9: Dynamic hedge tightening (Reyaansh)
                if config.HEDGE_TIGHTEN_ENABLED:
                    idx_name = pos.get("index_name", "NIFTY")
                    closer_pts = (config.HEDGE_TIGHTEN_CLOSER_POINTS_NIFTY
                                  if idx_name == "NIFTY"
                                  else config.HEDGE_TIGHTEN_CLOSER_POINTS_BANKNIFTY)

                    # Check call hedge (long call) — has it decayed significantly?
                    cl_entry = pos["call_long"].get("entry_premium", 0)
                    if cl_entry > 0 and call_long_prem <= cl_entry * config.HEDGE_TIGHTEN_THRESHOLD_PCT:
                        # Hedge has decayed to 40% of entry — simulate tightening
                        old_strike = pos["call_long"].get("strike", 0)
                        new_strike = old_strike - closer_pts
                        logger.info("HEDGE TIGHTEN %s call: ₹%.0f → moved %dpt closer (strike %d→%d)",
                                    idx_name, call_long_prem, closer_pts, old_strike, new_strike)
                        pos["call_long"]["strike"] = new_strike
                        pos["call_long"]["entry_premium"] = call_long_prem  # reset entry to current
                        pos["_hedge_tightened_call"] = True

                    # Check put hedge (long put)
                    pl_entry = pos["put_long"].get("entry_premium", 0)
                    if pl_entry > 0 and put_long_prem <= pl_entry * config.HEDGE_TIGHTEN_THRESHOLD_PCT:
                        old_strike = pos["put_long"].get("strike", 0)
                        new_strike = old_strike + closer_pts
                        logger.info("HEDGE TIGHTEN %s put: ₹%.0f → moved %dpt closer (strike %d→%d)",
                                    idx_name, put_long_prem, closer_pts, old_strike, new_strike)
                        pos["put_long"]["strike"] = new_strike
                        pos["put_long"]["entry_premium"] = put_long_prem
                        pos["_hedge_tightened_put"] = True

                unrealized_pnl += condor_pnl
                logger.info("%s condor%s: spot ₹%.0f  P&L: ₹%.0f (%.1f%%)  [HOLD]",
                            pos.get("index_name", "NIFTY"),
                            " (monthly)" if pos.get("strategy") == "monthly_condor" else "",
                            underlying_ltp, condor_pnl, pnl_pct)
            continue

        # --- IV Crush monitoring ---
        is_iv_crush = pos.get("instrument") == "IV_CRUSH"
        if is_iv_crush:
            underlying_ltp = get_ltp(smart_api, "Nifty 50", NIFTY_TOKEN)
            time.sleep(config.API_DELAY)
            if underlying_ltp is None:
                logger.warning("IV crush: could not fetch LTP, skipping")
                continue

            # Estimate premiums for all 4 legs (same as condor)
            cs_prem = estimate_premium_from_underlying(
                {"long_leg": pos["call_short"], "underlying_at_entry": pos.get("underlying_at_entry", 0),
                 "expiry": pos["expiry"]},
                "long", underlying_ltp, today)
            cl_prem = estimate_premium_from_underlying(
                {"long_leg": pos["call_long"], "underlying_at_entry": pos.get("underlying_at_entry", 0),
                 "expiry": pos["expiry"]},
                "long", underlying_ltp, today)
            ps_prem = estimate_premium_from_underlying(
                {"long_leg": pos["put_short"], "underlying_at_entry": pos.get("underlying_at_entry", 0),
                 "expiry": pos["expiry"]},
                "long", underlying_ltp, today)
            pl_prem = estimate_premium_from_underlying(
                {"long_leg": pos["put_long"], "underlying_at_entry": pos.get("underlying_at_entry", 0),
                 "expiry": pos["expiry"]},
                "long", underlying_ltp, today)

            # Calculate P&L
            cost_to_close = (cs_prem - cl_prem + ps_prem - pl_prem) * pos["quantity"]
            crush_pnl = pos["net_credit"] - cost_to_close
            pnl_pct = (crush_pnl / pos["allocated"]) * 100 if pos["allocated"] > 0 else 0

            # Set _current_value for check_iv_crush_exit
            pos["_current_value"] = cost_to_close

            try:
                from iv_crush import check_iv_crush_exit
                should_exit, reason = check_iv_crush_exit(
                    pos, smart_api=smart_api, _today=today,
                    _underlying_ltp=underlying_ltp,
                )
            except Exception as e:
                logger.debug("IV crush exit check failed: %s", e)
                should_exit, reason = False, "hold"

            if should_exit:
                logger.info("EXIT IV crush %s: P&L ₹%.0f (%.1f%%), reason=%s",
                            pos.get("event_type", ""), crush_pnl, pnl_pct, reason)
                close_position(portfolio, pos, underlying_ltp, reason, smart_api=smart_api)
                exits += 1
            else:
                unrealized_pnl += crush_pnl
                logger.info("IV crush %s: Nifty ₹%.0f  P&L: ₹%.0f (%.1f%%)  [HOLD]",
                            pos.get("event_type", ""), underlying_ltp, crush_pnl, pnl_pct)

            # Clean up temporary field
            pos.pop("_current_value", None)
            continue

        # --- Momentum option monitoring ---
        is_momentum = pos.get("instrument") == "MOMENTUM"
        if is_momentum:
            # Get current option premium via underlying estimation
            underlying_ltp = get_ltp(smart_api, "Nifty 50", NIFTY_TOKEN)
            time.sleep(config.API_DELAY)
            if underlying_ltp is None:
                logger.warning("NIFTY momentum: could not fetch LTP, skipping")
                continue

            # Estimate current premium
            strike = pos.get("strike", 0)
            opt_type = pos.get("option_type", "CE")
            if opt_type == "CE":
                intrinsic = max(0, underlying_ltp - strike)
            else:
                intrinsic = max(0, strike - underlying_ltp)

            # Simple estimate: intrinsic + time value decay
            entry_premium = pos["entry_price"]
            entry_intrinsic = max(0, (pos.get("underlying_at_entry", underlying_ltp) - strike) if opt_type == "CE"
                                  else (strike - pos.get("underlying_at_entry", underlying_ltp)))
            time_value_at_entry = max(0, entry_premium - entry_intrinsic)

            dte = days_to_expiry(pos.get("expiry", ""))
            entry_dte = max(dte + 1, 1)  # rough estimate
            tv_remaining = time_value_at_entry * (dte / entry_dte) * 0.7
            current_premium = max(0.05, intrinsic + tv_remaining)

            # Update peak
            if current_premium > pos.get("peak_premium", 0):
                pos["peak_premium"] = current_premium

            exit_reason = check_momentum_exit(pos, current_premium, today)

            pnl = (current_premium - entry_premium) * pos["quantity"]
            pnl_pct = (current_premium - entry_premium) / entry_premium * 100 if entry_premium > 0 else 0

            if exit_reason:
                # Claude exit eval for momentum
                try:
                    from claude_intel import evaluate_exit
                    should_exit, exit_reasoning = evaluate_exit(
                        pos, exit_reason, current_premium, pnl, pnl_pct, portfolio)
                    if not should_exit:
                        logger.info("CLAUDE HOLD NIFTY momentum: overriding %s — %s", exit_reason, exit_reasoning)
                        unrealized_pnl += pnl
                        continue
                    pos["_claude_exit_reasoning"] = exit_reasoning
                except Exception as e:
                    logger.debug("Claude exit eval failed for momentum: %s", e)

                logger.info("EXIT NIFTY momentum %s: P&L ₹%.0f (%.1f%%), reason=%s",
                            pos.get("direction", ""), pnl, pnl_pct, exit_reason)
                close_position(portfolio, pos, current_premium, exit_reason, smart_api=smart_api)
                exits += 1
            else:
                unrealized_pnl += pnl
                logger.info("NIFTY momentum %s: prem ₹%.2f  P&L: ₹%.0f (%.1f%%)  [HOLD]",
                            pos.get("direction", ""), current_premium, pnl, pnl_pct)
            continue

        # Check max hold expiry first (doesn't need LTP)
        expired = today > pos["max_hold_date"]

        # Check DTE-based forced exit for options
        dte_expired = False
        dte = None
        if is_option:
            dte = days_to_expiry(pos["expiry"])
            if dte < MIN_DTE_TO_HOLD:
                dte_expired = True

        # Fetch current LTP with session-refresh retry
        ltp = None
        for _attempt in range(2):
            try:
                if is_option or is_futures:
                    ltp = get_ltp_nfo(smart_api, pos.get("contract_symbol", symbol), pos["token"])
                else:
                    ltp = get_ltp(smart_api, symbol, pos["token"])
                    if ltp is None:
                        ltp = get_ltp(smart_api, f"{symbol}-EQ", pos["token"])
                break  # no session error, proceed
            except SessionExpiredError:
                if _attempt == 0 and not session_refreshed:
                    logger.info("Session expired, re-authenticating...")
                    try:
                        refresh_session(smart_api)
                        session_refreshed = True
                        time.sleep(config.API_DELAY)
                        continue  # retry LTP fetch
                    except Exception as e:
                        logger.warning("Session refresh failed: %s", e)
                        break
                else:
                    break  # already refreshed once, give up
        time.sleep(config.API_DELAY)

        if ltp is None:
            logger.warning("%s: could not fetch LTP, skipping", symbol)
            continue

        # V3: Gap-through-SL detection — immediate exit if gapped past SL
        if not is_option and not is_spread and detect_gap_through_sl(pos, ltp):
            gap_pct = abs(ltp - pos["stoploss_price"]) / pos["entry_price"] * 100
            logger.warning("GAP THROUGH SL: %s gapped to ₹%.2f (SL was ₹%.2f, gap=%.1f%%)",
                          symbol, ltp, pos["stoploss_price"], gap_pct)
            gap_instrument = pos.get("instrument", "EQ")
            exit_price = apply_slippage(ltp, gap_instrument, "sell" if pos["direction"] == "bullish" else "buy")
            pnl_direction = pos["direction"]
            if pnl_direction == "bullish":
                pnl = round((exit_price - pos["entry_price"]) * abs(pos["quantity"]), 2)
            else:
                pnl = round((pos["entry_price"] - exit_price) * abs(pos["quantity"]), 2)
            pnl_pct = calc_pnl_pct(pos["entry_price"], exit_price, pnl_direction)
            logger.warning("FORCE EXIT %s: gap through SL → P&L ₹%.0f (%.1f%%)", symbol, pnl, pnl_pct)
            pos["_claude_exit_reasoning"] = "SL exit (gap_through_sl) — non-negotiable risk management"
            close_position(portfolio, pos, exit_price, "gap_through_sl", smart_api=smart_api)
            exits += 1
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

        # --- Auction window buffer: widen thresholds during 9:15-9:30 / 3:15-3:30 ---
        auction_atr_buffer = 0.0
        try:
            from smart_monitor import get_auction_atr_buffer
            auction_atr_buffer = get_auction_atr_buffer()
        except Exception:
            pass

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
                # ATR-based: price thresholds + ATR-based trailing stop
                if pos["direction"] == "bullish":
                    if ltp >= pos["target_price"]:
                        reason = "target"
                    else:
                        # V3: Enhanced trailing after activation threshold
                        unrealized_pct = calc_pnl_pct(pos["entry_price"], ltp, "bullish")
                        if unrealized_pct >= config.TRAILING_SL_ACTIVATION_PCT:
                            # Tighter trail: 1.5x ATR from high water mark, floored at entry
                            trailing_sl = peak - config.TRAILING_SL_ATR_MULT * atr_at_entry
                            trailing_sl = max(trailing_sl, pos["entry_price"])
                        else:
                            # Standard trail before activation
                            trailing_sl = peak - ATR_TRAILING_MULTIPLIER * atr_at_entry
                        fixed_sl = pos["stoploss_price"]
                        effective_sl = max(trailing_sl, fixed_sl)
                        # Auction window: widen SL by 0.5x ATR to avoid false exits
                        if auction_atr_buffer > 0:
                            effective_sl -= auction_atr_buffer * atr_at_entry
                        if ltp <= effective_sl:
                            reason = "trailing_stop" if trailing_sl > fixed_sl else "stoploss"
                else:
                    if ltp <= pos["target_price"]:
                        reason = "target"
                    else:
                        # V3: Enhanced trailing after activation threshold
                        unrealized_pct = calc_pnl_pct(pos["entry_price"], ltp, "bearish")
                        if unrealized_pct >= config.TRAILING_SL_ACTIVATION_PCT:
                            trailing_sl = peak + config.TRAILING_SL_ATR_MULT * atr_at_entry
                            trailing_sl = min(trailing_sl, pos["entry_price"])
                        else:
                            trailing_sl = peak + ATR_TRAILING_MULTIPLIER * atr_at_entry
                        fixed_sl = pos["stoploss_price"]
                        effective_sl = min(trailing_sl, fixed_sl)
                        # Auction window: widen SL by 0.5x ATR to avoid false exits
                        if auction_atr_buffer > 0:
                            effective_sl += auction_atr_buffer * atr_at_entry
                        if ltp >= effective_sl:
                            reason = "trailing_stop" if trailing_sl < fixed_sl else "stoploss"
            else:
                # Legacy percentage-based (no ATR data)
                target_pct = TARGET_PCT
                stoploss_pct = STOPLOSS_PCT

                if pos["direction"] == "bullish":
                    trailing_sl = peak * (1 - LEGACY_TRAILING_STOP_PCT / 100)
                    fixed_sl = pos.get("stoploss_price", pos["entry_price"] * (1 + stoploss_pct / 100))
                    effective_sl = max(trailing_sl, fixed_sl)

                    if pnl_pct >= target_pct:
                        reason = "target"
                    elif ltp <= effective_sl:
                        reason = "trailing_stop" if trailing_sl > fixed_sl else "stoploss"
                else:
                    trailing_sl = peak * (1 + LEGACY_TRAILING_STOP_PCT / 100)
                    fixed_sl = pos.get("stoploss_price", pos["entry_price"] * (1 - stoploss_pct / 100))
                    effective_sl = min(trailing_sl, fixed_sl)

                    if pnl_pct >= target_pct:
                        reason = "target"
                    elif ltp >= effective_sl:
                        reason = "trailing_stop" if trailing_sl < fixed_sl else "stoploss"

            # V4: Time-based SL tightening — as max_hold approaches, tighten stops
            if reason is None and config.TIME_SL_ENABLED and not is_option:
                atr_time = pos.get("atr_at_entry")
                if atr_time is not None:
                    day_num_t = _trading_days_between(pos["entry_date"], today)
                    max_days_t = _trading_days_between(pos["entry_date"], pos["max_hold_date"])
                    if max_days_t > 0:
                        progress = day_num_t / max_days_t
                        if progress >= 0.75:
                            # At 75%+ of hold period: tighten to breakeven or 0.5x ATR
                            if pos["direction"] == "bullish":
                                time_sl = max(pos["entry_price"], peak - config.TIME_SL_THREE_QUARTER_MULT * atr_time)
                                if ltp <= time_sl and time_sl > pos.get("stoploss_price", 0):
                                    reason = "time_tightened_sl"
                            else:
                                time_sl = min(pos["entry_price"], peak + config.TIME_SL_THREE_QUARTER_MULT * atr_time)
                                if ltp >= time_sl and time_sl < pos.get("stoploss_price", float("inf")):
                                    reason = "time_tightened_sl"
                        elif progress >= 0.5:
                            # At 50-75%: tighten to 0.75x ATR
                            if pos["direction"] == "bullish":
                                time_sl = peak - config.TIME_SL_HALF_LIFE_MULT * atr_time
                                if ltp <= time_sl and time_sl > pos.get("stoploss_price", 0):
                                    reason = "time_tightened_sl"
                            else:
                                time_sl = peak + config.TIME_SL_HALF_LIFE_MULT * atr_time
                                if ltp >= time_sl and time_sl < pos.get("stoploss_price", float("inf")):
                                    reason = "time_tightened_sl"

            if reason is None and expired:
                reason = "expiry"

        # --- Claude Intelligence: evaluate exit decision ---
        if reason is not None:
            try:
                from claude_intel import evaluate_exit
                # Compute P&L for Claude eval
                qty_eval = abs(pos["quantity"])
                if is_option:
                    pnl_eval = (ltp - pos["entry_price"]) * qty_eval
                elif pos["direction"] == "bullish":
                    pnl_eval = (ltp - pos["entry_price"]) * qty_eval
                else:
                    pnl_eval = (pos["entry_price"] - ltp) * qty_eval
                should_exit, exit_reasoning = evaluate_exit(
                    pos, reason, ltp, pnl_eval, pnl_pct, portfolio)
                if not should_exit:
                    logger.info("CLAUDE HOLD %s: overriding %s exit — %s", symbol, reason, exit_reasoning)
                    reason = None  # reset to trigger HOLD path below
                else:
                    pos["_claude_exit_reasoning"] = exit_reasoning
            except Exception as e:
                logger.debug("Claude exit eval failed for %s: %s", symbol, e)

        if reason is None:
            # HOLD — calculate unrealized rupee P&L
            qty = abs(pos["quantity"])
            if is_option:
                pos_pnl = (ltp - pos["entry_price"]) * qty  # bought put, long position
            elif pos["direction"] == "bullish":
                pos_pnl = (ltp - pos["entry_price"]) * qty
            else:
                pos_pnl = (pos["entry_price"] - ltp) * qty
            unrealized_pnl += pos_pnl

            day_num = _trading_days_between(pos["entry_date"], today)
            max_days = _trading_days_between(pos["entry_date"], pos["max_hold_date"])
            if is_option:
                logger.info(f"{symbol} PE{int(pos['strike'])}: Prem ₹{ltp:.2f}  P&L: {pnl_pct:+.1f}% (₹{pos_pnl:+,.0f})  DTE:{dte}  Day {day_num}/{max_days}  [HOLD]")
            else:
                # Compute effective trailing SL for display
                atr_hold = pos.get("atr_at_entry")
                if atr_hold is not None:
                    if pos["direction"] == "bullish":
                        trail_display = max(peak - ATR_TRAILING_MULTIPLIER * atr_hold, pos["stoploss_price"])
                    else:
                        trail_display = min(peak + ATR_TRAILING_MULTIPLIER * atr_hold, pos["stoploss_price"])
                else:
                    if pos["direction"] == "bullish":
                        trail_display = max(peak * (1 - LEGACY_TRAILING_STOP_PCT / 100),
                                            pos.get("stoploss_price", pos["entry_price"] * (1 + STOPLOSS_PCT / 100)))
                    else:
                        trail_display = min(peak * (1 + LEGACY_TRAILING_STOP_PCT / 100),
                                            pos.get("stoploss_price", pos["entry_price"] * (1 - STOPLOSS_PCT / 100)))
                instr_tag = f" FUT {pos['direction'].upper()}" if is_futures else ""
                logger.info(f"{symbol}{instr_tag}: LTP ₹{ltp:.2f}  P&L: {pnl_pct:+.1f}% (₹{pos_pnl:+,.0f})  Trail: ₹{trail_display:.2f}  Day {day_num}/{max_days}  [HOLD]")
            continue

        # --- Partial exit for equity/futures target ---
        if reason == "target" and not is_option and not pos.get("partial_exit_done"):
            if is_futures:
                # Futures: partial exit in lot-size increments
                lot_size = pos.get("lot_size", 1)
                total_lots = pos.get("num_lots", 1)
                partial_lots = max(1, total_lots // 2)
                partial_qty = partial_lots * lot_size
                if partial_lots < total_lots:
                    exit_side = "buy" if pos["direction"] == "bearish" else "sell"
                    exit_price = apply_slippage(ltp, "FUT", exit_side)
                    close_position(portfolio, pos, exit_price, "partial_target", close_qty=partial_qty, smart_api=smart_api)
                    pos["partial_exit_done"] = True
                    pos["num_lots"] = total_lots - partial_lots
                    logger.info("%s FUT: PARTIAL EXIT (%d/%d lots) @ ₹%.2f  [remaining holds]",
                                symbol, partial_lots, total_lots, exit_price)
                    continue
            else:
                partial_qty = abs(pos["quantity"]) // 2
                if partial_qty >= 1:
                    exit_price = apply_slippage(ltp, "EQ", "sell")
                    close_position(portfolio, pos, exit_price, "partial_target", close_qty=partial_qty, smart_api=smart_api)
                    pos["partial_exit_done"] = True
                    logger.info("%s: PARTIAL EXIT (%d/%d) @ ₹%.2f  [remaining holds]",
                                symbol, partial_qty, abs(pos["quantity"]) + partial_qty, exit_price)
                    continue

        # --- V4: Partial profit booking on options ---
        if is_option and not pos.get("opt_partial_done") and pnl_pct >= config.OPT_PARTIAL_PROFIT_PCT:
            total_lots = pos.get("num_lots", 1)
            partial_lots = max(1, int(total_lots * config.OPT_PARTIAL_RATIO))
            if partial_lots < total_lots:
                partial_qty = partial_lots * pos.get("lot_size", abs(pos["quantity"]))
                if partial_qty < abs(pos["quantity"]):
                    exit_price = apply_slippage(ltp, "OPT", "sell")
                    close_position(portfolio, pos, exit_price, "opt_partial_profit", close_qty=partial_qty, smart_api=smart_api)
                    pos["opt_partial_done"] = True
                    pos["num_lots"] = total_lots - partial_lots
                    logger.info("%s PE%d: OPT PARTIAL PROFIT (%d/%d lots) @ ₹%.2f (+%.1f%%)  [remaining holds]",
                                symbol, int(pos.get("strike", 0)), partial_lots, total_lots, exit_price, pnl_pct)
                    continue

        # Apply slippage on exit (short futures exit = buy side)
        instrument = pos.get("instrument", "EQ")
        exit_side = "buy" if (is_futures and pos["direction"] == "bearish") else "sell"
        exit_price = apply_slippage(ltp, instrument, exit_side)

        closed = close_position(portfolio, pos, exit_price, reason, smart_api=smart_api)
        tag = reason.upper().replace("_", " ")
        if is_option:
            logger.info("%s PE%d: EXIT (%s) @ ₹%.2f  P&L: %+.1f%% (₹%+.0f)",
                        symbol, int(pos['strike']), tag, exit_price, closed['pnl_pct'], closed['pnl'])
        elif is_futures:
            logger.info("%s FUT %s: EXIT (%s) @ ₹%.2f  P&L: %+.1f%% (₹%+.0f)",
                        symbol, pos["direction"].upper(), tag, exit_price, closed['pnl_pct'], closed['pnl'])
        else:
            logger.info("%s: EXIT (%s) @ ₹%.2f  P&L: %+.1f%% (₹%+.0f)",
                        symbol, tag, exit_price, closed['pnl_pct'], closed['pnl'])
        exits += 1

    # V3: Friday afternoon risk reduction — tighten stops for weekend
    if _is_friday_afternoon():
        for pos in [p for p in portfolio["positions"] if p["status"] == "open"]:
            is_fno = pos.get("instrument") in ("OPT", "FUT", "SPREAD", "CONDOR", "MOMENTUM", "IV_CRUSH")
            if is_fno:
                # Close marginal F&O positions (less than 1% unrealized gain)
                # We'd need LTP here but it was already fetched above — log warning for now
                logger.warning("FRIDAY RISK: F&O position %s open over weekend — monitor closely", pos["symbol"])
            else:
                # Tighten equity trailing SL to breakeven if in profit
                if pos.get("peak_price", pos["entry_price"]) > pos["entry_price"] * 1.01:
                    old_sl = pos.get("stoploss_price", 0)
                    new_sl = max(old_sl, pos["entry_price"])
                    if new_sl > old_sl:
                        pos["stoploss_price"] = new_sl
                        logger.info("FRIDAY RISK: %s SL tightened to breakeven ₹%.2f (was ₹%.2f)",
                                   pos["symbol"], new_sl, old_sl)

    # ── Greeks computation for option positions ─────────────────────────
    remaining_open = [p for p in portfolio["positions"] if p["status"] == "open"]
    opt_positions = [p for p in remaining_open if p.get("instrument") in ("OPT", "MOMENTUM")]
    if opt_positions:
        try:
            from greeks import compute_position_greeks, portfolio_greeks_summary
            # Build spot prices map from underlying tokens
            spot_map = {}
            for p in opt_positions:
                sym = p["symbol"]
                if sym not in spot_map:
                    token = p.get("token", "")
                    if token:
                        try:
                            sp = get_ltp(smart_api, sym, token)
                            if sp:
                                spot_map[sym] = sp
                            time.sleep(config.API_DELAY)
                        except Exception:
                            pass
                    # Fallback: use underlying_price_at_entry if LTP unavailable
                    if sym not in spot_map and p.get("underlying_price_at_entry"):
                        spot_map[sym] = p["underlying_price_at_entry"]

            if spot_map:
                pg = compute_position_greeks(opt_positions, spot_map, config.RISK_FREE_RATE)
                if pg:
                    summary = portfolio_greeks_summary(pg)
                    logger.info("Greeks: Delta=%+.2f  Gamma=%+.4f  Theta=₹%+.0f/day  Vega=%+.2f  Premium@Risk=₹%,.0f",
                                summary["net_delta"], summary["net_gamma"],
                                summary["theta_decay_daily_rupees"], summary["net_vega"],
                                summary["total_premium_at_risk"])
                    if summary["total_premium_at_risk"] > 0:
                        theta_pct = abs(summary["theta_decay_daily_rupees"]) / summary["total_premium_at_risk"] * 100
                        if theta_pct >= config.GREEKS_THETA_WARN_PCT:
                            logger.warning("THETA WARN: Daily decay %.1f%% of premium at risk", theta_pct)
        except Exception as e:
            logger.debug("Greeks computation skipped: %s", e)

    # Clean up closed positions from the open list
    portfolio["positions"] = [p for p in portfolio["positions"] if p["status"] == "open"]

    return exits, unrealized_pnl


def close_all_positions(smart_api, portfolio: dict) -> int:
    """Force-close all open positions at current LTP."""
    open_pos = [p for p in portfolio["positions"] if p["status"] == "open"]
    if not open_pos:
        logger.info("No open positions to close.")
        return 0

    exits = 0
    session_refreshed = False
    for pos in open_pos:
        is_option = pos.get("instrument") == "OPT"
        ltp = None
        for _attempt in range(2):
            try:
                if is_option:
                    ltp = get_ltp_nfo(smart_api, pos.get("contract_symbol", pos["symbol"]), pos["token"])
                else:
                    ltp = get_ltp(smart_api, pos["symbol"], pos["token"])
                    if ltp is None:
                        ltp = get_ltp(smart_api, f"{pos['symbol']}-EQ", pos["token"])
                break
            except SessionExpiredError:
                if _attempt == 0 and not session_refreshed:
                    logger.info("Session expired, re-authenticating...")
                    try:
                        refresh_session(smart_api)
                        session_refreshed = True
                        time.sleep(config.API_DELAY)
                        continue
                    except Exception as e:
                        logger.warning("Session refresh failed: %s", e)
                        break
                else:
                    break
        time.sleep(config.API_DELAY)

        if ltp is None:
            logger.warning("%s: could not fetch LTP, using entry price", pos['symbol'])
            ltp = pos["entry_price"]

        instrument = pos.get("instrument", "EQ")
        exit_price = apply_slippage(ltp, instrument, "sell")
        closed = close_position(portfolio, pos, exit_price, "manual", smart_api=smart_api)
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
                try:
                    if is_option:
                        ltp = get_ltp_nfo(smart_api, pos.get("contract_symbol", pos["symbol"]), pos["token"])
                    else:
                        ltp = get_ltp(smart_api, pos["symbol"], pos["token"])
                        if ltp is None:
                            ltp = get_ltp(smart_api, f"{pos['symbol']}-EQ", pos["token"])
                except SessionExpiredError:
                    pass  # dashboard is display-only, skip on session error
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


def _is_weekly_theta_day(d: date = None) -> bool:
    """True if today is Friday (4) or Monday (0) — good days to sell weekly theta."""
    if d is None:
        d = datetime.now(IST).date()
    return d.weekday() in (0, 4)  # Monday or Friday


def check_weekly_theta_exit(pos: dict, current_call_prem: float, current_put_prem: float) -> str | None:
    """Check exit conditions for weekly theta (short strangle).

    Returns exit reason or None.
    """
    total_current = current_call_prem + current_put_prem
    total_entry = pos["total_credit"]

    # Target: 50% premium decay
    if total_current <= total_entry * config.WEEKLY_THETA_TARGET_PCT:
        return "target"

    # Stop loss: premium doubles
    if total_current >= total_entry * config.WEEKLY_THETA_SL_MULTIPLIER:
        return "stoploss"

    return None


# ---------------------------------------------------------------------------
# Index-Level Strategies (Iron Condor + Momentum)
# ---------------------------------------------------------------------------


def _try_index_strategies(smart_api, portfolio: dict) -> int:
    """Attempt index-level strategies on all configured indices (Nifty + BankNifty).

    Runs iron condor and momentum options on each index.
    Returns total number of positions opened.
    """
    opened = 0
    today = _today_ist()
    near_macro = is_near_macro_event(today)

    # Check which instruments are already open
    open_instruments = [p.get("instrument") for p in portfolio["positions"] if p["status"] == "open"]
    open_index_names = [p.get("index_name", "NIFTY") for p in portfolio["positions"] if p["status"] == "open"]

    available = portfolio["available_capital"]

    for idx_cfg in INDEX_CONFIGS:
        idx_name = idx_cfg["name"]
        idx_token = idx_cfg["token"]
        lot_size = idx_cfg["lot_size"]
        vix_mult = idx_cfg["vix_mult"]

        # Fetch candles for this index
        candles = fetch_daily_candles(smart_api, idx_token, days=50)
        if not candles:
            continue

        vix_ltp = get_ltp(smart_api, "India VIX", VIX_TOKEN)
        time.sleep(config.API_DELAY)

        idx_ltp = get_ltp(smart_api, idx_name, idx_token)
        time.sleep(config.API_DELAY)

        if not idx_ltp or not vix_ltp:
            continue

        # Regime detection (using per-index candles for regime)
        idx_df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        regime_result = classify_regime_v2(idx_df, vix_ltp)
        regime = regime_result["regime"]
        iv_percentile = regime_result.get("iv_percentile", 50)
        confidence = regime_result.get("confidence", 0)

        # Scale VIX thresholds for this index
        condor_min_vix = config.CONDOR_MIN_VIX * vix_mult
        condor_max_vix = config.CONDOR_MAX_VIX * vix_mult

        # Check if this index already has positions
        has_condor = any(i == "CONDOR" and n == idx_name
                         for i, n in zip(open_instruments, open_index_names))
        has_momentum = any(i == "MOMENTUM" and n == idx_name
                            for i, n in zip(open_instruments, open_index_names))

        # --- Iron Condor: SIDEWAYS + VIX in range + no macro event + Mon-Wed ---
        if (not has_condor
            and regime == "SIDEWAYS"
            and confidence >= 0.70
            and condor_min_vix <= vix_ltp <= condor_max_vix
            and not near_macro
            and datetime.now(IST).weekday() < 3
        ):
            max_condor_alloc = available * config.ALLOC_IRON_CONDOR_MAX
            max_loss_budget = TOTAL_CAPITAL * config.CONDOR_MAX_RISK_PCT

            try:
                from agent_with_options import select_iron_condor_strikes

                chain = fetch_option_chain(smart_api, idx_name)
                time.sleep(config.API_DELAY)

                if chain:
                    expiry = get_nearest_expiry(chain, min_dte=config.CONDOR_TIME_EXIT_DAYS + 5)
                    condor_data = select_iron_condor_strikes(
                        chain, spot=idx_ltp, lot_size=lot_size,
                        max_loss_budget=max_loss_budget,
                        min_credit_per_lot=config.CONDOR_MIN_CREDIT_PER_LOT,
                    )

                    if condor_data:
                        condor_data["spot"] = idx_ltp
                        condor_data["vix_at_entry"] = vix_ltp
                        condor_data["index_name"] = idx_name
                        allocation = min(condor_data["max_loss"], max_condor_alloc)

                        # Claude entry eval for condor
                        try:
                            from claude_intel import evaluate_entry
                            condor_candidate = {
                                "symbol": idx_name, "direction": "neutral",
                                "score": 0, "categories": ["IronCondor"],
                                "sector": "Index",
                            }
                            approved, reasoning, adj = evaluate_entry(
                                condor_candidate, "CONDOR", regime, vix_ltp,
                                None, 1.0, idx_ltp, portfolio, nifty_ltp=idx_ltp,
                                extra_context=f"Credit=₹{condor_data['net_credit']:.0f}, MaxLoss=₹{condor_data['max_loss']:.0f}, VIX={vix_ltp:.1f}")
                            if not approved:
                                logger.info("CLAUDE SKIP condor on %s: %s", idx_name, reasoning)
                                continue
                            condor_data["claude_reasoning"] = reasoning
                            condor_data["claude_conviction"] = "medium"
                        except Exception as e:
                            logger.debug("Claude entry eval failed for condor: %s", e)
                            reasoning = ""

                        pos = _open_iron_condor(
                            portfolio, condor_data, allocation, regime, iv_percentile,
                            expiry or "",
                        )
                        if pos:
                            pos["index_name"] = idx_name
                            if condor_data.get("claude_reasoning"):
                                pos["claude_reasoning"] = condor_data["claude_reasoning"]
                            opened += 1
                            logger.info("OPENED iron condor on %s: credit=₹%.0f, max_loss=₹%.0f",
                                        idx_name, condor_data["net_credit"], condor_data["max_loss"])
            except Exception as e:
                logger.debug("Iron condor attempt on %s failed: %s", idx_name, e)

        # --- Momentum: breakout + TRENDING/VOLATILE regime ---
        if (not has_momentum
            and regime in ("TRENDING_UP", "TRENDING_DOWN", "VOLATILE")
            and iv_percentile < 80
        ):
            breakout = check_nifty_breakout(candles, period=20)

            if breakout:
                valid = False
                if breakout == "bullish" and regime in ("TRENDING_UP", "VOLATILE"):
                    valid = True
                elif breakout == "bearish" and regime in ("TRENDING_DOWN", "VOLATILE"):
                    valid = True

                if valid:
                    max_momentum_alloc = TOTAL_CAPITAL * config.MOMENTUM_MAX_RISK_PCT

                    try:
                        chain = fetch_option_chain(smart_api, idx_name)
                        time.sleep(config.API_DELAY)

                        if chain:
                            expiry = get_nearest_expiry(chain, min_dte=10)
                            opt_type = "CE" if breakout == "bullish" else "PE"

                            strikes = sorted(chain, key=lambda x: abs(x["strikePrice"] - idx_ltp))
                            atm = strikes[0] if strikes else None

                            if atm and atm.get(opt_type, {}).get("lastPrice"):
                                premium = atm[opt_type]["lastPrice"]
                                strike = atm["strikePrice"]
                                total_cost = premium * lot_size

                                if total_cost <= max_momentum_alloc and premium > 0:
                                    # Claude entry eval for momentum
                                    momentum_reasoning = ""
                                    try:
                                        from claude_intel import evaluate_entry
                                        mom_candidate = {
                                            "symbol": idx_name, "direction": breakout,
                                            "score": 0, "categories": ["Momentum", "Breakout"],
                                            "sector": "Index",
                                        }
                                        approved, momentum_reasoning, adj = evaluate_entry(
                                            mom_candidate, "MOMENTUM", regime, vix_ltp,
                                            None, 1.0, idx_ltp, portfolio, nifty_ltp=idx_ltp,
                                            extra_context=f"{opt_type} {strike} @ ₹{premium:.2f}, cost=₹{total_cost:.0f}")
                                        if not approved:
                                            logger.info("CLAUDE SKIP momentum %s on %s: %s", breakout, idx_name, momentum_reasoning)
                                            continue
                                    except Exception as e:
                                        logger.debug("Claude entry eval failed for momentum: %s", e)

                                    option_data = {
                                        "strike": strike,
                                        "premium": premium,
                                        "lot_size": lot_size,
                                        "option_type": opt_type,
                                        "index_name": idx_name,
                                    }
                                    pos = _open_momentum_position(
                                        portfolio, breakout, option_data,
                                        round(total_cost, 2), regime, iv_percentile,
                                        expiry or "",
                                    )
                                    if pos:
                                        pos["index_name"] = idx_name
                                        if momentum_reasoning:
                                            pos["claude_reasoning"] = momentum_reasoning
                                        opened += 1
                                        logger.info("OPENED momentum %s on %s: %s %s @ ₹%.2f",
                                                    breakout, idx_name, opt_type, strike, premium)
                    except Exception as e:
                        logger.debug("Momentum attempt on %s failed: %s", idx_name, e)

    # --- V9: Monthly Iron Condor (Reyaansh style) ---
    if config.MONTHLY_CONDOR_ENABLED:
        today_date = datetime.now(IST).date()
        day_of_month = today_date.day

        if config.MONTHLY_CONDOR_ENTRY_DAY_MIN <= day_of_month <= config.MONTHLY_CONDOR_ENTRY_DAY_MAX:
            # Check if we already have a monthly condor open
            has_monthly_condor = any(
                p.get("strategy") == "monthly_condor" and p.get("status") == "open"
                for p in portfolio["positions"]
            )

            if not has_monthly_condor:
                for idx_cfg in INDEX_CONFIGS:
                    idx_name = idx_cfg["name"]
                    lot_size = idx_cfg["lot_size"]
                    hedge_gap = (config.MONTHLY_CONDOR_HEDGE_GAP_NIFTY
                                 if idx_name == "NIFTY"
                                 else config.MONTHLY_CONDOR_HEDGE_GAP_BANKNIFTY)

                    try:
                        vix_now = get_ltp(smart_api, "India VIX", VIX_TOKEN)
                        time.sleep(config.API_DELAY)

                        # Only enter monthly condor when VIX is in sweet spot (12-20)
                        if not vix_now or vix_now < 10 or vix_now > 20:
                            logger.info("SKIP monthly condor %s: VIX %.1f outside [10, 20]", idx_name, vix_now or 0)
                            continue

                        idx_ltp = get_ltp(smart_api, idx_name, idx_cfg["token"])
                        time.sleep(config.API_DELAY)
                        if not idx_ltp:
                            continue

                        chain = fetch_option_chain(smart_api, idx_name)
                        time.sleep(config.API_DELAY)
                        if not chain:
                            continue

                        # Get monthly expiry (>= 20 DTE)
                        expiry = get_nearest_expiry(chain, min_dte=20)
                        if not expiry:
                            logger.info("SKIP monthly condor %s: no expiry with >= 20 DTE", idx_name)
                            continue

                        # Select 25-delta strikes
                        from agent_with_options import select_iron_condor_strikes
                        max_loss_budget = TOTAL_CAPITAL * config.MONTHLY_CONDOR_MAX_RISK_PCT
                        condor_data = select_iron_condor_strikes(
                            chain, spot=idx_ltp, lot_size=lot_size,
                            max_loss_budget=max_loss_budget,
                            min_credit_per_lot=config.CONDOR_MIN_CREDIT_PER_LOT,
                        )

                        if condor_data:
                            condor_data["spot"] = idx_ltp
                            condor_data["vix_at_entry"] = vix_now
                            condor_data["index_name"] = idx_name
                            allocation = min(condor_data["max_loss"], max_loss_budget)

                            # Claude entry eval for monthly condor
                            try:
                                from claude_intel import evaluate_entry
                                mc_candidate = {
                                    "symbol": idx_name, "direction": "neutral",
                                    "score": 0, "categories": ["MonthlyCondor"],
                                    "sector": "Index",
                                }
                                approved, mc_reasoning, _ = evaluate_entry(
                                    mc_candidate, "CONDOR", "SIDEWAYS", vix_now,
                                    None, 1.0, idx_ltp, portfolio, nifty_ltp=idx_ltp,
                                    extra_context=f"Monthly condor. Credit=₹{condor_data['net_credit']:.0f}, MaxLoss=₹{condor_data['max_loss']:.0f}, VIX={vix_now:.1f}")
                                if not approved:
                                    logger.info("CLAUDE SKIP monthly condor on %s: %s", idx_name, mc_reasoning)
                                    continue
                                condor_data["claude_reasoning"] = mc_reasoning
                            except Exception as e:
                                logger.debug("Claude entry eval failed for monthly condor: %s", e)

                            pos = _open_iron_condor(
                                portfolio, condor_data, allocation, "SIDEWAYS",
                                vix_to_iv_percentile(vix_now), expiry or "",
                            )
                            if pos:
                                pos["strategy"] = "monthly_condor"
                                pos["index_name"] = idx_name
                                pos["monthly_sl_cap"] = round(TOTAL_CAPITAL * config.MONTHLY_CONDOR_MONTHLY_SL_PCT, 2)
                                if condor_data.get("claude_reasoning"):
                                    pos["claude_reasoning"] = condor_data["claude_reasoning"]
                                opened += 1
                                logger.info("OPENED monthly condor on %s: credit=₹%.0f, max_loss=₹%.0f, expiry=%s",
                                            idx_name, condor_data["net_credit"], condor_data["max_loss"], expiry)
                    except Exception as e:
                        logger.debug("Monthly condor attempt on %s failed: %s", idx_name, e)

    # --- V7: IV Crush event strategy (Nifty only) ---
    has_iv_crush = any(
        p.get("instrument") == "IV_CRUSH" and p.get("status") == "open"
        for p in portfolio["positions"]
    )
    if not has_iv_crush:
        try:
            from iv_crush import should_enter_iv_crush, open_iv_crush_condor

            crush_setup = should_enter_iv_crush(
                smart_api,
                _vix=get_ltp(smart_api, "India VIX", VIX_TOKEN),
                _regime=regime_result.get("regime", "UNKNOWN") if 'regime_result' in dir() else "UNKNOWN",
                _portfolio=portfolio,
            )

            if crush_setup:
                from agent_with_options import select_iron_condor_strikes

                chain = fetch_option_chain(smart_api, "NIFTY")
                time.sleep(config.API_DELAY)

                if chain:
                    nifty_ltp = get_ltp(smart_api, "NIFTY", NIFTY_TOKEN)
                    time.sleep(config.API_DELAY)

                    expiry = get_nearest_expiry(
                        chain,
                        min_dte=max(crush_setup.get("days_until_event", 2) + 3, 7),
                    )

                    max_loss_budget = TOTAL_CAPITAL * config.CONDOR_MAX_RISK_PCT * 0.5

                    condor_data = select_iron_condor_strikes(
                        chain, spot=nifty_ltp or 0, lot_size=75,
                        max_loss_budget=max_loss_budget,
                        min_credit_per_lot=40,
                    )

                    if condor_data:
                        condor_data["spot"] = nifty_ltp
                        condor_data["lot_size"] = 75
                        condor_data["expiry"] = expiry or ""

                        pos = open_iv_crush_condor(portfolio, crush_setup, condor_data)
                        if pos:
                            opened += 1
                            logger.info(
                                "OPENED IV crush condor for %s (%s): credit=₹%.0f",
                                crush_setup["event"], crush_setup["event_date"],
                                pos["net_credit"],
                            )
        except Exception as e:
            logger.debug("IV crush check failed: %s", e)

    return opened


# ---------------------------------------------------------------------------
# Weekly Performance Report
# ---------------------------------------------------------------------------


def generate_weekly_report(portfolio: dict) -> str:
    """Generate weekly performance report for Telegram.

    Analyzes trades from the past 7 days, computes per-strategy win rates,
    P&L attribution, and portfolio health metrics.
    """
    today = _today_ist()
    week_ago = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")

    closed = portfolio.get("closed_trades", [])
    weekly_trades = [t for t in closed if t.get("exit_date", "") >= week_ago]

    lines = ["<b>📊 Weekly Performance Report</b>\n"]
    lines.append(f"Period: {week_ago} to {today}\n")

    if not weekly_trades:
        lines.append("No trades closed this week.")
        # Still show portfolio status
        stats = portfolio.get("stats", {})
        lines.append(f"\nCapital: ₹{portfolio.get('capital', 0):,.0f}")
        lines.append(f"Total P&amp;L: ₹{stats.get('total_pnl', 0):+,.0f}")
        open_count = len([p for p in portfolio.get("positions", []) if p.get("status") == "open"])
        lines.append(f"Open positions: {open_count}")
        return "\n".join(lines)

    # Overall stats this week
    total_pnl = sum(t["pnl"] for t in weekly_trades)
    wins = [t for t in weekly_trades if t["pnl"] >= 0]
    losses = [t for t in weekly_trades if t["pnl"] < 0]
    win_rate = len(wins) / len(weekly_trades) * 100 if weekly_trades else 0

    lines.append(f"<b>Week Summary:</b>")
    lines.append(f"Trades: {len(weekly_trades)}  |  Win Rate: {win_rate:.0f}%")
    lines.append(f"P&amp;L: ₹{total_pnl:+,.0f}")
    if wins:
        avg_win = sum(t["pnl_pct"] for t in wins) / len(wins)
        lines.append(f"Avg Win: {avg_win:+.1f}%")
    if losses:
        avg_loss = sum(t["pnl_pct"] for t in losses) / len(losses)
        lines.append(f"Avg Loss: {avg_loss:+.1f}%")

    # Per-strategy breakdown
    strategy_stats = {}
    for t in weekly_trades:
        strat = t.get("instrument", "EQ")
        if t.get("exit_reason", "").startswith("opt_partial"):
            strat = "OPT"
        if strat not in strategy_stats:
            strategy_stats[strat] = {"trades": 0, "wins": 0, "pnl": 0.0}
        strategy_stats[strat]["trades"] += 1
        strategy_stats[strat]["pnl"] += t["pnl"]
        if t["pnl"] >= 0:
            strategy_stats[strat]["wins"] += 1

    lines.append(f"\n<b>Per-Strategy:</b>")
    for strat, data in sorted(strategy_stats.items()):
        wr = data["wins"] / data["trades"] * 100 if data["trades"] > 0 else 0
        lines.append(f"  {strat}: {data['trades']} trades, {wr:.0f}% WR, ₹{data['pnl']:+,.0f}")

    # Exit reason breakdown
    reason_counts = {}
    for t in weekly_trades:
        reason = t.get("exit_reason", "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    lines.append(f"\n<b>Exit Reasons:</b>")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {reason}: {count}")

    # Best and worst trades
    if weekly_trades:
        best = max(weekly_trades, key=lambda t: t["pnl_pct"])
        worst = min(weekly_trades, key=lambda t: t["pnl_pct"])
        lines.append(f"\nBest: {best['symbol']} ({best['pnl_pct']:+.1f}%, ₹{best['pnl']:+,.0f})")
        lines.append(f"Worst: {worst['symbol']} ({worst['pnl_pct']:+.1f}%, ₹{worst['pnl']:+,.0f})")

    # Portfolio health
    stats = portfolio.get("stats", {})
    analytics = calc_performance_analytics(closed, portfolio.get("capital", TOTAL_CAPITAL))
    if analytics:
        lines.append(f"\n<b>Portfolio Health:</b>")
        pf = f"{analytics['profit_factor']:.2f}" if analytics["profit_factor"] != float("inf") else "∞"
        lines.append(f"Sharpe: {analytics['sharpe_ratio']:.2f}  |  PF: {pf}")
        lines.append(f"Max DD: {analytics['max_drawdown_pct']:.1f}%  |  Avg Hold: {analytics['avg_holding_days']:.1f}d")

    open_count = len([p for p in portfolio.get("positions", []) if p.get("status") == "open"])
    lines.append(f"\nCapital: ₹{portfolio.get('capital', 0):,.0f}  |  Open: {open_count}")
    lines.append(f"All-time P&amp;L: ₹{stats.get('total_pnl', 0):+,.0f} ({stats.get('total_pnl_pct', 0):+.1f}%)")

    # Tax estimate
    total_pnl = stats.get("total_pnl", 0)
    if total_pnl > 0:
        eq_trades = [t for t in closed if t.get("instrument", "EQ") == "EQ"]
        fo_trades = [t for t in closed if t.get("instrument", "EQ") != "EQ"]
        eq_pnl = sum(t["pnl"] for t in eq_trades)
        fo_pnl = sum(t["pnl"] for t in fo_trades)
        stcg_eq = max(0, eq_pnl) * 0.15  # 15% STCG on equity
        stcg_fo = max(0, fo_pnl) * 0.30  # ~30% slab rate estimate for F&O
        total_tax = stcg_eq + stcg_fo
        lines.append(f"\n<b>Tax Estimate:</b>")
        lines.append(f"Equity STCG (15%): ₹{stcg_eq:,.0f}  |  F&amp;O (30% est): ₹{stcg_fo:,.0f}")
        lines.append(f"Net after tax: ₹{total_pnl - total_tax:+,.0f}")

    # Journal stats
    try:
        from journal import get_journal_stats
        jstats = get_journal_stats()
        if jstats.get("closed", 0) > 0:
            lines.append(f"\n<b>Journal:</b> Thesis accuracy {jstats['thesis_accuracy']:.0f}%")
    except Exception:
        pass

    return "\n".join(lines)


def send_weekly_report() -> None:
    """Send weekly performance report via Telegram."""
    portfolio = load_portfolio()
    report = generate_weekly_report(portfolio)
    _telegram_send(report)
    logger.info("Weekly report sent to Telegram.")


def archive_and_reset() -> str:
    """Archive current portfolio and start fresh.

    Saves current portfolio to data/paper_trades/archive_YYYY-MM-DD.json,
    then resets to empty portfolio with full capital.
    Returns path to archive file.
    """
    portfolio = load_portfolio()
    today = _today_ist()

    # Archive
    archive_file = PORTFOLIO_DIR / f"archive_{today}.json"
    archive_file.write_text(json.dumps(portfolio, indent=2, default=str))

    # Summary of archived period
    stats = portfolio.get("stats", {})
    total_trades = stats.get("total_trades", 0)
    total_pnl = stats.get("total_pnl", 0)
    logger.info("Archived portfolio: %d trades, P&L ₹%+.0f → %s",
                total_trades, total_pnl, archive_file)

    # Reset
    new_portfolio = _empty_portfolio()
    save_portfolio(new_portfolio)
    logger.info("Portfolio reset to ₹%.0f capital.", TOTAL_CAPITAL)

    return str(archive_file)


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

        # YouTube market intel (Mode A)
        yt_market = None
        try:
            from youtube_intel import fetch_market_intel
            yt_market = fetch_market_intel()
            if yt_market:
                logger.info("YouTube intel: bias=%s, confidence=%s",
                            yt_market.get("market_bias", "unknown"),
                            yt_market.get("confidence", "unknown"))
        except Exception as e:
            logger.debug("YouTube intel unavailable: %s", e)

        logger.info("\nOpening positions (available capital: ₹%.0f)...", portfolio['available_capital'])
        positions_before = {p["symbol"] for p in portfolio["positions"] if p["status"] == "open"}
        opened = open_positions(smart_api, portfolio, top_buffer, yt_market=yt_market)

        # --- Index-level strategies: Iron Condor + Momentum ---
        # These don't need screener candidates — they use Nifty index directly
        try:
            index_opened = _try_index_strategies(smart_api, portfolio)
            opened += index_opened
        except Exception as e:
            logger.debug("Index strategies error: %s", e)

        logger.info("\n  Opened %d position(s).", opened)

        # Telegram alert for new entries (with Claude reasoning)
        if opened > 0:
            new_positions = [p for p in portfolio["positions"]
                            if p["status"] == "open" and p["symbol"] not in positions_before]
            try:
                from claude_intel import format_entry_telegram
                for pos in new_positions:
                    entry_msg = format_entry_telegram(
                        pos, pos.get("claude_reasoning", ""),
                        regime=regime, vix=vix_ltp,
                    )
                    _telegram_send(entry_msg)
            except Exception:
                _telegram_notify_entry(new_positions)

        save_portfolio(portfolio)
        print_portfolio_status(portfolio)

        # Telegram cron summary (every run)
        try:
            from claude_intel import generate_cron_summary
            summary = generate_cron_summary("open", portfolio, regime=regime, vix=vix_ltp, opened=opened)
            _telegram_send(summary)
        except Exception:
            pass

        # Daily summary after opening
        _telegram_daily_summary(portfolio)

    elif mode == "monitor":
        logger.info("\n=== PAPER TRADE: Monitoring positions ===\n")

        # Smart Monitor: sector correlation check + Nifty session tracking
        try:
            from smart_monitor import (check_sector_correlation, should_send_hourly_digest,
                                       generate_hourly_digest, _load_state, _save_state,
                                       mark_checked)
            sm_state = _load_state()

            # Track Nifty at session start (for correlation checks)
            nifty_ltp_sm = get_ltp(smart_api, "Nifty 50", NIFTY_TOKEN)
            time.sleep(config.API_DELAY)
            if nifty_ltp_sm and not sm_state.get("nifty_at_session_start"):
                sm_state["nifty_at_session_start"] = nifty_ltp_sm
                # Reset at start of new day
                today_str = datetime.now(IST).strftime("%Y-%m-%d")
                if sm_state.get("daily_realized_date") != today_str:
                    sm_state["nifty_at_session_start"] = nifty_ltp_sm

            # Sector correlation: tighten SLs if Nifty moves >1.5% against correlated positions
            corr_alerts = check_sector_correlation(portfolio, nifty_ltp_sm, sm_state)
            if corr_alerts:
                corr_msg = (f"<b>⚠️ Correlation Alert</b>\n"
                           f"Nifty moved significantly — tightened SLs on: {', '.join(corr_alerts)}")
                _telegram_send(corr_msg)

            _save_state(sm_state)
        except Exception as e:
            logger.debug("Smart monitor pre-checks failed: %s", e)
            sm_state = None

        exits, unrealized_pnl = monitor_positions(smart_api, portfolio)
        if exits:
            logger.info("\n  %d position(s) closed.", exits)
        save_portfolio(portfolio)

        # Smart Monitor: mark checked positions + hourly digest
        try:
            if sm_state is not None:
                open_syms = [p["symbol"] for p in portfolio["positions"] if p["status"] == "open"]
                mark_checked(sm_state, open_syms)

                # Hourly digest (10:00-14:00 on the hour)
                if should_send_hourly_digest(sm_state):
                    vix_digest = get_ltp(smart_api, "India VIX", VIX_TOKEN)
                    time.sleep(config.API_DELAY)
                    digest = generate_hourly_digest(portfolio, vix_digest, regime)
                    _telegram_send(digest)
                    sm_state["last_hourly_digest"] = datetime.now(IST).isoformat()

                _save_state(sm_state)
        except Exception as e:
            logger.debug("Smart monitor post-checks failed: %s", e)

        # Print brief status
        open_count = len([p for p in portfolio["positions"] if p["status"] == "open"])
        realized_pnl = portfolio['stats']['total_pnl']
        logger.info(f"Open: {open_count}  |  Unrealized: ₹{unrealized_pnl:+,.0f}  |  Realized: ₹{realized_pnl:+,.0f}  |  Total: ₹{unrealized_pnl + realized_pnl:+,.0f}")

        # Telegram cron summary (every monitor run)
        try:
            from claude_intel import generate_cron_summary
            summary = generate_cron_summary("monitor", portfolio, exited=exits, unrealized_pnl=unrealized_pnl)
            _telegram_send(summary)
        except Exception:
            pass

        # Record expected move data at EOD (after 3:15 PM IST)
        now_ist = datetime.now(IST)
        if now_ist.hour >= 15 and now_ist.minute >= 15:
            try:
                from expected_move import record_daily_move
                nifty_ltp = get_ltp(smart_api, "NIFTY", NIFTY_TOKEN)
                vix_ltp = get_ltp(smart_api, "India VIX", VIX_TOKEN)
                if nifty_ltp and vix_ltp:
                    # Fetch today's Nifty open from candle data
                    today_str = now_ist.strftime("%Y-%m-%d")
                    try:
                        candle_resp = smart_api.getCandleData({
                            "exchange": "NSE",
                            "symboltoken": NIFTY_TOKEN,
                            "interval": "ONE_DAY",
                            "fromdate": f"{today_str} 09:15",
                            "todate": f"{today_str} 15:30",
                        })
                        if candle_resp and candle_resp.get("data"):
                            candle = candle_resp["data"][0]
                            nifty_open = float(candle[1])  # [timestamp, open, high, low, close]
                            record_daily_move(nifty_open, nifty_ltp, vix_ltp, today_str)
                    except Exception as e:
                        logger.debug("Expected move EOD recording failed: %s", e)
            except Exception as e:
                logger.debug("Expected move import/recording failed: %s", e)

    elif mode == "status":
        print_portfolio_status(portfolio, smart_api)

    elif mode == "close-all":
        logger.info("\n=== PAPER TRADE: Force closing all positions ===\n")
        exits = close_all_positions(smart_api, portfolio)
        logger.info("\n  Closed %d position(s).", exits)
        save_portfolio(portfolio)
        print_portfolio_status(portfolio)

    elif mode == "weekly-report":
        logger.info("\n=== PAPER TRADE: Weekly Performance Report ===\n")
        send_weekly_report()

    elif mode == "reset":
        logger.info("\n=== PAPER TRADE: Archive & Reset Portfolio ===\n")
        msg = archive_and_reset()
        logger.info(msg)

    elif mode == "journal":
        from journal import generate_journal_report
        report = generate_journal_report()
        logger.info(report)

    elif mode == "greeks":
        logger.info("\n=== PAPER TRADE: Greeks Dashboard ===\n")
        opt_positions = [p for p in portfolio["positions"]
                         if p["status"] == "open" and p.get("instrument") in ("OPT", "MOMENTUM")]
        if not opt_positions:
            logger.info("No open option positions.")
        else:
            from greeks import compute_position_greeks, portfolio_greeks_summary, generate_greeks_report
            spot_map = {}
            for p in opt_positions:
                sym = p["symbol"]
                if sym not in spot_map:
                    token = p.get("token", "")
                    if token:
                        try:
                            sp = get_ltp(smart_api, sym, token)
                            if sp:
                                spot_map[sym] = sp
                            time.sleep(config.API_DELAY)
                        except Exception:
                            pass
                    if sym not in spot_map and p.get("underlying_price_at_entry"):
                        spot_map[sym] = p["underlying_price_at_entry"]

            pg = compute_position_greeks(opt_positions, spot_map, config.RISK_FREE_RATE)
            summary = portfolio_greeks_summary(pg)
            report = generate_greeks_report(pg, summary)
            # Print plain text (strip HTML tags for console)
            import re
            plain = re.sub(r"<[^>]+>", "", report)
            print(plain)

    elif mode == "attribution":
        logger.info("\n=== PAPER TRADE: Performance Attribution ===\n")
        from attribution import compute_attribution, generate_attribution_report
        from journal import _load_journal
        journal = _load_journal()
        closed = portfolio["closed_trades"]
        attr = compute_attribution(journal, closed)
        report = generate_attribution_report(attr)
        logger.info(report)

    elif mode == "tax-report":
        logger.info("\n=== PAPER TRADE: Tax-Ready P&L Report ===\n")
        from tax_report import generate_tax_report, generate_tax_pnl_statement
        tax_data = generate_tax_report(portfolio["closed_trades"])
        statement = generate_tax_pnl_statement(portfolio["closed_trades"])
        logger.info(statement)

    elif mode == "briefing":
        from premarket_briefing import generate_premarket_briefing
        msg = generate_premarket_briefing(smart_api)
        _telegram_send(msg)
        logger.info("Pre-market briefing sent.")

    elif mode == "expected-move":
        logger.info("\n=== PAPER TRADE: Expected Move Accuracy ===\n")
        from expected_move import generate_move_report
        report = generate_move_report()
        import re
        plain = re.sub(r"<[^>]+>", "", report)
        print(plain)

    else:
        logger.error("Unknown mode: %s", mode)
        logger.info("Usage: python paper_trade.py [open|monitor|status|close-all|weekly-report|greeks|attribution|tax-report]")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Paper trading engine for F&O screener signals")
    parser.add_argument("mode", choices=["open", "monitor", "status", "close-all",
                                          "weekly-report", "reset", "journal", "greeks",
                                          "attribution", "tax-report", "briefing",
                                          "expected-move"],
                        help="open: run screener + open positions, "
                             "monitor: check exits, "
                             "status: print dashboard, "
                             "close-all: force close everything, "
                             "weekly-report: send weekly performance summary, "
                             "reset: archive portfolio + start fresh, "
                             "journal: print trade journal report, "
                             "greeks: show Greeks for option positions, "
                             "attribution: performance attribution analysis, "
                             "tax-report: tax-ready P&L statement, "
                             "briefing: send pre-market briefing via Telegram, "
                             "expected-move: IV expected move accuracy report")
    args = parser.parse_args()

    logger.info("Connecting to AngelOne SmartAPI...")
    smart_api = get_session()

    run_paper_trade(smart_api, args.mode)


if __name__ == "__main__":
    main()
