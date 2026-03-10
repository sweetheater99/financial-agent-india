"""
Loads credentials from .env file.

SmartAPI credentials are read from environment variables.
Anthropic auth is resolved in this order:
  1. ANTHROPIC_API_KEY env var / .env  (explicit API key → uses SDK directly)
  2. Claude Code CLI  (Max subscription → uses `claude -p` under the hood)
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import date
from dotenv import load_dotenv

load_dotenv()

# ── V6 Claude-First Architecture ──────────────────────────────────────────
V6_CLAUDE_FIRST = True  # Set False to revert to V5 rule-based decisions

ANGELONE_API_KEY = os.getenv("ANGELONE_API_KEY")
ANGELONE_CLIENT_ID = os.getenv("ANGELONE_CLIENT_ID")
ANGELONE_PASSWORD = os.getenv("ANGELONE_PASSWORD")
ANGELONE_TOTP_SECRET = os.getenv("ANGELONE_TOTP_SECRET")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Centralized defaults — change here to update all modules at once
CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_MODEL_LIGHT = "claude-haiku-4-5-20251001"  # cheap model for classification/formatting
API_DELAY = 1.5  # seconds between SmartAPI calls to avoid rate-limiting

# ── Zerodha Kite Connect ─────────────────────────────────────────────────
KITE_API_KEY = os.getenv("KITE_API_KEY")
KITE_API_SECRET = os.getenv("KITE_API_SECRET")
KITE_ACCESS_TOKEN_FILE = os.path.join(os.path.dirname(__file__), "data", ".kite_access_token")

# Data source: "angelone" (default) or "kite"
DATA_SOURCE = os.getenv("DATA_SOURCE", "kite")

# Kite rate limit — 3 req/sec for historical, 1 req/sec for quotes
KITE_API_DELAY = 0.35  # seconds between Kite calls

# Whether claude CLI is available (for Max subscription users)
CLAUDE_CLI_AVAILABLE = shutil.which("claude") is not None


def get_anthropic_client():
    """
    Create an Anthropic client using the best available auth method.
    Returns a real Anthropic client if API key is set, otherwise a
    ClaudeCLIClient wrapper that uses the `claude` CLI (Max subscription).
    """
    if ANTHROPIC_API_KEY:
        import anthropic
        return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    if CLAUDE_CLI_AVAILABLE:
        return ClaudeCLIClient()

    print("No Anthropic credentials found.")
    print("Either set ANTHROPIC_API_KEY in .env, or install/login to Claude Code (Max subscription).")
    sys.exit(1)


class ClaudeCLIClient:
    """
    Drop-in wrapper that calls the `claude` CLI instead of the Anthropic SDK.
    Uses your Max subscription auth — no API key needed.
    """

    def __init__(self):
        self.messages = self._Messages()

    class _Messages:
        def create(self, *, model=None, max_tokens=1024,
                   system=None, messages=None, **kwargs):
            # Build the user prompt from messages
            user_text = ""
            for msg in (messages or []):
                if msg["role"] == "user":
                    user_text += msg["content"] + "\n"

            cmd = ["claude", "-p", "--output-format", "json", "--no-session-persistence"]

            # Map model names to claude CLI model aliases
            model_name = model or CLAUDE_MODEL
            model_map = {
                "claude-sonnet-4-20250514": "sonnet",
                "claude-opus-4-20250514": "opus",
                "claude-haiku-3-5-20241022": "haiku",
                "claude-haiku-4-5-20251001": "haiku",
            }
            cmd.extend(["--model", model_map.get(model_name, model_name)])

            if system:
                cmd.extend(["--system-prompt", system])

            cmd.append(user_text.strip())

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
            )

            if result.returncode != 0:
                raise RuntimeError(f"claude CLI failed: {result.stderr.strip()}")

            # Parse the JSON output from claude CLI
            try:
                output = json.loads(result.stdout)
                text = output.get("result", result.stdout)
            except json.JSONDecodeError:
                text = result.stdout.strip()

            return _CLIResponse(text)


class _CLIResponse:
    """Mimics anthropic.types.Message enough for agent.py to work."""

    def __init__(self, text: str):
        self.content = [_TextBlock(text)]


class _TextBlock:
    def __init__(self, text: str):
        self.text = text


def validate() -> bool:
    """Check that all required credentials are present. Prints what's missing."""
    missing = []
    if not ANGELONE_API_KEY:
        missing.append("ANGELONE_API_KEY")
    if not ANGELONE_CLIENT_ID:
        missing.append("ANGELONE_CLIENT_ID")
    if not ANGELONE_PASSWORD:
        missing.append("ANGELONE_PASSWORD")
    if not ANGELONE_TOTP_SECRET:
        missing.append("ANGELONE_TOTP_SECRET")
    if not ANTHROPIC_API_KEY and not CLAUDE_CLI_AVAILABLE:
        missing.append("ANTHROPIC_API_KEY (or Claude Code CLI)")

    if missing:
        print("Missing credentials:")
        for var in missing:
            print(f"  - {var}")
        print("\nFor AngelOne: copy .env.example to .env and fill in your values.")
        print("For Anthropic: set ANTHROPIC_API_KEY in .env, or just use Claude Code (Max).")
        return False
    return True


def is_market_open(_now=None) -> tuple[bool, str]:
    """Check if NSE market is currently open.

    NSE trading hours: Mon-Fri 9:15 AM – 3:30 PM IST (UTC+5:30).
    Returns (is_open, status_message).
    """
    from datetime import datetime, timezone, timedelta

    ist = timezone(timedelta(hours=5, minutes=30))
    now = _now if _now is not None else datetime.now(ist)

    # Weekend check
    if now.weekday() >= 5:  # 5=Saturday, 6=Sunday
        day_name = "Saturday" if now.weekday() == 5 else "Sunday"
        return (False, f"Market closed ({day_name})")

    # Holiday check
    today = now.date() if hasattr(now, 'date') and callable(now.date) else now
    if today in NSE_HOLIDAYS_2026:
        return (False, "Market closed (NSE holiday)")

    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

    if now < market_open:
        return (False, f"Market closed (opens at 9:15 AM IST, current time {now.strftime('%H:%M')} IST)")
    if now >= market_close:
        return (False, f"Market closed (after 3:30 PM IST, current time {now.strftime('%H:%M')} IST)")

    return (True, "Market is open")


# ── V2 Trading System Constants ──────────────────────────────────────────────

NSE_HOLIDAYS_2026 = {
    date(2026, 1, 26),   # Republic Day
    date(2026, 2, 26),   # Maha Shivaratri
    date(2026, 3, 3),    # Holi
    date(2026, 3, 30),   # Id-Ul-Fitr
    date(2026, 4, 2),    # Ram Navami
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 6, 5),    # Id-Ul-Adha
    date(2026, 7, 6),    # Muharram
    date(2026, 8, 15),   # Independence Day
    date(2026, 8, 19),   # Janmashtami
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    date(2026, 10, 20),  # Dussehra
    date(2026, 11, 9),   # Diwali (Laxmi Pujan)
    date(2026, 11, 30),  # Guru Nanak Jayanti
}


def is_trading_day(d: date) -> bool:
    """Returns True if the given date is a valid NSE trading day (not weekend, not holiday)."""
    if d.weekday() >= 5:  # Saturday or Sunday
        return False
    if d in NSE_HOLIDAYS_2026:
        return False
    return True


def trading_days_between(start: date, end: date) -> int:
    """Count trading days between two dates (exclusive of start, inclusive of end)."""
    from datetime import timedelta
    if start >= end:
        return 0
    count = 0
    current = start + timedelta(days=1)
    while current <= end:
        if is_trading_day(current):
            count += 1
        current += timedelta(days=1)
    return count


def add_trading_days(d: date, n: int) -> date:
    """Add n trading days to date d. If n is negative, subtract."""
    from datetime import timedelta
    step = 1 if n > 0 else -1
    remaining = abs(n)
    current = d
    while remaining > 0:
        current += timedelta(days=step)
        if is_trading_day(current):
            remaining -= 1
    return current


def next_trading_day(d: date) -> date:
    """Return the next trading day after d."""
    return add_trading_days(d, 1)


def prev_trading_day(d: date) -> date:
    """Return the previous trading day before d."""
    return add_trading_days(d, -1)


VIX_TIERS = {
    "extreme_low": {"max_vix": 12, "size_multiplier": 0.50, "iron_condor": False},
    "normal":      {"min_vix": 12, "max_vix": 18, "size_multiplier": 1.00, "iron_condor": True},
    "elevated":    {"min_vix": 18, "max_vix": 22, "size_multiplier": 0.75, "iron_condor": False},
    "high":        {"min_vix": 22, "max_vix": 28, "size_multiplier": 0.50, "iron_condor": False},
    "crisis":      {"min_vix": 28, "size_multiplier": 0.00, "iron_condor": False},
}


def get_vix_tier(vix: float) -> dict:
    """Return the VIX tier dict matching the given VIX value."""
    if vix < 12:
        return VIX_TIERS["extreme_low"]
    elif vix < 18:
        return VIX_TIERS["normal"]
    elif vix < 22:
        return VIX_TIERS["elevated"]
    elif vix < 28:
        return VIX_TIERS["high"]
    else:
        return VIX_TIERS["crisis"]


# Capital allocation limits (F&O only — no equity delivery)
ALLOC_EQUITY_MAX = 0.00
ALLOC_SPREADS_MAX = 0.40
ALLOC_IRON_CONDOR_MAX = 0.15
ALLOC_MOMENTUM_MAX = 0.20
ALLOC_CASH_MIN = 0.05
MAX_CONCURRENT_POSITIONS = 8
MAX_SAME_SECTOR = 2
MAX_SIMULTANEOUS_LOSS_PCT = 20  # V9: loosened from 10% — V8's wide 3.5x ATR SL needs more room

# Spread strategy constants
SPREAD_MIN_DTE = 30
SPREAD_MAX_DTE = 45
SPREAD_MAX_RISK_PCT = 0.02  # 2% of capital
SPREAD_TARGET_ATR_MULT = 2.0
SPREAD_SL_ATR_MULT = 1.5
SPREAD_PROFIT_CAP_PCT = 0.80
SPREAD_TIME_EXIT_DAYS = 5
SPREAD_MIN_OI = 500
SPREAD_MIN_VOLUME = 100
SPREAD_MAX_BID_ASK_PCT = 0.05

# Credit spread config
CREDIT_SPREAD_MIN_IV_PERCENTILE = 50
CREDIT_SPREAD_PREFER_IV_PERCENTILE = 70
DEBIT_SPREAD_MAX_IV_PERCENTILE = 70
CREDIT_SPREAD_TARGET_PCT = 0.50
CREDIT_SPREAD_SL_MULTIPLIER = 2.0
CREDIT_SPREAD_TIME_EXIT_DAYS = 5
CREDIT_SPREAD_MAX_DELTA = 0.35
CREDIT_SPREAD_MIN_VIX = 15
CREDIT_SPREAD_MAX_VIX = 25

# Momentum options constants
MOMENTUM_MAX_RISK_PCT = 0.01  # 1% of capital
MOMENTUM_SL_PCT = 0.35
MOMENTUM_TARGET_PCT = 0.90
MOMENTUM_TIME_EXIT_DAYS = 3
MOMENTUM_MIN_RR = 2.0

# Iron condor constants
CONDOR_MAX_RISK_PCT = 0.02
CONDOR_MIN_VIX = 12
CONDOR_MAX_VIX = 18
CONDOR_TARGET_PCT = 0.50
CONDOR_SL_MULTIPLIER = 2.0
CONDOR_DELTA_EXIT = 0.30
CONDOR_TIME_EXIT_DAYS = 5
CONDOR_MIN_CREDIT_PER_LOT = 50

# Drawdown circuit breakers
DRAWDOWN_DAILY_HALT = 0.03             # V6: to be removed when referencing code is updated (risk_manager.py)
DRAWDOWN_WEEKLY_REDUCE = 0.05          # V6: to be removed when referencing code is updated (risk_manager.py)
DRAWDOWN_MONTHLY_HALT = 0.08           # V6: to be removed when referencing code is updated (risk_manager.py)
DRAWDOWN_TOTAL_STOP = 0.15             # V6: to be removed when referencing code is updated (risk_manager.py, paper_trade.py)
CONSECUTIVE_LOSS_PAUSE = 3             # V6: to be removed when referencing code is updated (risk_manager.py)

# CASH regime triggers
CASH_TRIGGER_MONTHLY_DRAWDOWN_PCT = 8.0   # V6: to be removed when referencing code is updated (regime.py)
CASH_TRIGGER_CONSECUTIVE_LOSSES = 5       # V6: to be removed when referencing code is updated (regime.py)
CASH_TRIGGER_VIX = 25                     # V6: to be removed when referencing code is updated (regime.py)

# Slippage
PAPER_TRADE_SLIPPAGE_PCT = 0.005  # 0.5%

# Risk-free rate for Black-Scholes
RISK_FREE_RATE = 0.065

# Greeks monitoring
GREEKS_THETA_WARN_PCT = 2.0  # warn if daily theta > 2% of premium at risk

# ── V3 Global Macro Intelligence ────────────────────────────────────────────

# US Market hard gates
US_CRASH_BLOCK_PCT = 2.0               # V6: to be removed when referencing code is updated (global_intel.py)
US_SEVERE_CRASH_PCT = 3.0              # V6: to be removed when referencing code is updated (global_intel.py)
US_MILD_RED_PCT = 1.0                  # V6: to be removed when referencing code is updated (global_intel.py)
NASDAQ_IT_CRASH_PCT = 3.0              # V6: to be removed when referencing code is updated (global_intel.py)

# GIFT Nifty hard gates
GIFT_GAP_REDUCE_PCT = 1.5              # V6: to be removed when referencing code is updated (global_intel.py)
GIFT_GAP_BLOCK_PCT = 2.5               # V6: to be removed when referencing code is updated (global_intel.py)

# FII/DII hard gates (values in crores)
FII_HEAVY_SELL_CRORES = 5000           # V6: to be removed when referencing code is updated (global_intel.py)
FII_EXTREME_SELL_CRORES = 10000        # V6: to be removed when referencing code is updated (global_intel.py)
DII_SUPPORT_MODERATE_PCT = 0.25        # V6: to be removed when referencing code is updated (global_intel.py)

# PCR gates
PCR_EUPHORIA = 0.5                     # V6: to be removed when referencing code is updated (global_intel.py)
PCR_EXTREME_CALL = 0.7                 # V6: to be removed when referencing code is updated (global_intel.py)
PCR_EXTREME_PUT = 1.3                  # V6: to be removed when referencing code is updated (global_intel.py)

# Supertrend
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0
SUPERTREND_DISAGREE_REDUCTION = 0.25   # V6: to be removed when referencing code is updated (paper_trade.py)

# CPR thresholds
CPR_NARROW_PCT = 0.3                   # V6: to be removed when referencing code is updated (indicators_v3.py)
CPR_WIDE_PCT = 0.8                     # V6: to be removed when referencing code is updated (indicators_v3.py)

# Trailing SL enhancement
TRAILING_SL_ACTIVATION_PCT = 3.0    # don't trail until 3% profit — V8 param sweep confirmed
TRAILING_SL_ATR_MULT = 2.5          # tighter trail once activated — V8 param sweep confirmed

# Weekly theta strategy
WEEKLY_THETA_MIN_VIX = 14.0
WEEKLY_THETA_TARGET_PCT = 0.50
WEEKLY_THETA_SL_MULTIPLIER = 2.0
WEEKLY_THETA_MAX_RISK_PCT = 0.02
WEEKLY_THETA_OTM_POINTS_NIFTY = 250
WEEKLY_THETA_OTM_POINTS_BANKNIFTY = 600

# BankNifty
BANKNIFTY_TOKEN = "99926009"
BANKNIFTY_LOT_SIZE = 30
BANKNIFTY_VIX_MULTIPLIER = 1.3

# Updated allocations (V3 — F&O only)
ALLOC_EQUITY_MAX_V3 = 0.00
ALLOC_SPREADS_MAX_V3 = 0.35
ALLOC_IRON_CONDOR_MAX_V3 = 0.15
ALLOC_MOMENTUM_MAX_V3 = 0.15
ALLOC_WEEKLY_THETA_MAX = 0.15
ALLOC_CASH_MIN_V3 = 0.05

# V9: PR Sundar portfolio discipline — keep 20% cash
CASH_RESERVE_PCT = 0.20             # never deploy more than 80% of capital
MAX_STRATEGY_ALLOC_PCT = 0.12       # max 12% of capital per strategy
WEEKLY_PORTFOLIO_SL_PCT = 0.04      # 4% weekly portfolio SL → stop all new entries

# V9: Monthly iron condor (Reyaansh style)
MONTHLY_CONDOR_ENABLED = True
MONTHLY_CONDOR_ENTRY_DAY_MIN = 1    # enter 1st-5th of month
MONTHLY_CONDOR_ENTRY_DAY_MAX = 5
MONTHLY_CONDOR_DELTA = 0.25         # 25-delta strikes
MONTHLY_CONDOR_HEDGE_GAP_NIFTY = 500   # hedge wing gap in points (Nifty)
MONTHLY_CONDOR_HEDGE_GAP_BANKNIFTY = 700  # hedge wing gap in points (BankNifty)
MONTHLY_CONDOR_TARGET_PCT = 0.50    # take profit at 50% premium captured
MONTHLY_CONDOR_MONTHLY_SL_PCT = 0.04  # 4% monthly SL → stop the condor
MONTHLY_CONDOR_MAX_RISK_PCT = 0.03  # max 3% of capital at risk

# V9: Dynamic hedge tightening (Reyaansh)
HEDGE_TIGHTEN_ENABLED = True
HEDGE_TIGHTEN_THRESHOLD_PCT = 0.40  # tighten when hedge drops to 40% of entry premium
HEDGE_TIGHTEN_CLOSER_POINTS_NIFTY = 200  # re-buy hedge 200pt closer (Nifty)
HEDGE_TIGHTEN_CLOSER_POINTS_BANKNIFTY = 200  # re-buy hedge 200pt closer (BankNifty)

# ---------------------------------------------------------------------------
# Overnight Hedge Protection (V10)
# ---------------------------------------------------------------------------
OVERNIGHT_HEDGE_ENABLED = True
OVERNIGHT_HEDGE_TIME_START = "15:15"
OVERNIGHT_HEDGE_TIME_END = "15:25"
OVERNIGHT_MIN_GAIN_FOR_NAKED_CARRY = 0.50   # 50% gain required for naked carry
OVERNIGHT_MIN_GAIN_FOR_HEDGE = 0.30          # below 30% gain -> close, don't hedge
OVERNIGHT_VIX_NAKED_BLOCK = 20               # VIX above this -> no naked carry allowed
OVERNIGHT_EXPIRY_CLOSE_DAYS = 2              # close if <= 2 trading days to expiry
OVERNIGHT_HEDGE_MAX_COST_PCT = 0.01          # max 1% of position value for hedge premium
OVERNIGHT_HEDGE_OTM_POINTS_NIFTY = 250       # OTM distance for protective leg
OVERNIGHT_HEDGE_OTM_POINTS_BANKNIFTY = 500
OVERNIGHT_STOP_TIGHTEN_PCT = 0.20            # tighten SL by 20% on naked carry approval
OVERNIGHT_NAKED_INSTRUMENTS = ("MOMENTUM", "OPT", "FUT")  # instruments considered naked

# V9: PCR entry filter (Nitin Murarka)
PCR_FILTER_ENABLED = True
PCR_BULLISH_CONTRARIAN = 1.3        # PCR > 1.3 → contrarian bullish (too many puts)
PCR_BEARISH_CONTRARIAN = 0.7        # PCR < 0.7 → contrarian bearish (too many calls)

# V9: Max pain gravity near expiry
MAX_PAIN_FILTER_ENABLED = True
MAX_PAIN_PROXIMITY_PCT = 2.0        # use max pain if price is >2% away with ≤2 DTE

# X/Twitter intel
X_SEARCH_QUERIES = [
    "Nifty today",
    "GIFT Nifty",
    "India VIX",
    "FII selling OR FII buying",
    "RBI rate",
    "war India OR sanctions India",
]
X_CACHE_HOURS = 1
X_MIN_LIKES = 50

# Volatility-adjusted sizing
VOL_SIZING_ENABLED = True
VOL_SIZING_BASELINE_ATR_PCT = 2.0  # ATR% considered "normal" volatility
VOL_SIZING_MIN_MULT = 0.4          # minimum size multiplier for very volatile stocks
VOL_SIZING_MAX_MULT = 1.5          # maximum size multiplier for very steady stocks

# Time-based SL tightening
TIME_SL_ENABLED = True
TIME_SL_HALF_LIFE_MULT = 0.75      # at 50% of max_hold, tighten to 0.75x ATR
TIME_SL_THREE_QUARTER_MULT = 0.5   # at 75% of max_hold, tighten to breakeven or 0.5x ATR

# Partial profit on options
OPT_PARTIAL_PROFIT_PCT = 30.0      # book half when option is up 30%
OPT_PARTIAL_RATIO = 0.5            # book 50% of lots

# Weekly performance report
WEEKLY_REPORT_DAY = 6              # Sunday (0=Mon, 6=Sun)

# ── Smart Monitoring (V5) ─────────────────────────────────────────────────
DAILY_LOSS_CIRCUIT_BREAKER_PCT = 3.0   # % of capital — block ALL new entries for rest of day
SECTOR_CORRELATION_NIFTY_PCT = 1.5     # Nifty move % to trigger correlation protection
SECTOR_CORRELATION_TIGHTEN_ATR = 0.5   # ATR multiplier to tighten SLs on correlated positions
HOT_CHECK_INTERVAL_MIN = 5            # minutes between checks for HOT positions
WARM_CHECK_INTERVAL_MIN = 10          # minutes between checks for WARM positions
COLD_CHECK_INTERVAL_MIN = 30          # minutes between checks when no positions

# Sector rotation
SECTOR_ROTATION_BOOST = 0.15       # +15% allocation for strengthening sectors  # V6: to be removed when referencing code is updated (paper_trade.py)
SECTOR_ROTATION_CUT = 0.15         # -15% allocation for weakening sectors  # V6: to be removed when referencing code is updated (paper_trade.py)

# Unusual OI
OI_UNUSUAL_CAUTIONARY_CUT = 0.10   # -10% allocation on cautionary OI signal  # V6: to be removed when referencing code is updated (paper_trade.py)

# Global intel caching
GLOBAL_INTEL_CACHE_HOURS = 6
FII_DII_CACHE_HOURS = 12

# YouTube intel config
YT_CHANNEL_ALLOWLIST = [
    "PR Sundar", "CA Rachana Ranade", "Power of Stocks",
    "Pranjal Kamra", "Nitin Bhatia",
]
YT_CACHE_HOURS = 12


# ── Risk Guardrails ─────────────────────────────────────────────────────────
MONTHLY_DRAWDOWN_BLOCK_PCT = 0.08      # 8% monthly DD → block all trades
WEEKLY_DRAWDOWN_REDUCE_PCT = 0.04      # 4% weekly DD → reduce 50%
DAILY_LOSS_LIMIT_PCT = 0.02            # 2% daily loss → block
MAX_CONSECUTIVE_LOSSES_PAUSE = 3       # pause 1 day after 3 losses
MAX_CONSECUTIVE_LOSSES_EXTENDED = 5    # pause 2 days after 5 losses
MAX_SECTOR_POSITIONS = 2               # max positions per sector
MAX_DIRECTION_POSITIONS = 3            # max same-direction positions
MAX_PORTFOLIO_HEAT_PCT = 0.06          # 6% of capital at risk max
EXPIRY_DAY_ALLOCATION_MULT = 0.5       # 50% allocation on expiry days
BUDGET_BLOCK_DATES = ["2026-02-01", "2027-02-01"]
RBI_POLICY_DATES = [
    "2026-02-07", "2026-04-09", "2026-06-06",
    "2026-08-08", "2026-10-08", "2026-12-05",
]


# ── Live Execution ─────────────────────────────────────────────────────────
LIVE_MODE = False                       # MUST be explicitly enabled — places real orders
LIVE_MAX_ORDER_VALUE = 15000            # safety cap per order (INR)
LIVE_CONFIRM_TELEGRAM = True            # send Telegram alert before every live order
LIVE_MAX_PRICE_DEVIATION_PCT = 2.0      # reject if price deviates >2% from LTP
LIVE_CONSECUTIVE_FAIL_KILL = 3          # disable LIVE_MODE after 3 consecutive failures


if __name__ == "__main__":
    if validate():
        auth_method = "API key" if ANTHROPIC_API_KEY else "Claude Code CLI (Max subscription)"
        print(f"All credentials loaded. Anthropic auth: {auth_method}")
    else:
        sys.exit(1)
