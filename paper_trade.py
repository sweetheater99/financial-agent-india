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
import math
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config
from connect import get_session
from agent_with_options import fetch_option_chain, get_nearest_expiry
from screener import run_screener, fetch_all_signals, score_signals

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
        print(f"  WARNING: searchScrip failed for {symbol}: {e}")
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
        print(f"  WARNING: LTP fetch failed for {symbol}: {e}")
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
        print(f"  WARNING: NFO LTP fetch failed for {symbol}: {e}")
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
        print(f"  WARNING: searchScrip NFO failed for {query}: {e}")
    return None


def _open_put_position(smart_api, symbol: str, eq_token: str, spot_price: float,
                       allocation: float, alloc: dict) -> dict | None:
    """
    Open a put option position for a bearish candidate.

    Returns position dict or None if skipped.
    """
    expiry = get_nearest_expiry()
    if not expiry:
        print(f"  SKIP {symbol}: could not determine option expiry")
        return None

    dte = days_to_expiry(expiry)
    if dte < MIN_DTE_TO_OPEN:
        print(f"  SKIP {symbol}: expiry {expiry} only {dte}d away (min {MIN_DTE_TO_OPEN}d)")
        return None

    # Fetch option chain
    option_chain = fetch_option_chain(smart_api, symbol, eq_token)
    time.sleep(config.API_DELAY)
    if not option_chain:
        print(f"  SKIP {symbol}: option chain fetch failed")
        return None

    # Select strike
    strike_info = select_put_strike(option_chain, spot_price)
    if not strike_info:
        print(f"  SKIP {symbol}: no suitable put strike found")
        return None

    strike = strike_info["strike"]
    premium = strike_info["pe_ltp"]

    # Resolve NFO contract for token + lot size
    contract = resolve_option_contract(smart_api, symbol, strike, expiry)
    time.sleep(config.API_DELAY)
    if not contract:
        print(f"  SKIP {symbol}: could not resolve PE contract")
        return None

    lot_size = contract["lot_size"]
    # Size: how many lots can we afford?
    cost_per_lot = premium * lot_size
    if cost_per_lot <= 0:
        print(f"  SKIP {symbol}: zero premium cost")
        return None

    num_lots = max(1, math.floor(allocation / cost_per_lot))
    actual_allocated = round(num_lots * cost_per_lot, 2)

    # Verify live premium from NFO
    live_premium = get_ltp_nfo(smart_api, contract["trading_symbol"], contract["token"])
    time.sleep(config.API_DELAY)
    if live_premium and live_premium > 0:
        premium = live_premium
        actual_allocated = round(num_lots * premium * lot_size, 2)

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
        "entry_price": premium,
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
        "score": alloc["score"],
        "categories": alloc.get("categories", []),
        "entry_date": today,
        "max_hold_date": max_hold_date,
        "status": "open",
    }

    print(f"  OPENED {symbol} PE{int(strike)} x{num_lots}lot ({num_lots * lot_size}qty) "
          f"@ ₹{premium:,.2f} (₹{actual_allocated:,.0f}, score={alloc['score']:.1f}, DTE:{dte})")

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
# Open Positions
# ---------------------------------------------------------------------------

def open_positions(smart_api, portfolio: dict, candidates: list[dict]) -> int:
    """
    Open paper positions for top candidates. Returns number of positions opened.

    Skips candidates already in open positions or if capital is insufficient.
    """
    if portfolio["available_capital"] < MIN_CAPITAL:
        print(f"  Insufficient capital (₹{portfolio['available_capital']:,.0f} < ₹{MIN_CAPITAL:,.0f}). Skipping.")
        return 0

    # Filter out stocks already held
    open_symbols = {p["symbol"] for p in portfolio["positions"] if p["status"] == "open"}
    eligible = [c for c in candidates if c["symbol"] not in open_symbols]

    if not eligible:
        print("  No new eligible candidates (all already in portfolio).")
        return 0

    top = eligible[:TOP_N]
    allocations = compute_allocations(top, portfolio["available_capital"])
    today = _today_ist()
    opened = 0

    for alloc in allocations:
        symbol = alloc["symbol"]
        direction = alloc["direction"]

        # Resolve equity token + LTP (needed for both paths)
        resolved = resolve_token(smart_api, symbol)
        if not resolved:
            print(f"  SKIP {symbol}: token resolution failed")
            continue
        trading_symbol, token = resolved
        time.sleep(config.API_DELAY)

        ltp = get_ltp(smart_api, trading_symbol, token)
        if ltp is None or ltp <= 0:
            print(f"  SKIP {symbol}: could not get LTP")
            continue
        time.sleep(config.API_DELAY)

        allocation = alloc["allocation"]

        if direction == "bearish":
            # --- Put option path ---
            position = _open_put_position(smart_api, symbol, token, ltp, allocation, alloc)
            if position is None:
                continue
        else:
            # --- Equity buy path ---
            quantity = math.floor(allocation / ltp)
            if quantity < 1:
                print(f"  SKIP {symbol}: allocation ₹{allocation:,.0f} < 1 share at ₹{ltp:,.2f}")
                continue

            actual_allocated = round(quantity * ltp, 2)
            target_price = round(ltp * (1 + TARGET_PCT / 100), 2)
            stoploss_price = round(ltp * (1 + STOPLOSS_PCT / 100), 2)
            max_hold_date = _add_trading_days(today, MAX_HOLD_DAYS)

            position = {
                "symbol": symbol,
                "token": token,
                "direction": direction,
                "instrument": "EQ",
                "entry_price": ltp,
                "quantity": quantity,
                "allocated": actual_allocated,
                "score": alloc["score"],
                "categories": alloc.get("categories", []),
                "entry_date": today,
                "target_price": target_price,
                "stoploss_price": stoploss_price,
                "max_hold_date": max_hold_date,
                "status": "open",
            }

            print(f"  OPENED {symbol} BUY x{quantity} @ ₹{ltp:,.2f} "
                  f"(₹{actual_allocated:,.0f}, score={alloc['score']:.1f})")

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


def close_position(portfolio: dict, pos: dict, exit_price: float, reason: str) -> dict:
    """
    Close a position: move to closed_trades, update stats, free capital.
    Returns the closed trade record.
    """
    direction = pos["direction"]
    entry_price = pos["entry_price"]
    quantity = pos["quantity"]
    abs_qty = abs(quantity)

    # Option positions: we bought the put, so P&L = premium change (bullish calc)
    is_option = pos.get("instrument") == "OPT"
    pnl_direction = "bullish" if is_option else direction

    pnl_pct = calc_pnl_pct(entry_price, exit_price, pnl_direction)
    if pnl_direction == "bullish":
        pnl = round((exit_price - entry_price) * abs_qty, 2)
    else:
        pnl = round((entry_price - exit_price) * abs_qty, 2)

    closed = {
        "symbol": pos["symbol"],
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "quantity": quantity,
        "pnl": pnl,
        "pnl_pct": round(pnl_pct, 2),
        "entry_date": pos["entry_date"],
        "exit_date": _today_ist(),
        "exit_reason": reason,
    }

    portfolio["closed_trades"].append(closed)

    # macOS notification
    emoji = "+" if pnl >= 0 else ""
    reason_label = {"target": "Target hit", "stoploss": "Stop-loss hit",
                    "expiry": "Max hold expired", "manual": "Manual close"}.get(reason, reason)
    _notify("Paper Trade Exit",
            f"{pos['symbol']} {reason_label}: {pnl_pct:+.1f}% (₹{emoji}{pnl:,.0f})")

    # Remove from open positions
    pos["status"] = "closed"

    # Free allocated capital
    portfolio["available_capital"] = round(portfolio["available_capital"] + pos["allocated"], 2)

    # Update cumulative stats
    stats = portfolio["stats"]
    stats["total_trades"] += 1
    stats["total_pnl"] = round(stats["total_pnl"] + pnl, 2)
    stats["total_pnl_pct"] = round(stats["total_pnl"] / portfolio["capital"] * 100, 2)

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
    open_positions = [p for p in portfolio["positions"] if p["status"] == "open"]
    if not open_positions:
        print("  No open positions to monitor.")
        return 0

    today = _today_ist()
    exits = 0

    for pos in open_positions:
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
            print(f"  {symbol}: could not fetch LTP, skipping")
            continue

        # P&L: options use "bullish" direction (bought the put)
        pnl_direction = "bullish" if is_option else pos["direction"]
        pnl_pct = calc_pnl_pct(pos["entry_price"], ltp, pnl_direction)

        # Thresholds depend on instrument
        target_pct = PUT_TARGET_PCT if is_option else TARGET_PCT
        stoploss_pct = PUT_STOPLOSS_PCT if is_option else STOPLOSS_PCT

        if pnl_pct >= target_pct:
            reason = "target"
        elif pnl_pct <= stoploss_pct:
            reason = "stoploss"
        elif dte_expired:
            reason = "expiry"
        elif expired:
            reason = "expiry"
        else:
            day_num = _trading_days_between(pos["entry_date"], today)
            max_days = _trading_days_between(pos["entry_date"], pos["max_hold_date"])
            if is_option:
                label = (f"  {symbol} PE{int(pos['strike'])}: "
                         f"Prem ₹{ltp:,.2f}  P&L: {pnl_pct:+.1f}%  "
                         f"DTE:{dte}  Day {day_num}/{max_days}  [HOLD]")
            else:
                label = (f"  {symbol}: LTP ₹{ltp:,.2f}  P&L: {pnl_pct:+.1f}%  "
                         f"Day {day_num}/{max_days}  [HOLD]")
            print(label)
            continue

        closed = close_position(portfolio, pos, ltp, reason)
        tag = "TARGET" if reason == "target" else "STOPLOSS" if reason == "stoploss" else "EXPIRY"
        if is_option:
            print(f"  {symbol} PE{int(pos['strike'])}: EXIT ({tag}) @ ₹{ltp:,.2f}  "
                  f"P&L: {closed['pnl_pct']:+.1f}% (₹{closed['pnl']:+,.0f})")
        else:
            print(f"  {symbol}: EXIT ({tag}) @ ₹{ltp:,.2f}  "
                  f"P&L: {closed['pnl_pct']:+.1f}% (₹{closed['pnl']:+,.0f})")
        exits += 1

    # Clean up closed positions from the open list
    portfolio["positions"] = [p for p in portfolio["positions"] if p["status"] == "open"]

    return exits


def close_all_positions(smart_api, portfolio: dict) -> int:
    """Force-close all open positions at current LTP."""
    open_pos = [p for p in portfolio["positions"] if p["status"] == "open"]
    if not open_pos:
        print("  No open positions to close.")
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
            print(f"  {pos['symbol']}: could not fetch LTP, using entry price")
            ltp = pos["entry_price"]

        closed = close_position(portfolio, pos, ltp, "manual")
        print(f"  {pos['symbol']}: CLOSED @ ₹{ltp:,.2f}  P&L: {closed['pnl_pct']:+.1f}% (₹{closed['pnl']:+,.0f})")
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
        print("\n=== PAPER TRADE: Opening new positions ===\n")

        is_open, status_msg = config.is_market_open()
        if not is_open:
            print(f"  Note: {status_msg}")
            print(f"  Opening anyway (LTP will be previous close).\n")

        print("Running screener (raw mode)...")
        result = run_screener(smart_api, top_n=TOP_N, raw_only=True)

        if "error" in result:
            print(f"  Screener error: {result['error']}")
            return

        candidates = result.get("candidates", [])
        if not candidates:
            print("  No candidates from screener.")
            return

        print(f"\nTop {min(TOP_N, len(candidates))} candidates for paper trading:")
        for i, c in enumerate(candidates[:TOP_N], 1):
            dir_tag = "BULL" if c["direction"] == "bullish" else "BEAR"
            print(f"  {i}. {c['symbol']:<15} [{dir_tag}]  Score: {c['score']:.1f}")

        print(f"\nOpening positions (available capital: ₹{portfolio['available_capital']:,.0f})...")
        opened = open_positions(smart_api, portfolio, candidates)
        print(f"\n  Opened {opened} position(s).")

        save_portfolio(portfolio)
        print_portfolio_status(portfolio)

    elif mode == "monitor":
        print("\n=== PAPER TRADE: Monitoring positions ===\n")

        exits = monitor_positions(smart_api, portfolio)
        if exits:
            print(f"\n  {exits} position(s) closed.")
        save_portfolio(portfolio)

        # Print brief status
        open_count = len([p for p in portfolio["positions"] if p["status"] == "open"])
        print(f"  Open: {open_count}  |  "
              f"Total P&L: ₹{portfolio['stats']['total_pnl']:+,.0f}")

    elif mode == "status":
        print_portfolio_status(portfolio, smart_api)

    elif mode == "close-all":
        print("\n=== PAPER TRADE: Force closing all positions ===\n")
        exits = close_all_positions(smart_api, portfolio)
        print(f"\n  Closed {exits} position(s).")
        save_portfolio(portfolio)
        print_portfolio_status(portfolio)

    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python paper_trade.py [open|monitor|status|close-all]")
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

    print("Connecting to AngelOne SmartAPI...")
    smart_api = get_session()

    run_paper_trade(smart_api, args.mode)


if __name__ == "__main__":
    main()
