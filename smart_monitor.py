"""Smart Tiered Monitoring — replaces flat 30-min cron.

Classifies positions by urgency, checks only what's due, manages circuit
breaker, hourly digests, morning briefing timing, and sector correlation.

Called every 5 min by cron. Decides internally what to do each tick.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import config

logger = logging.getLogger("paper_trade")

STATE_FILE = Path("data/paper_trades/monitor_state.json")
LESSONS_FILE = Path("data/trade_journal/lessons.json")

IST = None  # imported lazily from paper_trade


def _get_ist():
    global IST
    if IST is None:
        from paper_trade import IST as _IST
        IST = _IST
    return IST


def _now_ist():
    return datetime.now(_get_ist())


def _load_state() -> dict:
    """Load monitor state, falling back to .bak if main file is corrupt."""
    bak = STATE_FILE.with_suffix(".json.bak")
    for path in (STATE_FILE, bak):
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt state file: %s — trying next", path)
    return {
        "last_check": {},
        "last_regime_check": "",
        "last_hourly_digest": "",
        "circuit_breaker_active": False,
        "circuit_breaker_date": "",
        "daily_realized_loss": 0.0,
        "daily_realized_date": "",
        "nifty_at_session_start": None,
    }


def _save_state(state: dict):
    """Persist monitor state with backup + atomic write."""
    import shutil

    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Backup existing file before overwriting
    if STATE_FILE.exists():
        shutil.copy2(STATE_FILE, STATE_FILE.with_suffix(".json.bak"))

    # Atomic write: write to .tmp, then rename
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.replace(STATE_FILE)


def cleanup_closed_positions(portfolio: dict, state: dict):
    """Remove closed/stale symbols from state tracking dicts.

    Keeps only symbols that have an open position in portfolio.
    """
    open_symbols = {
        p["symbol"]
        for p in portfolio.get("positions", [])
        if p.get("status") == "open"
    }

    for key in ("last_check", "position_assessments"):
        if key in state:
            state[key] = {
                sym: val for sym, val in state[key].items()
                if sym in open_symbols
            }


# ---------------------------------------------------------------------------
# Position Urgency Classification
# ---------------------------------------------------------------------------

def classify_urgency(pos: dict, vix: float | None = None) -> str:
    """Classify position urgency: HOT, WARM, or COLD.

    HOT (5 min): near target/SL, low DTE options, big P&L move, VIX spike day
    WARM (10 min): normal open position
    """
    instrument = pos.get("instrument", "EQ")

    # Options with low DTE are always hot
    if instrument in ("OPT", "MOMENTUM"):
        from paper_trade import days_to_expiry
        dte = days_to_expiry(pos.get("expiry", ""))
        if dte is not None and dte < 5:
            return "HOT"

    # Check distance to target/SL
    entry = pos.get("entry_price", 0)
    target = pos.get("target_price", 0)
    sl = pos.get("stoploss_price", 0)
    atr = pos.get("atr_at_entry")
    ltp = pos.get("_last_ltp")  # set during previous check

    if ltp and atr and atr > 0:
        dist_to_target = abs(target - ltp)
        dist_to_sl = abs(ltp - sl)
        # Within 2x ATR of either exit
        if dist_to_target < 2 * atr or dist_to_sl < 2 * atr:
            return "HOT"

    # High VIX day
    if vix and vix > 22:
        return "HOT"

    # Big unrealized move
    if ltp and entry > 0:
        pnl_pct = abs(ltp - entry) / entry * 100
        if pnl_pct > 5:
            return "HOT"

    return "WARM"


def get_positions_due(portfolio: dict, state: dict, vix: float | None = None) -> list[dict]:
    """Return positions that are due for a check based on urgency + last check time."""
    now = _now_ist()
    due = []

    for pos in portfolio.get("positions", []):
        if pos.get("status") != "open":
            continue

        symbol = pos.get("symbol", "?")
        urgency = classify_urgency(pos, vix)

        # Check interval based on urgency
        interval_min = 5 if urgency == "HOT" else 10

        last_str = state.get("last_check", {}).get(symbol, "")
        if last_str:
            try:
                last_dt = datetime.fromisoformat(last_str)
                if (now - last_dt) < timedelta(minutes=interval_min):
                    continue  # not due yet
            except (ValueError, TypeError):
                pass

        pos["_urgency"] = urgency
        due.append(pos)

    return due


def mark_checked(state: dict, symbols: list[str]):
    """Update last_check timestamps."""
    now_str = _now_ist().isoformat()
    if "last_check" not in state:
        state["last_check"] = {}
    for sym in symbols:
        state["last_check"][sym] = now_str


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

def check_circuit_breaker(portfolio: dict, state: dict) -> bool:
    """Check if daily loss circuit breaker should be active.

    Returns True if trading should be blocked.
    """
    today = _now_ist().strftime("%Y-%m-%d")

    # Reset at start of new day
    if state.get("daily_realized_date") != today:
        state["daily_realized_loss"] = 0.0
        state["daily_realized_date"] = today
        state["circuit_breaker_active"] = False
        state["circuit_breaker_date"] = ""

    # Compute today's realized losses from closed trades
    daily_loss = 0.0
    for trade in portfolio.get("closed_trades", []):
        if trade.get("exit_date") == today and trade.get("pnl", 0) < 0:
            daily_loss += abs(trade["pnl"])

    state["daily_realized_loss"] = round(daily_loss, 2)

    capital = portfolio.get("capital", 100_000)
    threshold = capital * (getattr(config, "DAILY_LOSS_CIRCUIT_BREAKER_PCT", 3.0) / 100)

    if daily_loss >= threshold and not state.get("circuit_breaker_active"):
        state["circuit_breaker_active"] = True
        state["circuit_breaker_date"] = today
        logger.warning("CIRCUIT BREAKER: daily loss ₹%.0f >= threshold ₹%.0f (%.1f%%)",
                       daily_loss, threshold, daily_loss / capital * 100)
        return True

    return state.get("circuit_breaker_active", False)


# ---------------------------------------------------------------------------
# Sector Correlation Protection
# ---------------------------------------------------------------------------

def check_sector_correlation(portfolio: dict, nifty_ltp: float | None, state: dict):
    """Check if Nifty moved significantly against open positions.

    If Nifty moves > 1.5% against position direction with 3+ correlated positions,
    tighten SLs by 0.5x ATR.
    """
    if not nifty_ltp:
        return []

    nifty_start = state.get("nifty_at_session_start")
    if not nifty_start or nifty_start <= 0:
        return []

    nifty_change_pct = ((nifty_ltp - nifty_start) / nifty_start) * 100
    alerts = []

    # Check bullish positions when Nifty drops
    if nifty_change_pct < -1.5:
        bullish = [p for p in portfolio.get("positions", [])
                   if p.get("status") == "open" and p.get("direction") == "bullish"
                   and p.get("instrument") in ("EQ", "FUT")]
        if len(bullish) >= 3:
            for pos in bullish:
                atr = pos.get("atr_at_entry")
                if atr and atr > 0:
                    old_sl = pos.get("stoploss_price", 0)
                    new_sl = old_sl + 0.5 * atr  # tighten by 0.5x ATR
                    if new_sl > old_sl:
                        pos["stoploss_price"] = round(new_sl, 2)
                        pos["_sl_tightened_correlation"] = True
                        alerts.append(pos["symbol"])
            if alerts:
                logger.warning("CORRELATION ALERT: Nifty %.1f%% — tightened SL on %d bullish: %s",
                               nifty_change_pct, len(alerts), ", ".join(alerts))

    # Check bearish positions when Nifty rallies
    elif nifty_change_pct > 1.5:
        bearish = [p for p in portfolio.get("positions", [])
                   if p.get("status") == "open" and p.get("direction") == "bearish"
                   and p.get("instrument") in ("EQ", "FUT")]
        if len(bearish) >= 3:
            for pos in bearish:
                atr = pos.get("atr_at_entry")
                if atr and atr > 0:
                    old_sl = pos.get("stoploss_price", float("inf"))
                    new_sl = old_sl - 0.5 * atr
                    if new_sl < old_sl:
                        pos["stoploss_price"] = round(new_sl, 2)
                        pos["_sl_tightened_correlation"] = True
                        alerts.append(pos["symbol"])
            if alerts:
                logger.warning("CORRELATION ALERT: Nifty +%.1f%% — tightened SL on %d bearish: %s",
                               nifty_change_pct, len(alerts), ", ".join(alerts))

    return alerts


# ---------------------------------------------------------------------------
# Hourly Digest
# ---------------------------------------------------------------------------

def should_send_hourly_digest(state: dict) -> bool:
    """Check if it's time for an hourly digest (on the hour, 10-14)."""
    now = _now_ist()
    if now.hour < 10 or now.hour > 14:
        return False
    # Only at the top of the hour (within first 10 min)
    if now.minute > 10:
        return False
    last = state.get("last_hourly_digest", "")
    if last:
        try:
            last_dt = datetime.fromisoformat(last)
            if (now - last_dt) < timedelta(minutes=50):
                return False
        except (ValueError, TypeError):
            pass
    return True


def generate_hourly_digest(portfolio: dict, vix: float | None, regime: str | None) -> str:
    """Generate hourly position digest for Telegram."""
    now = _now_ist()
    open_pos = [p for p in portfolio.get("positions", []) if p.get("status") == "open"]
    total_pnl = portfolio.get("stats", {}).get("total_pnl", 0)

    lines = [f"<b>📊 {now.strftime('%H:%M')} Hourly Update</b>"]

    if regime:
        vix_str = f" | VIX {vix:.1f}" if vix else ""
        lines.append(f"Regime: {regime}{vix_str}")

    if not open_pos:
        lines.append("No open positions")
    else:
        unrealized = 0
        for p in open_pos:
            ltp = p.get("_last_ltp", p["entry_price"])
            entry = p["entry_price"]
            qty = abs(p.get("quantity", 1))

            if p.get("instrument") == "OPT" or p.get("direction") == "bullish":
                pos_pnl = (ltp - entry) * qty
            else:
                pos_pnl = (entry - ltp) * qty
            pnl_pct = ((ltp - entry) / entry * 100) if entry > 0 else 0
            unrealized += pos_pnl

            urgency = p.get("_urgency", "")
            hot_flag = " 🔥" if urgency == "HOT" else ""
            arrow = "🟢" if p["direction"] == "bullish" else "🔴"

            # Distance to exits
            target = p.get("target_price", 0)
            sl = p.get("stoploss_price", 0)
            dist_tgt = f"{((target - ltp) / ltp * 100):+.1f}%" if target and ltp else ""
            dist_sl = f"{((sl - ltp) / ltp * 100):+.1f}%" if sl and ltp else ""

            lines.append(
                f"  {arrow} <b>{p['symbol']}</b> {p.get('instrument', 'EQ')}"
                f" {pnl_pct:+.1f}% (₹{pos_pnl:+,.0f}){hot_flag}"
                f"\n    Tgt: {dist_tgt} | SL: {dist_sl}"
            )

        lines.append(f"\nUnrealized: ₹{unrealized:+,.0f} | Realized: ₹{total_pnl:+,.0f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# EOD Wrap
# ---------------------------------------------------------------------------

def generate_eod_wrap(portfolio: dict) -> str:
    """Generate end-of-day summary for Telegram."""
    today = _now_ist().strftime("%Y-%m-%d")
    total_pnl = portfolio.get("stats", {}).get("total_pnl", 0)

    # Today's trades
    today_trades = [t for t in portfolio.get("closed_trades", []) if t.get("exit_date") == today]
    today_pnl = sum(t.get("pnl", 0) for t in today_trades)

    open_pos = [p for p in portfolio.get("positions", []) if p.get("status") == "open"]

    lines = [f"<b>📋 EOD Wrap — {today}</b>"]

    if today_trades:
        lines.append(f"\n<b>Today's Trades ({len(today_trades)}):</b>")
        for t in today_trades:
            emoji = "✅" if t.get("pnl", 0) >= 0 else "❌"
            lines.append(f"  {emoji} {t['symbol']} {t.get('exit_reason', '')}: {t.get('pnl_pct', 0):+.1f}% (₹{t.get('pnl', 0):+,.0f})")
        lines.append(f"  Day P&L: ₹{today_pnl:+,.0f}")
    else:
        lines.append("No trades closed today")

    lines.append(f"\n<b>Portfolio:</b> ₹{total_pnl:+,.0f} cumulative")

    if open_pos:
        lines.append(f"\n<b>Carrying Overnight ({len(open_pos)}):</b>")
        for p in open_pos:
            lines.append(f"  {p['symbol']} {p.get('instrument', 'EQ')} {p['direction']}")
    else:
        lines.append("\nNo positions overnight — flat book")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Post-Trade Learning
# ---------------------------------------------------------------------------

def record_trade_lesson(pos: dict, closed: dict, portfolio: dict):
    """After a trade closes, ask Claude to review it and extract a lesson."""
    try:
        from claude_intel import _call_claude, _parse_json
    except ImportError:
        return

    symbol = closed.get("symbol", "?")
    entry_price = closed.get("entry_price", 0)
    exit_price = closed.get("exit_price", 0)
    pnl_pct = closed.get("pnl_pct", 0)
    exit_reason = closed.get("exit_reason", "?")
    instrument = closed.get("instrument", "EQ")
    direction = closed.get("direction", "?")
    hold_days = 0
    try:
        from datetime import datetime
        entry_dt = datetime.strptime(closed["entry_date"], "%Y-%m-%d")
        exit_dt = datetime.strptime(closed["exit_date"], "%Y-%m-%d")
        hold_days = (exit_dt - entry_dt).days
    except Exception:
        pass

    categories = pos.get("categories", [])
    regime_at_entry = pos.get("market_regime", "?")
    score = pos.get("score", 0)
    claude_reasoning = pos.get("claude_reasoning", "")

    prompt = f"""POST-TRADE REVIEW: {symbol} ({instrument}, {direction})

ENTRY: ₹{entry_price:,.2f} | Score: {score} | Categories: {categories}
  Regime at entry: {regime_at_entry}
  Claude entry reasoning: {claude_reasoning or 'N/A'}

EXIT: ₹{exit_price:,.2f} | {exit_reason} | P&L: {pnl_pct:+.1f}% | Held: {hold_days}d

Was this a good trade? What's the ONE key lesson?
Respond JSON only:
{{"grade": "A"/"B"/"C"/"D", "lesson": "one sentence", "category": "entry_timing"/"exit_timing"/"sizing"/"direction"/"instrument_choice"}}"""

    result = _call_claude(prompt, max_tokens=256)
    if not result:
        return

    parsed = _parse_json(result)
    if not parsed:
        return

    lesson = {
        "symbol": symbol,
        "instrument": instrument,
        "direction": direction,
        "pnl_pct": pnl_pct,
        "exit_reason": exit_reason,
        "regime": regime_at_entry,
        "grade": parsed.get("grade", "?"),
        "lesson": parsed.get("lesson", ""),
        "category": parsed.get("category", ""),
        "date": _now_ist().strftime("%Y-%m-%d"),
    }

    # Append to lessons file (keep last 100)
    LESSONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    lessons = []
    if LESSONS_FILE.exists():
        try:
            lessons = json.loads(LESSONS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    lessons.append(lesson)
    lessons = lessons[-100:]  # keep last 100
    LESSONS_FILE.write_text(json.dumps(lessons, indent=2))
    logger.info("TRADE LESSON [%s] %s: %s — %s", lesson["grade"], symbol, lesson["category"], lesson["lesson"])


def get_recent_lessons(instrument: str = None, direction: str = None, limit: int = 5) -> list[dict]:
    """Get recent lessons, optionally filtered by instrument/direction."""
    if not LESSONS_FILE.exists():
        return []
    try:
        lessons = json.loads(LESSONS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    if instrument:
        lessons = [l for l in lessons if l.get("instrument") == instrument]
    if direction:
        lessons = [l for l in lessons if l.get("direction") == direction]

    return lessons[-limit:]


# ---------------------------------------------------------------------------
# Auction Window Protection
# ---------------------------------------------------------------------------

def is_auction_window() -> bool:
    """Check if we're in opening/closing auction (volatile, widen thresholds)."""
    now = _now_ist()
    mins = now.hour * 60 + now.minute
    # 9:15-9:30 or 3:15-3:30
    return (555 <= mins <= 570) or (915 <= mins <= 930)


def get_auction_atr_buffer() -> float:
    """Extra ATR buffer during auction windows to avoid false exits."""
    return 0.5 if is_auction_window() else 0.0
