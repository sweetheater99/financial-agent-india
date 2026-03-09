"""Overnight hedge guardrail functions (V10).

Protects naked F&O positions at EOD by applying a cascade of rules:
1. Positions in loss → close
2. Near expiry → close
3. High VIX → block naked carry
4. Low gain → close
5. Moderate/high gain → ask Claude for hedge vs carry decision
"""

import json
import logging
from datetime import datetime, timezone, timedelta

from config import (
    OVERNIGHT_NAKED_INSTRUMENTS,
    OVERNIGHT_MIN_GAIN_FOR_NAKED_CARRY,
    OVERNIGHT_MIN_GAIN_FOR_HEDGE,
    OVERNIGHT_VIX_NAKED_BLOCK,
    OVERNIGHT_EXPIRY_CLOSE_DAYS,
    OVERNIGHT_HEDGE_MAX_COST_PCT,
    OVERNIGHT_HEDGE_OTM_POINTS_NIFTY,
    OVERNIGHT_HEDGE_OTM_POINTS_BANKNIFTY,
    OVERNIGHT_STOP_TIGHTEN_PCT,
)

logger = logging.getLogger("paper_trade")


def is_naked_fno(pos: dict) -> bool:
    """Check if a position is a naked (unhedged) F&O position.

    Returns True when:
    - instrument is in OVERNIGHT_NAKED_INSTRUMENTS
    - status is 'open'
    - no hedge_leg attached
    """
    if pos.get("status") != "open":
        return False
    if pos.get("instrument") not in OVERNIGHT_NAKED_INSTRUMENTS:
        return False
    if pos.get("hedge_leg"):
        return False
    return True


def _calc_gain_pct(entry_price: float, current_price: float, direction: str = "bullish") -> float:
    """Calculate unrealised gain as a fraction (0.5 = 50% gain).

    For bullish: (current - entry) / entry
    For bearish: (entry - current) / entry
    """
    if entry_price <= 0:
        return 0.0
    if direction == "bearish":
        return (entry_price - current_price) / entry_price
    return (current_price - entry_price) / entry_price


def apply_guardrails(pos: dict, current_price: float, vix: float, days_to_expiry: int) -> dict:
    """Apply overnight guardrail cascade to a position.

    Returns {"action": "close"|"ask_claude", "reason": str}
    """
    direction = pos.get("direction", "bullish")
    entry_price = pos.get("entry_price", 0)
    gain_pct = _calc_gain_pct(entry_price, current_price, direction)

    # 1. Position in loss → close
    if gain_pct < 0:
        return {"action": "close", "reason": "position in loss — no overnight carry"}

    # 2. Near expiry → close
    if days_to_expiry <= OVERNIGHT_EXPIRY_CLOSE_DAYS:
        return {"action": "close", "reason": f"only {days_to_expiry} day(s) to expiry — close"}

    # 3. Low gain → close (below hedge threshold)
    if gain_pct < OVERNIGHT_MIN_GAIN_FOR_HEDGE:
        return {"action": "close", "reason": f"gain {gain_pct:.0%} below {OVERNIGHT_MIN_GAIN_FOR_HEDGE:.0%} hedge threshold — close"}

    # 4. High VIX → block naked carry (still ask Claude for hedge decision)
    if vix >= OVERNIGHT_VIX_NAKED_BLOCK:
        return {"action": "ask_claude", "reason": f"VIX {vix} >= {OVERNIGHT_VIX_NAKED_BLOCK} — naked carry blocked, ask Claude for hedge"}

    # 5. Moderate/high gain → ask Claude
    return {"action": "ask_claude", "reason": f"gain {gain_pct:.0%}, VIX {vix}, {days_to_expiry} DTE — ask Claude"}


def enforce_carry_naked_threshold(decision: dict, gain_pct: float) -> dict:
    """Override Claude's carry_naked decision if gain is below threshold.

    If Claude says carry_naked but gain < OVERNIGHT_MIN_GAIN_FOR_NAKED_CARRY,
    downgrade to hedge instead.  Close decisions pass through unchanged.
    """
    if decision.get("action") == "close":
        return decision

    if decision.get("action") == "carry_naked" and gain_pct < OVERNIGHT_MIN_GAIN_FOR_NAKED_CARRY:
        return {
            "action": "hedge",
            "reason": f"carry_naked overridden — gain {gain_pct:.0%} below {OVERNIGHT_MIN_GAIN_FOR_NAKED_CARRY:.0%} threshold",
        }

    return decision


def check_hedge_cost(position_value: float, hedge_premium: float, quantity: int) -> dict:
    """Check if the hedge cost is within budget.

    Returns {"affordable": bool, "cost": float, "cost_pct": float}
    """
    total_cost = hedge_premium * quantity
    cost_pct = total_cost / position_value if position_value > 0 else float("inf")
    return {
        "affordable": cost_pct <= OVERNIGHT_HEDGE_MAX_COST_PCT,
        "cost": total_cost,
        "cost_pct": cost_pct,
    }


# ---------------------------------------------------------------------------
# PARSE CLAUDE OVERNIGHT DECISION
# ---------------------------------------------------------------------------

def parse_overnight_decision(raw_response: str | None) -> dict:
    """Parse Claude's overnight decision response into a standardised dict.

    Returns {"action": "close"|"hedge"|"carry_naked", "reasoning": str}
    """
    if raw_response is None:
        return {"action": "close", "reasoning": "No Claude response — closing as fail-safe"}

    valid_actions = {"close", "hedge", "carry_naked"}

    # Try to parse JSON (handle markdown fences)
    try:
        cleaned = raw_response.strip()
        if "```" in cleaned:
            parts = cleaned.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    cleaned = part
                    break
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and parsed.get("action") in valid_actions:
            return {
                "action": parsed["action"],
                "reasoning": parsed.get("reasoning", "no reasoning provided"),
            }
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # Try to find JSON within the text
    try:
        start = raw_response.find("{")
        end = raw_response.rfind("}")
        if start != -1 and end > start:
            parsed = json.loads(raw_response[start : end + 1])
            if isinstance(parsed, dict) and parsed.get("action") in valid_actions:
                return {
                    "action": parsed["action"],
                    "reasoning": parsed.get("reasoning", "no reasoning provided"),
                }
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return {"action": "hedge", "reasoning": "Could not parse Claude response — defaulting to hedge"}


# ---------------------------------------------------------------------------
# HEDGE EXECUTION LOGIC
# ---------------------------------------------------------------------------

def build_hedge_leg(pos: dict, spot_price: float) -> dict:
    """Build a protective option leg for a naked position.

    Bullish → buy OTM PE (put)
    Bearish → buy OTM CE (call)
    """
    symbol = pos.get("symbol", "NIFTY")
    direction = pos.get("direction", "bullish")
    quantity = abs(pos.get("quantity", 0))

    if "BANKNIFTY" in symbol.upper():
        otm_points = OVERNIGHT_HEDGE_OTM_POINTS_BANKNIFTY
        rounding = 100
    else:
        otm_points = OVERNIGHT_HEDGE_OTM_POINTS_NIFTY
        rounding = 50

    rounded_spot = round(spot_price / rounding) * rounding

    if direction == "bearish":
        option_type = "CE"
        strike = rounded_spot + otm_points
    else:
        option_type = "PE"
        strike = rounded_spot - otm_points

    return {
        "option_type": option_type,
        "strike": int(strike),
        "symbol": symbol,
        "quantity": quantity,
    }


def tighten_stop_loss(pos: dict, current_price: float) -> float:
    """Tighten stop-loss by reducing the gap between current price and SL.

    Reduces the gap by OVERNIGHT_STOP_TIGHTEN_PCT (default 20%).
    Returns new stop-loss price rounded to 2 decimal places.
    """
    current_sl = pos.get("stoploss_price", 0)
    gap = current_price - current_sl
    new_gap = gap * (1 - OVERNIGHT_STOP_TIGHTEN_PCT)
    new_sl = round(current_price - new_gap, 2)
    return new_sl


def execute_close(portfolio: dict, pos: dict, smart_api, current_price: float, reason: str) -> dict:
    """Close a position as part of overnight risk management.

    Delegates to paper_trade.close_position with an overnight_risk exit reason.
    """
    from paper_trade import close_position

    return close_position(
        portfolio=portfolio,
        pos=pos,
        exit_price=current_price,
        reason=f"overnight_risk: {reason}",
        smart_api=smart_api,
    )


def execute_hedge(portfolio: dict, pos: dict, smart_api, hedge_leg: dict, hedge_premium: float) -> bool:
    """Attach a hedge leg to an existing position.

    Adds hedge_leg dict with option details, premium, cost, and date.
    Returns True on success.
    """
    IST = timezone(timedelta(hours=5, minutes=30))
    quantity = hedge_leg.get("quantity", 0)
    pos["hedge_leg"] = {
        "option_type": hedge_leg["option_type"],
        "strike": hedge_leg["strike"],
        "premium": hedge_premium,
        "quantity": quantity,
        "cost": round(hedge_premium * quantity, 2),
        "added_date": datetime.now(IST).strftime("%Y-%m-%d"),
    }
    logger.info(
        "HEDGE attached to %s: %s %s @ ₹%.2f (cost ₹%.2f)",
        pos.get("symbol", "?"),
        hedge_leg["option_type"],
        hedge_leg["strike"],
        hedge_premium,
        hedge_premium * quantity,
    )
    return True


def execute_carry_naked(pos: dict, current_price: float, reasoning: str) -> dict:
    """Approve naked carry with tightened stop-loss.

    Tightens SL and records approval metadata on the position.
    Returns the updated position.
    """
    IST = timezone(timedelta(hours=5, minutes=30))
    new_sl = tighten_stop_loss(pos, current_price)
    old_sl = pos.get("stoploss_price", 0)
    pos["stoploss_price"] = new_sl
    pos["overnight_carry_approved"] = {
        "date": datetime.now(IST).strftime("%Y-%m-%d"),
        "reasoning": reasoning,
        "tightened_sl": new_sl,
    }
    logger.info(
        "CARRY NAKED approved for %s: SL tightened %.2f → %.2f — %s",
        pos.get("symbol", "?"),
        old_sl,
        new_sl,
        reasoning,
    )
    return pos
