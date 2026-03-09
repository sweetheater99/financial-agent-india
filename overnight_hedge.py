"""Overnight hedge guardrail functions (V10).

Protects naked F&O positions at EOD by applying a cascade of rules:
1. Positions in loss → close
2. Near expiry → close
3. High VIX → block naked carry
4. Low gain → close
5. Moderate/high gain → ask Claude for hedge vs carry decision
"""

from config import (
    OVERNIGHT_NAKED_INSTRUMENTS,
    OVERNIGHT_MIN_GAIN_FOR_NAKED_CARRY,
    OVERNIGHT_MIN_GAIN_FOR_HEDGE,
    OVERNIGHT_VIX_NAKED_BLOCK,
    OVERNIGHT_EXPIRY_CLOSE_DAYS,
    OVERNIGHT_HEDGE_MAX_COST_PCT,
)


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
