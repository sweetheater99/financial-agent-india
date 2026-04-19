# v7/strategist.py
"""Claude-powered Strategist for V7.

Generates and adapts the daily trading playbook.
Calls Claude at scheduled times:
  8:45 AM  — Pre-market playbook (Sonnet)
  9:45 AM  — Opening read (Sonnet)
  10:30 AM — Check-in 1 (Sonnet)
  1:00 PM  — Check-in 2 (Sonnet)
  On exception — Emergency (Sonnet)
  3:30 PM  — EOD review (Haiku)

Uses ClaudeCLIClient from config.py which wraps either the Anthropic SDK
or the `claude` CLI depending on available auth.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

from v7.types import (
    Playbook, Setup, SetupType, DayClassification, Conviction,
    RiskBudget, PacingStatus, CarryRules, MarketContext,
)

log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# Max stock plans per playbook (spec: 2-3)
MAX_STOCK_PLANS = 3

# ── System Prompts ──────────────────────────────────────────────────────


STRATEGIST_SYSTEM = """You are the head of a professional Indian F&O trading desk managing ₹3-5L capital.

Your job: generate a structured daily trading playbook that the mechanical executor will follow exactly.

Rules:
- Max 2-3 setups per instrument (Plan A and Plan B, not an encyclopedia)
- Max 2-3 stock plans per day (pick the best)
- Every setup has a SPECIFIC price trigger level (not "buy if bullish")
- No-trade zones explicitly defined
- Priority ranking determines execution order
- Conviction: "high" (2% risk), "medium" (1.5% risk), "low" (0.75% risk)
- All trigger levels must be numbers, not text
- Stock plans MUST have trigger, target, stoploss as concrete price numbers (not 0) — if you can't find a good stock setup, return an empty stock_plans array instead of stub entries
- Respond with ONLY a JSON playbook — no commentary before or after

Strategy guidance (from backtested optimization + trade review):

SETUP TYPE BY DAY CLASSIFICATION:
- LIKELY_TREND_UP/DOWN → breakout_long or breakout_short ONLY. This is where breakouts work.
- LIKELY_RANGE / UNCERTAIN → PREFER support_bounce and resistance_fade. Breakouts get chopped in range days.
  If you can't identify clear S/R levels, declare "no trade" rather than forcing breakout entries.
- EVENT_DAY → credit spreads and theta are highest edge. Low-to-medium conviction. VIX >= 18 means rich premiums — SELL them, do not sit out.

CRITICAL RULES:
- MAX 2 directional option buys per day. Remaining slots should be credit spreads or skip.
- Directional buys allowed until 1:00 PM (any conviction). 1:00-2:30 PM: MEDIUM+ conviction only. After 2:30 PM: theta/spreads only.
- AVOID repeating a losing symbol same day unless the setup is clearly different (e.g., morning breakout failed, afternoon reversal forming).
- Iron condors / credit spreads when VIX >= 16 — this is the HIGHEST EDGE strategy. Research shows option sellers have 65-70% win rates vs 30% for buyers in Indian markets.
- Theta plan MUST be "enter" when VIX >= 16 and day is Friday or Monday. theta_details should specify: sell 0.18 delta OTM, 200pt wings, 50% profit target.
- On UNCERTAIN days with VIX 16+, theta income should be the PRIMARY strategy, not a backup. Directional buys are secondary.
- Premium sweet spot: ₹15-80 at normal VIX (15). At elevated VIX (20+), premiums scale — acceptable range is ₹15-120. The executor auto-adjusts the cap based on VIX. Do NOT avoid setups just because VIX makes premiums above ₹80.
- MINIMUM: Always output at least 1 NIFTY setup with trigger_level, target, stoploss (non-zero). An empty nifty_setups array is NEVER acceptable when trade slots remain.
- Grade A entries only: trigger + volume confirmation. Don't force B-grade entries to fill your setup slots.
- USE 60-MIN OPENING RANGE (9:15-10:15) for breakout levels — backtested 88.8% win rate vs 55% for 15-min.
  Wait for the full 60-min candle to close before defining breakout levels. First hour = observe, then act.
- BEST directional window: 9:30-11:00 AM. Full window: 9:30 AM-1:00 PM. Acceptable: 1:00-2:30 PM for MEDIUM+ conviction.
- Monday-Wednesday are the best directional days. Friday = mean reversion bias, skip breakouts.
- If a symbol has been losing for 2+ consecutive days, DROP IT from the watchlist for the day.

Instrument universe: NIFTY, BANKNIFTY, RELIANCE, HDFCBANK, ICICIBANK, TCS, TMPV, BAJFINANCE, SBIN, INFY"""

OPENING_READ_SYSTEM = """You are updating the morning playbook after the first 30 minutes of price discovery.
Review the opening range, gap behavior, volume, and OI shifts.
Output an updated playbook JSON with:
- Day type confirmation or override
- Opening range levels added
- Setups adjusted if levels invalidated
- Possible: "no good setups today, theta only"
Respond with ONLY the updated JSON playbook."""

CHECKIN_SYSTEM = """You are doing a mid-session check-in. Your job is to KEEP THE AGENT TRADING, not to find reasons to stop.

Review: current P&L, open positions, fired/unfired setups, level tests, OI changes, VIX.

MANDATORY OUTPUT RULES:
1. REMOVE no_trade_conditions that are no longer true (VIX dropped? remove VIX block. Chop settled? remove chop block). Start fresh — only include conditions that are CURRENTLY valid RIGHT NOW.
2. For each unfired setup: confirm trigger is still valid OR replace with a better one. Do NOT just cancel — replace.
3. If a stock was banned but the reason expired (news absorbed, results digested), UNBAN it.
4. Add 1-2 NEW setups if clear opportunities emerged. The market moves — new levels form.
5. Mark setups as "cancelled": true ONLY if the level was definitively broken against the setup direction.
6. If zero trades have fired today and it's past 11 AM, actively look for the BEST available setup even if conviction is medium. One disciplined trade > zero trades.
7. Maximum 5 no_trade_conditions. If you have more, keep only the 5 most important.
8. MINIMUM SETUP RULE: You MUST output at least 1 NIFTY setup with ALL fields filled (trigger_level, target, stoploss — never zero). High VIX = rich premiums = high edge for credit spreads. If VIX > 20, at least one setup MUST be a credit_spread (bull or bear). No excuse for empty nifty_setups when there are trade slots remaining.
9. Every setup MUST have numeric target and stoploss. If you cannot determine exact levels, use the opening range high/low +/- 50 points. Never output target=0 or stoploss=0.

BIAS: A day with 1 disciplined trade that loses on SL is BETTER than a day with 0 trades. Do not optimize for zero-loss days. High VIX is an OPPORTUNITY for premium sellers, not a reason to stop trading.

Respond with ONLY the updated JSON playbook."""

EXCEPTION_SYSTEM = """You are handling an EXCEPTION in the trading session.
Something unexpected happened that the playbook doesn't cover.
Respond with ONLY a JSON object:
{
  "action": "flatten_all" | "hold_no_new" | "adjust_sl" | "specific_action",
  "details": "what to do",
  "new_sl_levels": {"SYMBOL": new_sl_price} (if action is adjust_sl),
  "close_symbols": ["SYMBOL"] (if action is flatten_all or specific_action)
}"""

EOD_SYSTEM = """You are grading today's trades for the trading journal.
For each trade, assign:
- entry_grade: A (trigger + confirmation) / B (trigger, weak confirmation) / C (FOMO)
- exit_grade: A (plan followed) / B (minor deviation) / C (panic/held too long)
- lesson: one sentence

Also provide a day summary.
Respond with ONLY a JSON object."""


# ── Prompt Building ─────────────────────────────────────────────────────


def build_premarket_prompt(
    us_close: dict,
    gift_nifty: str,
    prev_vix: float,
    fii_dii: str,
    events_today: list[str],
    events_this_week: list[str],
    level_memory: dict,
    edge_tracker: dict,
    risk_state: dict,
    fo_ban_list: list[str],
    recent_lessons: list[str],
    computed_levels: dict | None = None,
    oi_context: dict | None = None,
) -> str:
    """Build the pre-market prompt with all inputs."""
    parts = [
        "Generate today's trading playbook.\n",
        f"## Market Data\n",
        f"- US Close: {json.dumps(us_close)}",
        f"- GIFT Nifty: {gift_nifty}",
        f"- India VIX (prev close): {prev_vix}",
        f"- FII/DII: {fii_dii}",
        f"- Events today: {events_today if events_today else 'None'}",
        f"- Events this week: {events_this_week if events_this_week else 'None'}",
        f"- F&O ban list: {fo_ban_list if fo_ban_list else 'None'}",
        "",
        f"## Key Levels (from memory)\n",
        json.dumps(level_memory, indent=2) if level_memory else "No levels stored yet.",
        "",
    ]

    if computed_levels:
        parts.append("## Computed Levels\n")
        parts.append(json.dumps(computed_levels, indent=2))
        parts.append("")

    if oi_context:
        parts.append("## Open Interest Context\n")
        parts.append(json.dumps(oi_context, indent=2))
        parts.append("")

    parts.extend([
        f"## Edge Tracker (historical performance)\n",
        json.dumps(edge_tracker, indent=2) if edge_tracker else "No trade history yet.",
        "",
        f"## Risk State\n",
        f"- MTD P&L: {risk_state.get('mtd_pnl_pct', 0):.1f}%",
        f"- Pacing: {risk_state.get('pacing', 'on_track')}",
    ])

    if risk_state.get("survival_mode"):
        parts.append("\n**SURVIVAL MODE ACTIVE**: MTD drawdown > 5%. No directional trades allowed. Theta only. Generate a theta-only playbook with wider wings.")

    if recent_lessons:
        parts.append(f"\n## Recent Lessons\n")
        for lesson in recent_lessons[-5:]:  # last 5 lessons
            parts.append(f"- {lesson}")

    parts.append("\n## Output Format")
    parts.append("Respond with a JSON playbook matching this schema:")
    parts.append(_playbook_schema_hint())

    return "\n".join(parts)


def build_opening_read_prompt(
    current_playbook: dict,
    opening_range_high: float,
    opening_range_low: float,
    gap_direction: str,
    gap_behavior: str,
    first_30min_volume_ratio: float,
    oi_changes: dict,
) -> str:
    """Build the opening read prompt (9:45 AM)."""
    return "\n".join([
        "Update the playbook after first 30 minutes.\n",
        f"## Current Playbook\n{json.dumps(current_playbook, indent=2)}\n",
        f"## Opening Data",
        f"- Opening range: {opening_range_low:.2f} — {opening_range_high:.2f}",
        f"- Gap: {gap_direction} ({gap_behavior})",
        f"- First 30-min volume vs 20-day avg: {first_30min_volume_ratio:.1%}",
        f"- OI changes from previous close: {json.dumps(oi_changes)}\n",
        "Update the playbook JSON. Add opening_range levels. Confirm or override day type.",
    ])


def build_checkin_prompt(
    current_playbook: dict,
    daily_pnl: float,
    open_positions: list[dict],
    setups_fired: list[str],
    levels_tested: list[dict],
    oi_changes: dict,
    current_vix: float,
    checkin_number: int,
    extra_sections: list[str] | None = None,
) -> str:
    """Build check-in prompt (10:30 AM or 1:00 PM)."""
    parts = [
        f"Check-in #{checkin_number}. Update the playbook.\n",
        f"## Current Playbook\n{json.dumps(current_playbook, indent=2)}\n",
        f"## Session Status",
        f"- Daily P&L: ₹{daily_pnl:.0f}",
        f"- Open positions: {json.dumps(open_positions) if open_positions else 'None'}",
        f"- Setups fired: {setups_fired if setups_fired else 'None yet'}",
        f"- Levels tested since last update: {json.dumps(levels_tested) if levels_tested else 'None'}",
        f"- OI changes: {json.dumps(oi_changes)}",
        f"- Current VIX: {current_vix}",
        "",
    ]
    if extra_sections:
        for section in extra_sections:
            if section:
                parts.append(section)
                parts.append("")
    parts.append("Update the playbook JSON. Confirm/modify remaining setups.")
    return "\n".join(parts)


def build_exception_prompt(
    exception_type: str,
    details: dict,
    current_playbook: dict,
    open_positions: list[dict],
) -> str:
    """Build exception prompt for unexpected events."""
    return "\n".join([
        f"EXCEPTION: {exception_type}\n",
        f"## Details\n{json.dumps(details, indent=2)}\n",
        f"## Open Positions\n{json.dumps(open_positions, indent=2)}\n",
        f"## Current Playbook\n{json.dumps(current_playbook, indent=2)}\n",
        "What action should we take?",
    ])


def build_eod_prompt(
    trades_today: list[dict],
    daily_pnl: float,
    day_classification_predicted: str,
    day_classification_actual: str,
) -> str:
    """Build EOD review prompt (3:30 PM, Haiku)."""
    return "\n".join([
        "Grade today's trades and provide a day summary.\n",
        f"## Trades\n{json.dumps(trades_today, indent=2)}\n",
        f"## Summary",
        f"- Daily P&L: ₹{daily_pnl:.0f}",
        f"- Day type predicted: {day_classification_predicted}",
        f"- Day type actual: {day_classification_actual}",
        "",
        "Grade each trade (entry_grade, exit_grade, lesson). Provide day summary.",
    ])


def _playbook_schema_hint() -> str:
    return """```json
{
  "date": "YYYY-MM-DD",
  "day_classification": "LIKELY_TREND_UP|LIKELY_TREND_DOWN|LIKELY_RANGE|UNCERTAIN|EVENT_DAY|NO_TRADE",
  "nifty_plan": {
    "bias": "bullish|bearish|neutral",
    "key_levels": {"resistance_1": float, "support_1": float, ...},
    "setups": [
      {
        "id": "N1",
        "priority": 1,
        "type": "breakout_long|breakout_short|support_bounce|resistance_fade|credit_spread_bull|credit_spread_bear",
        "trigger": "specific condition with PRICE LEVEL number",
        "instrument": "NIFTY CE|PE",
        "strike_logic": "delta description",
        "target": float,
        "stoploss": float,
        "max_risk_pct": float,
        "conviction": "high|medium|low",
        "cancelled": false
      }
    ],
    "no_trade_zone": "low-high"
  },
  "stock_plans": [
    {
      "id": "X1",
      "priority": 3,
      "type": "breakout_long|breakout_short|support_bounce|resistance_fade",
      "symbol": "RELIANCE",
      "trigger": "specific condition with PRICE LEVEL number e.g. breaks above 1280",
      "instrument": "RELIANCE CE|PE",
      "strike_logic": "delta description",
      "target": float,
      "stoploss": float,
      "max_risk_pct": float,
      "conviction": "high|medium|low"
    }
  ],
  "risk_budget": {
    "max_capital_at_risk_today_pct": 4.0,
    "max_trades_today": 4,
    "max_per_trade_risk_pct": 1.5,
    "survival_mode": false
  },
  "no_trade_conditions": ["condition1", ...],
  "carry_rules": {"carry_if": "conditions"},
  "theta_plan": {"action": "hold|enter|adjust|exit", "details": "..."},
  "market_context": {"us_close": "...", "gift_nifty": "...", "vix": float, "fii_dii": "..."}
}
```"""


# ── Response Parsing ────────────────────────────────────────────────────


def _extract_json(text: str) -> dict | None:
    """Extract JSON from Claude's response, handling markdown code blocks."""
    # Try direct JSON parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` block
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { ... last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


def _extract_trigger_level(trigger_text: str, target: float, stoploss: float) -> float:
    """Extract a numeric trigger level from the trigger description.

    Looks for numbers in the trigger text. Falls back to midpoint of
    target and stoploss if no number found.
    """
    numbers = re.findall(r"[\d]+(?:\.[\d]+)?", trigger_text)
    # Filter for price-like numbers (> 100 for Nifty-scale, or > 10 for stock premium)
    candidates = [float(n) for n in numbers if float(n) > 10]

    if candidates:
        # Pick the number closest to the midpoint of target/stoploss
        midpoint = (target + stoploss) / 2
        return min(candidates, key=lambda x: abs(x - midpoint))

    # Fallback: midpoint
    return (target + stoploss) / 2


def _parse_setup(d: dict, symbol: str = "NIFTY") -> Setup:
    """Parse a setup dict from Claude's response into a Setup object."""
    target = float(d.get("target", 0))
    stoploss = float(d.get("stoploss", 0))
    trigger_text = d.get("trigger", d.get("entry_condition", d.get("trigger_condition", "")))

    # Map type string to SetupType enum
    type_map = {
        "breakout_long": SetupType.BREAKOUT_LONG,
        "breakout_short": SetupType.BREAKOUT_SHORT,
        "support_bounce": SetupType.SUPPORT_BOUNCE,
        "resistance_fade": SetupType.RESISTANCE_FADE,
        "range_fade": SetupType.RESISTANCE_FADE,
        "credit_spread_bull": SetupType.CREDIT_SPREAD_BULL,
        "credit_spread_bear": SetupType.CREDIT_SPREAD_BEAR,
        "credit_spread": SetupType.CREDIT_SPREAD_BEAR,  # infer from direction below
        "iron_condor": SetupType.IRON_CONDOR,
    }
    raw_type = d.get("type", "")
    setup_type = type_map.get(raw_type, SetupType.BREAKOUT_LONG)
    # Handle "credit_spread" with direction field
    if raw_type == "credit_spread":
        direction = d.get("direction", "bear")
        if "bull" in direction:
            setup_type = SetupType.CREDIT_SPREAD_BULL

    # STRICT VALIDATION: resolve direction from target/stoploss orientation (LLM numbers are more
    # reliable than LLM labels). Bear setups: target < stoploss. Bull: target > stoploss.
    bullish_types = {SetupType.BREAKOUT_LONG, SetupType.SUPPORT_BOUNCE, SetupType.CREDIT_SPREAD_BULL}
    bearish_types = {SetupType.BREAKOUT_SHORT, SetupType.RESISTANCE_FADE, SetupType.CREDIT_SPREAD_BEAR}
    if target > 0 and stoploss > 0 and target != stoploss:
        numbers_bullish = target > stoploss
        label_bullish = setup_type in bullish_types
        if numbers_bullish != label_bullish:
            import logging
            _logger = logging.getLogger(__name__)
            _logger.warning(
                "Setup %s: type=%s says %s but target=%.1f/stoploss=%.1f says %s — flipping to match numbers",
                d.get("id", d.get("setup_id", "?")), raw_type,
                "bullish" if label_bullish else "bearish",
                target, stoploss,
                "bullish" if numbers_bullish else "bearish",
            )
            # Flip to the orientation the numbers imply, preferring spread types if name has "credit"/"spread"
            setup_id_lower = str(d.get("id", d.get("setup_id", ""))).lower()
            is_spread = "spread" in setup_id_lower or "credit" in setup_id_lower or setup_type in {SetupType.CREDIT_SPREAD_BULL, SetupType.CREDIT_SPREAD_BEAR}
            if numbers_bullish:
                setup_type = SetupType.CREDIT_SPREAD_BULL if is_spread else SetupType.BREAKOUT_LONG
            else:
                setup_type = SetupType.CREDIT_SPREAD_BEAR if is_spread else SetupType.BREAKOUT_SHORT

    conviction_map = {
        "high": Conviction.HIGH,
        "medium": Conviction.MEDIUM,
        "low": Conviction.LOW,
    }
    conviction = conviction_map.get(d.get("conviction", "medium"), Conviction.MEDIUM)

    # Handle direct trigger_level or extract from text
    raw_trigger = d.get("trigger_level")
    if raw_trigger is not None and float(raw_trigger) > 0:
        trigger_level = float(raw_trigger)
    else:
        trigger_level = _extract_trigger_level(trigger_text, target, stoploss)

    return Setup(
        id=d.get("id", d.get("setup_id", "X1")),
        priority=int(d.get("priority", 99)),
        type=setup_type,
        symbol=d.get("symbol", symbol),
        trigger_level=trigger_level,
        trigger_condition=trigger_text,
        instrument=d.get("instrument", f"{symbol} CE"),
        strike_logic=d.get("strike_logic", "delta 0.45"),
        target=target,
        stoploss=stoploss,
        max_risk_pct=float(d.get("max_risk_pct", 1.5)),
        conviction=conviction,
        fired=bool(d.get("fired", False)),
        cancelled=bool(d.get("cancelled", False)),
    )


def _build_market_context(data: dict | None) -> MarketContext:
    """Build a MarketContext from a raw dict (possibly from Claude's response)."""
    if not data:
        return MarketContext()
    if isinstance(data, MarketContext):
        return data
    return MarketContext(
        us_close=str(data.get("us_close", "")),
        gift_nifty=str(data.get("gift_nifty", "")),
        vix=float(data.get("vix", 0.0)),
        fii_dii=str(data.get("fii_dii", "")),
        events_today=data.get("events_today", []),
        events_this_week=data.get("events_this_week", []),
        fo_ban_list=data.get("fo_ban_list", []),
    )


def parse_playbook_response(raw: str, today: date | None = None) -> Playbook | None:
    """Parse Claude's response into a Playbook object.

    Returns None if parsing fails.
    """
    today = today or date.today()
    data = _extract_json(raw)
    if data is None:
        log.error("Failed to extract JSON from Strategist response")
        return None

    try:
        # Day classification
        dc_map = {
            "LIKELY_TREND_UP": DayClassification.LIKELY_TREND_UP,
            "LIKELY_TREND_DOWN": DayClassification.LIKELY_TREND_DOWN,
            "LIKELY_RANGE": DayClassification.LIKELY_RANGE,
            "UNCERTAIN": DayClassification.UNCERTAIN,
            "EVENT_DAY": DayClassification.EVENT_DAY,
            "NO_TRADE": DayClassification.NO_TRADE,
        }
        day_class = dc_map.get(
            data.get("day_classification", "UNCERTAIN"),
            DayClassification.UNCERTAIN,
        )

        # Nifty setups — handle both nifty_plan.setups and direct nifty_setups
        nifty_plan = data.get("nifty_plan", {})
        nifty_raw = data.get("nifty_setups", [])
        if not nifty_raw:
            nifty_raw = nifty_plan.get("setups", [])
        nifty_setups = [
            _parse_setup(s, symbol="NIFTY")
            for s in nifty_raw
        ]

        # Stock plans (cap at MAX_STOCK_PLANS, drop stubs with no levels)
        stock_plans_raw = data.get("stock_plans", [])[:MAX_STOCK_PLANS]
        stock_plans = []
        for s in stock_plans_raw:
            # Skip stub entries where Claude didn't provide actual levels
            if float(s.get("target", 0)) == 0 and float(s.get("stoploss", 0)) == 0:
                log.warning("Dropping stock plan %s/%s — no target/stoploss", s.get("symbol"), s.get("id"))
                continue
            stock_plans.append(_parse_setup(s, symbol=s.get("symbol", "UNKNOWN")))

        # Risk budget
        rb_data = data.get("risk_budget", {})
        risk_budget = RiskBudget(
            max_capital_at_risk_today_pct=rb_data.get("max_capital_at_risk_today_pct", 4.0),
            max_trades_today=rb_data.get("max_trades_today", 4),
            max_per_trade_risk_pct=rb_data.get("max_per_trade_risk_pct", 1.5),
            survival_mode=rb_data.get("survival_mode", False),
        )

        # Carry rules
        carry_data = data.get("carry_rules", {})
        carry_rules = CarryRules(
            min_profit_pct=carry_data.get("min_profit_pct", 1.5),
            max_vix=carry_data.get("max_vix", 20.0),
            min_dte=carry_data.get("min_dte", 3),
            max_hedge_cost=carry_data.get("max_hedge_cost", 500.0),
        )

        # Theta action
        theta_plan = data.get("theta_plan", {})
        theta_action = theta_plan.get("action", "hold")
        theta_details = theta_plan.get("details", "")

        # Market context — must be a MarketContext object, not raw dict
        market_context = _build_market_context(data.get("market_context", {}))

        playbook = Playbook(
            date=today,
            day_classification=day_class,
            nifty_bias=nifty_plan.get("bias", "neutral"),
            nifty_setups=nifty_setups,
            stock_plans=stock_plans,
            risk_budget=risk_budget,
            no_trade_conditions=data.get("no_trade_conditions", [])[:5],
            carry_rules=carry_rules,
            market_context=market_context,
            theta_action=theta_action,
            theta_details=theta_details,
        )

        return playbook

    except Exception as e:
        log.error(f"Failed to parse playbook: {e}")
        return None


# ── Fallback Playbook ───────────────────────────────────────────────────


def build_fallback_playbook(
    today: date | None = None,
    prev_playbook: Playbook | None = None,
) -> Playbook:
    """Conservative fallback when Claude is unreachable.

    If previous playbook available: reuse with halved risk budgets, index only.
    Otherwise: no-trade playbook (theta only).
    """
    today = today or date.today()

    if prev_playbook:
        # Reuse previous playbook with conservative adjustments
        return Playbook(
            date=today,
            day_classification=DayClassification.UNCERTAIN,
            nifty_bias=prev_playbook.nifty_bias,
            nifty_setups=[
                Setup(
                    id=s.id, priority=s.priority, type=s.type,
                    symbol=s.symbol, trigger_level=s.trigger_level,
                    trigger_condition=s.trigger_condition,
                    instrument=s.instrument, strike_logic=s.strike_logic,
                    target=s.target, stoploss=s.stoploss,
                    max_risk_pct=s.max_risk_pct * 0.5,  # halved
                    conviction=Conviction.LOW,  # downgraded
                )
                for s in prev_playbook.nifty_setups[:2]  # max 2 setups
            ],
            stock_plans=[],  # no stock setups without fresh analysis
            risk_budget=RiskBudget(
                max_capital_at_risk_today_pct=2.0,  # halved
                max_trades_today=2,  # halved
                max_per_trade_risk_pct=0.75,  # halved
                survival_mode=False,
            ),
            no_trade_conditions=prev_playbook.no_trade_conditions + ["Claude API unavailable — conservative mode"],
            carry_rules=CarryRules(),
            market_context=MarketContext(us_close="Fallback playbook — Claude unavailable"),
            theta_action="hold",
        )

    # No previous playbook — absolute minimum
    return Playbook(
        date=today,
        day_classification=DayClassification.NO_TRADE,
        nifty_bias="neutral",
        nifty_setups=[],
        stock_plans=[],
        risk_budget=RiskBudget(
            max_capital_at_risk_today_pct=0.0,
            max_trades_today=0,
            max_per_trade_risk_pct=0.0,
            survival_mode=False,
        ),
        no_trade_conditions=["Claude API unavailable — no playbook generated"],
        carry_rules=CarryRules(),
        market_context=MarketContext(us_close="No-trade fallback — Claude unavailable, no prior playbook"),
        theta_action="hold",
    )


# ── Exception Response Parsing ──────────────────────────────────────────


def parse_exception_response(raw: str) -> dict | None:
    """Parse Claude's exception response."""
    data = _extract_json(raw)
    if data is None:
        return None
    if "action" not in data:
        return None
    return data


def default_exception_action(exception_type: str) -> dict:
    """Default action when Claude is unreachable during an exception."""
    defaults = {
        "vix_spike": {"action": "hold_no_new", "details": "VIX spike — hold positions, no new trades. Close 50% if VIX > 25."},
        "flash_crash": {"action": "flatten_all", "details": "Flash crash — close all positions immediately."},
        "3_sl_hits": {"action": "hold_no_new", "details": "3 SL hits — stop trading for the day."},
        "margin_warning": {"action": "hold_no_new", "details": "High margin — no new positions until margin frees."},
        "stock_spike": {"action": "hold_no_new", "details": "Large stock move — hold, check for news."},
    }
    return defaults.get(exception_type, {"action": "hold_no_new", "details": f"Unknown exception: {exception_type}. Defaulting to hold."})


def _backfill_or_levels(playbook: "Playbook", or_high: float, or_low: float) -> None:
    """Fill in target/stoploss/trigger for OR setups that Claude left at 0.

    Uses opening range high/low to derive sensible levels based on setup ID.
    Also fixes setup types that may have been misassigned.
    """
    if or_high <= 0 or or_low <= 0:
        return

    or_range = or_high - or_low
    type_fixes = {
        "NF_OR_HIGH_FADE": SetupType.RESISTANCE_FADE,
        "NF_OR_LOW_BOUNCE": SetupType.SUPPORT_BOUNCE,
        "NF_OR_BREAKOUT_LONG": SetupType.BREAKOUT_LONG,
        "NF_OR_BREAKDOWN_SHORT": SetupType.BREAKOUT_SHORT,
    }

    for s in playbook.nifty_setups:
        # Fix misassigned setup types
        if s.id in type_fixes and s.type != type_fixes[s.id]:
            log.info("Fixing setup %s type: %s -> %s", s.id, s.type.value, type_fixes[s.id].value)
            s.type = type_fixes[s.id]

        # Backfill trigger/target/stoploss from OR levels
        if s.target == 0 or s.stoploss == 0:
            if "HIGH_FADE" in s.id or "RESISTANCE" in s.id:
                s.trigger_level = or_high
                s.target = or_low  # fade from high to low
                s.stoploss = or_high + or_range * 0.3  # SL above OR high
            elif "LOW_BOUNCE" in s.id or "SUPPORT" in s.id:
                s.trigger_level = or_low
                s.target = or_high  # bounce from low to high
                s.stoploss = or_low - or_range * 0.3  # SL below OR low
            elif "BREAKOUT_LONG" in s.id:
                s.trigger_level = or_high
                s.target = or_high + or_range  # 1:1 R:R from OR range
                s.stoploss = or_high - or_range * 0.5  # SL inside OR
            elif "BREAKDOWN_SHORT" in s.id:
                s.trigger_level = or_low
                s.target = or_low - or_range  # 1:1 R:R from OR range
                s.stoploss = or_low + or_range * 0.5  # SL inside OR
            else:
                # Generic: skip — cannot infer levels
                log.warning("Setup %s has target=0 but no OR-level inference rule — skipping backfill", s.id)
                continue
            log.info("Backfilled OR levels for %s: trigger=%.2f target=%.2f sl=%.2f",
                     s.id, s.trigger_level, s.target, s.stoploss)


# ── Strategist Class ────────────────────────────────────────────────────


class Strategist:
    """Claude-powered playbook generator.

    Handles all scheduled Claude calls and fallback logic.
    """

    def __init__(self, state_dir: str | None = None,
                 data_feed=None, edge_tracker=None, risk_engine=None):
        from config import get_anthropic_client, CLAUDE_MODEL, CLAUDE_MODEL_LIGHT
        self._client = get_anthropic_client()
        self._model = CLAUDE_MODEL
        self._model_light = CLAUDE_MODEL_LIGHT
        self._max_retries = 1
        self._retry_delay = 5  # seconds

        from v7.state import StateManager
        from pathlib import Path
        self._data_dir = Path(state_dir or "data/v7")
        self._state = StateManager(self._data_dir)
        self._data_feed = data_feed
        self._edge_tracker = edge_tracker
        self._risk_engine = risk_engine

    def _call_claude(
        self, prompt: str, system: str, model: str | None = None,
        max_tokens: int = 4096,
    ) -> str | None:
        """Call Claude with retry logic. Returns response text or None."""
        import time as time_mod

        model = model or self._model
        for attempt in range(self._max_retries):
            try:
                response = self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except Exception as e:
                log.warning(f"Claude call failed (attempt {attempt + 1}/{self._max_retries}): {e}")
                if attempt < self._max_retries - 1:
                    time_mod.sleep(self._retry_delay)
        return None

    def premarket(self) -> Playbook:
        """No-arg wrapper: gathers all market intel, then generates playbook."""
        from v7.market_intel import gather_premarket_intel
        intel = gather_premarket_intel(
            data_feed=self._data_feed,
            edge_tracker=self._edge_tracker,
            risk_engine=self._risk_engine,
            state=self._state,
            data_dir=self._data_dir,
        )
        return self.generate_premarket_playbook(**intel)

    def generate_premarket_playbook(
        self,
        us_close: dict, gift_nifty: str, prev_vix: float,
        fii_dii: str, events_today: list[str], events_this_week: list[str],
        level_memory: dict, edge_tracker: dict, risk_state: dict,
        fo_ban_list: list[str], recent_lessons: list[str],
        computed_levels: dict | None = None,
        oi_context: dict | None = None,
    ) -> Playbook:
        """Generate the pre-market playbook (8:45 AM call).

        Falls back to conservative playbook if Claude is unreachable.
        """
        prompt = build_premarket_prompt(
            us_close=us_close, gift_nifty=gift_nifty, prev_vix=prev_vix,
            fii_dii=fii_dii, events_today=events_today,
            events_this_week=events_this_week, level_memory=level_memory,
            edge_tracker=edge_tracker, risk_state=risk_state,
            fo_ban_list=fo_ban_list, recent_lessons=recent_lessons,
            computed_levels=computed_levels, oi_context=oi_context,
        )

        raw = self._call_claude(prompt, system=STRATEGIST_SYSTEM)
        if raw:
            playbook = parse_playbook_response(raw)
            if playbook:
                self._state.save_playbook(playbook)
                return playbook
            log.error("Claude returned unparseable playbook — using fallback")

        # Fallback
        prev = self._state.load_playbook()
        fallback = build_fallback_playbook(prev_playbook=prev)
        self._state.save_playbook(fallback)
        return fallback

    def opening_read(self) -> Playbook | None:
        """Update playbook after first 30 min (9:45 AM).

        Auto-fetches opening range, gap, volume, and OI from data feed.
        Returns updated playbook or current playbook if Claude unavailable.
        """
        current = self._state.load_playbook()
        if current is None:
            log.warning("No current playbook for opening read")
            return None

        # Gather opening data from data feed
        opening_range_high = 0.0
        opening_range_low = 0.0
        gap_direction = "flat"
        gap_behavior = "unknown"
        first_30min_volume_ratio = 1.0
        oi_changes = {}

        if self._data_feed and self._data_feed.can_trade():
            try:
                candles = self._data_feed.get_candles("NIFTY", interval="5minute", days=1)
                if candles:
                    # First 30 min = first 6 five-minute candles (9:15-9:45)
                    first_30 = candles[:6] if len(candles) >= 6 else candles
                    highs = [c["high"] for c in first_30]
                    lows = [c["low"] for c in first_30]
                    opening_range_high = max(highs) if highs else 0
                    opening_range_low = min(lows) if lows else 0

                    # Volume ratio vs 20-day average (approximate from today's volume)
                    today_vol = sum(c.get("volume", 0) for c in first_30)
                    if today_vol > 0:
                        first_30min_volume_ratio = 1.0  # TODO: compare with 20-day avg

                    # Gap direction from prev close
                    prev_close = current.market_context.gift_nifty if current.market_context else ""
                    open_price = first_30[0]["open"] if first_30 else 0
                    if opening_range_low > 0 and open_price > 0:
                        # Use playbook's key levels to determine gap
                        nifty_levels = {}
                        for s in current.nifty_setups:
                            if s.trigger_level > 0:
                                nifty_levels[s.id] = s.trigger_level
                        # Simple gap classification
                        range_pct = (opening_range_high - opening_range_low) / opening_range_low * 100 if opening_range_low else 0
                        if range_pct > 0.5:
                            gap_behavior = "wide_range"
                        else:
                            gap_behavior = "narrow_range"
            except Exception as e:
                log.warning("Opening read data fetch failed: %s", e)

            # OI changes
            try:
                from v7.oi_pipeline import OIPipeline
                oi = OIPipeline(data_dir=self._data_dir)
                oi_changes_raw = oi.compute_oi_changes("NIFTY")
                # Trim to top 5 strikes to reduce prompt size
                if isinstance(oi_changes_raw, dict):
                    oi_changes = dict(list(oi_changes_raw.items())[:3])
                else:
                    oi_changes = oi_changes_raw
            except Exception:
                pass

        prompt = build_opening_read_prompt(
            current_playbook=current.to_dict(),
            opening_range_high=opening_range_high,
            opening_range_low=opening_range_low,
            gap_direction=gap_direction,
            gap_behavior=gap_behavior,
            first_30min_volume_ratio=first_30min_volume_ratio,
            oi_changes=oi_changes,
        )

        raw = self._call_claude(prompt, system=OPENING_READ_SYSTEM)
        if raw:
            playbook = parse_playbook_response(raw)
            if playbook:
                playbook.opening_range = {
                    "high": opening_range_high,
                    "low": opening_range_low,
                }
                _backfill_or_levels(playbook, opening_range_high, opening_range_low)
                self._state.save_playbook(playbook)
                return playbook

        # If Claude fails, just add opening range to existing playbook
        current.opening_range = {
            "high": opening_range_high,
            "low": opening_range_low,
        }
        _backfill_or_levels(current, opening_range_high, opening_range_low)
        self._state.save_playbook(current)
        return current

    def checkin(self, checkin_number: int = 1) -> dict:
        """Check-in update (10:30 AM or 1:00 PM).

        Auto-fetches daily P&L, positions, VIX, OI from state + data feed.
        Returns dict with plan_changed and summary.
        """
        current = self._state.load_playbook()
        if current is None:
            return {"plan_changed": False, "summary": "No playbook loaded"}

        # Gather state
        daily = self._state.load_daily_state()
        daily_pnl = daily.get("daily_pnl", 0.0)
        positions = self._state.load_positions()
        open_positions = [
            {"symbol": p.symbol, "instrument": p.instrument,
             "direction": p.direction, "entry_price": p.entry_price,
             "stoploss": p.stoploss, "target": p.target}
            for p in positions
        ]

        # Setups fired
        setups_fired = [s.id for s in current.all_setups() if s.fired]

        # VIX
        current_vix = 0.0
        if self._data_feed:
            try:
                current_vix = self._data_feed.get_vix() or 0.0
            except Exception:
                pass

        # OI changes
        oi_changes = {}
        try:
            from v7.oi_pipeline import OIPipeline
            oi = OIPipeline(data_dir=self._data_dir)
            oi_changes_raw = oi.compute_oi_changes("NIFTY")
            # Trim to top 5 strikes to reduce prompt size
            if isinstance(oi_changes_raw, dict):
                oi_changes = dict(list(oi_changes_raw.items())[:3])
            else:
                oi_changes = oi_changes_raw
        except Exception:
            pass

        # Levels tested (simplified — would need candle analysis for full impl)
        levels_tested = []

        # Build extra context sections for enriched checkin
        extra_sections: list[str] = []

        # Setup performance from edge tracker
        from v7.market_intel import setup_performance_summary
        if self._edge_tracker:
            setup_perf = setup_performance_summary(self._edge_tracker)
            if setup_perf:
                extra_sections.append(setup_perf)

        # Open position health scores
        if positions:
            health_lines = ["## Open Position Health"]
            for pos in positions:
                status = (
                    "healthy" if pos.health_score > 70
                    else "warning" if pos.health_score > 40
                    else "critical"
                )
                health_lines.append(
                    f"- {pos.symbol} {pos.instrument} | Health: {pos.health_score:.0f}/100 ({status}) | "
                    f"P&L: {pos.unrealized_pnl_pct(pos.entry_price):.1f}%"
                )
            extra_sections.append("\n".join(health_lines))

        # Trade hunger: show trade activity vs target
        trades_today = daily.get("trades_today", 0)
        banned_count = len(daily.get("banned_symbols", []))
        hunger_lines = ["## Trade Activity"]
        hunger_lines.append(f"- Trades today: {trades_today} (target: 1-2)")
        hunger_lines.append(f"- Banned symbols: {banned_count}")
        if trades_today == 0:
            hunger_lines.append("- WARNING: Zero trades today. Bias toward finding the best available setup.")
        extra_sections.append("\n".join(hunger_lines))

        # Add current spot prices so LLM knows where the market IS
        if self._data_feed:
            spot_lines = ["## Current Spot Prices"]
            try:
                ltps = self._data_feed.get_batch_ltp(["NIFTY", "BANKNIFTY"])
                for sym, ltp in ltps.items():
                    if ltp:
                        spot_lines.append(f"- {sym}: {ltp:.2f}")
            except Exception:
                pass
            if len(spot_lines) > 1:
                extra_sections.append("\n".join(spot_lines))

        prompt = build_checkin_prompt(
            current_playbook=current.to_dict(),
            daily_pnl=daily_pnl, open_positions=open_positions,
            setups_fired=setups_fired, levels_tested=levels_tested,
            oi_changes=oi_changes, current_vix=current_vix,
            checkin_number=checkin_number,
            extra_sections=extra_sections,
        )

        raw = self._call_claude(prompt, system=CHECKIN_SYSTEM, model=self._model_light)
        if raw:
            # Debug: log raw response to see what the LLM actually returned
            with open("/tmp/checkin_raw.txt", "w") as _f:
                _f.write(raw)
            log.info("CHECKIN RAW dumped to /tmp/checkin_raw.txt (%d bytes)", len(raw))
            playbook = parse_playbook_response(raw)
            if playbook:
                playbook.opening_range = current.opening_range
                self._state.save_playbook(playbook)
                return {"plan_changed": True, "summary": f"Updated to {playbook.day_classification.value}"}

        return {"plan_changed": False, "summary": "No changes (Claude unavailable or no updates)"}

    def handle_exception(
        self,
        exception_type: str, details: dict,
        open_positions: list[dict],
    ) -> dict:
        """Handle an exception event.

        Returns action dict. Falls back to default if Claude unreachable.
        """
        current = self._state.load_playbook()
        prompt = build_exception_prompt(
            exception_type=exception_type,
            details=details,
            current_playbook=current.to_dict() if current else {},
            open_positions=open_positions,
        )

        raw = self._call_claude(prompt, system=EXCEPTION_SYSTEM)
        if raw:
            action = parse_exception_response(raw)
            if action:
                return action

        return default_exception_action(exception_type)

    def eod_review(self, trades_today: list[dict], daily_pnl: float,
                   predicted_day_type: str, actual_day_type: str) -> dict | None:
        """EOD review (3:30 PM, Haiku). Returns grades dict or None."""
        prompt = build_eod_prompt(
            trades_today=trades_today, daily_pnl=daily_pnl,
            day_classification_predicted=predicted_day_type,
            day_classification_actual=actual_day_type,
        )

        raw = self._call_claude(prompt, system=EOD_SYSTEM, model=self._model_light)
        if raw:
            return _extract_json(raw)
        return None
