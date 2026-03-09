# Overnight Hedge Protection — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add EOD protection for naked F&O positions — Claude-decided close/hedge/carry with hard guardrails.

**Architecture:** Standalone `overnight_hedge.py` module called at 3:15 PM from cron. Scans naked F&O, applies deterministic guardrails, sends survivors to Claude for decision, executes close/hedge/carry_naked. Morning unwind removes hedge legs.

**Tech Stack:** Python 3.13, AngelOne SmartAPI, Claude CLI/API (via claude_intel.py), pytest

---

### Task 1: Add config constants

**Files:**
- Modify: `config.py` (append after line ~414, after HEDGE_TIGHTEN block)

**Step 1: Add overnight hedge config block**

Add this block to `config.py` after the existing `HEDGE_TIGHTEN_CLOSER_POINTS_BANKNIFTY` line:

```python
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
```

**Step 2: Verify config loads**

Run: `cd ~/financial-agent-india && python -c "import config; print(config.OVERNIGHT_HEDGE_ENABLED)"`
Expected: `True`

**Step 3: Commit**

```bash
git add config.py
git commit -m "feat: add overnight hedge protection config constants"
```

---

### Task 2: Write test for guardrail logic

**Files:**
- Create: `tests/test_overnight_hedge.py`

**Step 1: Write failing tests for all guardrail rules**

```python
"""Tests for overnight hedge guardrail logic."""

import pytest


def _make_position(instrument="MOMENTUM", direction="bullish", entry_price=100.0,
                   quantity=75, expiry="09MAR2026", **overrides):
    """Helper to create a test position dict."""
    pos = {
        "id": f"test_{instrument.lower()}",
        "strategy": instrument.lower(),
        "symbol": "NIFTY",
        "instrument": instrument,
        "direction": direction,
        "entry_date": "2026-03-06",
        "expiry": expiry,
        "entry_price": entry_price,
        "quantity": quantity,
        "allocated": entry_price * quantity,
        "target_price": entry_price * 1.9,
        "stoploss_price": entry_price * 0.65,
        "status": "open",
    }
    pos.update(overrides)
    return pos


class TestApplyGuardrails:
    """Test deterministic guardrail rules (no Claude needed)."""

    def test_position_in_loss_returns_close(self):
        from overnight_hedge import apply_guardrails
        pos = _make_position(entry_price=100.0)
        result = apply_guardrails(pos, current_price=90.0, vix=15.0, days_to_expiry=10)
        assert result["action"] == "close"
        assert "loss" in result["reason"].lower()

    def test_high_vix_blocks_naked_carry(self):
        from overnight_hedge import apply_guardrails
        pos = _make_position(entry_price=100.0)
        result = apply_guardrails(pos, current_price=200.0, vix=22.0, days_to_expiry=10)
        assert result["action"] in ("close", "hedge")
        assert result["action"] != "carry_naked"

    def test_low_gain_returns_close(self):
        from overnight_hedge import apply_guardrails
        pos = _make_position(entry_price=100.0)
        # 20% gain — below OVERNIGHT_MIN_GAIN_FOR_HEDGE (30%)
        result = apply_guardrails(pos, current_price=120.0, vix=15.0, days_to_expiry=10)
        assert result["action"] == "close"
        assert "gain" in result["reason"].lower()

    def test_near_expiry_returns_close(self):
        from overnight_hedge import apply_guardrails
        pos = _make_position(entry_price=100.0)
        result = apply_guardrails(pos, current_price=200.0, vix=15.0, days_to_expiry=1)
        assert result["action"] == "close"
        assert "expiry" in result["reason"].lower()

    def test_moderate_gain_returns_ask_claude(self):
        from overnight_hedge import apply_guardrails
        pos = _make_position(entry_price=100.0)
        # 40% gain, low VIX, far from expiry -> should ask Claude
        result = apply_guardrails(pos, current_price=140.0, vix=15.0, days_to_expiry=10)
        assert result["action"] == "ask_claude"

    def test_high_gain_returns_ask_claude(self):
        from overnight_hedge import apply_guardrails
        pos = _make_position(entry_price=100.0)
        # 60% gain, low VIX, far from expiry -> should ask Claude
        result = apply_guardrails(pos, current_price=160.0, vix=15.0, days_to_expiry=10)
        assert result["action"] == "ask_claude"

    def test_spread_position_is_skipped(self):
        from overnight_hedge import is_naked_fno
        pos = _make_position(instrument="SPREAD")
        assert is_naked_fno(pos) is False

    def test_condor_position_is_skipped(self):
        from overnight_hedge import is_naked_fno
        pos = _make_position(instrument="CONDOR")
        assert is_naked_fno(pos) is False

    def test_momentum_is_naked(self):
        from overnight_hedge import is_naked_fno
        pos = _make_position(instrument="MOMENTUM")
        assert is_naked_fno(pos) is True

    def test_opt_is_naked(self):
        from overnight_hedge import is_naked_fno
        pos = _make_position(instrument="OPT")
        assert is_naked_fno(pos) is True

    def test_fut_is_naked(self):
        from overnight_hedge import is_naked_fno
        pos = _make_position(instrument="FUT")
        assert is_naked_fno(pos) is True

    def test_already_hedged_position_is_skipped(self):
        from overnight_hedge import is_naked_fno
        pos = _make_position(instrument="MOMENTUM", hedge_leg={"strike": 24800})
        assert is_naked_fno(pos) is False


class TestClaudeOverride:
    """Test that Claude's carry_naked is overridden when gain < 50%."""

    def test_carry_naked_below_threshold_becomes_hedge(self):
        from overnight_hedge import enforce_carry_naked_threshold
        decision = {"action": "carry_naked", "reasoning": "strong trend"}
        result = enforce_carry_naked_threshold(decision, gain_pct=0.40)
        assert result["action"] == "hedge"

    def test_carry_naked_above_threshold_allowed(self):
        from overnight_hedge import enforce_carry_naked_threshold
        decision = {"action": "carry_naked", "reasoning": "strong trend"}
        result = enforce_carry_naked_threshold(decision, gain_pct=0.55)
        assert result["action"] == "carry_naked"

    def test_close_decision_unchanged(self):
        from overnight_hedge import enforce_carry_naked_threshold
        decision = {"action": "close", "reasoning": "weak"}
        result = enforce_carry_naked_threshold(decision, gain_pct=0.10)
        assert result["action"] == "close"


class TestHedgeCost:
    """Test hedge cost cap logic."""

    def test_hedge_too_expensive_falls_back_to_close(self):
        from overnight_hedge import check_hedge_cost
        # Position value = 100 * 75 = 7500, hedge cost = 100 (1.3%) > 1% cap
        result = check_hedge_cost(position_value=7500.0, hedge_premium=100.0, quantity=75)
        assert result["affordable"] is False

    def test_hedge_within_budget(self):
        from overnight_hedge import check_hedge_cost
        # Position value = 7500, hedge cost = 40 (0.53%) < 1% cap
        result = check_hedge_cost(position_value=7500.0, hedge_premium=40.0, quantity=75)
        assert result["affordable"] is True
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_overnight_hedge.py -v 2>&1 | head -40`
Expected: All tests FAIL with `ModuleNotFoundError: No module named 'overnight_hedge'`

**Step 3: Commit**

```bash
git add tests/test_overnight_hedge.py
git commit -m "test: add failing tests for overnight hedge guardrails"
```

---

### Task 3: Implement core guardrail functions in `overnight_hedge.py`

**Files:**
- Create: `overnight_hedge.py`

**Step 1: Implement `is_naked_fno`, `apply_guardrails`, `enforce_carry_naked_threshold`, `check_hedge_cost`**

```python
"""Overnight Hedge Protection — EOD scan for naked F&O positions.

Runs at 3:15 PM IST. For each naked position:
1. Apply hard guardrails (deterministic close/hedge rules)
2. Send surviving positions to Claude for decision
3. Execute: close, add protective hedge leg, or approve naked carry

Usage:
    python overnight_hedge.py          # full scan
    python overnight_hedge.py --dry    # scan only, no execution
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import config

logger = logging.getLogger("paper_trade")

IST = timezone(timedelta(hours=5, minutes=30))


def is_naked_fno(pos: dict) -> bool:
    """Check if a position is a naked (unhedged) F&O position."""
    if pos.get("status") != "open":
        return False
    if pos.get("hedge_leg"):
        return False
    instrument = pos.get("instrument", "")
    return instrument in config.OVERNIGHT_NAKED_INSTRUMENTS


def _calc_gain_pct(entry_price: float, current_price: float, direction: str) -> float:
    """Calculate gain as a fraction (0.30 = 30%)."""
    if direction == "bearish":
        return (entry_price - current_price) / entry_price
    return (current_price - entry_price) / entry_price


def apply_guardrails(pos: dict, current_price: float, vix: float,
                     days_to_expiry: int) -> dict:
    """Apply deterministic guardrails. Returns action dict.

    Returns:
        {"action": "close"|"hedge"|"ask_claude", "reason": str}
    """
    entry_price = pos["entry_price"]
    direction = pos.get("direction", "bullish")
    gain_pct = _calc_gain_pct(entry_price, current_price, direction)

    # Rule 1: Position in loss -> CLOSE
    if gain_pct < 0:
        return {"action": "close", "reason": f"Position in loss ({gain_pct:.1%}). No overnight carry for losers."}

    # Rule 2: Near expiry -> CLOSE
    if days_to_expiry <= config.OVERNIGHT_EXPIRY_CLOSE_DAYS:
        return {"action": "close", "reason": f"Expiry in {days_to_expiry} day(s). Theta/gamma risk too high."}

    # Rule 3: Low gain -> CLOSE (not worth hedging)
    if gain_pct < config.OVERNIGHT_MIN_GAIN_FOR_HEDGE:
        return {"action": "close", "reason": f"Gain {gain_pct:.1%} below {config.OVERNIGHT_MIN_GAIN_FOR_HEDGE:.0%} threshold. Not worth hedging."}

    # Rule 4: High VIX -> HEDGE or CLOSE only (no naked carry)
    if vix > config.OVERNIGHT_VIX_NAKED_BLOCK:
        return {"action": "ask_claude", "reason": f"VIX {vix:.1f} > {config.OVERNIGHT_VIX_NAKED_BLOCK}. Claude decides hedge vs close (no naked carry)."}

    # Passed all hard rules -> ask Claude
    return {"action": "ask_claude", "reason": f"Gain {gain_pct:.1%}, VIX {vix:.1f}, {days_to_expiry} DTE. Eligible for Claude decision."}


def enforce_carry_naked_threshold(decision: dict, gain_pct: float) -> dict:
    """Override Claude's carry_naked if gain below threshold."""
    if decision["action"] == "carry_naked" and gain_pct < config.OVERNIGHT_MIN_GAIN_FOR_NAKED_CARRY:
        return {
            "action": "hedge",
            "reasoning": f"OVERRIDE: Claude said carry_naked but gain {gain_pct:.1%} < {config.OVERNIGHT_MIN_GAIN_FOR_NAKED_CARRY:.0%}. Forcing hedge.",
        }
    return decision


def check_hedge_cost(position_value: float, hedge_premium: float, quantity: int) -> dict:
    """Check if hedge cost is within budget."""
    total_cost = hedge_premium * quantity
    cost_pct = total_cost / position_value if position_value > 0 else 1.0
    return {
        "affordable": cost_pct <= config.OVERNIGHT_HEDGE_MAX_COST_PCT,
        "cost": round(total_cost, 2),
        "cost_pct": round(cost_pct, 4),
    }
```

**Step 2: Run tests**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_overnight_hedge.py -v 2>&1 | tail -20`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add overnight_hedge.py
git commit -m "feat: implement overnight hedge guardrail functions"
```

---

### Task 4: Add Claude overnight decision prompt to `claude_intel.py`

**Files:**
- Modify: `claude_intel.py` (add new function after `evaluate_exit` ~line 620)

**Step 1: Write test for Claude decision parsing**

Add to `tests/test_overnight_hedge.py`:

```python
class TestParseClaudeOvernightDecision:
    """Test parsing Claude's overnight hedge response."""

    def test_parse_valid_close(self):
        from overnight_hedge import parse_overnight_decision
        raw = '{"action": "close", "reasoning": "Weak momentum, close before gap risk"}'
        result = parse_overnight_decision(raw)
        assert result["action"] == "close"

    def test_parse_valid_hedge(self):
        from overnight_hedge import parse_overnight_decision
        raw = '{"action": "hedge", "reasoning": "Strong trend but overnight risk"}'
        result = parse_overnight_decision(raw)
        assert result["action"] == "hedge"

    def test_parse_valid_carry_naked(self):
        from overnight_hedge import parse_overnight_decision
        raw = '{"action": "carry_naked", "reasoning": "Very strong trend, low vol"}'
        result = parse_overnight_decision(raw)
        assert result["action"] == "carry_naked"

    def test_parse_invalid_defaults_to_hedge(self):
        from overnight_hedge import parse_overnight_decision
        raw = "I think you should hedge this position"
        result = parse_overnight_decision(raw)
        assert result["action"] == "hedge"

    def test_parse_none_defaults_to_close(self):
        from overnight_hedge import parse_overnight_decision
        result = parse_overnight_decision(None)
        assert result["action"] == "close"
```

**Step 2: Run test to verify it fails**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_overnight_hedge.py::TestParseClaudeOvernightDecision -v`
Expected: FAIL — `parse_overnight_decision` doesn't exist yet

**Step 3: Implement `evaluate_overnight` in `claude_intel.py` and `parse_overnight_decision` in `overnight_hedge.py`**

Add to `claude_intel.py` after `evaluate_exit`:

```python
def evaluate_overnight(pos: dict, current_price: float, gain_pct: float,
                       vix: float, regime: str, market_context: str = "") -> dict:
    """Ask Claude whether to close, hedge, or carry a naked F&O position overnight.

    Returns: {"action": "close"|"hedge"|"carry_naked", "reasoning": str}
    """
    prompt = f"""OVERNIGHT RISK DECISION

You have a naked F&O position going into close. Decide: close, hedge, or carry_naked.

POSITION:
- Symbol: {pos['symbol']}
- Instrument: {pos.get('instrument')}
- Direction: {pos['direction']}
- Entry: {pos['entry_price']} -> Current: {current_price}
- Gain: {gain_pct:.1%}
- Strike: {pos.get('strike', 'N/A')}
- Expiry: {pos.get('expiry', 'N/A')}
- Option type: {pos.get('option_type', 'N/A')}

MARKET:
- VIX: {vix:.1f}
- Regime: {regime}
{market_context}

RULES:
- "close": exit the position now. Use when signals weakening or risk not worth reward.
- "hedge": add a protective OTM option (we handle strike selection). Use when position has merit but overnight gap risk is real.
- "carry_naked": no hedge needed. ONLY if position is deeply profitable (>50%), trend is strong, and no event risk overnight.

Respond with JSON only:
{{"action": "close|hedge|carry_naked", "reasoning": "1-2 sentence explanation"}}"""

    response = _call_claude(prompt, max_tokens=256)
    parsed = _parse_json(response) if response else None

    _save_decision_log("overnight", pos.get("symbol", "UNKNOWN"), prompt,
                       response or "", parsed or {}, {"gain_pct": gain_pct, "vix": vix})

    if parsed and parsed.get("action") in ("close", "hedge", "carry_naked"):
        return {"action": parsed["action"], "reasoning": parsed.get("reasoning", "")}

    # Default: hedge if profitable, close if Claude fails
    return {"action": "hedge" if gain_pct > 0 else "close",
            "reasoning": "Claude response parse failed — defaulting to safe action"}
```

Add to `overnight_hedge.py`:

```python
def parse_overnight_decision(raw_response: str | None) -> dict:
    """Parse Claude's overnight decision response. Fail-safe defaults."""
    if raw_response is None:
        return {"action": "close", "reasoning": "No Claude response — closing as fail-safe"}

    try:
        import json
        # Try to extract JSON
        cleaned = raw_response
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
        if parsed.get("action") in ("close", "hedge", "carry_naked"):
            return {"action": parsed["action"], "reasoning": parsed.get("reasoning", "")}
    except (json.JSONDecodeError, AttributeError):
        pass

    # Default to hedge (safer than naked, less wasteful than close)
    return {"action": "hedge", "reasoning": "Could not parse Claude response — defaulting to hedge"}
```

**Step 4: Run tests**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_overnight_hedge.py -v 2>&1 | tail -25`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add claude_intel.py overnight_hedge.py tests/test_overnight_hedge.py
git commit -m "feat: add Claude overnight decision evaluation and parsing"
```

---

### Task 5: Implement hedge execution logic

**Files:**
- Modify: `overnight_hedge.py` (add execution functions)

**Step 1: Write test for hedge leg creation**

Add to `tests/test_overnight_hedge.py`:

```python
class TestBuildHedgeLeg:
    """Test protective hedge leg construction."""

    def test_bullish_position_gets_put_hedge(self):
        from overnight_hedge import build_hedge_leg
        pos = _make_position(instrument="MOMENTUM", direction="bullish", strike=25000)
        leg = build_hedge_leg(pos, spot_price=25200.0)
        assert leg["option_type"] == "PE"
        assert leg["strike"] == 25200 - 250  # OTM_POINTS_NIFTY

    def test_bearish_position_gets_call_hedge(self):
        from overnight_hedge import build_hedge_leg
        pos = _make_position(instrument="MOMENTUM", direction="bearish", strike=25000)
        leg = build_hedge_leg(pos, spot_price=24800.0)
        assert leg["option_type"] == "CE"
        assert leg["strike"] == 24800 + 250

    def test_banknifty_uses_wider_otm(self):
        from overnight_hedge import build_hedge_leg
        pos = _make_position(instrument="MOMENTUM", direction="bullish",
                             strike=52000, symbol="BANKNIFTY")
        leg = build_hedge_leg(pos, spot_price=52200.0)
        assert leg["option_type"] == "PE"
        assert leg["strike"] == 52200 - 500

    def test_futures_bullish_gets_put(self):
        from overnight_hedge import build_hedge_leg
        pos = _make_position(instrument="FUT", direction="bullish", symbol="NIFTY")
        leg = build_hedge_leg(pos, spot_price=25000.0)
        assert leg["option_type"] == "PE"


class TestTightenStopLoss:
    """Test stop-loss tightening for naked carry."""

    def test_tighten_bullish_raises_stop(self):
        from overnight_hedge import tighten_stop_loss
        pos = _make_position(entry_price=100.0, stoploss_price=65.0)
        current = 160.0  # 60% gain
        new_sl = tighten_stop_loss(pos, current)
        # Should tighten by 20%: new SL = current - (current - old_sl) * (1 - 0.20)
        # = 160 - (160 - 65) * 0.80 = 160 - 76 = 84
        assert new_sl > 65.0
        assert new_sl < current
```

**Step 2: Run to verify fail**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_overnight_hedge.py::TestBuildHedgeLeg -v`
Expected: FAIL

**Step 3: Implement `build_hedge_leg`, `tighten_stop_loss`, and execution functions**

Add to `overnight_hedge.py`:

```python
def build_hedge_leg(pos: dict, spot_price: float) -> dict:
    """Build a protective option leg for the position.

    Bullish positions get OTM puts, bearish get OTM calls.
    Returns hedge leg spec (not yet executed).
    """
    symbol = pos.get("symbol", "NIFTY")
    direction = pos.get("direction", "bullish")

    if symbol == "BANKNIFTY":
        otm_points = config.OVERNIGHT_HEDGE_OTM_POINTS_BANKNIFTY
    else:
        otm_points = config.OVERNIGHT_HEDGE_OTM_POINTS_NIFTY

    # Round spot to nearest 50 (Nifty) or 100 (BankNifty)
    rounding = 100 if symbol == "BANKNIFTY" else 50
    rounded_spot = round(spot_price / rounding) * rounding

    if direction == "bullish":
        # Buy protective put
        option_type = "PE"
        strike = rounded_spot - otm_points
    else:
        # Buy protective call
        option_type = "CE"
        strike = rounded_spot + otm_points

    return {
        "option_type": option_type,
        "strike": strike,
        "symbol": symbol,
        "quantity": pos.get("quantity", 75),
    }


def tighten_stop_loss(pos: dict, current_price: float) -> float:
    """Tighten stop-loss by OVERNIGHT_STOP_TIGHTEN_PCT for naked carry approval.

    Moves stop-loss closer to current price by reducing the gap by tighten %.
    """
    old_sl = pos.get("stoploss_price", 0)
    gap = current_price - old_sl
    new_gap = gap * (1 - config.OVERNIGHT_STOP_TIGHTEN_PCT)
    return round(current_price - new_gap, 2)


def execute_close(portfolio: dict, pos: dict, smart_api, current_price: float, reason: str) -> dict:
    """Close a position via paper_trade.close_position."""
    from paper_trade import close_position
    return close_position(portfolio, pos, current_price, f"overnight_risk: {reason}",
                          smart_api=smart_api)


def execute_hedge(portfolio: dict, pos: dict, smart_api, hedge_leg: dict,
                  hedge_premium: float) -> bool:
    """Add a protective hedge leg to the position.

    Returns True if hedge was added, False if it failed.
    """
    pos["hedge_leg"] = {
        "option_type": hedge_leg["option_type"],
        "strike": hedge_leg["strike"],
        "premium": hedge_premium,
        "quantity": hedge_leg["quantity"],
        "cost": round(hedge_premium * hedge_leg["quantity"], 2),
        "added_date": datetime.now(IST).strftime("%Y-%m-%d"),
    }
    return True


def execute_carry_naked(pos: dict, current_price: float, reasoning: str) -> None:
    """Approve naked carry: tighten stop-loss, log reasoning."""
    new_sl = tighten_stop_loss(pos, current_price)
    pos["stoploss_price"] = new_sl
    pos["overnight_carry_approved"] = {
        "date": datetime.now(IST).strftime("%Y-%m-%d"),
        "reasoning": reasoning,
        "tightened_sl": new_sl,
    }
```

**Step 4: Run all tests**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_overnight_hedge.py -v 2>&1 | tail -30`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add overnight_hedge.py tests/test_overnight_hedge.py
git commit -m "feat: implement hedge execution, stop tightening, and hedge leg builder"
```

---

### Task 6: Implement main scan orchestrator

**Files:**
- Modify: `overnight_hedge.py` (add `run_overnight_hedge_scan` and `main`)

**Step 1: Implement the orchestrator and CLI**

Add to `overnight_hedge.py`:

```python
def _format_telegram_message(results: list[dict], vix: float, regime: str) -> str:
    """Format overnight hedge results for Telegram (HTML parse_mode)."""
    closed = [r for r in results if r["action"] == "close"]
    hedged = [r for r in results if r["action"] == "hedge"]
    carried = [r for r in results if r["action"] == "carry_naked"]

    lines = ["<b>OVERNIGHT HEDGE SCAN</b>\n"]

    if closed:
        lines.append(f"<b>CLOSED ({len(closed)}):</b>")
        for r in closed:
            lines.append(f"- {r['pos_id']} | {r['reason']}")
        lines.append("")

    if hedged:
        lines.append(f"<b>HEDGED ({len(hedged)}):</b>")
        for r in hedged:
            cost_str = f"₹{r.get('hedge_cost', 0):,.0f}" if r.get('hedge_cost') else ""
            lines.append(f"- {r['pos_id']} | +{r.get('gain_pct', 0):.0%} | {r.get('hedge_desc', '')}")
            if cost_str:
                lines.append(f"  Cost: {cost_str} ({r.get('cost_pct', 0):.1%} of position)")
        lines.append("")

    if carried:
        lines.append(f"<b>CARRY NAKED ({len(carried)}):</b>")
        for r in carried:
            lines.append(f"- {r['pos_id']} | +{r.get('gain_pct', 0):.0%} | SL tightened")
            lines.append(f"  Reason: {r.get('reasoning', '')[:100]}")
        lines.append("")

    if not results:
        lines.append("No naked F&O positions found.")

    lines.append(f"VIX: {vix:.1f} | Regime: {regime}")
    return "\n".join(lines)


def run_overnight_hedge_scan(smart_api, dry_run: bool = False) -> list[dict]:
    """Main orchestrator: scan naked F&O, apply guardrails, ask Claude, execute.

    Args:
        smart_api: authenticated SmartConnect session
        dry_run: if True, log decisions but don't execute

    Returns:
        list of decision dicts [{pos_id, action, reason, ...}]
    """
    from paper_trade import (load_portfolio, save_portfolio, get_ltp, get_ltp_nfo,
                             _telegram_send, calc_pnl_pct)
    from regime import classify_regime
    from connect import get_session

    if not config.OVERNIGHT_HEDGE_ENABLED:
        logger.info("Overnight hedge protection disabled")
        return []

    portfolio = load_portfolio()
    open_positions = [p for p in portfolio["positions"] if p["status"] == "open"]
    naked_positions = [p for p in open_positions if is_naked_fno(p)]

    if not naked_positions:
        logger.info("No naked F&O positions — skipping overnight hedge scan")
        return []

    logger.info("Overnight hedge scan: %d naked positions found", len(naked_positions))

    # Get VIX and regime
    try:
        import yfinance as yf
        vix_data = yf.Ticker("^INDIAVIX").history(period="1d")
        vix = float(vix_data["Close"].iloc[-1]) if not vix_data.empty else 15.0
    except Exception:
        vix = 15.0
        logger.warning("VIX fetch failed, using default 15.0")

    try:
        regime_result = classify_regime(smart_api, vix=vix)
        regime = regime_result.get("regime", "UNKNOWN")
    except Exception:
        regime = "UNKNOWN"

    results = []
    for pos in naked_positions:
        pos_id = pos.get("id", pos.get("symbol", "???"))
        instrument = pos.get("instrument", "")
        direction = pos.get("direction", "bullish")

        # Fetch current price
        try:
            if instrument in ("OPT", "MOMENTUM"):
                # Need NFO token for options
                nfo_token = pos.get("nfo_token", pos.get("token", ""))
                trading_sym = pos.get("trading_symbol", "")
                if nfo_token and trading_sym:
                    current_price = get_ltp_nfo(smart_api, trading_sym, nfo_token)
                else:
                    current_price = None
            elif instrument == "FUT":
                fut_token = pos.get("nfo_token", pos.get("token", ""))
                trading_sym = pos.get("trading_symbol", "")
                current_price = get_ltp_nfo(smart_api, trading_sym, fut_token) if fut_token else None
            else:
                current_price = None

            if current_price is None:
                logger.warning("Could not fetch LTP for %s — closing as fail-safe", pos_id)
                if not dry_run:
                    execute_close(portfolio, pos, smart_api, pos["entry_price"], "LTP fetch failed")
                results.append({"pos_id": pos_id, "action": "close", "reason": "LTP fetch failed"})
                continue

        except Exception as e:
            logger.warning("LTP error for %s: %s — closing as fail-safe", pos_id, e)
            if not dry_run:
                execute_close(portfolio, pos, smart_api, pos["entry_price"], f"LTP error: {e}")
            results.append({"pos_id": pos_id, "action": "close", "reason": f"LTP error: {e}"})
            continue

        gain_pct = _calc_gain_pct(pos["entry_price"], current_price, direction)

        # Calculate days to expiry
        try:
            from paper_trade import _add_trading_days, _today_ist
            today = _today_ist()
            expiry_str = pos.get("expiry", "")
            if expiry_str:
                from datetime import datetime as dt
                expiry_date = dt.strptime(expiry_str, "%d%b%Y").date()
                today_date = dt.strptime(today, "%Y-%m-%d").date()
                days_to_expiry = (expiry_date - today_date).days
            else:
                days_to_expiry = 30  # assume far expiry if unknown
        except Exception:
            days_to_expiry = 30

        # Apply guardrails
        guardrail = apply_guardrails(pos, current_price, vix, days_to_expiry)

        if guardrail["action"] == "close":
            if not dry_run:
                execute_close(portfolio, pos, smart_api, current_price, guardrail["reason"])
            results.append({"pos_id": pos_id, "action": "close", "reason": guardrail["reason"],
                            "gain_pct": gain_pct})
            continue

        if guardrail["action"] == "ask_claude":
            # Ask Claude
            try:
                from claude_intel import evaluate_overnight
                high_vix = vix > config.OVERNIGHT_VIX_NAKED_BLOCK
                decision = evaluate_overnight(pos, current_price, gain_pct, vix, regime)

                # Enforce carry_naked threshold
                decision = enforce_carry_naked_threshold(decision, gain_pct)

                # If VIX high, block carry_naked even if Claude says so
                if high_vix and decision["action"] == "carry_naked":
                    decision = {"action": "hedge",
                                "reasoning": f"VIX {vix:.1f} too high for naked carry — overriding to hedge"}

            except Exception as e:
                logger.warning("Claude overnight eval failed for %s: %s", pos_id, e)
                decision = {"action": "hedge" if gain_pct > 0.30 else "close",
                            "reasoning": f"Claude failed: {e}"}

            action = decision["action"]
            reasoning = decision.get("reasoning", "")

            if action == "close":
                if not dry_run:
                    execute_close(portfolio, pos, smart_api, current_price, reasoning)
                results.append({"pos_id": pos_id, "action": "close", "reason": reasoning,
                                "gain_pct": gain_pct})

            elif action == "hedge":
                # Build and price the hedge leg
                try:
                    # Get spot price for hedge strike calculation
                    eq_token = pos.get("token", "")
                    spot_price = get_ltp(smart_api, pos["symbol"], eq_token) if eq_token else current_price

                    hedge_leg = build_hedge_leg(pos, spot_price or current_price)

                    # Try to get hedge option premium
                    from paper_trade import resolve_option_contract
                    expiry_str = pos.get("expiry", "")
                    contract = resolve_option_contract(
                        smart_api, hedge_leg["symbol"],
                        hedge_leg["strike"], expiry_str
                    ) if expiry_str else None

                    if contract:
                        hedge_premium = get_ltp_nfo(smart_api, contract["trading_symbol"],
                                                     contract["token"])
                        time.sleep(config.API_DELAY)
                    else:
                        hedge_premium = None

                    if hedge_premium is None:
                        logger.warning("Could not price hedge for %s — closing", pos_id)
                        if not dry_run:
                            execute_close(portfolio, pos, smart_api, current_price,
                                          "Hedge pricing failed")
                        results.append({"pos_id": pos_id, "action": "close",
                                        "reason": "Hedge pricing failed", "gain_pct": gain_pct})
                        continue

                    # Check cost cap
                    position_value = pos["entry_price"] * abs(pos["quantity"])
                    cost_check = check_hedge_cost(position_value, hedge_premium,
                                                  hedge_leg["quantity"])

                    if not cost_check["affordable"]:
                        logger.info("Hedge too expensive for %s (%.1f%%) — closing",
                                    pos_id, cost_check["cost_pct"] * 100)
                        if not dry_run:
                            execute_close(portfolio, pos, smart_api, current_price,
                                          f"Hedge too expensive ({cost_check['cost_pct']:.1%})")
                        results.append({"pos_id": pos_id, "action": "close",
                                        "reason": f"Hedge too expensive ({cost_check['cost_pct']:.1%})",
                                        "gain_pct": gain_pct})
                        continue

                    # Execute hedge
                    if not dry_run:
                        execute_hedge(portfolio, pos, smart_api, hedge_leg, hedge_premium)

                    hedge_desc = f"bought {hedge_leg['strike']} {hedge_leg['option_type']} @ ₹{hedge_premium:.0f}"
                    results.append({
                        "pos_id": pos_id, "action": "hedge", "reason": reasoning,
                        "gain_pct": gain_pct, "hedge_desc": hedge_desc,
                        "hedge_cost": cost_check["cost"], "cost_pct": cost_check["cost_pct"],
                    })

                except Exception as e:
                    logger.warning("Hedge execution failed for %s: %s — closing", pos_id, e)
                    if not dry_run:
                        execute_close(portfolio, pos, smart_api, current_price, f"Hedge failed: {e}")
                    results.append({"pos_id": pos_id, "action": "close",
                                    "reason": f"Hedge failed: {e}", "gain_pct": gain_pct})

            elif action == "carry_naked":
                if not dry_run:
                    execute_carry_naked(pos, current_price, reasoning)
                results.append({"pos_id": pos_id, "action": "carry_naked",
                                "reason": reasoning, "gain_pct": gain_pct,
                                "reasoning": reasoning})

    # Save portfolio and send Telegram
    if not dry_run and results:
        save_portfolio(portfolio)

    if results:
        msg = _format_telegram_message(results, vix, regime)
        if not dry_run:
            _telegram_send(msg)
        else:
            print(msg)

    return results


def main():
    parser = argparse.ArgumentParser(description="Overnight hedge protection for naked F&O")
    parser.add_argument("--dry", action="store_true", help="Scan only, no execution")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from connect import get_session
    smart_api = get_session()

    results = run_overnight_hedge_scan(smart_api, dry_run=args.dry)
    logger.info("Overnight hedge scan complete: %d decisions", len(results))


if __name__ == "__main__":
    main()
```

**Step 2: Verify module runs without errors**

Run: `cd ~/financial-agent-india && python -c "from overnight_hedge import run_overnight_hedge_scan; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add overnight_hedge.py
git commit -m "feat: implement overnight hedge scan orchestrator with CLI"
```

---

### Task 7: Add morning unwind logic to `paper_trade.py`

**Files:**
- Modify: `paper_trade.py` (add hedge leg unwind in monitor flow)

**Step 1: Write test for morning unwind**

Add to `tests/test_overnight_hedge.py`:

```python
class TestMorningUnwind:
    """Test that hedge legs get unwound on the next trading morning."""

    def test_position_with_hedge_leg_detected(self):
        from overnight_hedge import find_positions_with_hedge_legs
        positions = [
            _make_position(instrument="MOMENTUM", hedge_leg={"strike": 24800, "premium": 45}),
            _make_position(instrument="MOMENTUM"),
        ]
        hedged = find_positions_with_hedge_legs(positions)
        assert len(hedged) == 1

    def test_unwind_records_insurance_cost(self):
        from overnight_hedge import calc_hedge_pnl
        # Bought hedge at 45, current premium is 30 (lost 15)
        result = calc_hedge_pnl(entry_premium=45.0, exit_premium=30.0, quantity=75)
        assert result["pnl"] < 0
        assert result["label"] == "insurance_cost"

    def test_unwind_records_hedge_saved(self):
        from overnight_hedge import calc_hedge_pnl
        # Bought hedge at 45, gap against us, hedge now worth 120
        result = calc_hedge_pnl(entry_premium=45.0, exit_premium=120.0, quantity=75)
        assert result["pnl"] > 0
        assert result["label"] == "hedge_saved"
```

**Step 2: Run to verify fail**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_overnight_hedge.py::TestMorningUnwind -v`
Expected: FAIL

**Step 3: Implement morning unwind functions**

Add to `overnight_hedge.py`:

```python
def find_positions_with_hedge_legs(positions: list[dict]) -> list[dict]:
    """Find open positions that have a hedge_leg to unwind."""
    return [p for p in positions if p.get("status") == "open" and p.get("hedge_leg")]


def calc_hedge_pnl(entry_premium: float, exit_premium: float, quantity: int) -> dict:
    """Calculate P&L from unwinding a hedge leg."""
    pnl = round((exit_premium - entry_premium) * quantity, 2)
    return {
        "pnl": pnl,
        "label": "hedge_saved" if pnl > 0 else "insurance_cost",
    }


def unwind_hedge_legs(smart_api, portfolio: dict) -> list[dict]:
    """Unwind all hedge legs on open positions. Called at 9:30 AM next trading day.

    Returns list of unwind results.
    """
    from paper_trade import get_ltp_nfo, resolve_option_contract, _telegram_send

    hedged = find_positions_with_hedge_legs(portfolio["positions"])
    if not hedged:
        return []

    results = []
    for pos in hedged:
        leg = pos["hedge_leg"]
        pos_id = pos.get("id", pos.get("symbol", "???"))

        try:
            # Get current premium for the hedge option
            contract = resolve_option_contract(
                smart_api, leg["symbol"] if "symbol" in leg else pos["symbol"],
                leg["strike"], pos.get("expiry", "")
            )
            if contract:
                exit_premium = get_ltp_nfo(smart_api, contract["trading_symbol"],
                                            contract["token"])
            else:
                exit_premium = None

            if exit_premium is None:
                logger.warning("Could not price hedge unwind for %s — will retry", pos_id)
                continue

            pnl_result = calc_hedge_pnl(leg["premium"], exit_premium, leg["quantity"])

            # Record hedge P&L in the position
            pos["hedge_pnl_history"] = pos.get("hedge_pnl_history", [])
            pos["hedge_pnl_history"].append({
                "date": leg.get("added_date", ""),
                "entry_premium": leg["premium"],
                "exit_premium": exit_premium,
                "pnl": pnl_result["pnl"],
                "label": pnl_result["label"],
            })

            # Remove hedge leg
            del pos["hedge_leg"]
            # Also clean up overnight carry approval if present
            pos.pop("overnight_carry_approved", None)

            results.append({
                "pos_id": pos_id,
                "pnl": pnl_result["pnl"],
                "label": pnl_result["label"],
                "entry": leg["premium"],
                "exit": exit_premium,
            })

            logger.info("Hedge unwind %s: %s ₹%.0f (bought %.0f, sold %.0f)",
                        pos_id, pnl_result["label"], pnl_result["pnl"],
                        leg["premium"], exit_premium)

        except Exception as e:
            logger.warning("Hedge unwind failed for %s: %s", pos_id, e)

    if results:
        lines = ["<b>MORNING HEDGE UNWIND</b>\n"]
        for r in results:
            emoji = "+" if r["pnl"] > 0 else ""
            lines.append(f"- {r['pos_id']} | {r['label']} | {emoji}₹{r['pnl']:,.0f}")
        _telegram_send("\n".join(lines))

    return results
```

**Step 4: Add unwind call to paper_trade.py monitor flow**

Find the monitor section in `paper_trade.py` (around the start of the monitor command handler). Add at the top of the monitor logic, before position checks:

```python
# Unwind overnight hedge legs (morning only, 9:25-9:35 AM)
now_ist = datetime.now(IST)
if now_ist.hour == 9 and 25 <= now_ist.minute <= 35:
    try:
        from overnight_hedge import unwind_hedge_legs
        unwind_results = unwind_hedge_legs(smart_api, portfolio)
        if unwind_results:
            save_portfolio(portfolio)
            logger.info("Morning hedge unwind: %d legs closed", len(unwind_results))
    except Exception as e:
        logger.warning("Morning hedge unwind failed: %s", e)
```

**Step 5: Run all tests**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_overnight_hedge.py -v 2>&1 | tail -30`
Expected: All PASS

**Step 6: Commit**

```bash
git add overnight_hedge.py paper_trade.py tests/test_overnight_hedge.py
git commit -m "feat: add morning hedge unwind with P&L tracking"
```

---

### Task 8: Update cron script

**Files:**
- Modify: `scripts/paper_trade_cron.sh`

**Step 1: Add the 3:15 PM hedge scan entry**

Insert before the `# --- 3:35 PM: EOD wrap` block (around line 116):

```bash
# --- 3:15-3:25 PM: Overnight hedge scan ---
if [ "$HOUR" -eq 15 ] && [ "$MIN" -ge 13 ] && [ "$MIN" -le 27 ]; then
    echo "[HEDGE] Overnight hedge scan" >> "$LOG"
    python overnight_hedge.py >> "$LOG" 2>&1
    HEDGE_EXIT=$?
    if [ "$HEDGE_EXIT" -ne 0 ]; then
        echo "[HEDGE] ERROR: overnight hedge scan failed (exit $HEDGE_EXIT)" >> "$LOG"
    fi
fi
```

**Step 2: Verify cron script syntax**

Run: `bash -n ~/financial-agent-india/scripts/paper_trade_cron.sh && echo "Syntax OK"`
Expected: `Syntax OK`

**Step 3: Commit**

```bash
git add scripts/paper_trade_cron.sh
git commit -m "feat: add 3:15 PM overnight hedge scan to cron"
```

---

### Task 9: Integration test — full scan dry run

**Files:**
- Modify: `tests/test_overnight_hedge.py` (add integration test)

**Step 1: Write integration test with mocked SmartAPI**

Add to `tests/test_overnight_hedge.py`:

```python
class TestIntegrationDryRun:
    """Integration test: full scan with mocked dependencies."""

    def test_full_scan_with_naked_momentum(self, monkeypatch, tmp_path):
        import paper_trade
        import overnight_hedge
        import config

        # Point portfolio to temp
        pf_file = tmp_path / "portfolio.json"
        monkeypatch.setattr(paper_trade, "PORTFOLIO_FILE", pf_file)
        monkeypatch.setattr(paper_trade, "PORTFOLIO_DIR", tmp_path)

        # Create portfolio with one naked momentum position
        portfolio = {
            "capital": 500_000,
            "available_capital": 450_000,
            "positions": [
                _make_position(instrument="MOMENTUM", direction="bullish",
                               entry_price=100.0, quantity=75, expiry="30MAR2026",
                               id="momentum_test", status="open",
                               nfo_token="12345", trading_symbol="NIFTY30MAR25000CE"),
            ],
            "closed_trades": [],
        }
        pf_file.write_text(json.dumps(portfolio))

        # Mock SmartAPI
        class MockSmartAPI:
            def ltpData(self, exchange, symbol, token):
                return {"data": {"ltp": 85.0}}  # position in loss

        # Mock Telegram
        sent = []
        monkeypatch.setattr(paper_trade, "_telegram_send", lambda msg, **kw: sent.append(msg))

        # Run scan
        results = overnight_hedge.run_overnight_hedge_scan(MockSmartAPI(), dry_run=True)

        assert len(results) == 1
        assert results[0]["action"] == "close"
        assert "loss" in results[0]["reason"].lower()

    def test_full_scan_no_naked_positions(self, monkeypatch, tmp_path):
        import paper_trade

        pf_file = tmp_path / "portfolio.json"
        monkeypatch.setattr(paper_trade, "PORTFOLIO_FILE", pf_file)
        monkeypatch.setattr(paper_trade, "PORTFOLIO_DIR", tmp_path)

        portfolio = {
            "capital": 500_000,
            "available_capital": 500_000,
            "positions": [
                _make_position(instrument="CONDOR", status="open"),
            ],
            "closed_trades": [],
        }
        pf_file.write_text(json.dumps(portfolio))

        import overnight_hedge
        results = overnight_hedge.run_overnight_hedge_scan(None, dry_run=True)
        assert len(results) == 0
```

**Step 2: Run integration tests**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_overnight_hedge.py::TestIntegrationDryRun -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/test_overnight_hedge.py
git commit -m "test: add integration tests for overnight hedge scan"
```

---

### Task 10: Final verification

**Step 1: Run full test suite**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_overnight_hedge.py -v`
Expected: All tests PASS

**Step 2: Verify module runs standalone**

Run: `cd ~/financial-agent-india && python overnight_hedge.py --dry 2>&1 | head -10`
Expected: Either "No naked F&O positions" or a dry-run output (no errors)

**Step 3: Verify cron script**

Run: `bash -n ~/financial-agent-india/scripts/paper_trade_cron.sh && echo "OK"`
Expected: `OK`

**Step 4: Verify imports**

Run: `cd ~/financial-agent-india && python -c "from overnight_hedge import run_overnight_hedge_scan, unwind_hedge_legs, is_naked_fno, apply_guardrails, build_hedge_leg; print('All imports OK')"`
Expected: `All imports OK`

**Step 5: Final commit (if any remaining changes)**

```bash
git add -A
git status
# Only commit if there are changes
```
