"""Overnight hedge guardrail functions (V10).

Protects naked F&O positions at EOD by applying a cascade of rules:
1. Positions in loss → close
2. Near expiry → close
3. High VIX → block naked carry
4. Low gain → close
5. Moderate/high gain → ask Claude for hedge vs carry decision
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone, timedelta

from config import (
    OVERNIGHT_HEDGE_ENABLED,
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

IST = timezone(timedelta(hours=5, minutes=30))


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
    Handles both bullish (SL below price) and bearish (SL above price).
    Returns new stop-loss price rounded to 2 decimal places.
    """
    current_sl = pos.get("stoploss_price", 0)
    gap = abs(current_price - current_sl)
    new_gap = gap * (1 - OVERNIGHT_STOP_TIGHTEN_PCT)
    direction = pos.get("direction", "bullish")
    if direction == "bearish":
        new_sl = round(current_price + new_gap, 2)
    else:
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


# ---------------------------------------------------------------------------
# MORNING UNWIND LOGIC
# ---------------------------------------------------------------------------


def find_positions_with_hedge_legs(positions: list[dict]) -> list[dict]:
    """Return open positions that have a hedge_leg attached."""
    return [p for p in positions if p.get("status") == "open" and p.get("hedge_leg")]


def calc_hedge_pnl(entry_premium: float, exit_premium: float, quantity: int) -> dict:
    """Calculate P&L from unwinding a hedge leg.

    Returns {"pnl": float, "label": "hedge_saved"|"insurance_cost"}
    """
    pnl = (exit_premium - entry_premium) * quantity
    label = "hedge_saved" if pnl > 0 else "insurance_cost"
    return {"pnl": round(pnl, 2), "label": label}


def unwind_hedge_legs(smart_api, portfolio: dict) -> list[dict]:
    """Unwind all hedge legs at market open.

    For each hedged position: get current premium, calc P&L, record history,
    remove hedge_leg and overnight_carry_approved, send Telegram summary.
    Returns list of unwind result dicts.
    """
    from paper_trade import get_ltp_nfo, resolve_option_contract, _telegram_send

    positions = portfolio.get("positions", [])
    hedged = find_positions_with_hedge_legs(positions)

    if not hedged:
        return []

    results = []

    for pos in hedged:
        hedge = pos["hedge_leg"]
        symbol = pos.get("symbol", "?")
        strike = hedge.get("strike", 0)
        option_type = hedge.get("option_type", "")
        entry_premium = hedge.get("premium", 0)
        quantity = hedge.get("quantity", 0)

        # Resolve option contract to get current premium
        try:
            contract = resolve_option_contract(smart_api, symbol, strike, pos.get("expiry", ""))
            if contract:
                exit_premium = get_ltp_nfo(smart_api, contract["trading_symbol"], contract["token"])
            else:
                exit_premium = None
        except Exception as e:
            logger.warning("Hedge unwind LTP failed for %s: %s", symbol, e)
            exit_premium = None

        if exit_premium is None:
            # Can't get exit price — use entry as fallback (zero P&L)
            exit_premium = entry_premium
            logger.warning("Using entry premium as fallback for %s hedge unwind", symbol)

        pnl_result = calc_hedge_pnl(entry_premium, exit_premium, quantity)

        # Record in hedge_pnl_history
        if "hedge_pnl_history" not in pos:
            pos["hedge_pnl_history"] = []
        pos["hedge_pnl_history"].append({
            "date": datetime.now(IST).strftime("%Y-%m-%d"),
            "option_type": option_type,
            "strike": strike,
            "entry_premium": entry_premium,
            "exit_premium": exit_premium,
            "quantity": quantity,
            **pnl_result,
        })

        # Remove hedge leg and overnight carry approval
        del pos["hedge_leg"]
        pos.pop("overnight_carry_approved", None)

        results.append({
            "symbol": symbol,
            "option_type": option_type,
            "strike": strike,
            **pnl_result,
        })

        logger.info(
            "HEDGE UNWIND %s: %s %s — P&L ₹%.2f (%s)",
            symbol, option_type, strike, pnl_result["pnl"], pnl_result["label"],
        )

    # Send Telegram summary
    if results:
        lines = ["<b>Morning Hedge Unwind</b>", ""]
        total_pnl = 0.0
        for r in results:
            emoji = "+" if r["pnl"] > 0 else ""
            lines.append(
                f"  {r['symbol']} {r['option_type']} {r['strike']}: "
                f"{emoji}{r['pnl']:.0f} ({r['label']})"
            )
            total_pnl += r["pnl"]
        lines.append("")
        label = "Net saved" if total_pnl > 0 else "Insurance cost"
        lines.append(f"<b>{label}: {total_pnl:+,.0f}</b>")
        try:
            _telegram_send("\n".join(lines))
        except Exception as e:
            logger.error("Telegram send failed for hedge unwind: %s", e)

    return results


# ---------------------------------------------------------------------------
# TELEGRAM FORMATTING
# ---------------------------------------------------------------------------

def _format_telegram_message(results: list[dict], vix: float, regime: str) -> str:
    """Format overnight hedge scan results for Telegram (HTML parse_mode)."""
    if not results:
        return "<b>Overnight Hedge Scan</b>\n\nNo naked F&O positions found."

    closed = [r for r in results if r["action"] == "close"]
    hedged = [r for r in results if r["action"] == "hedge"]
    carried = [r for r in results if r["action"] == "carry_naked"]

    lines = ["<b>Overnight Hedge Scan</b>", ""]

    if closed:
        lines.append(f"<b>CLOSED ({len(closed)})</b>")
        for r in closed:
            gain_str = f"{r.get('gain_pct', 0):.0%}"
            lines.append(f"  {r['pos_id']} — gain {gain_str} — {r['reason']}")
        lines.append("")

    if hedged:
        lines.append(f"<b>HEDGED ({len(hedged)})</b>")
        for r in hedged:
            gain_str = f"{r.get('gain_pct', 0):.0%}"
            desc = r.get("hedge_desc", "")
            cost = r.get("hedge_cost", 0)
            cost_pct = r.get("cost_pct", 0)
            lines.append(f"  {r['pos_id']} — gain {gain_str} — {r['reason']}")
            if desc:
                lines.append(f"    hedge: {desc} (cost ₹{cost:.0f}, {cost_pct:.2%})")
        lines.append("")

    if carried:
        lines.append(f"<b>CARRY NAKED ({len(carried)})</b>")
        for r in carried:
            gain_str = f"{r.get('gain_pct', 0):.0%}"
            lines.append(f"  {r['pos_id']} — gain {gain_str} — {r['reason']}")
            if r.get("reasoning"):
                lines.append(f"    reasoning: {r['reasoning']}")
        lines.append("")

    lines.append(f"VIX: {vix:.1f} | Regime: {regime}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MAIN SCAN ORCHESTRATOR
# ---------------------------------------------------------------------------

def _parse_expiry_days(expiry_str: str) -> int:
    """Parse expiry string (DDMONYYYY) and return days to expiry."""
    try:
        exp_date = datetime.strptime(expiry_str, "%d%b%Y").date()
        today = datetime.now(IST).date()
        return (exp_date - today).days
    except (ValueError, TypeError):
        return 0  # treat unparseable as expired → will trigger close


def run_overnight_hedge_scan(smart_api, dry_run: bool = False) -> list[dict]:
    """Main orchestrator for overnight hedge protection.

    Scans all naked F&O positions and applies guardrails + Claude decisions.
    Returns list of result dicts with actions taken.
    """
    import config
    from paper_trade import (
        load_portfolio,
        save_portfolio,
        _telegram_send,
        get_ltp,
        get_ltp_nfo,
        resolve_option_contract,
    )
    from claude_intel import evaluate_overnight
    from regime import classify_regime

    # 1. Check feature flag
    if not OVERNIGHT_HEDGE_ENABLED:
        logger.info("Overnight hedge scan disabled")
        return []

    # 2. Load portfolio and filter naked positions
    portfolio = load_portfolio()
    positions = portfolio.get("positions", [])
    naked = [p for p in positions if is_naked_fno(p)]

    if not naked:
        logger.info("No naked F&O positions — nothing to scan")
        return []

    logger.info("Found %d naked F&O position(s) to scan", len(naked))

    # 3. Fetch VIX
    try:
        import yfinance as yf
        vix = yf.Ticker("^INDIAVIX").history(period="1d")["Close"].iloc[-1]
        logger.info("India VIX: %.2f", vix)
    except Exception as e:
        logger.warning("VIX fetch failed, defaulting to 15.0: %s", e)
        vix = 15.0

    # 4. Fetch regime
    try:
        regime_data = classify_regime(smart_api, vix=vix)
        regime = regime_data.get("regime", "UNKNOWN")
        logger.info("Market regime: %s", regime)
    except Exception as e:
        logger.warning("Regime classification failed, defaulting to UNKNOWN: %s", e)
        regime = "UNKNOWN"

    results = []

    # 5. Process each naked position
    for pos in naked:
        pos_id = pos.get("symbol", "?")
        instrument = pos.get("instrument", "")
        direction = pos.get("direction", "bullish")
        entry_price = pos.get("entry_price", 0)
        expiry_str = pos.get("expiry", "")

        # 5a. Fetch current price
        try:
            nfo_token = pos.get("nfo_token")
            trading_symbol = pos.get("trading_symbol")
            if nfo_token and trading_symbol:
                current_price = get_ltp_nfo(smart_api, trading_symbol, nfo_token)
            else:
                current_price = get_ltp(smart_api, pos.get("symbol", ""), pos.get("token", ""))
            time.sleep(config.API_DELAY)
        except Exception as e:
            logger.error("LTP fetch failed for %s: %s", pos_id, e)
            current_price = None

        if current_price is None:
            # Fail-safe: close if we can't get price
            reason = "LTP fetch failed — closing as fail-safe"
            if not dry_run:
                execute_close(portfolio, pos, smart_api, entry_price, reason)
            results.append({
                "pos_id": pos_id,
                "action": "close",
                "reason": reason,
                "gain_pct": 0,
            })
            continue

        # 5b. Calculate gain and days to expiry
        gain_pct = _calc_gain_pct(entry_price, current_price, direction)
        days_to_expiry = _parse_expiry_days(expiry_str)

        # 5c. Apply guardrails
        guardrail = apply_guardrails(pos, current_price, vix, days_to_expiry)

        if guardrail["action"] == "close":
            # 5d. Guardrail says close
            if not dry_run:
                execute_close(portfolio, pos, smart_api, current_price, guardrail["reason"])
            results.append({
                "pos_id": pos_id,
                "action": "close",
                "reason": guardrail["reason"],
                "gain_pct": gain_pct,
            })
            continue

        # 5e. Guardrail says ask_claude
        try:
            claude_raw = evaluate_overnight(pos, current_price, gain_pct, vix, regime)
            decision = parse_overnight_decision(claude_raw.get("action", None) if isinstance(claude_raw, dict) else claude_raw)
            # If evaluate_overnight already returns a parsed dict, use it directly
            if isinstance(claude_raw, dict) and claude_raw.get("action") in ("close", "hedge", "carry_naked"):
                decision = claude_raw
            time.sleep(config.API_DELAY)
        except Exception as e:
            logger.error("Claude evaluate_overnight failed for %s: %s", pos_id, e)
            decision = {"action": "hedge", "reasoning": "Claude call failed — defaulting to hedge"}

        # Enforce thresholds
        decision = enforce_carry_naked_threshold(decision, gain_pct)

        # Block carry_naked if VIX is high
        if decision.get("action") == "carry_naked" and vix >= OVERNIGHT_VIX_NAKED_BLOCK:
            decision = {"action": "hedge", "reasoning": f"carry_naked blocked — VIX {vix:.1f} >= {OVERNIGHT_VIX_NAKED_BLOCK}"}

        reasoning = decision.get("reasoning", "")

        # 5f. Execute decision
        if decision["action"] == "close":
            if not dry_run:
                execute_close(portfolio, pos, smart_api, current_price, reasoning)
            results.append({
                "pos_id": pos_id,
                "action": "close",
                "reason": reasoning,
                "gain_pct": gain_pct,
                "reasoning": reasoning,
            })

        elif decision["action"] == "hedge":
            # Build hedge leg
            # Get spot price for hedge strike calculation
            try:
                spot_price = get_ltp(smart_api, pos.get("symbol", ""), pos.get("token", ""))
                time.sleep(config.API_DELAY)
            except Exception:
                spot_price = current_price  # fallback to current NFO price

            if spot_price is None:
                spot_price = current_price

            hedge_leg = build_hedge_leg(pos, spot_price)

            # Resolve option contract
            try:
                contract = resolve_option_contract(
                    smart_api,
                    hedge_leg["symbol"],
                    hedge_leg["strike"],
                    expiry_str,
                )
                time.sleep(config.API_DELAY)
            except Exception as e:
                logger.error("resolve_option_contract failed for %s: %s", pos_id, e)
                contract = None

            if contract is None:
                # Can't find hedge contract — close as fail-safe
                close_reason = "hedge contract unavailable — closing as fail-safe"
                if not dry_run:
                    execute_close(portfolio, pos, smart_api, current_price, close_reason)
                results.append({
                    "pos_id": pos_id,
                    "action": "close",
                    "reason": close_reason,
                    "gain_pct": gain_pct,
                })
                continue

            # Get hedge premium
            try:
                hedge_premium = get_ltp_nfo(
                    smart_api,
                    contract["trading_symbol"],
                    contract["token"],
                )
                time.sleep(config.API_DELAY)
            except Exception as e:
                logger.error("Hedge premium fetch failed for %s: %s", pos_id, e)
                hedge_premium = None

            if hedge_premium is None:
                close_reason = "hedge premium unavailable — closing as fail-safe"
                if not dry_run:
                    execute_close(portfolio, pos, smart_api, current_price, close_reason)
                results.append({
                    "pos_id": pos_id,
                    "action": "close",
                    "reason": close_reason,
                    "gain_pct": gain_pct,
                })
                continue

            # Check cost
            position_value = abs(entry_price * pos.get("quantity", 1))
            cost_check = check_hedge_cost(position_value, hedge_premium, hedge_leg["quantity"])

            if not cost_check["affordable"]:
                close_reason = f"hedge too expensive ({cost_check['cost_pct']:.2%}) — closing"
                if not dry_run:
                    execute_close(portfolio, pos, smart_api, current_price, close_reason)
                results.append({
                    "pos_id": pos_id,
                    "action": "close",
                    "reason": close_reason,
                    "gain_pct": gain_pct,
                })
                continue

            # Execute hedge
            hedge_desc = f"{hedge_leg['option_type']} {hedge_leg['strike']} @ ₹{hedge_premium:.2f}"
            if not dry_run:
                execute_hedge(portfolio, pos, smart_api, hedge_leg, hedge_premium)
            results.append({
                "pos_id": pos_id,
                "action": "hedge",
                "reason": reasoning,
                "gain_pct": gain_pct,
                "hedge_desc": hedge_desc,
                "hedge_cost": cost_check["cost"],
                "cost_pct": cost_check["cost_pct"],
                "reasoning": reasoning,
            })

        elif decision["action"] == "carry_naked":
            if not dry_run:
                execute_carry_naked(pos, current_price, reasoning)
            results.append({
                "pos_id": pos_id,
                "action": "carry_naked",
                "reason": reasoning,
                "gain_pct": gain_pct,
                "reasoning": reasoning,
            })

    # 6. Save portfolio
    if not dry_run and results:
        save_portfolio(portfolio)
        logger.info("Portfolio saved after overnight hedge scan")

    # 7. Send Telegram message
    msg = _format_telegram_message(results, vix, regime)
    if dry_run:
        logger.info("DRY RUN — Telegram message:\n%s", msg)
    else:
        try:
            _telegram_send(msg)
        except Exception as e:
            logger.error("Telegram send failed: %s", e)

    return results


# ---------------------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for overnight hedge scan."""
    parser = argparse.ArgumentParser(description="Overnight hedge scan for naked F&O positions")
    parser.add_argument("--dry", action="store_true", help="Dry run — no trades, no Telegram")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from connect import get_session

    logger.info("Connecting to SmartAPI...")
    smart_api = get_session()
    logger.info("Running overnight hedge scan (dry_run=%s)", args.dry)

    results = run_overnight_hedge_scan(smart_api, dry_run=args.dry)
    logger.info("Overnight hedge scan complete — %d position(s) processed", len(results))


if __name__ == "__main__":
    main()
