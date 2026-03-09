"""Claude Intelligence Module for Trading Decisions.

Claude acts as the final decision-maker for ALL trade entries and exits.
Every position open/close goes through Claude for approval with reasoning.
Reasoning is sent to Telegram so the user knows exactly why each action was taken.

Uses Haiku for speed/cost efficiency on the 30-min cron cycle.
"""

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger("paper_trade")

DECISION_LOG_DIR = Path("data/paper_trades/claude_decisions")


def _save_decision_log(decision_type: str, symbol: str, prompt: str, response: str, parsed: dict, extra: dict = None):
    """Save Claude decision for replay/debugging."""
    try:
        DECISION_LOG_DIR.mkdir(parents=True, exist_ok=True)
        IST = timezone(timedelta(hours=5, minutes=30))
        now = datetime.now(IST)
        filename = f"{now.strftime('%Y-%m-%d_%H-%M')}_{decision_type}_{symbol}.json"
        data = {
            "timestamp": now.isoformat(),
            "type": decision_type,
            "symbol": symbol,
            "prompt_sent": prompt,
            "claude_response": response,
            "parsed_action": parsed.get("action", ""),
            **(extra or {}),
        }
        (DECISION_LOG_DIR / filename).write_text(json.dumps(data, indent=2, default=str))
    except Exception as e:
        logger.debug("Decision log save failed: %s", e)


SYSTEM_PROMPT = """You are an expert Indian F&O trader with 15+ years of experience trading Nifty, BankNifty, and stock options/futures on NSE.

Your edge comes from:
- Reading OI buildup patterns (long/short buildup, unwinding) to gauge institutional positioning
- Using PCR as a contrarian indicator (>1.3 = market bottoming, <0.7 = market topping)
- Respecting FII flows — never fight sustained FII selling
- Understanding max pain gravity near expiry — price tends to drift toward max pain
- Volatility regime awareness — VIX >20 means reduce size, VIX >25 means only hedge trades
- Sector rotation — money flows between sectors, don't chase sectors that already ran up
- Risk-first thinking — always size positions assuming the worst case

You are decisive. You give clear TRADE or SKIP verdicts with specific reasoning.
Keep reasoning to 2-3 sentences max — pithy, actionable, like a trading desk callout."""


def _call_claude_cli(prompt: str) -> str | None:
    """Call Claude via Claude Code CLI (uses Max subscription, no API cost)."""
    import shutil
    import subprocess
    import tempfile

    claude_bin = shutil.which("claude")
    if not claude_bin:
        return None

    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
    try:
        # Write prompt to temp file to avoid shell escaping issues
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(full_prompt)
            prompt_file = f.name

        result = subprocess.run(
            [claude_bin, "-p", "--output-format", "text",
             "--max-turns", "1", "--no-session-persistence"],
            input=full_prompt,
            capture_output=True, text=True, timeout=45,
        )
        os.unlink(prompt_file)

        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        if result.stderr:
            logger.debug("Claude CLI stderr: %s", result.stderr[:200])
        return None
    except subprocess.TimeoutExpired:
        logger.warning("Claude CLI timed out")
        return None
    except Exception as e:
        logger.debug("Claude CLI failed: %s", e)
        return None


def _get_client():
    """Get Anthropic client, returns None if unavailable."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    try:
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    except Exception:
        return None


def _call_claude(prompt: str, max_tokens: int = 1024) -> str | None:
    """Call Claude — prefers CLI (Max subscription, free) over API (paid).

    Priority: Claude Code CLI → Anthropic API (Haiku) → None
    """
    # Try Claude Code CLI first (uses Max subscription — no API cost)
    cli_result = _call_claude_cli(prompt)
    if cli_result:
        return cli_result

    # Fallback to Anthropic API (paid)
    client = _get_client()
    if not client:
        return None
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        logger.warning("Claude API call failed: %s", e)
        return None


def _parse_json(text: str) -> list | dict | None:
    """Extract JSON from Claude response, handling markdown fences."""
    if not text:
        return None
    try:
        # Strip markdown code fences
        cleaned = text
        if "```" in cleaned:
            parts = cleaned.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("[") or part.startswith("{"):
                    cleaned = part
                    break
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON within the text
        for start_char, end_char in [("[", "]"), ("{", "}")]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    continue
        return None


def _validate_entry_response(parsed: dict) -> dict:
    """Normalize and validate Claude entry response."""
    defaults = {"action": "SKIP", "conviction": "medium", "allocation_adj": 1.0, "reasoning": "No reasoning provided"}
    result = {}
    for key, default in defaults.items():
        val = parsed.get(key, default)
        result[key] = default if val is None else val
    try:
        result["allocation_adj"] = max(0.5, min(1.5, float(result["allocation_adj"])))
    except (TypeError, ValueError):
        result["allocation_adj"] = 1.0
    result["action"] = result["action"].upper() if isinstance(result["action"], str) else "SKIP"
    if result["action"] not in ("TRADE", "SKIP"):
        result["action"] = "SKIP"
    return result


def _validate_exit_response(parsed: dict) -> dict:
    """Normalize and validate Claude exit response."""
    defaults = {"action": "EXIT", "reasoning": "No reasoning provided"}
    result = {}
    for key, default in defaults.items():
        val = parsed.get(key, default)
        result[key] = default if val is None else val
    result["action"] = result["action"].upper() if isinstance(result["action"], str) else "EXIT"
    if result["action"] not in ("EXIT", "HOLD", "PARTIAL", "ADD"):
        result["action"] = "EXIT"
    return result


def _build_portfolio_context(portfolio: dict) -> str:
    """Build compact portfolio context string."""
    open_positions = []
    for p in portfolio.get("positions", []):
        if p.get("status") != "open":
            continue
        entry = p["entry_price"]
        days = (datetime.now().date() - datetime.strptime(p["entry_date"], "%Y-%m-%d").date()).days
        open_positions.append(
            f"  {p['symbol']} {p['direction']} {p.get('instrument', 'EQ')} "
            f"entry=₹{entry:,.0f} days={days} "
            f"target=₹{p.get('target_price', 0):,.0f} sl=₹{p.get('stoploss_price', 0):,.0f}"
        )

    recent = portfolio.get("closed_trades", [])[-10:]
    wins = sum(1 for t in recent if t.get("pnl", 0) > 0)
    losses = len(recent) - wins
    total_pnl = portfolio.get("stats", {}).get("total_pnl", 0)
    capital = portfolio.get("capital", 100_000)
    available = portfolio.get("available_capital", capital)

    lines = [
        f"Capital: ₹{capital:,.0f} | Available: ₹{available:,.0f} | P&L: ₹{total_pnl:+,.0f}",
        f"Recent: {wins}W/{losses}L",
    ]
    if open_positions:
        lines.append("Open positions:\n" + "\n".join(open_positions))
    else:
        lines.append("Open positions: None")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ENTRY EVALUATION — Called before opening ANY position
# ---------------------------------------------------------------------------

def evaluate_entry(
    candidate: dict,
    instrument: str,  # "EQ", "PE", "FUT", "CONDOR", "SPREAD", "MOMENTUM"
    regime: str,
    vix: float | None,
    macro: dict | None,
    pcr: float,
    max_pain: float,
    portfolio: dict,
    nifty_ltp: float | None = None,
    extra_context: str = "",
) -> tuple[bool, str, float]:
    """Evaluate a single candidate for entry.

    Returns (approved, reasoning, allocation_multiplier).
    If Claude is unavailable, returns (True, "", 1.0) — fail-open.
    """
    symbol = candidate.get("symbol", "?")
    direction = candidate.get("direction", "?")
    score = candidate.get("score", 0)

    macro_str = ""
    if macro:
        macro_str = (
            f"S&P {macro.get('sp500_pct_change', 0):+.1f}% | "
            f"Nasdaq {macro.get('nasdaq_pct_change', 0):+.1f}% | "
            f"FII {macro.get('fii_net_crores', 0):+,.0f}cr | "
            f"DII {macro.get('dii_net_crores', 0):+,.0f}cr"
        )

    # Fetch recent lessons for learning feedback
    lessons_str = ""
    try:
        from smart_monitor import get_recent_lessons
        lessons = get_recent_lessons(instrument=instrument, direction=direction, limit=5)
        if lessons:
            lesson_lines = [f"  - [{l['grade']}] {l['symbol']} ({l['category']}): {l['lesson']}" for l in lessons]
            lessons_str = "\n\nRECENT LESSONS (learn from these):\n" + "\n".join(lesson_lines)
    except Exception:
        pass

    prompt = f"""ENTRY DECISION for {symbol} ({instrument}, {direction})

MARKET:
- Regime: {regime} | VIX: {vix or 'N/A'} | Nifty: {nifty_ltp or 'N/A'}
- PCR: {pcr:.2f} | Max Pain: {max_pain:.0f}
- Global: {macro_str or 'N/A'}

CANDIDATE:
- Score: {score} | Direction: {direction}
- Categories: {candidate.get('categories', [])}
- RSI: {candidate.get('rsi', 'N/A')} | Volume ratio: {candidate.get('volume_ratio', 'N/A')}x
- News: {candidate.get('news_sentiment', 'N/A')}
- Supertrend: {candidate.get('supertrend_signal', 'N/A')}
- Price change: {candidate.get('price_change_pct', 'N/A')}%
- Sector: {candidate.get('sector', 'Other')}
{f"- Extra: {extra_context}" if extra_context else ""}

PORTFOLIO:
{_build_portfolio_context(portfolio)}
{lessons_str}

Should I ENTER this trade? Respond with JSON only:
{{"action": "TRADE" or "SKIP", "conviction": "high"/"medium"/"low", "allocation_adj": 0.5-1.5, "reasoning": "2-3 sentences"}}"""

    result_text = _call_claude(prompt, max_tokens=512)
    if result_text is None:
        logger.debug("Claude entry eval unavailable for %s — auto-approve", symbol)
        return (True, "", 1.0)

    decision = _parse_json(result_text)
    if not decision:
        logger.warning("Claude entry eval: unparseable response for %s", symbol)
        return (True, "", 1.0)

    decision = _validate_entry_response(decision)
    _save_decision_log("entry", candidate.get("symbol", "?"), prompt, result_text, decision)
    action = decision["action"]
    reasoning = decision["reasoning"]
    conviction = decision["conviction"]
    adj = decision["allocation_adj"]

    if action == "SKIP":
        logger.info("CLAUDE SKIP %s (%s): %s", symbol, instrument, reasoning)
        return (False, reasoning, 0.0)

    logger.info("CLAUDE TRADE %s (%s) [%s]: %s (adj: %.1fx)",
                symbol, instrument, conviction, reasoning, adj)
    return (True, reasoning, adj)


def evaluate_candidates(
    candidates: list[dict],
    regime: str,
    regime_confidence: float,
    vix: float | None,
    macro: dict | None,
    pcr: float,
    max_pain: float,
    portfolio: dict,
    nifty_ltp: float | None = None,
) -> list[dict]:
    """Evaluate multiple candidates in a single Claude call (batch mode).

    Used for the main equity/options screening pass.
    Returns filtered list with Claude reasoning attached.
    """
    if not candidates:
        return []

    macro_str = ""
    if macro:
        macro_str = (
            f"S&P {macro.get('sp500_pct_change', 0):+.1f}% | "
            f"Nasdaq {macro.get('nasdaq_pct_change', 0):+.1f}% | "
            f"FII {macro.get('fii_net_crores', 0):+,.0f}cr | "
            f"DII {macro.get('dii_net_crores', 0):+,.0f}cr"
        )

    candidate_lines = []
    for c in candidates:
        candidate_lines.append(
            f"  {c['symbol']} [{c['direction']}] score={c.get('score', 0)} "
            f"RSI={c.get('rsi', '?')} vol={c.get('volume_ratio', '?')}x "
            f"categories={c.get('categories', [])} "
            f"supertrend={c.get('supertrend_signal', '?')} "
            f"news={c.get('news_sentiment', '?')} "
            f"sector={c.get('sector', 'Other')}"
        )

    prompt = f"""BATCH ENTRY EVALUATION

MARKET:
- Regime: {regime} (conf: {regime_confidence:.0%}) | VIX: {vix or 'N/A'} | Nifty: {nifty_ltp or 'N/A'}
- PCR: {pcr:.2f} | Max Pain: {max_pain:.0f}
- Global: {macro_str or 'N/A'}

PORTFOLIO:
{_build_portfolio_context(portfolio)}

CANDIDATES:
{chr(10).join(candidate_lines)}

For EACH candidate, decide TRADE or SKIP. Consider:
- Holistic signal alignment (OI + RSI + volume + news + macro)
- FII flow direction vs trade direction
- VIX level vs position sizing
- Sector overlap with existing positions
- Score quality (>6 = strong, 4-6 = moderate, <4 = weak)

Respond with JSON array only:
[{{"symbol": "X", "action": "TRADE"/"SKIP", "conviction": "high"/"medium"/"low", "allocation_adj": 0.5-1.5, "reasoning": "2-3 sentences"}}]"""

    result_text = _call_claude(prompt, max_tokens=1024)
    if result_text is None:
        logger.debug("Claude batch eval unavailable — passing all through")
        return candidates

    decisions = _parse_json(result_text)
    if not decisions or not isinstance(decisions, list):
        logger.warning("Claude batch eval: bad JSON — passing all through")
        return candidates

    logger.info("Claude intel: evaluated %d candidates", len(decisions))

    approved = []
    for c in candidates:
        decision = next((d for d in decisions if d.get("symbol") == c["symbol"]), None)
        if decision is None:
            logger.info("  %s: SKIP (not in Claude response)", c["symbol"])
            continue

        if decision.get("action") == "SKIP":
            logger.info("  CLAUDE SKIP %s: %s", c["symbol"], decision.get("reasoning", ""))
            continue

        adj = max(0.5, min(1.5, decision.get("allocation_adj", 1.0)))
        c["claude_allocation_adj"] = adj
        c["claude_conviction"] = decision.get("conviction", "medium")
        c["claude_reasoning"] = decision.get("reasoning", "")

        logger.info("  CLAUDE TRADE %s [%s]: %s (adj: %.1fx)",
                    c["symbol"], decision.get("conviction", "?"),
                    decision.get("reasoning", ""), adj)
        approved.append(c)

    return approved


# ---------------------------------------------------------------------------
# EXIT EVALUATION — Called before closing ANY position
# ---------------------------------------------------------------------------

def evaluate_exit(
    pos: dict,
    exit_reason: str,
    current_ltp: float,
    pnl: float,
    pnl_pct: float,
    portfolio: dict,
    vix: float | None = None,
    extra_context: str = "",
) -> tuple[bool, str]:
    """Evaluate whether to proceed with a triggered exit.

    Returns (should_exit, reasoning).
    If Claude is unavailable, returns (True, "") — always honor rule-based exits.

    Note: Stoploss exits are ALWAYS honored (never overridden by Claude).
    Claude can only suggest holding when target/trailing/time exits trigger.
    """
    symbol = pos.get("symbol", "?")
    instrument = pos.get("instrument", "EQ")
    direction = pos.get("direction", "?")
    entry = pos.get("entry_price", 0)
    days_held = 0
    try:
        days_held = (datetime.now().date() - datetime.strptime(pos["entry_date"], "%Y-%m-%d").date()).days
    except Exception:
        pass

    # NEVER let Claude override a stoploss — risk management is non-negotiable
    if exit_reason in ("stoploss", "gap_through_sl", "monthly_sl_cap",
                       "condor_stoploss", "credit_stoploss", "spread_stoploss"):
        reasoning = f"SL exit ({exit_reason}) — non-negotiable risk management"
        logger.info("Claude exit: %s %s — SL exit, auto-approved", symbol, exit_reason)
        return (True, reasoning)

    # Fetch recent lessons about exit timing
    lessons_str = ""
    try:
        from smart_monitor import get_recent_lessons
        lessons = get_recent_lessons(limit=5)
        exit_lessons = [l for l in lessons if l.get("category") == "exit_timing"]
        if exit_lessons:
            lesson_lines = [f"  - [{l['grade']}] {l['symbol']}: {l['lesson']}" for l in exit_lessons[-3:]]
            lessons_str = "\n\nRECENT EXIT LESSONS:\n" + "\n".join(lesson_lines)
    except Exception:
        pass

    prompt = f"""EXIT DECISION for {symbol} ({instrument}, {direction})

POSITION:
- Entry: ₹{entry:,.2f} | Current: ₹{current_ltp:,.2f}
- P&L: ₹{pnl:+,.0f} ({pnl_pct:+.1f}%)
- Days held: {days_held}
- Target: ₹{pos.get('target_price', 0):,.2f} | SL: ₹{pos.get('stoploss_price', 0):,.2f}
- ATR at entry: {pos.get('atr_at_entry', 'N/A')}

EXIT TRIGGER: {exit_reason}
VIX: {vix or 'N/A'}
{f"Context: {extra_context}" if extra_context else ""}

PORTFOLIO:
{_build_portfolio_context(portfolio)}
{lessons_str}

The rule-based system wants to EXIT this position because: {exit_reason}
Should I proceed with the exit or continue holding?

Consider:
- If target hit, is there momentum for more? Or book profits?
- If trailing stop, is this a temporary pullback or trend reversal?
- If time exit, is the thesis still valid?
- If expiry approaching on options, theta decay risk
- VIX spike = increased risk, favor exits

Respond with JSON only:
{{"action": "EXIT" or "HOLD", "reasoning": "2-3 sentences"}}"""

    result_text = _call_claude(prompt, max_tokens=384)
    if result_text is None:
        # Fail-safe: always honor rule-based exits
        return (True, f"Rule-based exit: {exit_reason}")

    decision = _parse_json(result_text)
    if not decision:
        return (True, f"Rule-based exit: {exit_reason}")

    decision = _validate_exit_response(decision)
    _save_decision_log("exit", pos.get("symbol", "?"), prompt, result_text, decision)
    action = decision["action"]
    reasoning = decision["reasoning"]

    if action == "HOLD":
        logger.info("CLAUDE HOLD %s: overriding %s exit — %s", symbol, exit_reason, reasoning)
        return (False, reasoning)

    logger.info("CLAUDE EXIT %s (%s): %s", symbol, exit_reason, reasoning)
    return (True, reasoning)


# ---------------------------------------------------------------------------
# TELEGRAM FORMATTING
# ---------------------------------------------------------------------------

def format_entry_telegram(
    pos: dict,
    claude_reasoning: str,
    regime: str = "",
    vix: float | None = None,
) -> str:
    """Format a rich Telegram entry notification with Claude's reasoning."""
    symbol = pos.get("symbol", "?")
    direction = pos.get("direction", "?")
    instrument = pos.get("instrument", "EQ")
    entry = pos.get("entry_price", 0)
    target = pos.get("target_price", 0)
    sl = pos.get("stoploss_price", 0)
    alloc = pos.get("allocated", 0)
    score = pos.get("score", 0)
    conviction = pos.get("claude_conviction", "")

    dir_emoji = "🟢" if direction == "bullish" else "🔴"
    instr_label = {"EQ": "Equity", "PE": "Put Option", "FUT": "Futures",
                   "SPREAD": "Spread", "CONDOR": "Iron Condor"}.get(instrument, instrument)

    lines = [
        f"<b>{dir_emoji} ENTRY: {symbol} ({instr_label})</b>",
        f"Direction: {direction.upper()} | Score: {score}",
        f"Entry: ₹{entry:,.2f} | Target: ₹{target:,.2f} | SL: ₹{sl:,.2f}",
        f"Allocated: ₹{alloc:,.0f}",
    ]
    if regime:
        lines.append(f"Regime: {regime}" + (f" | VIX: {vix:.1f}" if vix else ""))
    if conviction:
        lines.append(f"Conviction: {conviction.upper()}")
    if claude_reasoning:
        lines.append(f"\n<b>Why:</b> {claude_reasoning}")

    return "\n".join(lines)


def format_exit_telegram(
    pos: dict,
    exit_reason: str,
    pnl: float,
    pnl_pct: float,
    exit_price: float,
    claude_reasoning: str,
    portfolio: dict | None = None,
) -> str:
    """Format a rich Telegram exit notification with Claude's reasoning."""
    symbol = pos.get("symbol", "?")
    direction = pos.get("direction", "?")
    instrument = pos.get("instrument", "EQ")
    entry = pos.get("entry_price", 0)
    days = 0
    try:
        days = (datetime.now().date() - datetime.strptime(pos["entry_date"], "%Y-%m-%d").date()).days
    except Exception:
        pass

    pnl_emoji = "✅" if pnl >= 0 else "❌"
    reason_labels = {
        "target": "🎯 Target Hit", "stoploss": "🛑 Stop Loss",
        "trailing_stop": "📉 Trailing Stop", "expiry": "⏰ Expiry",
        "gap_through_sl": "⚡ Gap Through SL", "time_tightened_sl": "⏳ Time SL",
        "partial_target": "🎯 Partial Target", "theta_decay": "⏳ Theta Decay",
        "condor_target": "🎯 Condor Target", "condor_stoploss": "🛑 Condor SL",
        "condor_vix_spike": "📊 VIX Spike", "condor_time_exit": "⏰ Condor Time",
        "monthly_sl_cap": "🛑 Monthly SL Cap",
        "credit_target": "🎯 Credit Target", "credit_stoploss": "🛑 Credit SL",
        "spread_target": "🎯 Spread Target", "spread_stoploss": "🛑 Spread SL",
        "spread_time_exit": "⏰ Spread Time", "spread_expiry_safety": "⏰ Expiry Safety",
        "opt_partial_profit": "💰 Partial Profit",
    }
    reason_label = reason_labels.get(exit_reason, exit_reason)

    instr_label = {"EQ": "Equity", "PE": "Put Option", "FUT": "Futures",
                   "SPREAD": "Spread", "CONDOR": "Iron Condor"}.get(instrument, instrument)

    lines = [
        f"<b>{pnl_emoji} EXIT: {symbol} ({instr_label})</b>",
        f"{reason_label}",
        f"Entry: ₹{entry:,.2f} → Exit: ₹{exit_price:,.2f}",
        f"P&amp;L: ₹{pnl:+,.0f} ({pnl_pct:+.1f}%) | {days}d held",
    ]

    if portfolio:
        total_pnl = portfolio.get("stats", {}).get("total_pnl", 0)
        lines.append(f"Portfolio P&amp;L: ₹{total_pnl:+,.0f}")

    if claude_reasoning:
        lines.append(f"\n<b>Why:</b> {claude_reasoning}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CRON SUMMARY
# ---------------------------------------------------------------------------

def generate_cron_summary(
    mode: str,
    portfolio: dict,
    regime: str | None = None,
    vix: float | None = None,
    opened: int = 0,
    exited: int = 0,
    unrealized_pnl: float = 0,
    skipped_reasons: list[str] | None = None,
) -> str:
    """Generate a brief Telegram-friendly summary of what happened this cron run."""
    now = datetime.now()
    time_str = now.strftime("%H:%M")

    open_positions = [p for p in portfolio.get("positions", []) if p.get("status") == "open"]
    total_pnl = portfolio.get("stats", {}).get("total_pnl", 0)

    lines = [f"<b>📊 {time_str} | {mode.upper()}</b>"]

    if regime:
        vix_str = f" | VIX {vix:.1f}" if vix else ""
        lines.append(f"Regime: {regime}{vix_str}")

    if mode == "open":
        if opened > 0:
            lines.append(f"✅ Opened {opened} position(s)")
        else:
            lines.append("No new positions")
            if skipped_reasons:
                lines.append("Skips: " + ", ".join(skipped_reasons[:3]))

    if mode == "monitor":
        if exited > 0:
            lines.append(f"🔔 {exited} position(s) closed")
        else:
            lines.append("All positions holding")

    if open_positions:
        for p in open_positions:
            arrow = "🟢" if p["direction"] == "bullish" else "🔴"
            lines.append(f"  {arrow} {p['symbol']} {p.get('instrument', 'EQ')} @ ₹{p['entry_price']:,.0f}")

    lines.append(f"P&L: ₹{total_pnl + unrealized_pnl:+,.0f} (R: ₹{total_pnl:+,.0f} U: ₹{unrealized_pnl:+,.0f})")

    return "\n".join(lines)
