"""Debate System — Adversarial multi-agent entry analysis.

Three parallel agents (Bull, Bear, Context) analyze each top candidate.
A Moderator synthesizes their views, with an optional round 2 if contradictions
are detected. The result feeds into evaluate_candidates() as extra context.

Uses claude CLI only (Pro Max subscription, no API cost).
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger("paper_trade")

DEBATE_LOG_DIR = Path("data/paper_trades/debate_logs")
AGENT_TIMEOUT = 30  # seconds per agent thread


def _call_claude_cli_debate(prompt: str) -> str | None:
    """Call Claude via CLI for debate agents. Lighter system prompt than main eval."""
    import shutil
    import subprocess

    claude_bin = shutil.which("claude")
    if not claude_bin:
        return None

    try:
        result = subprocess.run(
            [claude_bin, "-p", "--output-format", "text",
             "--max-turns", "1", "--no-session-persistence",
             "--model", "haiku"],
            input=prompt,
            capture_output=True, text=True, timeout=AGENT_TIMEOUT,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None
    except subprocess.TimeoutExpired:
        logger.warning("Debate agent timed out")
        return None
    except Exception as e:
        logger.debug("Debate CLI failed: %s", e)
        return None


def _format_candidate_data(candidate: dict) -> str:
    """Format candidate data for agent prompts."""
    return (
        f"Symbol: {candidate['symbol']} | Direction: {candidate.get('direction', '?')} | "
        f"Score: {candidate.get('score', 0)} | RSI: {candidate.get('rsi', '?')} | "
        f"Volume: {candidate.get('volume_ratio', '?')}x | "
        f"Categories: {candidate.get('categories', [])} | "
        f"Supertrend: {candidate.get('supertrend_signal', '?')} | "
        f"Sector: {candidate.get('sector', 'Other')} | "
        f"MFI: {candidate.get('mfi', '?')} | ADX: {candidate.get('adx', '?')}"
    )


def _bull_agent(candidate: dict, caches: dict) -> str:
    """Bull perspective — find every reason TO ENTER."""
    symbol = candidate["symbol"]
    data = _format_candidate_data(candidate)
    prompt = f"""You are a BULL analyst for {symbol} on NSE. Your job is to find every reason
TO ENTER this trade. Be specific with price levels, not vague.

Data: {data}

Respond in 3-4 sentences:
1. Strongest bullish signal
2. Price target with reasoning
3. What would make this thesis even stronger"""

    result = _call_claude_cli_debate(prompt)
    return result or ""


def _bear_agent(candidate: dict, caches: dict) -> str:
    """Bear perspective — find every reason NOT TO ENTER."""
    symbol = candidate["symbol"]
    data = _format_candidate_data(candidate)
    prompt = f"""You are a BEAR analyst for {symbol} on NSE. Your job is to find every reason
NOT TO ENTER this trade. Be adversarial — assume the bull case is wrong.

Data: {data}

Respond in 3-4 sentences:
1. Strongest risk/red flag
2. Downside target or stop level
3. What specific event would invalidate the bull case"""

    result = _call_claude_cli_debate(prompt)
    return result or ""


def _context_agent(candidate: dict, caches: dict) -> str:
    """Context perspective — surface info Bull and Bear might miss."""
    symbol = candidate["symbol"]
    macro = caches.get("macro", {})
    vix_hist = caches.get("vix_history", [])
    prompt = f"""You are a CONTEXT analyst. Your job is to surface information the Bull and Bear
might miss — external factors, calendar events, macro shifts.

Symbol: {symbol}
Macro: FII={macro.get('fii_net_crores', 'N/A')}cr, DII={macro.get('dii_net_crores', 'N/A')}cr
Recent VIX: {vix_hist[-5:] if vix_hist else 'N/A'}

Respond in 2-3 sentences:
1. Any upcoming event (earnings, RBI, expiry) within 5 days
2. Relevant macro/sector signal
3. Any anomaly (unusual VIX move, FII pattern break)"""

    result = _call_claude_cli_debate(prompt)
    return result or ""


def _moderate(symbol: str, bull: str, bear: str, context: str) -> str:
    """Moderator round 1 — synthesize and detect contradictions."""
    prompt = f"""You are a MODERATOR synthesizing a Bull-Bear-Context debate for {symbol}.

Bull says: {bull or '[unavailable]'}
Bear says: {bear or '[unavailable]'}
Context says: {context or '[unavailable]'}

Synthesize in 3-4 sentences:
1. Where bull and bear AGREE (if anywhere)
2. The key point of DISAGREEMENT and which side has stronger evidence
3. How the context changes the picture
4. Your verdict: "bull-leaning", "bear-leaning", or "split"

If bull and bear directly contradict each other on a factual claim, end with:
CONTRADICTION: [describe it in one line]"""

    result = _call_claude_cli_debate(prompt)
    return result or ""


def _moderate_round2(symbol: str, bull: str, bear: str, contradiction: str) -> str:
    """Moderator round 2 — resolve contradiction."""
    prompt = f"""The debate for {symbol} has an unresolved contradiction:
{contradiction}

Bull: {bull}
Bear: {bear}

Which side has the stronger factual basis? Resolve this in 2 sentences,
then give your final verdict: "bull-leaning", "bear-leaning", or "split"."""

    result = _call_claude_cli_debate(prompt)
    return result or ""


def _format_summary(symbol: str, bull: str, bear: str, context: str,
                    verdict: str, rounds: int) -> str:
    """Format debate summary for injection into evaluate_candidates."""
    def truncate(text: str, max_words: int = 50) -> str:
        words = text.split()
        return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")

    return (
        f"DEBATE SUMMARY for {symbol}:\n"
        f"- Bull: {truncate(bull) if bull else '[unavailable]'}\n"
        f"- Bear: {truncate(bear) if bear else '[unavailable]'}\n"
        f"- Context: {truncate(context) if context else '[unavailable]'}\n"
        f"- Verdict: {verdict} ({rounds} round{'s' if rounds > 1 else ''})"
    )


def _run_single_debate(candidate: dict, caches: dict) -> dict:
    """Run a full debate for one candidate. Returns debate log dict."""
    symbol = candidate["symbol"]
    t0 = time.monotonic()

    # --- Run 3 agents in parallel ---
    agents = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_bull_agent, candidate, caches): "bull",
            pool.submit(_bear_agent, candidate, caches): "bear",
            pool.submit(_context_agent, candidate, caches): "context",
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                agents[name] = {
                    "response": future.result() or "",
                    "elapsed_ms": round((time.monotonic() - t0) * 1000),
                }
            except Exception as e:
                logger.debug("Debate agent %s failed: %s", name, e)
                agents[name] = {"response": "", "elapsed_ms": 0}

    bull = agents.get("bull", {}).get("response", "")
    bear = agents.get("bear", {}).get("response", "")
    context = agents.get("context", {}).get("response", "")

    # If no agents returned, skip debate
    if not any([bull, bear, context]):
        return {
            "symbol": symbol, "score": candidate.get("score", 0),
            "agents": agents,
            "moderator": {"round_1": None, "round_2": None},
            "final_summary": "",
            "total_elapsed_ms": round((time.monotonic() - t0) * 1000),
        }

    # --- Moderator round 1 ---
    r1 = _moderate(symbol, bull, bear, context)
    contradiction = None
    if "CONTRADICTION:" in r1:
        contradiction = r1.split("CONTRADICTION:")[-1].strip()

    # Extract verdict from round 1
    verdict = "split"
    for v in ("bull-leaning", "bear-leaning", "split"):
        if v in r1.lower():
            verdict = v
            break

    r2 = None
    rounds = 1

    # --- Moderator round 2 (only if contradiction) ---
    if contradiction and bull and bear:
        r2 = _moderate_round2(symbol, bull, bear, contradiction)
        rounds = 2
        for v in ("bull-leaning", "bear-leaning", "split"):
            if v in (r2 or "").lower():
                verdict = v
                break

    summary = _format_summary(symbol, bull, bear, context, verdict, rounds)

    IST = timezone(timedelta(hours=5, minutes=30))
    log = {
        "timestamp": datetime.now(IST).isoformat(),
        "symbol": symbol,
        "score": candidate.get("score", 0),
        "agents": agents,
        "moderator": {
            "round_1": {"response": r1, "verdict": verdict, "contradiction": contradiction},
            "round_2": {"response": r2} if r2 else None,
        },
        "final_summary": summary,
        "total_elapsed_ms": round((time.monotonic() - t0) * 1000),
    }

    # Save debate log
    try:
        DEBATE_LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(IST).strftime("%Y%m%d_%H%M%S")
        log_path = DEBATE_LOG_DIR / f"{symbol}_{ts}.json"
        log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))
        logger.info("Debate log saved: %s", log_path.name)
    except Exception as e:
        logger.debug("Failed to save debate log: %s", e)

    return log


def run_debates(candidates: list[dict], caches: dict) -> dict[str, str]:
    """Run debates for top candidates. Returns {symbol: summary_str}.

    Candidates are debated sequentially (each debate runs 3 agents in parallel).
    """
    import config

    if not getattr(config, "DEBATE_ENABLED", False):
        return {}

    top_n = getattr(config, "DEBATE_TOP_N", 3)
    to_debate = candidates[:top_n]

    results = {}
    for candidate in to_debate:
        try:
            log = _run_single_debate(candidate, caches)
            if log["final_summary"]:
                results[candidate["symbol"]] = log["final_summary"]
                logger.info("Debate for %s: %s (%.1fs)",
                            candidate["symbol"],
                            log["moderator"]["round_1"]["verdict"] if log["moderator"]["round_1"] else "skipped",
                            log["total_elapsed_ms"] / 1000)
        except Exception as e:
            logger.debug("Debate failed for %s: %s", candidate["symbol"], e)

    return results
