"""
Morning Screener Agent — scans F&O market for signals and produces an actionable watchlist.

Fetches all 8 gainersLosers data types from SmartAPI, cross-references stocks
across categories, scores them, enriches the top candidates with candle data,
and sends everything to Claude for a structured morning briefing.

Usage:
    python screener.py              # full screener with Claude analysis
    python screener.py --raw        # raw signal data only, skip Claude
    python screener.py --top 10     # analyze top 10 instead of 5
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta

import config
from connect import get_session

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# gainersLosers endpoint data types
GAINERS_LOSERS_TYPES = [
    "PercPriceGainers",
    "PercPriceLosers",
    "PercOIGainers",
    "PercOILosers",
]

# oIBuildup endpoint data types (separate API, space-separated names)
OI_BUILDUP_TYPES = [
    "Long Built Up",
    "Short Built Up",
    "Long Unwinding",
    "Short Covering",
]

# Canonical short names used internally (for scoring, display, and Claude prompt)
# Maps API data type -> canonical name
CANONICAL_NAMES = {
    "PercPriceGainers": "PercPriceGainers",
    "PercPriceLosers": "PercPriceLosers",
    "PercOIGainers": "PercOIGainers",
    "PercOILosers": "PercOILosers",
    "Long Built Up": "LongBuildUp",
    "Short Built Up": "ShortBuildUp",
    "Long Unwinding": "LongUnwinding",
    "Short Covering": "ShortCovering",
}

ALL_SIGNAL_NAMES = list(CANONICAL_NAMES.values())

# Reverse mapping: canonical name -> API name (for buildup types only)
# Used by tools.py to route canonical names back to the oIBuildup endpoint
BUILDUP_API_NAMES = {v: k for k, v in CANONICAL_NAMES.items() if k in OI_BUILDUP_TYPES}

# Bullish signals: price up, OI up (longs building), shorts covering
BULLISH_CATEGORIES = {"PercPriceGainers", "PercOIGainers", "LongBuildUp", "ShortCovering"}
BEARISH_CATEGORIES = {"PercPriceLosers", "PercOILosers", "ShortBuildUp", "LongUnwinding"}

# Weights — buildups strongest (price + OI confirming), OI-based next, price-only baseline
SIGNAL_WEIGHTS = {
    "LongBuildUp": 2.0,
    "ShortBuildUp": 2.0,
    "PercOIGainers": 1.5,
    "PercOILosers": 1.5,
    "ShortCovering": 1.5,
    "LongUnwinding": 1.5,
    "PercPriceGainers": 1.0,
    "PercPriceLosers": 1.0,
}

# Symbols to skip equity enrichment for (indices)
INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"}

# Regex to extract underlying stock symbol from F&O trading symbol
# e.g. RELIANCE26FEB26FUT -> RELIANCE, M&M26FEB26FUT -> M&M
SYMBOL_REGEX = re.compile(r"^([A-Z&]+)\d")

SYSTEM_PROMPT = """You are a derivatives market analyst specializing in Indian F&O (NSE).

You receive pre-processed screener data: stocks that appeared in today's gainers/losers
across price movers, OI changes, and buildup/unwinding patterns, plus short-term candle
data and index-level PCR for the top candidates.

Your job is to synthesize this into a clear, actionable morning briefing. Be direct and
specific — traders reading this need to know what to watch and why, not generic disclaimers.

Interpret the signals:
- LongBuildUp (price up + OI up) = strong bullish — fresh longs being added
- ShortBuildUp (price down + OI up) = strong bearish — fresh shorts being added
- ShortCovering (price up + OI down) = bullish but weaker — shorts exiting, not new longs
- LongUnwinding (price down + OI down) = bearish but weaker — longs exiting, not new shorts
- PercOIGainers/Losers = OI change without price direction context
- PercPriceGainers/Losers = pure price movement

A stock appearing in multiple same-direction categories is a stronger signal.

IMPORTANT: You MUST respond with ONLY a valid JSON object — nothing else. No reasoning,
no markdown, no code fences, no explanation before or after. Start your response with {
and end with }. The JSON must have exactly these fields:

{
    "market_mood": "bullish" | "bearish" | "mixed" | "neutral",
    "mood_reasoning": "<1-2 sentences explaining the overall mood based on the data>",
    "watchlist": [
        {
            "symbol": "HDFCBANK",
            "direction": "bullish" | "bearish" | "watch",
            "conviction": "high" | "medium" | "low",
            "signals": ["LongBuildUp", "PercPriceGainers"],
            "price_change_pct": 3.2,
            "oi_insight": "<what OI data tells us about this stock>",
            "action": "Watch for breakout above 1,720",
            "short_term_view": "<1 sentence based on recent candle data if available>"
        }
    ],
    "sector_patterns": "<any sector-level patterns you notice — e.g., banking stocks dominating bullish signals>",
    "risk_flags": ["<any concerns — e.g., 'broad OI unwinding suggests caution'>"],
    "summary": "<3-5 sentence morning briefing paragraph>"
}"""


# ---------------------------------------------------------------------------
# Phase 1: Data Collection
# ---------------------------------------------------------------------------

_signal_cache: dict = {"data": None, "timestamp": None}
SIGNAL_CACHE_TTL = 300  # 5 minutes


def fetch_all_signals(smart_api, use_cache: bool = True) -> dict[str, list]:
    """
    Fetch all 8 signal types from SmartAPI using two endpoints:
    - gainersLosers for price/OI gainers and losers
    - oIBuildup for buildup/unwinding patterns

    Returns a dict mapping canonical_name -> list of records.
    Skips types that fail and continues with available data.

    Results are cached for 5 minutes to avoid redundant API calls
    when watchdog and screener run close together.
    """
    if (
        use_cache
        and _signal_cache["data"] is not None
        and _signal_cache["timestamp"] is not None
        and (datetime.now() - _signal_cache["timestamp"]).total_seconds() < SIGNAL_CACHE_TTL
    ):
        total = sum(len(v) for v in _signal_cache["data"].values())
        print(f"  Using cached signals ({total} records, "
              f"{int((datetime.now() - _signal_cache['timestamp']).total_seconds())}s old)")
        return _signal_cache["data"]

    signals = {}

    # Fetch price/OI gainers and losers via gainersLosers endpoint
    for dtype in GAINERS_LOSERS_TYPES:
        canonical = CANONICAL_NAMES[dtype]
        try:
            response = smart_api.gainersLosers({
                "datatype": dtype,
                "expirytype": "NEAR",
            })
            if response and response.get("data"):
                signals[canonical] = response["data"]
                print(f"  {canonical}: {len(response['data'])} stocks")
            else:
                print(f"  {canonical}: no data")
                signals[canonical] = []
        except Exception as e:
            print(f"  {canonical}: FAILED ({e})")
            signals[canonical] = []
        time.sleep(1.5)

    # Fetch buildup/unwinding patterns via oIBuildup endpoint
    for dtype in OI_BUILDUP_TYPES:
        canonical = CANONICAL_NAMES[dtype]
        try:
            response = smart_api.oIBuildup({
                "expirytype": "NEAR",
                "datatype": dtype,
            })
            if response and response.get("data"):
                signals[canonical] = response["data"]
                print(f"  {canonical}: {len(response['data'])} stocks")
            else:
                print(f"  {canonical}: no data")
                signals[canonical] = []
        except Exception as e:
            print(f"  {canonical}: FAILED ({e})")
            signals[canonical] = []
        time.sleep(1.5)

    _signal_cache["data"] = signals
    _signal_cache["timestamp"] = datetime.now()
    return signals


def extract_underlying(trading_symbol: str) -> str:
    """
    Extract the underlying stock symbol from an F&O trading symbol.

    e.g. RELIANCE26FEB26FUT -> RELIANCE
         M&M26FEB26FUT -> M&M
         BANKNIFTY26FEB26FUT -> BANKNIFTY
    """
    match = SYMBOL_REGEX.match(trading_symbol)
    if match:
        return match.group(1)
    # Fallback: strip trailing digits and known suffixes
    cleaned = re.sub(r"\d.*$", "", trading_symbol)
    return cleaned if cleaned else trading_symbol


# ---------------------------------------------------------------------------
# Phase 2: Cross-Reference & Score
# ---------------------------------------------------------------------------

def score_signals(signals: dict[str, list]) -> list[dict]:
    """
    Map each underlying symbol to the categories it appears in, compute a
    directional score, and return ranked candidates.

    Returns a list of dicts sorted by absolute score (strongest first):
    [{"symbol": "RELIANCE", "categories": [...], "score": 5.5, "direction": "bullish", ...}]
    """
    # Build per-symbol profile
    symbol_data = {}  # symbol -> {categories: set, details: {category: record}}

    for dtype, records in signals.items():
        for record in records:
            tsym = record.get("tradingSymbol", record.get("symbol", ""))
            underlying = extract_underlying(tsym)
            if not underlying:
                continue

            if underlying not in symbol_data:
                symbol_data[underlying] = {"categories": set(), "details": {}}

            symbol_data[underlying]["categories"].add(dtype)
            symbol_data[underlying]["details"][dtype] = record

    # Score each symbol
    candidates = []
    for symbol, data in symbol_data.items():
        categories = data["categories"]

        bullish_score = 0.0
        bearish_score = 0.0

        for cat in categories:
            weight = SIGNAL_WEIGHTS.get(cat, 1.0)
            if cat in BULLISH_CATEGORIES:
                bullish_score += weight
            elif cat in BEARISH_CATEGORIES:
                bearish_score += weight

        # Count same-direction categories for multi-signal bonus
        bullish_count = len(categories & BULLISH_CATEGORIES)
        bearish_count = len(categories & BEARISH_CATEGORIES)

        if bullish_count >= 3:
            bullish_score *= 1.5
        if bearish_count >= 3:
            bearish_score *= 1.5

        # Net directional score
        if bullish_score >= bearish_score:
            score = bullish_score
            direction = "bullish"
        else:
            score = bearish_score
            direction = "bearish"

        # Get price change from the best available record
        price_change_pct = None
        for cat in ["PercPriceGainers", "PercPriceLosers", "LongBuildUp",
                     "ShortBuildUp", "ShortCovering", "LongUnwinding"]:
            if cat in data["details"]:
                rec = data["details"][cat]
                pct = rec.get("percentChange")
                if pct is None:
                    pct = rec.get("perChange")
                if pct is not None:
                    try:
                        price_change_pct = float(pct)
                    except (ValueError, TypeError):
                        pass
                    break

        candidates.append({
            "symbol": symbol,
            "categories": sorted(categories),
            "score": round(score, 2),
            "direction": direction,
            "bullish_score": round(bullish_score, 2),
            "bearish_score": round(bearish_score, 2),
            "price_change_pct": price_change_pct,
            "details": data["details"],
        })

    # Sort by absolute score, then by number of categories
    candidates.sort(key=lambda c: (c["score"], len(c["categories"])), reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Phase 3: Enrichment
# ---------------------------------------------------------------------------

def enrich_candidates(smart_api, candidates: list[dict]) -> list[dict]:
    """
    For each candidate, fetch equity token via searchScrip and 5 days of
    daily candles for short-term context. Modifies candidates in-place.
    """
    for cand in candidates:
        symbol = cand["symbol"]

        if symbol in INDEX_SYMBOLS:
            cand["candles"] = None
            continue

        # Resolve equity token
        try:
            search_resp = smart_api.searchScrip("NSE", symbol)
            if search_resp and search_resp.get("data"):
                # Prefer -EQ suffix (equity segment), fall back to first result
                token = None
                for match in search_resp["data"]:
                    if match.get("tradingsymbol", "").endswith("-EQ"):
                        token = match.get("symboltoken")
                        break
                if not token:
                    token = search_resp["data"][0].get("symboltoken")

                if token:
                    cand["equity_token"] = token
                    # Fetch 5 days of candles
                    to_date = datetime.now()
                    from_date = to_date - timedelta(days=7)  # 7 calendar days ~ 5 trading
                    candle_resp = smart_api.getCandleData({
                        "exchange": "NSE",
                        "symboltoken": token,
                        "interval": "ONE_DAY",
                        "fromdate": from_date.strftime("%Y-%m-%d 09:15"),
                        "todate": to_date.strftime("%Y-%m-%d 15:30"),
                    })
                    if candle_resp and candle_resp.get("data"):
                        cand["candles"] = candle_resp["data"]
                        print(f"  {symbol}: {len(candle_resp['data'])} candles")
                    else:
                        cand["candles"] = None
                        print(f"  {symbol}: no candle data")
                else:
                    cand["candles"] = None
                    print(f"  {symbol}: token not found")
            else:
                cand["candles"] = None
                print(f"  {symbol}: search returned nothing")
        except Exception as e:
            cand["candles"] = None
            print(f"  {symbol}: enrichment failed ({e})")

        time.sleep(1.5)

    return candidates


def fetch_index_pcr(smart_api) -> dict | None:
    """Fetch put-call ratio data. Returns NIFTY PCR if found, else first available."""
    try:
        response = smart_api.putCallRatio()
        if response and response.get("data"):
            data = response["data"]
            # Find NIFTY in the list
            for item in data:
                if "NIFTY" in item.get("tradingSymbol", "").upper():
                    return item
            # Fallback: return all PCR data (usually just a handful)
            return data[:5] if isinstance(data, list) else data
    except Exception as e:
        print(f"  PCR: failed ({e})")
    return None


# ---------------------------------------------------------------------------
# Phase 4: Claude Analysis
# ---------------------------------------------------------------------------

def build_claude_prompt(candidates: list[dict], pcr_data: dict | None,
                        all_signals: dict[str, list]) -> str:
    """Build the user message for Claude with all screener data."""
    lines = []

    # Summary counts
    lines.append("=== F&O SCREENER DATA ===\n")
    lines.append("Signal counts across the F&O universe:")
    for dtype in ALL_SIGNAL_NAMES:
        count = len(all_signals.get(dtype, []))
        lines.append(f"  {dtype}: {count} stocks")
    lines.append("")

    # Top candidates
    lines.append(f"=== TOP {len(candidates)} CANDIDATES (ranked by signal strength) ===\n")
    for i, cand in enumerate(candidates, 1):
        lines.append(f"--- #{i}: {cand['symbol']} ---")
        lines.append(f"  Direction: {cand['direction']}")
        lines.append(f"  Score: {cand['score']} (bullish={cand['bullish_score']}, bearish={cand['bearish_score']})")
        lines.append(f"  Signals: {', '.join(cand['categories'])}")
        if cand["price_change_pct"] is not None:
            lines.append(f"  Price Change: {cand['price_change_pct']:+.2f}%")

        # Include raw detail data for each signal
        for cat in cand["categories"]:
            if cat in cand.get("details", {}):
                rec = cand["details"][cat]
                oi_val = rec.get("opnInterest")
                if oi_val is None:
                    oi_val = rec.get("openInterest")
                if oi_val is None:
                    oi_val = rec.get("oi")
                pct = rec.get("percentChange")
                if pct is None:
                    pct = rec.get("perChange")
                ltp = rec.get("ltp")
                if ltp is None:
                    ltp = rec.get("lastTradedPrice")
                detail_parts = []
                if ltp is not None:
                    detail_parts.append(f"LTP={ltp}")
                if pct is not None:
                    detail_parts.append(f"Change={pct}%")
                if oi_val is not None:
                    detail_parts.append(f"OI={oi_val}")
                if detail_parts:
                    lines.append(f"    {cat}: {', '.join(detail_parts)}")

        # Candle data
        if cand.get("candles"):
            lines.append(f"  Recent candles (5d):")
            lines.append(f"    {'Date':<12} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>12}")
            for candle in cand["candles"]:
                date_str = candle[0][:10] if isinstance(candle[0], str) else str(candle[0])[:10]
                lines.append(
                    f"    {date_str:<12} {candle[1]:>10.2f} {candle[2]:>10.2f} "
                    f"{candle[3]:>10.2f} {candle[4]:>10.2f} {candle[5]:>12,}"
                )
        lines.append("")

    # NIFTY PCR
    if pcr_data:
        lines.append("=== NIFTY PUT-CALL RATIO ===")
        lines.append(json.dumps(pcr_data, indent=2, default=str))
        lines.append("")

    lines.append(
        "Analyze this screener data. What's the market mood? Which stocks deserve attention "
        "and why? Any sector patterns? Give me an actionable morning briefing."
    )

    return "\n".join(lines)


def analyze_with_claude(candidates: list[dict], pcr_data: dict | None,
                        all_signals: dict[str, list]) -> dict:
    """Send screener data to Claude and get a structured morning briefing."""
    client = config.get_anthropic_client()
    user_message = build_claude_prompt(candidates, pcr_data, all_signals)

    print("Sending screener data to Claude for analysis...")

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
    except Exception as e:
        print(f"Claude API call failed: {e}")
        sys.exit(1)

    raw_text = response.content[0].text.strip()

    # Strip markdown code fences if present
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1]
        raw_text = raw_text.rsplit("```", 1)[0]
        raw_text = raw_text.strip()

    # Try direct parse first
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # Fallback: extract JSON object from mixed text (Claude sometimes reasons first)
    json_match = re.search(r"\{[\s\S]*\}", raw_text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    print("Claude returned invalid JSON:")
    print(raw_text[:500])
    sys.exit(1)


# ---------------------------------------------------------------------------
# Terminal Output
# ---------------------------------------------------------------------------

def wrap_text(text: str, width: int = 60, indent: str = "  ") -> list[str]:
    """Word-wrap text to the given width with indent."""
    if not text:
        return []
    words = text.split()
    lines = []
    current = indent
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = indent + word
        else:
            if current == indent:
                current += word
            else:
                current += " " + word
    if current.strip():
        lines.append(current)
    return lines


def print_raw_signals(signals: dict[str, list], candidates: list[dict]) -> None:
    """Print raw screener data without Claude analysis."""
    border = "═" * 60
    sep = "─" * 60

    print(f"\n{border}")
    print(f"  MORNING SCREENER — RAW SIGNALS")
    print(f"  {datetime.now().strftime('%d %b %Y, %H:%M')}")
    print(border)

    # Signal counts
    print(f"\n  Signal Counts:")
    print(f"  {sep[:40]}")
    for dtype in ALL_SIGNAL_NAMES:
        count = len(signals.get(dtype, []))
        label = dtype.ljust(20)
        print(f"  {label} {count:>3} stocks")

    # Top candidates
    print(f"\n{border}")
    print(f"  TOP CANDIDATES (by signal strength)")
    print(border)

    for i, cand in enumerate(candidates[:20], 1):
        direction_tag = "BULL" if cand["direction"] == "bullish" else "BEAR"
        pct = f"{cand['price_change_pct']:+.2f}%" if cand["price_change_pct"] is not None else "N/A"
        print(f"\n  {i:>2}. {cand['symbol']:<15} [{direction_tag}]  Score: {cand['score']:.1f}  Change: {pct}")
        print(f"      Signals: {', '.join(cand['categories'])}")

    print(f"\n{border}\n")


def print_briefing(analysis: dict) -> None:
    """Print the Claude-analyzed morning briefing."""
    border = "═" * 60
    sep = "─" * 60

    print(f"\n{border}")
    print(f"  MORNING SCREENER BRIEFING")
    print(f"  {datetime.now().strftime('%d %b %Y, %H:%M')}")
    print(border)

    # Market mood
    mood = analysis.get("market_mood", "unknown").upper()
    print(f"\n  Market Mood:  {mood}")
    for line in wrap_text(analysis.get("mood_reasoning", ""), 56, "  "):
        print(f"  {line}")

    # Watchlist
    watchlist = analysis.get("watchlist", [])
    if watchlist:
        print(f"\n  {sep}")
        print(f"  WATCHLIST ({len(watchlist)} stocks)")
        print(f"  {sep}")

        for i, item in enumerate(watchlist, 1):
            symbol = item.get("symbol", "?")
            direction = item.get("direction", "?").upper()
            conviction = item.get("conviction", "?").upper()
            signals = ", ".join(item.get("signals", []))
            pct = item.get("price_change_pct")
            pct_str = f" ({pct:+.1f}%)" if pct is not None else ""

            print(f"\n  {i}. {symbol}{pct_str}")
            print(f"     Direction: {direction}  |  Conviction: {conviction}")
            print(f"     Signals: {signals}")

            action = item.get("action", "")
            if action:
                for line in wrap_text(action, 52, ""):
                    print(f"     Action: {line}")

            oi_insight = item.get("oi_insight", "")
            if oi_insight:
                for line in wrap_text(oi_insight, 52, ""):
                    print(f"     OI: {line}")

            stv = item.get("short_term_view", "")
            if stv:
                for line in wrap_text(stv, 52, ""):
                    print(f"     View: {line}")

    # Sector patterns
    sector = analysis.get("sector_patterns", "")
    if sector:
        print(f"\n  {sep}")
        print(f"  SECTOR PATTERNS")
        for line in wrap_text(sector, 56, "  "):
            print(f"  {line}")

    # Risk flags
    risks = analysis.get("risk_flags", [])
    if risks:
        print(f"\n  {sep}")
        print(f"  RISK FLAGS")
        for flag in risks:
            for line in wrap_text(f"- {flag}", 56, "  "):
                print(f"  {line}")

    # Summary
    summary = analysis.get("summary", "")
    if summary:
        print(f"\n  {sep}")
        print(f"  SUMMARY")
        for line in wrap_text(summary, 56, "  "):
            print(f"  {line}")

    print(f"\n{border}\n")


# ---------------------------------------------------------------------------
# Public API (for tools.py integration)
# ---------------------------------------------------------------------------

def run_screener(smart_api, top_n: int = 5, raw_only: bool = False) -> dict:
    """
    Run the full morning screener pipeline.

    Args:
        smart_api: Authenticated SmartConnect session
        top_n: Number of top candidates to enrich and analyze
        raw_only: If True, skip Claude analysis and return raw signals

    Returns:
        dict with either raw signals or Claude analysis results
    """
    # Phase 1: Data Collection
    print("\nPhase 1: Fetching F&O signals...")
    all_signals = fetch_all_signals(smart_api)

    total = sum(len(v) for v in all_signals.values())
    if total == 0:
        return {"error": "No signal data returned from any category. Market may be closed."}

    # Phase 2: Cross-Reference & Score
    print("\nPhase 2: Scoring signals...")
    candidates = score_signals(all_signals)
    print(f"  {len(candidates)} unique stocks found across all categories")

    if not candidates:
        return {"error": "No scorable candidates found."}

    if raw_only:
        print_raw_signals(all_signals, candidates)
        return {
            "mode": "raw",
            "signal_counts": {dt: len(all_signals.get(dt, [])) for dt in ALL_SIGNAL_NAMES},
            "candidates": [
                {k: v for k, v in c.items() if k != "details"}
                for c in candidates[:20]
            ],
        }

    # Phase 3: Enrichment
    top = candidates[:top_n]
    print(f"\nPhase 3: Enriching top {len(top)} candidates...")
    enrich_candidates(smart_api, top)

    print("  Fetching NIFTY PCR...")
    pcr_data = fetch_index_pcr(smart_api)

    # Phase 4: Claude Analysis
    print("\nPhase 4: Claude analysis...")
    analysis = analyze_with_claude(top, pcr_data, all_signals)

    print_briefing(analysis)

    return {
        "mode": "full",
        "analysis": analysis,
        "candidates_analyzed": len(top),
    }


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Morning F&O screener — scan market for signals")
    parser.add_argument("--raw", action="store_true",
                        help="Raw signal data only, skip Claude analysis")
    parser.add_argument("--top", type=int, default=5,
                        help="Number of top candidates to analyze (default: 5)")
    args = parser.parse_args()

    print("Connecting to AngelOne SmartAPI...")
    smart_api = get_session()

    run_screener(smart_api, top_n=args.top, raw_only=args.raw)


if __name__ == "__main__":
    main()
