"""
Portfolio Watchdog & Deep Dive Agent — makes screener data actionable for your portfolio.

Features:
  1. Watchdog — Cross-references portfolio holdings against today's F&O signals.
     Flags stocks to EXIT, HOLD/ADD, or MONITOR with urgency levels.
  2. Deep Dive — Comprehensive single-stock analysis combining candles + option chain
     + OI signals + PCR into an actionable trade plan.

Usage:
    python portfolio_analysis.py watchdog              # match holdings against signals
    python portfolio_analysis.py watchdog --detailed   # + candle enrichment for flagged stocks
    python portfolio_analysis.py deepdive RELIANCE     # auto-resolve token
    python portfolio_analysis.py deepdive RELIANCE --token 2885  # explicit token
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta

import config
from connect import get_session
from screener import (
    BULLISH_CATEGORIES,
    BEARISH_CATEGORIES,
    SIGNAL_WEIGHTS,
    ALL_SIGNAL_NAMES,
    INDEX_SYMBOLS,
    extract_underlying,
    fetch_all_signals,
    fetch_index_pcr,
)
from fetch_data import fetch_candles
from agent_with_options import (
    fetch_option_chain,
    format_candles_for_prompt,
    format_options_for_prompt,
    get_nearest_expiry,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def parse_claude_json(raw_text: str) -> dict:
    """Parse JSON from Claude's response, stripping markdown fences if present."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: extract JSON object from mixed text
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from Claude response: {text[:300]}")


def resolve_equity_token(smart_api, symbol: str) -> str | None:
    """Resolve a stock symbol to its NSE equity token via searchScrip."""
    try:
        resp = smart_api.searchScrip("NSE", symbol)
        if resp and resp.get("data"):
            # Prefer -EQ suffix (equity segment), fall back to first result
            token = None
            for match in resp["data"]:
                if match.get("tradingsymbol", "").endswith("-EQ"):
                    token = match.get("symboltoken")
                    break
            if not token:
                token = resp["data"][0].get("symboltoken")
            return token
    except Exception as e:
        print(f"  searchScrip failed for {symbol}: {e}")
    return None


# ---------------------------------------------------------------------------
# Feature 1: Portfolio Watchdog
# ---------------------------------------------------------------------------

WATCHDOG_SYSTEM_PROMPT = """You are a portfolio risk analyst specializing in Indian F&O (NSE).

You receive the user's current holdings and open positions, cross-referenced against today's
F&O signals (OI buildups, price/OI movers). Your job is to flag actionable items — what to
exit, what to hold, what needs monitoring.

Signal interpretation:
- LongBuildUp (price up + OI up) = strong bullish — fresh longs being added
- ShortBuildUp (price down + OI up) = strong bearish — fresh shorts being added
- ShortCovering (price up + OI down) = bullish but weaker — shorts exiting
- LongUnwinding (price down + OI down) = bearish but weaker — longs exiting
- PercOIGainers/Losers = OI change without price direction context
- PercPriceGainers/Losers = pure price movement

For LONG positions:
- LongUnwinding or ShortBuildUp = danger signals → consider EXIT/REDUCE
- LongBuildUp or ShortCovering = confirmation → HOLD/ADD
- Price/OI movers only = MONITOR

For SHORT/PUT positions, signals reverse.

IMPORTANT: Respond with ONLY a valid JSON object. No markdown, no code fences. The JSON must
have exactly these fields:

{
    "portfolio_health": "strong" | "caution" | "warning",
    "health_reasoning": "1-2 sentences on overall portfolio health",
    "action_items": [
        {
            "symbol": "RELIANCE",
            "current_position": {"quantity": 100, "avg_price": 1280, "ltp": 1295, "pnl_pct": 1.17},
            "signals": ["LongBuildUp", "PercOIGainers"],
            "signal_direction": "bullish" | "bearish" | "neutral",
            "action": "HOLD/ADD" | "EXIT/REDUCE" | "MONITOR",
            "reasoning": "why this action",
            "urgency": "high" | "medium" | "low"
        }
    ],
    "unmatched_holdings": ["TCS", "WIPRO"],
    "summary": "2-3 sentences"
}"""


def normalize_holding_symbol(trading_symbol: str) -> str:
    """Normalize a holdings symbol: strip exchange suffix (e.g. -EQ, -BE, -BL)."""
    sym = trading_symbol.strip()
    if "-" in sym:
        base = sym.rsplit("-", 1)[0]
        if base:
            return base
    return sym


def fetch_portfolio(smart_api) -> tuple[list[dict], list[dict]]:
    """
    Fetch holdings and positions from SmartAPI.
    Returns (holdings_list, positions_list) with normalized symbols.
    """
    holdings = []
    positions = []

    # Holdings (delivery)
    try:
        resp = smart_api.allholding()
        if resp and resp.get("data"):
            raw = resp["data"].get("holdings", resp["data"])
            if isinstance(raw, list):
                for h in raw:
                    sym = h.get("tradingsymbol", h.get("symbolname", ""))
                    holdings.append({
                        "raw_symbol": sym,
                        "symbol": normalize_holding_symbol(sym),
                        "quantity": h.get("quantity", 0),
                        "avg_price": h.get("averageprice", 0),
                        "ltp": h.get("ltp", 0),
                        "pnl": h.get("profitandloss", 0),
                        "pnl_pct": h.get("pnlpercentage", 0),
                    })
    except Exception as e:
        print(f"  Holdings fetch failed: {e}")

    time.sleep(1.5)

    # Positions (F&O / intraday)
    try:
        resp = smart_api.position()
        if resp and resp.get("data"):
            for p in resp["data"]:
                tsym = p.get("tradingsymbol", "")
                positions.append({
                    "raw_symbol": tsym,
                    "symbol": extract_underlying(tsym),
                    "trading_symbol": tsym,
                    "quantity": int(p.get("netqty", 0)),
                    "avg_price": float(p.get("averageprice", 0)),
                    "ltp": float(p.get("ltp", 0)),
                    "pnl": float(p.get("pnl", 0)),
                    "product_type": p.get("producttype", ""),
                    "exchange": p.get("exchange", ""),
                })
    except Exception as e:
        print(f"  Positions fetch failed: {e}")

    return holdings, positions


def match_signals_to_portfolio(
    holdings: list[dict],
    positions: list[dict],
    signals: dict[str, list],
) -> list[dict]:
    """
    Match each portfolio stock against the 8 signal categories.
    Returns a list of match records with signal info attached.
    """
    # Build signal lookup: underlying symbol -> set of categories + details
    signal_map: dict[str, dict] = {}
    for category, records in signals.items():
        for rec in records:
            tsym = rec.get("tradingSymbol", rec.get("symbol", ""))
            underlying = extract_underlying(tsym)
            if underlying not in signal_map:
                signal_map[underlying] = {"categories": set(), "details": {}}
            signal_map[underlying]["categories"].add(category)
            signal_map[underlying]["details"][category] = rec

    # Merge holdings and positions into unique symbols
    portfolio_items = []
    seen = set()

    for h in holdings:
        sym = h["symbol"]
        if sym in seen or not sym:
            continue
        seen.add(sym)
        portfolio_items.append({
            "symbol": sym,
            "source": "holding",
            "quantity": h["quantity"],
            "avg_price": h["avg_price"],
            "ltp": h["ltp"],
            "pnl_pct": h["pnl_pct"],
        })

    for p in positions:
        sym = p["symbol"]
        if sym in seen or not sym:
            continue
        seen.add(sym)
        pnl_pct = 0
        if p["avg_price"] and p["avg_price"] != 0:
            pnl_pct = round((p["ltp"] - p["avg_price"]) / p["avg_price"] * 100, 2)
        portfolio_items.append({
            "symbol": sym,
            "source": "position",
            "quantity": p["quantity"],
            "avg_price": p["avg_price"],
            "ltp": p["ltp"],
            "pnl_pct": pnl_pct,
        })

    # Match
    matched = []
    unmatched = []

    for item in portfolio_items:
        sym = item["symbol"]
        if sym in signal_map:
            cats = sorted(signal_map[sym]["categories"])
            # Determine direction
            bullish = signal_map[sym]["categories"] & BULLISH_CATEGORIES
            bearish = signal_map[sym]["categories"] & BEARISH_CATEGORIES
            if bullish and not bearish:
                direction = "bullish"
            elif bearish and not bullish:
                direction = "bearish"
            elif bullish and bearish:
                direction = "mixed"
            else:
                direction = "neutral"

            matched.append({
                **item,
                "signals": cats,
                "signal_direction": direction,
                "signal_details": signal_map[sym]["details"],
            })
        else:
            unmatched.append(sym)

    return matched, unmatched


def build_watchdog_prompt(
    matched: list[dict],
    unmatched: list[str],
    signals: dict[str, list],
    candle_data: dict[str, list] | None = None,
) -> str:
    """Build the user message for the watchdog Claude call."""
    lines = []

    lines.append("=== PORTFOLIO WATCHDOG ===\n")
    lines.append(f"Date: {datetime.now().strftime('%d %b %Y, %H:%M')}\n")

    # Signal summary
    lines.append("Today's F&O signal counts:")
    for name in ALL_SIGNAL_NAMES:
        lines.append(f"  {name}: {len(signals.get(name, []))} stocks")
    lines.append("")

    # Portfolio matches
    lines.append(f"=== PORTFOLIO STOCKS WITH SIGNALS ({len(matched)}) ===\n")
    for item in matched:
        lines.append(f"--- {item['symbol']} ({item['source']}) ---")
        lines.append(f"  Position: {item['quantity']} qty @ avg ₹{item['avg_price']:.2f}, "
                     f"LTP ₹{item['ltp']:.2f}, P&L {item['pnl_pct']:+.2f}%")
        lines.append(f"  Signals: {', '.join(item['signals'])}")
        lines.append(f"  Signal direction: {item['signal_direction']}")

        # Raw signal details
        for cat in item["signals"]:
            det = item["signal_details"].get(cat, {})
            pct = det.get("percentChange")
            if pct is None:
                pct = det.get("perChange")
            oi = det.get("opnInterest")
            if oi is None:
                oi = det.get("openInterest")
            ltp = det.get("ltp")
            if ltp is None:
                ltp = det.get("lastTradedPrice")
            parts = []
            if ltp is not None:
                parts.append(f"LTP={ltp}")
            if pct is not None:
                parts.append(f"Change={pct}%")
            if oi is not None:
                parts.append(f"OI={oi}")
            if parts:
                lines.append(f"    {cat}: {', '.join(parts)}")

        # Candle data if available
        if candle_data and item["symbol"] in candle_data:
            candles = candle_data[item["symbol"]]
            lines.append(f"  Recent candles ({len(candles)}d):")
            lines.append(f"    {'Date':<12} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>12}")
            for c in candles:
                d = c[0][:10] if isinstance(c[0], str) else str(c[0])[:10]
                lines.append(f"    {d:<12} {c[1]:>10.2f} {c[2]:>10.2f} {c[3]:>10.2f} {c[4]:>10.2f} {c[5]:>12,}")

        lines.append("")

    if unmatched:
        lines.append(f"=== PORTFOLIO STOCKS WITH NO SIGNALS ({len(unmatched)}) ===")
        lines.append(f"  {', '.join(unmatched)}")
        lines.append("")

    lines.append(
        "Analyze my portfolio against today's signals. For each matched stock, "
        "recommend an action (HOLD/ADD, EXIT/REDUCE, or MONITOR) with urgency. "
        "Assess overall portfolio health."
    )

    return "\n".join(lines)


def enrich_flagged_with_candles(
    smart_api, matched: list[dict],
) -> dict[str, list]:
    """Fetch 5-day candles for stocks that have bearish signals (flagged for exit/reduce)."""
    candle_data = {}
    flagged = [
        m for m in matched
        if m["signal_direction"] in ("bearish", "mixed")
        and m["symbol"] not in INDEX_SYMBOLS
    ]

    if not flagged:
        print("  No flagged stocks to enrich with candles.")
        return candle_data

    print(f"  Enriching {len(flagged)} flagged stocks with candle data...")
    for item in flagged:
        sym = item["symbol"]
        token = resolve_equity_token(smart_api, sym)
        if token:
            time.sleep(1.5)
            candles = fetch_candles(smart_api, symbol=sym, token=token, days=7)
            if candles:
                candle_data[sym] = candles
                print(f"    {sym}: {len(candles)} candles")
            else:
                print(f"    {sym}: no candle data")
        else:
            print(f"    {sym}: could not resolve token")
        time.sleep(1.5)

    return candle_data


def run_watchdog(smart_api, detailed: bool = False) -> dict:
    """
    Run the portfolio watchdog pipeline.

    Args:
        smart_api: Authenticated SmartConnect session
        detailed: If True, fetch candle data for flagged stocks

    Returns:
        dict with watchdog analysis results
    """
    # Step 1: Fetch portfolio
    print("\nStep 1: Fetching portfolio...")
    holdings, positions = fetch_portfolio(smart_api)
    total = len(holdings) + len(positions)
    print(f"  {len(holdings)} holdings, {len(positions)} positions")

    if total == 0:
        msg = "No holdings or positions found. Portfolio is empty."
        print(f"  {msg}")
        return {"error": msg, "suggestion": "Run the screener to find opportunities."}

    # Step 2: Fetch F&O signals
    print("\nStep 2: Fetching F&O signals...")
    signals = fetch_all_signals(smart_api)
    signal_total = sum(len(v) for v in signals.values())
    if signal_total == 0:
        return {"error": "No signal data returned. Market may be closed."}

    # Step 3: Match signals to portfolio
    print("\nStep 3: Matching signals to portfolio...")
    matched, unmatched = match_signals_to_portfolio(holdings, positions, signals)
    print(f"  {len(matched)} stocks matched, {len(unmatched)} without signals")

    # Step 4: Optional candle enrichment
    candle_data = None
    if detailed and matched:
        print("\nStep 4: Enriching flagged stocks with candles...")
        candle_data = enrich_flagged_with_candles(smart_api, matched)
    elif detailed:
        print("\nStep 4: Skipped — no matched stocks to enrich.")

    # Step 5: Claude analysis
    step = "5" if detailed else "4"
    print(f"\nStep {step}: Sending to Claude for analysis...")
    prompt = build_watchdog_prompt(matched, unmatched, signals, candle_data)

    client = config.get_anthropic_client()
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=WATCHDOG_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        print(f"Claude API call failed: {e}")
        return {"error": f"Claude analysis failed: {e}"}

    try:
        analysis = parse_claude_json(response.content[0].text)
    except ValueError as e:
        print(str(e))
        return {"error": "Claude returned invalid JSON"}

    print_watchdog_report(analysis)
    return {"mode": "watchdog", "analysis": analysis, "matched": len(matched), "unmatched": unmatched}


def print_watchdog_report(analysis: dict) -> None:
    """Print the watchdog report to terminal."""
    border = "═" * 60
    sep = "─" * 60

    print(f"\n{border}")
    print(f"  PORTFOLIO WATCHDOG")
    print(f"  {datetime.now().strftime('%d %b %Y, %H:%M')}")
    print(border)

    # Health
    health = analysis.get("portfolio_health", "unknown").upper()
    health_icon = {"STRONG": "+", "CAUTION": "~", "WARNING": "!"}.get(health, "?")
    print(f"\n  [{health_icon}] Portfolio Health: {health}")
    reasoning = analysis.get("health_reasoning", "")
    if reasoning:
        for line in _wrap(reasoning):
            print(f"  {line}")

    # Action items
    items = analysis.get("action_items", [])
    if items:
        print(f"\n  {sep}")
        print(f"  ACTION ITEMS ({len(items)} stocks)")
        print(f"  {sep}")

        for i, item in enumerate(items, 1):
            sym = item.get("symbol", "?")
            action = item.get("action", "?")
            urgency = item.get("urgency", "?").upper()
            direction = item.get("signal_direction", "?")
            signals = ", ".join(item.get("signals", []))

            pos = item.get("current_position", {})
            qty = pos.get("quantity", "?")
            pnl = pos.get("pnl_pct")
            pnl_str = f" ({pnl:+.2f}%)" if pnl is not None else ""

            print(f"\n  {i}. {sym}{pnl_str}")
            print(f"     Action: {action}  |  Urgency: {urgency}  |  Direction: {direction}")
            print(f"     Signals: {signals}")

            reason = item.get("reasoning", "")
            if reason:
                for line in _wrap(reason, 52):
                    print(f"     {line}")

    # Unmatched
    unmatched = analysis.get("unmatched_holdings", [])
    if unmatched:
        print(f"\n  {sep}")
        print(f"  NO SIGNALS: {', '.join(unmatched)}")

    # Summary
    summary = analysis.get("summary", "")
    if summary:
        print(f"\n  {sep}")
        print(f"  SUMMARY")
        for line in _wrap(summary, 56):
            print(f"  {line}")

    print(f"\n{border}\n")


# ---------------------------------------------------------------------------
# Feature 2: Deep Dive Agent
# ---------------------------------------------------------------------------

DEEPDIVE_SYSTEM_PROMPT = """You are a derivatives trading strategist specializing in Indian F&O (NSE).

You receive comprehensive data for a single stock: 30-day candle data, option chain with Greeks,
OI buildup signals, and put-call ratio. Your job is to synthesize all of this into an actionable
trade plan with specific entry, targets, and stop-loss levels.

Signal interpretation:
- LongBuildUp (price up + OI up) = strong bullish — fresh longs being added
- ShortBuildUp (price down + OI up) = strong bearish — fresh shorts being added
- ShortCovering (price up + OI down) = bullish but weaker — shorts exiting
- LongUnwinding (price down + OI down) = bearish but weaker — longs exiting

Option chain insights:
- High call OI at a strike = resistance (call writers don't expect price above)
- High put OI at a strike = support (put writers don't expect price below)
- Max pain = strike where option buyers lose the most (price tends to gravitate here)
- IV skew reveals directional fear — higher put IV = downside fear

IMPORTANT: Respond with ONLY a valid JSON object. No markdown, no code fences. The JSON must
have exactly these fields:

{
    "overall_bias": "bullish" | "bearish" | "neutral",
    "confidence": "high" | "medium" | "low",
    "current_price": 1295.0,
    "price_analysis": {
        "trend": "bullish" | "bearish" | "sideways",
        "support_levels": [1278, 1250],
        "resistance_levels": [1315, 1340],
        "volume_profile": "description of volume pattern"
    },
    "options_analysis": {
        "sentiment": "bullish" | "bearish" | "neutral",
        "iv_assessment": "description of IV levels and skew",
        "max_pain": 1300,
        "key_insight": "1-2 sentences on what options market is pricing"
    },
    "oi_signals": {
        "categories": ["LongBuildUp"],
        "interpretation": "what the OI buildup pattern means"
    },
    "trade_plan": {
        "direction": "LONG" | "SHORT" | "WAIT",
        "entry_range": [1290, 1295],
        "targets": [1315, 1340],
        "stop_loss": 1275,
        "risk_reward": "1:2.5",
        "holding_period": "intraday" | "swing" | "positional"
    },
    "key_risks": ["risk 1", "risk 2"],
    "summary": "3-4 sentences"
}"""


def fetch_stock_signals(signals: dict[str, list], symbol: str) -> list[str]:
    """Check which of the 8 signal categories contain this stock."""
    found = []
    for category, records in signals.items():
        for rec in records:
            tsym = rec.get("tradingSymbol", rec.get("symbol", ""))
            if extract_underlying(tsym) == symbol:
                found.append(category)
                break
    return sorted(found)


def fetch_stock_pcr(smart_api, symbol: str) -> dict | None:
    """Fetch PCR data for a specific symbol."""
    try:
        resp = smart_api.putCallRatio()
        if resp and resp.get("data"):
            for item in resp["data"]:
                if symbol.upper() in item.get("tradingSymbol", "").upper():
                    return item
    except Exception as e:
        print(f"  PCR fetch failed: {e}")
    return None


def build_deepdive_prompt(
    symbol: str,
    candles: list | None,
    options_data: list | None,
    stock_signals: list[str],
    pcr_data: dict | None,
) -> str:
    """Build the user message for the deep dive Claude call."""
    lines = [f"=== DEEP DIVE: {symbol} ===\n"]

    # Price data
    if candles:
        lines.append(format_candles_for_prompt(candles, symbol))
        lines.append("")
    else:
        lines.append("No candle data available.\n")

    # Option chain
    if options_data:
        lines.append(format_options_for_prompt(options_data))
        lines.append("")
    else:
        lines.append("No option chain data available (stock may not have F&O contracts).\n")

    # OI signals
    lines.append("=== OI BUILDUP SIGNALS ===")
    if stock_signals:
        lines.append(f"  {symbol} appears in: {', '.join(stock_signals)}")
        # Interpret direction
        bullish = set(stock_signals) & BULLISH_CATEGORIES
        bearish = set(stock_signals) & BEARISH_CATEGORIES
        if bullish:
            lines.append(f"  Bullish signals: {', '.join(sorted(bullish))}")
        if bearish:
            lines.append(f"  Bearish signals: {', '.join(sorted(bearish))}")
    else:
        lines.append(f"  {symbol} not found in any OI buildup/unwinding category today.")
    lines.append("")

    # PCR
    lines.append("=== PUT-CALL RATIO ===")
    if pcr_data:
        lines.append(json.dumps(pcr_data, indent=2, default=str))
    else:
        lines.append(f"  No PCR data found for {symbol}.")
    lines.append("")

    lines.append(
        f"Provide a comprehensive analysis and trade plan for {symbol}. "
        f"Cross-reference price action, options chain, OI signals, and PCR. "
        f"Give specific entry, targets, and stop-loss levels with risk:reward ratio."
    )

    return "\n".join(lines)


def run_deep_dive(smart_api, symbol: str, token: str | None = None) -> dict:
    """
    Run the deep dive analysis pipeline for a single stock.

    Args:
        smart_api: Authenticated SmartConnect session
        symbol: Stock symbol (e.g. 'RELIANCE')
        token: SmartAPI token. If None, auto-resolved via searchScrip.

    Returns:
        dict with deep dive analysis results
    """
    symbol = symbol.upper().replace("-EQ", "")

    # Step 1: Resolve token
    if not token:
        print(f"\nStep 1: Resolving token for {symbol}...")
        token = resolve_equity_token(smart_api, symbol)
        if not token:
            return {"error": f"Could not resolve token for {symbol}. Check the symbol name."}
        print(f"  Token: {token}")
        time.sleep(1.5)
    else:
        print(f"\nStep 1: Using provided token {token} for {symbol}")

    # Step 2: Fetch 30-day candles
    print(f"\nStep 2: Fetching 30-day candle data...")
    candles = fetch_candles(smart_api, symbol=symbol, token=token, days=30)
    if candles:
        print(f"  Got {len(candles)} candles")
    else:
        print(f"  No candle data returned")
    time.sleep(1.5)

    # Step 3: Fetch option chain
    print(f"\nStep 3: Fetching option chain...")
    options_data = fetch_option_chain(smart_api, symbol, token)
    if options_data:
        print(f"  Got {len(options_data)} strikes")
    else:
        print(f"  No option chain (stock may not have F&O contracts)")
    time.sleep(1.5)

    # Step 4: Fetch F&O signals
    print(f"\nStep 4: Fetching F&O signals...")
    all_signals = fetch_all_signals(smart_api)
    stock_signals = fetch_stock_signals(all_signals, symbol)
    if stock_signals:
        print(f"  {symbol} found in: {', '.join(stock_signals)}")
    else:
        print(f"  {symbol} not in any signal category today")

    # Step 5: Fetch PCR
    print(f"\nStep 5: Fetching PCR...")
    pcr_data = fetch_stock_pcr(smart_api, symbol)
    if pcr_data:
        print(f"  PCR: {pcr_data.get('pcr', 'N/A')}")
    else:
        print(f"  No PCR data for {symbol}")
    time.sleep(1.5)

    # Step 6: Claude analysis
    print(f"\nStep 6: Sending to Claude for analysis...")
    prompt = build_deepdive_prompt(symbol, candles, options_data, stock_signals, pcr_data)

    if not candles and not options_data:
        return {"error": f"No price or options data available for {symbol}. Cannot analyze."}

    client = config.get_anthropic_client()
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=DEEPDIVE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        print(f"Claude API call failed: {e}")
        return {"error": f"Claude analysis failed: {e}"}

    try:
        analysis = parse_claude_json(response.content[0].text)
    except ValueError as e:
        print(str(e))
        return {"error": "Claude returned invalid JSON"}

    print_deepdive_report(analysis, symbol)
    return {"mode": "deep_dive", "symbol": symbol, "analysis": analysis}


def print_deepdive_report(analysis: dict, symbol: str) -> None:
    """Print the deep dive report to terminal."""
    border = "═" * 60
    sep = "─" * 60

    print(f"\n{border}")
    print(f"  DEEP DIVE: {symbol}")
    print(f"  {datetime.now().strftime('%d %b %Y, %H:%M')}")
    print(border)

    # Bias & confidence
    bias = analysis.get("overall_bias", "?").upper()
    confidence = analysis.get("confidence", "?").upper()
    price = analysis.get("current_price")
    price_str = f"₹{price:,.2f}" if price else "N/A"
    print(f"\n  Bias: {bias}  |  Confidence: {confidence}  |  Price: {price_str}")

    # Price analysis
    pa = analysis.get("price_analysis", {})
    if pa:
        print(f"\n  {sep}")
        print(f"  PRICE ANALYSIS")
        print(f"  Trend: {pa.get('trend', '?').upper()}")
        supports = pa.get("support_levels", [])
        resists = pa.get("resistance_levels", [])
        if supports:
            print(f"  Support:    {', '.join(f'₹{s:,.0f}' for s in supports)}")
        if resists:
            print(f"  Resistance: {', '.join(f'₹{r:,.0f}' for r in resists)}")
        vol = pa.get("volume_profile", "")
        if vol:
            for line in _wrap(vol, 52):
                print(f"  Volume: {line}")

    # Options analysis
    oa = analysis.get("options_analysis", {})
    if oa and oa.get("sentiment"):
        print(f"\n  {sep}")
        print(f"  OPTIONS ANALYSIS")
        print(f"  Sentiment: {oa.get('sentiment', '?').upper()}")
        mp = oa.get("max_pain")
        if mp:
            print(f"  Max Pain:  ₹{mp:,.0f}")
        iv = oa.get("iv_assessment", "")
        if iv:
            for line in _wrap(iv, 52):
                print(f"  IV: {line}")
        ki = oa.get("key_insight", "")
        if ki:
            for line in _wrap(ki, 52):
                print(f"  Insight: {line}")

    # OI signals
    oi = analysis.get("oi_signals", {})
    if oi:
        cats = oi.get("categories", [])
        interp = oi.get("interpretation", "")
        if cats or interp:
            print(f"\n  {sep}")
            print(f"  OI SIGNALS")
            if cats:
                print(f"  Categories: {', '.join(cats)}")
            if interp:
                for line in _wrap(interp, 52):
                    print(f"  {line}")

    # Trade plan
    tp = analysis.get("trade_plan", {})
    if tp:
        print(f"\n  {sep}")
        print(f"  TRADE PLAN")
        direction = tp.get("direction", "?")
        entry = tp.get("entry_range", [])
        targets = tp.get("targets", [])
        sl = tp.get("stop_loss")
        rr = tp.get("risk_reward", "?")
        period = tp.get("holding_period", "?")

        print(f"  Direction:      {direction}")
        if entry:
            print(f"  Entry:          ₹{entry[0]:,.0f} — ₹{entry[1]:,.0f}" if len(entry) == 2
                  else f"  Entry:          ₹{entry[0]:,.0f}")
        if targets:
            print(f"  Targets:        {', '.join(f'₹{t:,.0f}' for t in targets)}")
        if sl:
            print(f"  Stop Loss:      ₹{sl:,.0f}")
        print(f"  Risk:Reward:    {rr}")
        print(f"  Holding Period: {period}")

    # Risks
    risks = analysis.get("key_risks", [])
    if risks:
        print(f"\n  {sep}")
        print(f"  KEY RISKS")
        for risk in risks:
            for line in _wrap(f"- {risk}", 56):
                print(f"  {line}")

    # Summary
    summary = analysis.get("summary", "")
    if summary:
        print(f"\n  {sep}")
        print(f"  SUMMARY")
        for line in _wrap(summary, 56):
            print(f"  {line}")

    print(f"\n{border}\n")


# ---------------------------------------------------------------------------
# Shared formatting
# ---------------------------------------------------------------------------

def _wrap(text: str, width: int = 56) -> list[str]:
    """Word-wrap text to the given width."""
    if not text:
        return []
    words = text.split()
    lines = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        lines.append(current)
    return lines


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Portfolio Watchdog & Deep Dive Agent",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Watchdog subcommand
    watch_parser = subparsers.add_parser("watchdog", help="Check portfolio against F&O signals")
    watch_parser.add_argument("--detailed", action="store_true",
                              help="Fetch candle data for flagged stocks")

    # Deep dive subcommand
    dive_parser = subparsers.add_parser("deepdive", help="Comprehensive single-stock analysis")
    dive_parser.add_argument("symbol", help="Stock symbol (e.g. RELIANCE)")
    dive_parser.add_argument("--token", help="SmartAPI token (auto-resolved if omitted)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    print("Connecting to AngelOne SmartAPI...")
    smart_api = get_session()

    if args.command == "watchdog":
        run_watchdog(smart_api, detailed=args.detailed)
    elif args.command == "deepdive":
        run_deep_dive(smart_api, args.symbol, token=args.token)


if __name__ == "__main__":
    main()
