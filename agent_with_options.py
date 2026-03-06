"""
Enhanced financial agent with options chain analysis.

Does everything agent.py does, plus fetches option chain data (IV, OI, Greeks)
and asks Claude to cross-reference what the options market is pricing in
versus what the stock price action shows.

Usage:
    python agent_with_options.py
    python agent_with_options.py --symbol RELIANCE --token 2885
"""

import argparse
import json
import sys

import anthropic

import config
from connect import get_session
from utils import parse_claude_json
from fetch_data import fetch_candles

SYSTEM_PROMPT = """You are a financial analyst specializing in Indian equities and derivatives (NSE F&O).

You analyze both stock price data AND options chain data to give a complete picture. You are
practical and direct — you say what the data shows without excessive hedging.

When given OHLCV data and an options chain, you cross-reference:
- Price trend from the stock data
- What the options market is pricing in (from IV, OI distribution, put/call ratios)
- Whether options sentiment confirms or contradicts the price action
- Key support/resistance implied by max pain or high OI strikes

You MUST respond with ONLY a valid JSON object. No markdown, no code fences, no text outside
the JSON. The JSON must have exactly these fields:

{
    "trend": "bullish" | "bearish" | "sideways",
    "support": <number>,
    "resistance": <number>,
    "avg_volume": <number>,
    "volume_trend": "increasing" | "decreasing" | "stable",
    "options_sentiment": "bullish" | "bearish" | "neutral",
    "iv_percentile": "<description of where current IV sits — e.g., 'elevated', 'low', 'average'>",
    "pcr_signal": "<what the put/call ratio suggests>",
    "max_pain": <number — the max pain strike if identifiable, otherwise null>,
    "summary": "<3-4 sentences covering both price action and options analysis>",
    "options_insight": "<2-3 sentences specifically about what the options market is telling us>",
    "confidence": "high" | "medium" | "low"
}"""


def get_nearest_expiry(min_dte: int = 0) -> str:
    """Get the nearest monthly options expiry (last Thursday of the month) in DDMONYYYY format.

    Args:
        min_dte: minimum calendar days to expiry. If the nearest expiry is
                 closer than this, skip to the next month's expiry.
    """
    from datetime import date, timedelta
    import calendar

    today = date.today()
    # Check current month and next 2 months
    for month_offset in range(3):
        y = today.year + (today.month + month_offset - 1) // 12
        m = (today.month + month_offset - 1) % 12 + 1
        # Find last Thursday: start from last day, walk back
        last_day = calendar.monthrange(y, m)[1]
        d = date(y, m, last_day)
        while d.weekday() != 3:  # Thursday
            d -= timedelta(days=1)
        if d >= today and (d - today).days >= min_dte:
            return d.strftime("%d%b%Y").upper()  # e.g. 27FEB2026
    return ""


def fetch_option_chain(smart_api, symbol: str, token: str, exchange: str = "NFO") -> dict | None:
    """
    Fetch option chain data for a stock from SmartAPI.

    Returns the raw option chain response, or None on failure.
    """
    print(f"Fetching option chain for {symbol}...")

    is_open, market_status = config.is_market_open()
    if not is_open:
        print(f"  Warning: {market_status} — option chain data may be unavailable or stale.")

    expiry = get_nearest_expiry()
    print(f"Using expiry: {expiry}")

    try:
        response = smart_api.optionGreek({
            "name": symbol,
            "expirydate": expiry,
        })
    except Exception as e:
        print(f"Option chain API call failed: {e}")
        print("This might mean the stock doesn't have F&O contracts,")
        print("or SmartAPI's option chain endpoint is temporarily down.")
        return None

    if response is None:
        print("Got empty option chain response.")
        return None

    if response.get("status") is False:
        print(f"Option chain error: {response.get('message', 'Unknown')}")
        return None

    return response.get("data")


def select_spread_strikes(chain, spot, direction, atr, budget, lot_size, min_oi=None, min_rr=1.5):
    """Select strikes for a vertical debit spread.

    Bull call spread (direction="bullish"): Buy ATM/ITM call, sell OTM call
    Bear put spread (direction="bearish"): Buy ATM/ITM put, sell OTM put

    Args:
        chain: list of dicts from option chain (each has strikePrice, CE, PE)
        spot: current underlying price
        direction: "bullish" or "bearish"
        atr: average true range of the underlying
        budget: max debit (max risk) in rupees for the spread
        lot_size: contract lot size
        min_oi: minimum open interest per leg (default from config.SPREAD_MIN_OI)
        min_rr: minimum reward:risk ratio (default 1.5)

    Returns:
        dict with keys: long_strike, short_strike, long_premium, short_premium,
                        net_debit, max_profit, max_loss, rr_ratio, width,
                        long_option_type, short_option_type
        or None if no valid spread found
    """
    if min_oi is None:
        min_oi = config.SPREAD_MIN_OI

    if not chain or direction not in ("bullish", "bearish"):
        return None

    # Sort chain by strike price
    sorted_chain = sorted(chain, key=lambda r: r.get("strikePrice", 0))

    if direction == "bullish":
        opt_key = "CE"
        # Find long leg candidates: ATM or 1-strike ITM call (strike <= spot, OI filter)
        itm_strikes = [
            r for r in sorted_chain
            if r.get("strikePrice", 0) <= spot
            and r.get(opt_key, {}).get("openInterest", 0) >= min_oi
        ]
        if not itm_strikes:
            return None
        # Sort by distance from spot (closest first), take top 2 (ATM + 1 ITM)
        itm_strikes.sort(key=lambda r: abs(r["strikePrice"] - spot))
        long_candidates = itm_strikes[:2]

    else:  # bearish
        opt_key = "PE"
        # Find long leg candidates: ATM or 1-strike ITM put (strike >= spot, OI filter)
        itm_strikes = [
            r for r in sorted_chain
            if r.get("strikePrice", 0) >= spot
            and r.get(opt_key, {}).get("openInterest", 0) >= min_oi
        ]
        if not itm_strikes:
            return None
        # Sort by distance from spot (closest first), take top 2 (ATM + 1 ITM)
        itm_strikes.sort(key=lambda r: abs(r["strikePrice"] - spot))
        long_candidates = itm_strikes[:2]

    # Try all valid (long, short) combinations, pick best R:R that fits budget
    best = None

    for long_row in long_candidates:
        long_strike = long_row["strikePrice"]
        long_premium = long_row.get(opt_key, {}).get("lastPrice", 0) or 0

        if direction == "bullish":
            # Short leg: OTM strikes above long_strike with sufficient OI
            short_pool = [
                r for r in sorted_chain
                if r.get("strikePrice", 0) > long_strike
                and r.get(opt_key, {}).get("openInterest", 0) >= min_oi
            ]
        else:
            # Short leg: OTM strikes below long_strike with sufficient OI
            short_pool = [
                r for r in sorted_chain
                if r.get("strikePrice", 0) < long_strike
                and r.get(opt_key, {}).get("openInterest", 0) >= min_oi
            ]

        # Consider up to 4 short leg candidates
        for short_row in short_pool[:4]:
            short_strike = short_row["strikePrice"]
            short_premium = short_row.get(opt_key, {}).get("lastPrice", 0) or 0

            per_unit_debit = long_premium - short_premium
            if per_unit_debit <= 0:
                continue

            net_debit = per_unit_debit * lot_size
            width = abs(short_strike - long_strike)
            max_profit = width * lot_size - net_debit
            max_loss = net_debit

            if max_loss <= 0 or max_profit <= 0:
                continue
            if net_debit > budget:
                continue

            rr_ratio = max_profit / max_loss

            if rr_ratio < min_rr:
                continue

            if best is None or rr_ratio > best["rr_ratio"]:
                best = {
                    "long_strike": long_strike,
                    "short_strike": short_strike,
                    "long_premium": long_premium,
                    "short_premium": short_premium,
                    "net_debit": net_debit,
                    "max_profit": max_profit,
                    "max_loss": max_loss,
                    "rr_ratio": rr_ratio,
                    "width": width,
                    "long_option_type": opt_key,
                    "short_option_type": opt_key,
                }

    return best


def select_credit_spread_strikes(chain, spot, direction, max_loss_budget, lot_size,
                                  min_oi=500, min_credit_per_lot=30, target_delta=0.25):
    """Select strikes for a credit spread (bull put or bear call).

    Bull put: sell OTM put (delta ~0.25), buy further OTM put for protection.
    Bear call: sell OTM call (delta ~0.25), buy further OTM call for protection.
    Net credit received upfront. Max loss = width - credit.

    Returns dict with short_strike, long_strike, net_credit, max_loss, max_profit, rr_ratio
    or None if no valid spread found.
    """
    if direction == "bullish":
        opt_key = "PE"
        candidates = [s for s in chain if s["strikePrice"] < spot and s.get(opt_key, {}).get("openInterest", 0) >= min_oi]
        candidates.sort(key=lambda s: s["strikePrice"], reverse=True)
    else:
        opt_key = "CE"
        candidates = [s for s in chain if s["strikePrice"] > spot and s.get(opt_key, {}).get("openInterest", 0) >= min_oi]
        candidates.sort(key=lambda s: s["strikePrice"])

    if len(candidates) < 2:
        return None

    # Find short leg: closest to target delta (prefer closer to spot on ties)
    best_short = None
    best_delta_diff = float("inf")
    for c in candidates:
        delta = abs(c[opt_key].get("delta", 0))
        diff = abs(delta - target_delta)
        if diff < best_delta_diff - 1e-9:
            best_delta_diff = diff
            best_short = c
        elif abs(diff - best_delta_diff) < 1e-9 and best_short is not None:
            # Tie: prefer strike closer to spot (higher premium, more credit)
            if abs(c["strikePrice"] - spot) < abs(best_short["strikePrice"] - spot):
                best_short = c

    if not best_short:
        return None

    short_premium = best_short[opt_key]["lastPrice"]
    short_strike = best_short["strikePrice"]

    # Find long leg: 1-3 strikes further OTM from short
    if direction == "bullish":
        long_candidates = [c for c in candidates if c["strikePrice"] < short_strike]
    else:
        long_candidates = [c for c in candidates if c["strikePrice"] > short_strike]

    if not long_candidates:
        return None

    best_spread = None
    for lc in long_candidates[:3]:
        long_premium = lc[opt_key]["lastPrice"]
        long_strike = lc["strikePrice"]
        width = abs(short_strike - long_strike)

        net_credit = (short_premium - long_premium) * lot_size
        max_loss = (width * lot_size) - net_credit
        max_profit = net_credit

        per_unit_credit = short_premium - long_premium
        if net_credit <= 0 or per_unit_credit < min_credit_per_lot:
            continue
        if max_loss > max_loss_budget:
            continue
        if max_loss <= 0:
            continue

        rr = max_profit / max_loss

        if best_spread is None or rr > best_spread["rr_ratio"]:
            best_spread = {
                "short_strike": short_strike,
                "long_strike": long_strike,
                "short_premium": short_premium,
                "long_premium": long_premium,
                "net_credit": net_credit,
                "max_loss": max_loss,
                "max_profit": max_profit,
                "rr_ratio": round(rr, 2),
                "width": width,
            }

    return best_spread


def select_iron_condor_strikes(chain, spot, lot_size, max_loss_budget,
                                min_oi=500, min_credit_per_lot=50):
    """Select 4 strikes for an iron condor on Nifty.

    An iron condor is two credit spreads combined:
    - Bear call spread: sell OTM call (200-300 pts above), buy further OTM call (protection)
    - Bull put spread: sell OTM put (200-300 pts below), buy further OTM put (protection)

    Args:
        chain: list of dicts from option chain (each has strikePrice, CE, PE)
        spot: current underlying price
        lot_size: contract lot size (e.g. 75 for NIFTY)
        max_loss_budget: max acceptable loss in rupees
        min_oi: minimum open interest per leg
        min_credit_per_lot: minimum net credit per lot in rupees

    Returns:
        dict with call_short_strike, call_long_strike, put_short_strike, put_long_strike,
             call_short_premium, call_long_premium, put_short_premium, put_long_premium,
             net_credit, max_loss, width
        or None if no valid condor found.
    """
    if not chain:
        return None

    sorted_chain = sorted(chain, key=lambda r: r.get("strikePrice", 0))

    # Call side candidates: 200-400 pts above spot with sufficient OI
    call_candidates = [
        r for r in sorted_chain
        if 200 <= r.get("strikePrice", 0) - spot <= 400
        and r.get("CE", {}).get("openInterest", 0) >= min_oi
    ]

    # Put side candidates: 200-400 pts below spot with sufficient OI
    put_candidates = [
        r for r in sorted_chain
        if 200 <= spot - r.get("strikePrice", 0) <= 400
        and r.get("PE", {}).get("openInterest", 0) >= min_oi
    ]

    if not call_candidates or not put_candidates:
        return None

    # Pick short call closest to 250 pts OTM
    call_candidates.sort(key=lambda r: abs(r["strikePrice"] - spot - 250))
    short_call_row = call_candidates[0]
    short_call_strike = short_call_row["strikePrice"]
    short_call_premium = short_call_row.get("CE", {}).get("lastPrice", 0) or 0

    # Long call: 100-200 pts above short call, sufficient OI
    long_call_candidates = [
        r for r in sorted_chain
        if 100 <= r.get("strikePrice", 0) - short_call_strike <= 200
        and r.get("CE", {}).get("openInterest", 0) >= min_oi
    ]
    if not long_call_candidates:
        return None

    long_call_candidates.sort(key=lambda r: abs(r["strikePrice"] - short_call_strike - 100))
    long_call_row = long_call_candidates[0]
    long_call_strike = long_call_row["strikePrice"]
    long_call_premium = long_call_row.get("CE", {}).get("lastPrice", 0) or 0

    # Pick short put closest to 250 pts OTM (below spot)
    put_candidates.sort(key=lambda r: abs(spot - r["strikePrice"] - 250))
    short_put_row = put_candidates[0]
    short_put_strike = short_put_row["strikePrice"]
    short_put_premium = short_put_row.get("PE", {}).get("lastPrice", 0) or 0

    # Long put: 100-200 pts below short put, sufficient OI
    long_put_candidates = [
        r for r in sorted_chain
        if 100 <= short_put_strike - r.get("strikePrice", 0) <= 200
        and r.get("PE", {}).get("openInterest", 0) >= min_oi
    ]
    if not long_put_candidates:
        return None

    long_put_candidates.sort(key=lambda r: abs(short_put_strike - r["strikePrice"] - 100))
    long_put_row = long_put_candidates[0]
    long_put_strike = long_put_row["strikePrice"]
    long_put_premium = long_put_row.get("PE", {}).get("lastPrice", 0) or 0

    # Calculate net credit and max loss
    net_credit_per_unit = (short_call_premium + short_put_premium) - (long_call_premium + long_put_premium)
    net_credit = net_credit_per_unit * lot_size

    if net_credit <= 0:
        return None

    call_width = long_call_strike - short_call_strike
    put_width = short_put_strike - long_put_strike
    width = max(call_width, put_width)

    max_loss = width * lot_size - net_credit

    # Validate constraints
    if max_loss > max_loss_budget:
        return None

    if net_credit_per_unit < min_credit_per_lot:
        return None

    return {
        "call_short_strike": short_call_strike,
        "call_long_strike": long_call_strike,
        "put_short_strike": short_put_strike,
        "put_long_strike": long_put_strike,
        "call_short_premium": short_call_premium,
        "call_long_premium": long_call_premium,
        "put_short_premium": short_put_premium,
        "put_long_premium": long_put_premium,
        "net_credit": net_credit,
        "max_loss": max_loss,
        "width": width,
        "lot_size": lot_size,
    }


def format_candles_for_prompt(candles: list, symbol: str) -> str:
    """Format candle data as a text table."""
    lines = [f"Stock: {symbol} (NSE) — Last {len(candles)} trading days\n"]
    lines.append(f"{'Date':<12} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>12}")
    lines.append("─" * 68)

    for candle in candles:
        date_str = candle[0][:10] if isinstance(candle[0], str) else str(candle[0])[:10]
        lines.append(
            f"{date_str:<12} {candle[1]:>10.2f} {candle[2]:>10.2f} "
            f"{candle[3]:>10.2f} {candle[4]:>10.2f} {candle[5]:>12,}"
        )

    return "\n".join(lines)


def format_options_for_prompt(options_data: list) -> str:
    """
    Format option chain data as a readable table for Claude.

    SmartAPI returns a list of dicts, each with CE (call) and PE (put) data
    including strike price, OI, IV, and Greeks.
    """
    lines = ["Option Chain (nearest expiry):\n"]
    lines.append(
        f"{'Strike':>10} │ {'CE OI':>10} {'CE IV':>8} {'CE Delta':>9} │ "
        f"{'PE OI':>10} {'PE IV':>8} {'PE Delta':>9}"
    )
    lines.append("─" * 80)

    for row in options_data:
        strike = row.get("strikePrice", 0)

        ce = row.get("CE", {}) or {}
        pe = row.get("PE", {}) or {}

        ce_oi = ce.get("openInterest", 0) or 0
        ce_iv = ce.get("impliedVolatility", 0) or 0
        ce_delta = ce.get("delta", 0) or 0

        pe_oi = pe.get("openInterest", 0) or 0
        pe_iv = pe.get("impliedVolatility", 0) or 0
        pe_delta = pe.get("delta", 0) or 0

        lines.append(
            f"{strike:>10.2f} │ {ce_oi:>10,} {ce_iv:>8.2f} {ce_delta:>9.4f} │ "
            f"{pe_oi:>10,} {pe_iv:>8.2f} {pe_delta:>9.4f}"
        )

    return "\n".join(lines)


def analyze_with_options(candles: list, options_data: list | None, symbol: str) -> dict:
    """
    Send both price data and options data to Claude for cross-referenced analysis.
    """
    client = config.get_anthropic_client()

    price_table = format_candles_for_prompt(candles, symbol)

    if options_data:
        options_table = format_options_for_prompt(options_data)
        user_message = (
            f"Here is the recent daily OHLCV data for {symbol}:\n\n"
            f"{price_table}\n\n"
            f"And here is the current option chain:\n\n"
            f"{options_table}\n\n"
            f"Cross-reference the stock's price action with what the options market is pricing in. "
            f"What's the trend? Do options confirm or contradict? Where's the smart money positioned?"
        )
    else:
        user_message = (
            f"Here is the recent daily OHLCV data for {symbol}:\n\n"
            f"{price_table}\n\n"
            f"Option chain data was not available for this stock. "
            f"Analyze based on price action only. Set options-related fields to null or 'N/A'.\n\n"
            f"What's the trend? Where are support and resistance?"
        )

    print("Sending data to Claude for analysis...")

    try:
        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.AuthenticationError:
        print("Anthropic API key is invalid. Check ANTHROPIC_API_KEY in .env")
        sys.exit(1)
    except anthropic.RateLimitError:
        print("Anthropic rate limit hit. Wait a moment and try again.")
        sys.exit(1)
    except Exception as e:
        print(f"Claude API call failed: {e}")
        sys.exit(1)

    try:
        return parse_claude_json(response.content[0].text)
    except ValueError as e:
        print(f"Claude returned invalid JSON: {e}")
        sys.exit(1)


def print_report(analysis: dict, symbol: str) -> None:
    """Print the enhanced analysis report."""
    border = "═" * 55

    print(f"\n{border}")
    print(f"  ANALYSIS REPORT: {symbol} (with Options)")
    print(border)
    print()
    print(f"  Trend:              {analysis['trend'].upper()}")
    print(f"  Confidence:         {analysis['confidence'].capitalize()}")
    print()
    print(f"  Support:            ₹{analysis['support']:,.2f}")
    print(f"  Resistance:         ₹{analysis['resistance']:,.2f}")
    print()
    print(f"  Avg Volume:         {analysis['avg_volume']:,.0f}")
    print(f"  Volume Trend:       {analysis['volume_trend'].capitalize()}")
    print()

    # Options section
    options_sentiment = analysis.get("options_sentiment", "N/A")
    iv_pct = analysis.get("iv_percentile", "N/A")
    pcr = analysis.get("pcr_signal", "N/A")
    max_pain = analysis.get("max_pain")

    print(f"  Options Sentiment:  {options_sentiment.upper() if isinstance(options_sentiment, str) else 'N/A'}")
    print(f"  IV Level:           {iv_pct}")
    print(f"  PCR Signal:         {pcr}")
    if max_pain:
        print(f"  Max Pain:           ₹{max_pain:,.2f}")
    print()

    # Summary
    def wrap_print(label: str, text: str):
        if not text or text == "null":
            return
        words = text.split()
        lines = []
        current = " "
        for word in words:
            if len(current) + len(word) + 1 > 55:
                lines.append(current)
                current = "  " + word
            else:
                current += " " + word
        lines.append(current)
        print(f"  {label}:")
        for line in lines:
            print(f" {line}")
        print()

    wrap_print("Summary", analysis.get("summary", ""))
    wrap_print("Options Insight", analysis.get("options_insight", ""))
    print(border)


def main():
    parser = argparse.ArgumentParser(description="AI stock + options analysis for Indian markets")
    parser.add_argument("--symbol", default="RELIANCE", help="Stock symbol (default: RELIANCE)")
    parser.add_argument("--token", default="2885", help="SmartAPI symbol token (default: 2885)")
    args = parser.parse_args()

    print("Connecting to AngelOne SmartAPI...")
    smart_api = get_session()

    print(f"Fetching 30 days of data for {args.symbol} (token: {args.token})...")
    candles = fetch_candles(smart_api, symbol=args.symbol, token=args.token)

    if not candles:
        print("No price data to analyze.")
        sys.exit(1)

    options_data = fetch_option_chain(smart_api, args.symbol, args.token)

    analysis = analyze_with_options(candles, options_data, args.symbol)
    print_report(analysis, args.symbol)


if __name__ == "__main__":
    main()
