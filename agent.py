"""
Financial analysis agent for Indian equities.

Pulls real market data from AngelOne SmartAPI, sends it to Claude,
and gets back a structured analysis with trend, support/resistance,
volume analysis, and a plain English summary.

Usage:
    python agent.py
    python agent.py --symbol TCS --token 11536
"""

import argparse
import json
import sys

import anthropic

import config
from connect import get_session
from fetch_data import fetch_candles

# --- Claude setup ---

SYSTEM_PROMPT = """You are a financial analyst specializing in Indian equities (NSE/BSE).

You analyze stock price data and provide clear, actionable technical analysis. You are
practical and direct — no fluff, no hedging every sentence with disclaimers. Give your
honest read of what the data shows.

When given OHLCV data, you analyze:
- Price trend (direction, momentum, key patterns)
- Support and resistance levels (based on price action, not arbitrary round numbers)
- Volume patterns (is volume confirming or diverging from price?)
- Any notable patterns (breakouts, consolidations, reversals)

You MUST respond with ONLY a valid JSON object. No markdown, no code fences, no explanation
outside the JSON. The JSON must have exactly these fields:

{
    "trend": "bullish" | "bearish" | "sideways",
    "support": <number — key support level>,
    "resistance": <number — key resistance level>,
    "avg_volume": <number — average daily volume>,
    "volume_trend": "increasing" | "decreasing" | "stable",
    "summary": "<3-4 sentences of plain English analysis>",
    "confidence": "high" | "medium" | "low"
}"""


def format_candles_for_prompt(candles: list, symbol: str) -> str:
    """Format candle data as a readable text table for Claude."""
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


def analyze_stock(candles: list, symbol: str) -> dict:
    """
    Send stock data to Claude and get back a structured analysis.

    Returns a dict with trend, support, resistance, volume info, summary, and confidence.
    """
    client = config.get_anthropic_client()

    data_table = format_candles_for_prompt(candles, symbol)
    user_message = (
        f"Here is the recent daily OHLCV data for {symbol}:\n\n"
        f"{data_table}\n\n"
        f"Analyze this data. What's the trend? Where are support and resistance? "
        f"What's volume telling us? Give me your read."
    )

    print("Sending data to Claude for analysis...")

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
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

    # Parse the JSON response
    raw_text = response.content[0].text.strip()

    # Handle cases where Claude wraps JSON in code fences despite instructions
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1]  # remove first line
        raw_text = raw_text.rsplit("```", 1)[0]  # remove last fence
        raw_text = raw_text.strip()

    try:
        analysis = json.loads(raw_text)
    except json.JSONDecodeError:
        print("Claude returned something that isn't valid JSON:")
        print(raw_text[:500])
        sys.exit(1)

    return analysis


def print_report(analysis: dict, symbol: str) -> None:
    """Print a clean analysis report to the terminal."""
    border = "═" * 50

    print(f"\n{border}")
    print(f"  ANALYSIS REPORT: {symbol}")
    print(border)
    print()
    print(f"  Trend:          {analysis['trend'].upper()}")
    print(f"  Confidence:     {analysis['confidence'].capitalize()}")
    print()
    print(f"  Support:        ₹{analysis['support']:,.2f}")
    print(f"  Resistance:     ₹{analysis['resistance']:,.2f}")
    print()
    print(f"  Avg Volume:     {analysis['avg_volume']:,.0f}")
    print(f"  Volume Trend:   {analysis['volume_trend'].capitalize()}")
    print()

    # Word-wrap the summary at ~60 chars
    summary = analysis["summary"]
    words = summary.split()
    lines = []
    current_line = " "
    for word in words:
        if len(current_line) + len(word) + 1 > 60:
            lines.append(current_line)
            current_line = "  " + word
        else:
            current_line += " " + word
    lines.append(current_line)

    print("  Summary:")
    for line in lines:
        print(f" {line}")
    print()
    print(border)


def main():
    parser = argparse.ArgumentParser(description="AI-powered stock analysis for Indian markets")
    parser.add_argument("--symbol", default="RELIANCE", help="Stock symbol (default: RELIANCE)")
    parser.add_argument("--token", default="2885", help="SmartAPI symbol token (default: 2885)")
    args = parser.parse_args()

    print(f"Connecting to AngelOne SmartAPI...")
    smart_api = get_session()

    print(f"Fetching 30 days of data for {args.symbol} (token: {args.token})...")
    candles = fetch_candles(smart_api, symbol=args.symbol, token=args.token)

    if not candles:
        print("No data to analyze.")
        sys.exit(1)

    analysis = analyze_stock(candles, args.symbol)
    print_report(analysis, args.symbol)


if __name__ == "__main__":
    main()
