"""Shared utilities for the financial agent pipeline."""

import json
import re


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
    import os
    if os.getenv("DATA_SOURCE", "kite") == "kite":
        try:
            from kite_data import resolve_equity_token_kite
            token = resolve_equity_token_kite(symbol)
            if token is not None:
                return str(token)
        except Exception:
            pass
    try:
        resp = smart_api.searchScrip("NSE", symbol)
        if resp and resp.get("data"):
            # Prefer -EQ suffix (equity segment), fall back to first result
            for match in resp["data"]:
                if match.get("tradingsymbol", "").endswith("-EQ"):
                    return match.get("symboltoken")
            return resp["data"][0].get("symboltoken")
    except Exception as e:
        print(f"  searchScrip failed for {symbol}: {e}")
    return None
