"""Tests for portfolio_analysis.py pure functions."""

import pytest
from unittest.mock import MagicMock

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from portfolio_analysis import parse_claude_json, normalize_holding_symbol, resolve_equity_token


# ---------------------------------------------------------------------------
# parse_claude_json
# ---------------------------------------------------------------------------

def test_parse_claude_json_clean():
    raw = '{"market_mood": "bullish"}'
    result = parse_claude_json(raw)
    assert result == {"market_mood": "bullish"}


def test_parse_claude_json_markdown_fences():
    raw = '```json\n{"market_mood": "bearish"}\n```'
    result = parse_claude_json(raw)
    assert result == {"market_mood": "bearish"}


def test_parse_claude_json_mixed_text():
    raw = 'Here is my analysis:\n\n{"market_mood": "mixed"}\n\nHope that helps!'
    result = parse_claude_json(raw)
    assert result == {"market_mood": "mixed"}


def test_parse_claude_json_invalid():
    with pytest.raises(ValueError, match="Could not parse JSON"):
        parse_claude_json("this is not json at all")


# ---------------------------------------------------------------------------
# normalize_holding_symbol
# ---------------------------------------------------------------------------

def test_normalize_holding_symbol_eq():
    assert normalize_holding_symbol("HDFCBANK-EQ") == "HDFCBANK"


def test_normalize_holding_symbol_be():
    assert normalize_holding_symbol("RELIANCE-BE") == "RELIANCE"


def test_normalize_holding_symbol_no_suffix():
    assert normalize_holding_symbol("WIPRO") == "WIPRO"


def test_normalize_holding_symbol_empty():
    assert normalize_holding_symbol("") == ""


# ---------------------------------------------------------------------------
# resolve_equity_token
# ---------------------------------------------------------------------------

def test_resolve_equity_token_prefers_eq():
    mock_api = MagicMock()
    mock_api.searchScrip.return_value = {
        "data": [
            {"tradingsymbol": "RELIANCE", "symboltoken": "1111"},
            {"tradingsymbol": "RELIANCE-EQ", "symboltoken": "2885"},
        ]
    }
    token = resolve_equity_token(mock_api, "RELIANCE")
    assert token == "2885"
    mock_api.searchScrip.assert_called_once_with("NSE", "RELIANCE")


def test_resolve_equity_token_fallback():
    mock_api = MagicMock()
    mock_api.searchScrip.return_value = {
        "data": [
            {"tradingsymbol": "RELIANCE", "symboltoken": "1111"},
        ]
    }
    token = resolve_equity_token(mock_api, "RELIANCE")
    assert token == "1111"
