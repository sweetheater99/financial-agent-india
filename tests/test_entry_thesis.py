"""Tests for entry thesis storage and position-level Claude memory."""

import json
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Part A: Position assessment save/get
# ---------------------------------------------------------------------------

def test_position_assessment_save_and_get():
    """save_position_assessment stores and get_position_assessment retrieves."""
    from smart_monitor import save_position_assessment, get_position_assessment

    state = {}
    save_position_assessment(state, "RELIANCE", "HOLD", "Strong trend intact", ["RSI_oversold", "volume_spike"])

    assessment = get_position_assessment(state, "RELIANCE")
    assert assessment is not None
    assert assessment["action"] == "HOLD"
    assert assessment["reasoning"] == "Strong trend intact"
    assert "RSI_oversold" in assessment["signals_at_assessment"]
    assert "timestamp" in assessment

    # Non-existent symbol returns None
    assert get_position_assessment(state, "INFY") is None


def test_position_assessment_cleared_on_close():
    """cleanup_closed_positions removes assessments for closed symbols."""
    from smart_monitor import save_position_assessment, get_position_assessment, cleanup_closed_positions

    state = {}
    save_position_assessment(state, "RELIANCE", "HOLD", "reason1", [])
    save_position_assessment(state, "INFY", "EXIT", "reason2", [])

    portfolio = {
        "positions": [
            {"symbol": "RELIANCE", "status": "open"},
            {"symbol": "INFY", "status": "closed"},
        ]
    }

    cleanup_closed_positions(portfolio, state)

    # RELIANCE is still open — assessment kept
    assert get_position_assessment(state, "RELIANCE") is not None
    # INFY is closed — assessment removed
    assert get_position_assessment(state, "INFY") is None


# ---------------------------------------------------------------------------
# Part B: evaluate_entry returns 4-tuple with thesis
# ---------------------------------------------------------------------------

def test_evaluate_entry_returns_thesis():
    """evaluate_entry returns (approved, reasoning, adj, entry_thesis)."""
    from claude_intel import evaluate_entry

    mock_response = json.dumps({
        "action": "TRADE",
        "conviction": "high",
        "allocation_adj": 1.2,
        "reasoning": "Strong momentum with OI confirmation",
    })

    candidate = {
        "symbol": "RELIANCE",
        "direction": "bullish",
        "score": 7,
        "categories": ["LongBuildUp"],
        "sector": "Energy",
    }
    portfolio = {
        "positions": [],
        "closed_trades": [],
        "stats": {"total_pnl": 0},
        "capital": 100_000,
        "available_capital": 80_000,
    }

    with patch("claude_intel._call_claude", return_value=mock_response):
        result = evaluate_entry(
            candidate, "EQ", "TRENDING_UP", 14.5,
            None, 0.9, 24000, portfolio,
        )

    assert len(result) == 4
    approved, reasoning, adj, thesis = result
    assert approved is True
    assert "momentum" in reasoning.lower()
    assert adj == 1.2
    assert isinstance(thesis, dict)
    assert thesis["reasoning"] == "Strong momentum with OI confirmation"
    assert thesis["conviction"] == "high"
    assert thesis["key_conditions"]["regime"] == "TRENDING_UP"
    assert thesis["key_conditions"]["vix"] == 14.5
    assert thesis["key_conditions"]["pcr"] == 0.9


def test_evaluate_entry_unavailable_returns_empty_thesis():
    """When Claude is unavailable, evaluate_entry returns empty dict as thesis."""
    from claude_intel import evaluate_entry

    candidate = {
        "symbol": "INFY",
        "direction": "bullish",
        "score": 5,
        "categories": [],
        "sector": "IT",
    }
    portfolio = {
        "positions": [],
        "closed_trades": [],
        "stats": {"total_pnl": 0},
        "capital": 100_000,
        "available_capital": 80_000,
    }

    with patch("claude_intel._call_claude", return_value=None):
        result = evaluate_entry(
            candidate, "EQ", "SIDEWAYS", 16.0,
            None, 1.0, 24000, portfolio,
        )

    assert len(result) == 4
    approved, reasoning, adj, thesis = result
    assert approved is True  # fail-open
    assert thesis == {}


def test_evaluate_entry_skip_returns_thesis():
    """When Claude SKIPs, thesis is still returned."""
    from claude_intel import evaluate_entry

    mock_response = json.dumps({
        "action": "SKIP",
        "conviction": "low",
        "allocation_adj": 0.5,
        "reasoning": "Weak setup against regime",
    })

    candidate = {
        "symbol": "TCS",
        "direction": "bearish",
        "score": 3,
        "categories": [],
        "sector": "IT",
    }
    portfolio = {
        "positions": [],
        "closed_trades": [],
        "stats": {"total_pnl": 0},
        "capital": 100_000,
        "available_capital": 80_000,
    }

    with patch("claude_intel._call_claude", return_value=mock_response):
        approved, reasoning, adj, thesis = evaluate_entry(
            candidate, "EQ", "TRENDING_UP", 15.0,
            None, 1.0, 24000, portfolio,
        )

    assert approved is False
    assert thesis["reasoning"] == "Weak setup against regime"
    assert thesis["conviction"] == "low"


# ---------------------------------------------------------------------------
# Part D: evaluate_exit with thesis and previous_assessment
# ---------------------------------------------------------------------------

def test_evaluate_exit_with_thesis_and_assessment():
    """evaluate_exit includes entry thesis and previous assessment in prompt."""
    from claude_intel import evaluate_exit

    mock_response = json.dumps({
        "action": "HOLD",
        "reasoning": "Thesis still valid, trend intact",
    })

    pos = {
        "symbol": "RELIANCE",
        "instrument": "EQ",
        "direction": "bullish",
        "entry_price": 2500,
        "entry_date": "2026-03-01",
        "target_price": 2700,
        "stoploss_price": 2400,
        "atr_at_entry": 45,
        "entry_thesis": {
            "reasoning": "Strong momentum with OI confirmation",
            "conviction": "high",
            "key_conditions": {"regime": "TRENDING_UP", "vix": 14.5, "pcr": 0.9},
            "invalidation": "Break below 2400",
        },
    }
    portfolio = {
        "positions": [pos],
        "closed_trades": [],
        "stats": {"total_pnl": 0},
        "capital": 100_000,
        "available_capital": 80_000,
    }

    captured_prompts = []

    def mock_call(prompt, **kwargs):
        captured_prompts.append(prompt)
        return mock_response

    with patch("claude_intel._call_claude", side_effect=mock_call):
        should_exit, reasoning = evaluate_exit(
            pos, "trailing_stop", 2550, 5000, 2.0,
            portfolio, vix=15.0,
            previous_assessment="Last check: HOLD — momentum still strong",
        )

    assert should_exit is False
    assert "Thesis still valid" in reasoning

    # Verify the prompt included thesis and previous assessment
    prompt = captured_prompts[0]
    assert "ENTRY THESIS" in prompt
    assert "Strong momentum with OI confirmation" in prompt
    assert "Break below 2400" in prompt
    assert "PREVIOUS ASSESSMENT" in prompt
    assert "Last check: HOLD" in prompt


# ---------------------------------------------------------------------------
# _load_state defaults
# ---------------------------------------------------------------------------

def test_load_state_defaults():
    """_load_state returns all expected default keys."""
    from smart_monitor import _load_state
    import tempfile
    from pathlib import Path

    with patch("smart_monitor.STATE_FILE", Path(tempfile.mktemp(suffix=".json"))):
        state = _load_state()

    assert "position_assessments" in state
    assert state["position_assessments"] == {}
    assert "claude_consecutive_failures" in state
    assert state["claude_consecutive_failures"] == 0
    assert "claude_lockdown_active" in state
    assert state["claude_lockdown_active"] is False
