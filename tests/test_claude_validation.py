"""Tests for Claude JSON response validation in claude_intel.py."""

import sys

sys.path.insert(0, "/Users/aravindms/financial-agent-india")
from claude_intel import _validate_entry_response, _validate_exit_response


# --- Entry validation tests ---

def test_valid_entry_passes_through():
    inp = {"action": "TRADE", "conviction": "high", "allocation_adj": 1.2, "reasoning": "Strong setup"}
    result = _validate_entry_response(inp)
    assert result["action"] == "TRADE"
    assert result["conviction"] == "high"
    assert result["allocation_adj"] == 1.2
    assert result["reasoning"] == "Strong setup"


def test_missing_fields_get_defaults():
    result = _validate_entry_response({})
    assert result["action"] == "SKIP"
    assert result["conviction"] == "medium"
    assert result["allocation_adj"] == 1.0
    assert result["reasoning"] == "No reasoning provided"


def test_null_allocation_becomes_1():
    result = _validate_entry_response({"allocation_adj": None})
    assert result["allocation_adj"] == 1.0


def test_allocation_clamped_low():
    result = _validate_entry_response({"allocation_adj": 0.1})
    assert result["allocation_adj"] == 0.5


def test_allocation_clamped_high():
    result = _validate_entry_response({"allocation_adj": 3.0})
    assert result["allocation_adj"] == 1.5


def test_allocation_string_invalid():
    result = _validate_entry_response({"allocation_adj": "not a number"})
    assert result["allocation_adj"] == 1.0


def test_action_normalized_uppercase():
    result = _validate_entry_response({"action": "trade"})
    assert result["action"] == "TRADE"


def test_invalid_action_becomes_skip():
    result = _validate_entry_response({"action": "BUY"})
    assert result["action"] == "SKIP"


def test_non_string_action_becomes_skip():
    result = _validate_entry_response({"action": 123})
    assert result["action"] == "SKIP"


def test_null_action_becomes_skip():
    result = _validate_entry_response({"action": None})
    assert result["action"] == "SKIP"


# --- Exit validation tests ---

def test_valid_exit_passes_through():
    result = _validate_exit_response({"action": "HOLD", "reasoning": "Momentum strong"})
    assert result["action"] == "HOLD"
    assert result["reasoning"] == "Momentum strong"


def test_exit_missing_fields_defaults():
    result = _validate_exit_response({})
    assert result["action"] == "EXIT"
    assert result["reasoning"] == "No reasoning provided"


def test_exit_invalid_action_becomes_exit():
    result = _validate_exit_response({"action": "WAIT"})
    assert result["action"] == "EXIT"


def test_add_action_accepted_in_exit():
    result = _validate_exit_response({"action": "ADD"})
    assert result["action"] == "ADD"


def test_partial_action_accepted_in_exit():
    result = _validate_exit_response({"action": "PARTIAL"})
    assert result["action"] == "PARTIAL"


def test_exit_action_lowercase_normalized():
    result = _validate_exit_response({"action": "hold"})
    assert result["action"] == "HOLD"


def test_exit_non_string_action():
    result = _validate_exit_response({"action": [1, 2]})
    assert result["action"] == "EXIT"
