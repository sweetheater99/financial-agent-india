"""Tests for V5 → V6 migration script."""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from migrate_v5_to_v6 import migrate_portfolio, migrate_state


def _make_portfolio(*positions):
    return {"positions": list(positions), "schema_version": 5}


def _old_position(**overrides):
    base = {
        "symbol": "RELIANCE",
        "entry_date": "2026-01-15",
        "entry_price": 1200.0,
        "quantity": 10,
        "side": "long",
    }
    base.update(overrides)
    return base


# --- Portfolio tests ---


def test_migrate_adds_entry_thesis():
    pos = _old_position()
    portfolio = _make_portfolio(pos)
    result = migrate_portfolio(portfolio)

    thesis = result["positions"][0]["entry_thesis"]
    assert thesis["conviction"] == "medium"
    assert "Pre-V6" in thesis["reasoning"]
    assert thesis["key_conditions"] == {}
    assert thesis["invalidation"] == "Manual review required"
    assert result["positions"][0]["_add_count"] == 0
    assert result["schema_version"] == 6


def test_migrate_preserves_existing_thesis():
    existing_thesis = {
        "reasoning": "Strong breakout",
        "conviction": "high",
        "key_conditions": {"rsi": "> 60"},
        "invalidation": "Close below 1150",
        "expected_hold": "2 weeks",
        "target_scenario": "Retest of 1300",
    }
    pos = _old_position(entry_thesis=existing_thesis, _add_count=2)
    portfolio = _make_portfolio(pos)
    result = migrate_portfolio(portfolio)

    assert result["positions"][0]["entry_thesis"] == existing_thesis
    assert result["positions"][0]["_add_count"] == 2


# --- State tests ---


def test_migrate_state():
    state = {
        "last_check": {},
        "circuit_breaker_active": False,
        "schema_version": 5,
    }
    result = migrate_state(state)

    assert result["position_assessments"] == {}
    assert result["claude_consecutive_failures"] == 0
    assert result["claude_lockdown_active"] is False
    assert result["schema_version"] == 6
    # Existing fields preserved
    assert result["circuit_breaker_active"] is False


# --- Idempotency ---


def test_migrate_idempotent():
    pos = _old_position()
    portfolio = _make_portfolio(pos)

    first = migrate_portfolio(copy.deepcopy(portfolio))
    second = migrate_portfolio(copy.deepcopy(first))
    assert first == second

    state = {"last_check": {}, "schema_version": 5}
    first_state = migrate_state(copy.deepcopy(state))
    second_state = migrate_state(copy.deepcopy(first_state))
    assert first_state == second_state
