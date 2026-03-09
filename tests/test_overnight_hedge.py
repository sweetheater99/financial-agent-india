"""Tests for overnight hedge guardrail functions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from overnight_hedge import (
    apply_guardrails,
    check_hedge_cost,
    enforce_carry_naked_threshold,
    is_naked_fno,
)


def _make_position(
    instrument="MOMENTUM",
    direction="bullish",
    entry_price=100,
    status="open",
    hedge_leg=None,
):
    pos = {
        "symbol": "NIFTY",
        "instrument": instrument,
        "direction": direction,
        "entry_price": entry_price,
        "status": status,
    }
    if hedge_leg is not None:
        pos["hedge_leg"] = hedge_leg
    return pos


# ── TestIsNakedFno ───────────────────────────────────────────────────────────


class TestIsNakedFno:
    def test_spread_is_not_naked(self):
        pos = _make_position(instrument="SPREAD")
        assert is_naked_fno(pos) is False

    def test_condor_is_not_naked(self):
        pos = _make_position(instrument="CONDOR")
        assert is_naked_fno(pos) is False

    def test_momentum_is_naked(self):
        pos = _make_position(instrument="MOMENTUM")
        assert is_naked_fno(pos) is True

    def test_opt_is_naked(self):
        pos = _make_position(instrument="OPT")
        assert is_naked_fno(pos) is True

    def test_fut_is_naked(self):
        pos = _make_position(instrument="FUT")
        assert is_naked_fno(pos) is True

    def test_already_hedged_is_not_naked(self):
        pos = _make_position(instrument="MOMENTUM", hedge_leg={"symbol": "NIFTY25300PE"})
        assert is_naked_fno(pos) is False


# ── TestApplyGuardrails ──────────────────────────────────────────────────────


class TestApplyGuardrails:
    def test_position_in_loss_returns_close(self):
        pos = _make_position(entry_price=100)
        result = apply_guardrails(pos, current_price=90, vix=15, days_to_expiry=10)
        assert result["action"] == "close"

    def test_high_vix_blocks_naked_carry(self):
        pos = _make_position(entry_price=100)
        result = apply_guardrails(pos, current_price=160, vix=22, days_to_expiry=10)
        assert result["action"] != "carry_naked"

    def test_low_gain_returns_close(self):
        # 20% gain — below 30% hedge threshold
        pos = _make_position(entry_price=100)
        result = apply_guardrails(pos, current_price=120, vix=15, days_to_expiry=10)
        assert result["action"] == "close"

    def test_near_expiry_returns_close(self):
        pos = _make_position(entry_price=100)
        result = apply_guardrails(pos, current_price=160, vix=15, days_to_expiry=1)
        assert result["action"] == "close"

    def test_moderate_gain_returns_ask_claude(self):
        # 40% gain, VIX 15, 10 DTE
        pos = _make_position(entry_price=100)
        result = apply_guardrails(pos, current_price=140, vix=15, days_to_expiry=10)
        assert result["action"] == "ask_claude"

    def test_high_gain_returns_ask_claude(self):
        # 60% gain, VIX 15, 10 DTE
        pos = _make_position(entry_price=100)
        result = apply_guardrails(pos, current_price=160, vix=15, days_to_expiry=10)
        assert result["action"] == "ask_claude"


# ── TestClaudeOverride ───────────────────────────────────────────────────────


class TestClaudeOverride:
    def test_carry_naked_below_threshold_becomes_hedge(self):
        decision = {"action": "carry_naked", "reason": "Claude says carry"}
        result = enforce_carry_naked_threshold(decision, gain_pct=0.40)
        assert result["action"] == "hedge"

    def test_carry_naked_above_threshold_allowed(self):
        decision = {"action": "carry_naked", "reason": "Claude says carry"}
        result = enforce_carry_naked_threshold(decision, gain_pct=0.55)
        assert result["action"] == "carry_naked"

    def test_close_decision_unchanged(self):
        decision = {"action": "close", "reason": "position in loss"}
        result = enforce_carry_naked_threshold(decision, gain_pct=0.40)
        assert result["action"] == "close"


# ── TestHedgeCost ────────────────────────────────────────────────────────────


class TestHedgeCost:
    def test_hedge_too_expensive_falls_back(self):
        # cost = 13 * 100 = 1300, position = 100_000 → 1.3%
        result = check_hedge_cost(position_value=100_000, hedge_premium=13, quantity=100)
        assert result["affordable"] is False

    def test_hedge_within_budget(self):
        # cost = 5.3 * 10 = 53, position = 10_000 → 0.53%
        result = check_hedge_cost(position_value=10_000, hedge_premium=5.3, quantity=10)
        assert result["affordable"] is True
