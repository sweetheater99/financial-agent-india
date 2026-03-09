"""Tests for overnight hedge guardrail functions."""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from overnight_hedge import (
    apply_guardrails,
    build_hedge_leg,
    calc_hedge_pnl,
    check_hedge_cost,
    enforce_carry_naked_threshold,
    execute_carry_naked,
    execute_close,
    execute_hedge,
    find_positions_with_hedge_legs,
    is_naked_fno,
    parse_overnight_decision,
    run_overnight_hedge_scan,
    tighten_stop_loss,
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


# ── TestParseClaudeOvernightDecision ────────────────────────────────────────


class TestParseClaudeOvernightDecision:
    def test_parse_valid_close(self):
        raw = json.dumps({"action": "close", "reasoning": "risk too high"})
        result = parse_overnight_decision(raw)
        assert result["action"] == "close"
        assert result["reasoning"] == "risk too high"

    def test_parse_valid_hedge(self):
        raw = json.dumps({"action": "hedge", "reasoning": "protect gains"})
        result = parse_overnight_decision(raw)
        assert result["action"] == "hedge"
        assert result["reasoning"] == "protect gains"

    def test_parse_valid_carry_naked(self):
        raw = '```json\n{"action": "carry_naked", "reasoning": "strong trend"}\n```'
        result = parse_overnight_decision(raw)
        assert result["action"] == "carry_naked"
        assert result["reasoning"] == "strong trend"

    def test_parse_invalid_defaults_to_hedge(self):
        raw = "I think you should hedge this position because..."
        result = parse_overnight_decision(raw)
        assert result["action"] == "hedge"

    def test_parse_none_defaults_to_close(self):
        result = parse_overnight_decision(None)
        assert result["action"] == "close"
        assert "fail-safe" in result["reasoning"]


# ── TestBuildHedgeLeg ───────────────────────────────────────────────────────


class TestBuildHedgeLeg:
    def test_bullish_gets_put(self):
        pos = _make_position(direction="bullish", instrument="MOMENTUM")
        pos["quantity"] = 25
        result = build_hedge_leg(pos, spot_price=25200)
        assert result["option_type"] == "PE"
        # 25200 rounded to nearest 50 = 25200, minus 250 = 24950
        assert result["strike"] == 24950
        assert result["quantity"] == 25

    def test_bearish_gets_call(self):
        pos = _make_position(direction="bearish", instrument="MOMENTUM")
        pos["quantity"] = 25
        result = build_hedge_leg(pos, spot_price=24800)
        assert result["option_type"] == "CE"
        # 24800 rounded to nearest 50 = 24800, plus 250 = 25050
        assert result["strike"] == 25050

    def test_banknifty_wider_otm(self):
        pos = _make_position(direction="bullish", instrument="MOMENTUM")
        pos["symbol"] = "BANKNIFTY"
        pos["quantity"] = 15
        result = build_hedge_leg(pos, spot_price=52200)
        assert result["option_type"] == "PE"
        # 52200 rounded to nearest 100 = 52200, minus 500 = 51700
        assert result["strike"] == 51700

    def test_futures_bullish_gets_put(self):
        pos = _make_position(direction="bullish", instrument="FUT")
        pos["quantity"] = 25
        result = build_hedge_leg(pos, spot_price=25100)
        assert result["option_type"] == "PE"


# ── TestTightenStopLoss ─────────────────────────────────────────────────────


class TestTightenStopLoss:
    def test_tighten_raises_stop(self):
        pos = _make_position(entry_price=100)
        pos["stoploss_price"] = 65
        new_sl = tighten_stop_loss(pos, current_price=160)
        # Gap = 160 - 65 = 95, tightened by 20% → new gap = 76, new SL = 160 - 76 = 84
        assert new_sl > 65
        assert new_sl < 160
        assert new_sl == 84.0


# ── TestMorningUnwind ─────────────────────────────────────────────────────


class TestMorningUnwind:
    def test_find_positions_with_hedge_legs(self):
        hedged = _make_position(hedge_leg={"symbol": "NIFTY25300PE", "premium": 45, "quantity": 25})
        unhedged = _make_position()
        result = find_positions_with_hedge_legs([hedged, unhedged])
        assert len(result) == 1
        assert result[0] is hedged

    def test_calc_hedge_pnl_insurance_cost(self):
        # Bought at 45, current 30 → lost 15 per unit
        result = calc_hedge_pnl(entry_premium=45, exit_premium=30, quantity=25)
        assert result["pnl"] < 0
        assert result["label"] == "insurance_cost"

    def test_calc_hedge_pnl_hedge_saved(self):
        # Bought at 45, current 120 → gained 75 per unit
        result = calc_hedge_pnl(entry_premium=45, exit_premium=120, quantity=25)
        assert result["pnl"] > 0
        assert result["label"] == "hedge_saved"


# ── TestIntegrationDryRun ─────────────────────────────────────────────────


class MockSmartAPI:
    """Mock SmartAPI that returns configurable LTP values."""

    def __init__(self, ltp=100.0):
        self._ltp = ltp

    def ltpData(self, exchange, symbol, token):
        return {"data": {"ltp": self._ltp}}

    def getMarketData(self, mode=None, exchangeTokens=None):
        return {"data": {"fetched": [{"ltp": self._ltp}]}}


class TestIntegrationDryRun:
    def _write_portfolio(self, tmp_path, positions):
        """Write a portfolio JSON file to tmp_path and return the path."""
        portfolio = {"positions": positions}
        pf_file = tmp_path / "portfolio.json"
        pf_file.write_text(json.dumps(portfolio))
        return pf_file

    def test_full_scan_position_in_loss(self, tmp_path, monkeypatch):
        """Position in loss should be closed by guardrails."""
        import paper_trade
        import regime

        # Portfolio with one naked MOMENTUM position (entry=100, LTP will be 85)
        positions = [
            {
                "symbol": "NIFTY",
                "instrument": "MOMENTUM",
                "direction": "bullish",
                "entry_price": 100,
                "quantity": 75,
                "status": "open",
                "expiry": "30MAR2026",
                "token": "26000",
                "trading_symbol": "NIFTY30MAR26FUT",
            }
        ]
        self._write_portfolio(tmp_path, positions)

        # Monkeypatch portfolio paths
        monkeypatch.setattr(paper_trade, "PORTFOLIO_FILE", tmp_path / "portfolio.json")
        monkeypatch.setattr(paper_trade, "PORTFOLIO_DIR", tmp_path)

        # Mock regime to avoid SmartAPI calls
        monkeypatch.setattr(regime, "classify_regime", lambda *a, **kw: {"regime": "TRENDING_UP"})

        # Mock yfinance to avoid network calls — Ticker().history() raises so VIX defaults to 15.0
        import yfinance
        bad_ticker = MagicMock()
        bad_ticker.return_value.history.side_effect = Exception("mocked VIX")
        monkeypatch.setattr(yfinance, "Ticker", bad_ticker)

        # Mock _telegram_send
        sent_messages = []
        monkeypatch.setattr(paper_trade, "_telegram_send", lambda msg: sent_messages.append(msg))

        # Run scan with LTP=85 (position in loss: entry=100, current=85)
        mock_api = MockSmartAPI(ltp=85.0)
        results = run_overnight_hedge_scan(mock_api, dry_run=True)

        assert len(results) == 1
        assert results[0]["action"] == "close"
        assert "loss" in results[0]["reason"]

    def test_full_scan_no_naked_positions(self, tmp_path, monkeypatch):
        """Portfolio with only a CONDOR position should produce 0 results."""
        import paper_trade

        positions = [
            {
                "symbol": "NIFTY",
                "instrument": "CONDOR",
                "direction": "bullish",
                "entry_price": 100,
                "quantity": 75,
                "status": "open",
            }
        ]
        self._write_portfolio(tmp_path, positions)

        monkeypatch.setattr(paper_trade, "PORTFOLIO_FILE", tmp_path / "portfolio.json")
        monkeypatch.setattr(paper_trade, "PORTFOLIO_DIR", tmp_path)

        mock_api = MockSmartAPI(ltp=100.0)
        results = run_overnight_hedge_scan(mock_api, dry_run=True)

        assert results == []

    def test_full_scan_hedged_position_skipped(self, tmp_path, monkeypatch):
        """MOMENTUM position with hedge_leg should not be treated as naked."""
        import paper_trade

        positions = [
            {
                "symbol": "NIFTY",
                "instrument": "MOMENTUM",
                "direction": "bullish",
                "entry_price": 100,
                "quantity": 75,
                "status": "open",
                "hedge_leg": {"symbol": "NIFTY25300PE", "premium": 45, "quantity": 75},
            }
        ]
        self._write_portfolio(tmp_path, positions)

        monkeypatch.setattr(paper_trade, "PORTFOLIO_FILE", tmp_path / "portfolio.json")
        monkeypatch.setattr(paper_trade, "PORTFOLIO_DIR", tmp_path)

        mock_api = MockSmartAPI(ltp=100.0)
        results = run_overnight_hedge_scan(mock_api, dry_run=True)

        assert results == []
