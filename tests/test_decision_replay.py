"""Tests for decision replay logging in claude_intel.py."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from claude_intel import _save_decision_log, DECISION_LOG_DIR, evaluate_entry


class TestSaveDecisionLog:
    """Test _save_decision_log creates files with correct structure."""

    def test_save_decision_log(self, tmp_path):
        """Creates file with correct structure."""
        with patch("claude_intel.DECISION_LOG_DIR", tmp_path):
            _save_decision_log(
                decision_type="entry",
                symbol="RELIANCE",
                prompt="Should I buy?",
                response='{"action": "TRADE", "reasoning": "Strong momentum"}',
                parsed={"action": "TRADE", "reasoning": "Strong momentum"},
                extra={"conviction": "high"},
            )

        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1

        data = json.loads(files[0].read_text())
        assert data["type"] == "entry"
        assert data["symbol"] == "RELIANCE"
        assert data["prompt_sent"] == "Should I buy?"
        assert data["parsed_action"] == "TRADE"
        assert data["conviction"] == "high"
        assert "timestamp" in data
        assert "claude_response" in data

        # Filename format check
        assert "entry_RELIANCE" in files[0].name

    def test_save_decision_log_handles_failure(self, tmp_path):
        """Gracefully handles write failures (e.g., permission error)."""
        bad_path = tmp_path / "no_perms"
        bad_path.mkdir()
        bad_path.chmod(0o000)

        try:
            with patch("claude_intel.DECISION_LOG_DIR", bad_path / "subdir"):
                # Should not raise — just logs debug
                _save_decision_log(
                    decision_type="entry",
                    symbol="TEST",
                    prompt="test",
                    response="test",
                    parsed={"action": "SKIP"},
                )
        finally:
            bad_path.chmod(0o755)

    def test_save_decision_log_no_extra(self, tmp_path):
        """Works without extra dict."""
        with patch("claude_intel.DECISION_LOG_DIR", tmp_path):
            _save_decision_log(
                decision_type="exit",
                symbol="INFY",
                prompt="Should I exit?",
                response='{"action": "EXIT"}',
                parsed={"action": "EXIT"},
            )

        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["parsed_action"] == "EXIT"


class TestDecisionLogInEvaluateEntry:
    """Test that evaluate_entry wires into _save_decision_log."""

    @patch("claude_intel._save_decision_log")
    @patch("claude_intel._call_claude")
    def test_decision_log_in_evaluate_entry(self, mock_claude, mock_save):
        """Mock _call_claude, verify log created on entry eval."""
        mock_claude.return_value = json.dumps({
            "action": "TRADE",
            "conviction": "high",
            "allocation_adj": 1.2,
            "reasoning": "Strong OI buildup with volume confirmation",
        })

        candidate = {"symbol": "HDFCBANK", "direction": "bullish", "score": 7}
        portfolio = {"positions": [], "capital": 100000, "available_capital": 100000, "stats": {"total_pnl": 0}}

        result = evaluate_entry(
            candidate=candidate,
            instrument="EQ",
            regime="trending_up",
            vix=14.0,
            macro=None,
            pcr=1.1,
            max_pain=22000,
            portfolio=portfolio,
        )

        assert result[0] is True  # approved
        assert mock_save.call_count == 1

        call_args = mock_save.call_args
        assert call_args[1].get("decision_type", call_args[0][0]) == "entry"
        assert call_args[1].get("symbol", call_args[0][1]) == "HDFCBANK"
