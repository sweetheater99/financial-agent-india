"""Tests for debate system — Bull/Bear/Context agents + Moderator."""
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_bull_agent_returns_string():
    from debate import _bull_agent
    candidate = {"symbol": "RELIANCE", "direction": "bullish", "score": 6,
                 "rsi": 55, "volume_ratio": 1.5, "categories": ["LongBuildUp"]}
    with patch("debate._call_claude_cli_debate") as mock_cli:
        mock_cli.return_value = "Strong momentum with RSI at 55. Target 2800. Volume confirms."
        result = _bull_agent(candidate, {})
    assert isinstance(result, str)
    assert len(result) > 10
    mock_cli.assert_called_once()
    prompt_sent = mock_cli.call_args[0][0]
    assert "BULL" in prompt_sent
    assert "RELIANCE" in prompt_sent


def test_bear_agent_returns_string():
    from debate import _bear_agent
    candidate = {"symbol": "RELIANCE", "direction": "bullish", "score": 6,
                 "rsi": 55, "volume_ratio": 1.5, "categories": ["LongBuildUp"]}
    with patch("debate._call_claude_cli_debate") as mock_cli:
        mock_cli.return_value = "RSI divergence forming. Sector rotation risk. Downside to 2500."
        result = _bear_agent(candidate, {})
    assert isinstance(result, str)
    assert len(result) > 10
    prompt_sent = mock_cli.call_args[0][0]
    assert "BEAR" in prompt_sent


def test_context_agent_returns_string():
    from debate import _context_agent
    candidate = {"symbol": "RELIANCE", "direction": "bullish", "score": 6}
    caches = {"macro": {"fii_net_crores": -1200}, "vix_history": [14, 15, 16]}
    with patch("debate._call_claude_cli_debate") as mock_cli:
        mock_cli.return_value = "Earnings in 3 days. FII selling trend. No VIX anomaly."
        result = _context_agent(candidate, caches)
    assert isinstance(result, str)
    prompt_sent = mock_cli.call_args[0][0]
    assert "CONTEXT" in prompt_sent


def test_moderator_synthesizes():
    from debate import _moderate
    with patch("debate._call_claude_cli_debate") as mock_cli:
        mock_cli.return_value = (
            "Both agree on volume strength. Bear's RSI concern is weaker. "
            "Context notes earnings risk. Verdict: bull-leaning"
        )
        result = _moderate("RELIANCE", "bull view", "bear view", "context view")
    assert "bull-leaning" in result or "bear-leaning" in result or "split" in result


def test_moderator_detects_contradiction():
    from debate import _moderate
    with patch("debate._call_claude_cli_debate") as mock_cli:
        mock_cli.return_value = (
            "Bull says volume is rising, bear says volume is declining. "
            "CONTRADICTION: Volume trend disagreement"
        )
        result = _moderate("RELIANCE", "bull view", "bear view", "context view")
    assert "CONTRADICTION" in result


def test_moderator_round2_on_contradiction():
    """Patch individual agents to avoid thread-ordering fragility."""
    from debate import _run_single_debate
    mod_responses = iter([
        "Disagree on trend. CONTRADICTION: volume direction",
        "Bear has stronger evidence. Final verdict: bear-leaning",
    ])
    with patch("debate._bull_agent", return_value="Bull: strong buy"), \
         patch("debate._bear_agent", return_value="Bear: definite sell"), \
         patch("debate._context_agent", return_value="Context: earnings tomorrow"), \
         patch("debate._call_claude_cli_debate", side_effect=mod_responses):
        result = _run_single_debate(
            {"symbol": "TCS", "direction": "bullish", "score": 5,
             "rsi": 60, "volume_ratio": 1.2, "categories": ["LongBuildUp"]},
            {},
        )
    assert "bear-leaning" in result["final_summary"]
    assert result["moderator"]["round_2"] is not None


def test_agent_timeout_graceful():
    """Patch individual agents to avoid thread-ordering fragility."""
    from debate import _run_single_debate
    with patch("debate._bull_agent", return_value="Bull: strong buy"), \
         patch("debate._bear_agent", return_value=""), \
         patch("debate._context_agent", return_value="Context: no events"), \
         patch("debate._call_claude_cli_debate",
               return_value="Only bull and context available. Verdict: bull-leaning"):
        result = _run_single_debate(
            {"symbol": "INFY", "direction": "bullish", "score": 4,
             "rsi": 50, "volume_ratio": 1.0, "categories": []},
            {},
        )
    assert result["final_summary"] != ""
    assert result["agents"]["bear"]["response"] == ""


def test_all_agents_fail_returns_empty():
    from debate import _run_single_debate
    with patch("debate._call_claude_cli_debate") as mock_cli:
        mock_cli.return_value = None
        result = _run_single_debate(
            {"symbol": "INFY", "direction": "bullish", "score": 4,
             "rsi": 50, "volume_ratio": 1.0, "categories": []},
            {},
        )
    assert result["final_summary"] == ""


def test_run_debates_returns_dict():
    from debate import run_debates
    candidates = [
        {"symbol": "RELIANCE", "direction": "bullish", "score": 7,
         "rsi": 55, "volume_ratio": 1.5, "categories": ["LongBuildUp"]},
        {"symbol": "TCS", "direction": "bearish", "score": 5,
         "rsi": 30, "volume_ratio": 0.8, "categories": ["ShortBuildUp"]},
    ]
    with patch("debate._call_claude_cli_debate") as mock_cli:
        mock_cli.return_value = "Test response. Verdict: bull-leaning"
        result = run_debates(candidates, {})
    assert isinstance(result, dict)
    assert "RELIANCE" in result
    assert "TCS" in result
    assert "DEBATE SUMMARY" in result["RELIANCE"]


def test_run_debates_disabled():
    from debate import run_debates
    import config
    with patch.object(config, "DEBATE_ENABLED", False):
        result = run_debates([{"symbol": "TEST", "score": 5}], {})
        assert result == {}


def test_debate_log_written(tmp_path):
    from debate import _run_single_debate, DEBATE_LOG_DIR
    with patch("debate.DEBATE_LOG_DIR", tmp_path), \
         patch("debate._call_claude_cli_debate") as mock_cli:
        mock_cli.return_value = "Test. Verdict: split"
        _run_single_debate(
            {"symbol": "HDFC", "direction": "bullish", "score": 5,
             "rsi": 50, "volume_ratio": 1.0, "categories": []},
            {},
        )
    logs = list(tmp_path.glob("*.json"))
    assert len(logs) == 1
    data = json.loads(logs[0].read_text())
    assert data["symbol"] == "HDFC"
    assert "agents" in data
    assert "moderator" in data


def test_summary_format():
    from debate import _format_summary
    result = _format_summary(
        "RELIANCE",
        "Strong momentum, target 2800",
        "RSI divergence, downside 2500",
        "Earnings in 3 days",
        "bull-leaning",
        1,
    )
    assert result.startswith("DEBATE SUMMARY for RELIANCE:")
    assert "Bull:" in result
    assert "Bear:" in result
    assert "Context:" in result
    assert "Verdict: bull-leaning (1 round)" in result
