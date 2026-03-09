import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_entry_blocks_when_claude_down(monkeypatch):
    """evaluate_entry returns SKIP when Claude is unavailable (fail-safe)."""
    import claude_intel
    monkeypatch.setattr(claude_intel, "_call_claude", lambda *a, **kw: None)
    import config
    monkeypatch.setattr(config, "V6_CLAUDE_FIRST", True)

    result = claude_intel.evaluate_entry(
        {"symbol": "MCX", "direction": "bullish", "score": 7.0, "rsi": 55, "volume_ratio": 2.0},
        "EQ", "TRENDING_UP", 16.0, None, 0.85, 23800, {"positions": [], "capital": 100000}
    )
    approved, reasoning, alloc_adj = result[0], result[1], result[2]
    assert not approved, "Should block entry when Claude is down in V6"

def test_exit_honors_rule_when_claude_down(monkeypatch):
    """evaluate_exit returns EXIT when Claude is unavailable (execute triggered exit)."""
    import claude_intel
    monkeypatch.setattr(claude_intel, "_call_claude", lambda *a, **kw: None)

    pos = {"symbol": "MCX", "instrument": "EQ", "direction": "bullish", "entry_price": 100}
    should_exit, reasoning = claude_intel.evaluate_exit(pos, "trailing_stop", 95, -5, -5.0, {"positions": []})
    assert should_exit, "Should honor rule-based exit when Claude is down"

def test_failure_counter_tracks(monkeypatch):
    """Claude consecutive failures tracked in state."""
    import claude_intel
    state = {"claude_consecutive_failures": 0, "claude_lockdown_active": False}
    claude_intel._track_claude_failure(state)
    assert state["claude_consecutive_failures"] == 1

    claude_intel._track_claude_failure(state)
    claude_intel._track_claude_failure(state)
    assert state["claude_consecutive_failures"] == 3

def test_failure_counter_resets_on_success():
    """Counter resets when Claude succeeds."""
    import claude_intel
    state = {"claude_consecutive_failures": 5, "claude_lockdown_active": True}
    claude_intel._track_claude_success(state)
    assert state["claude_consecutive_failures"] == 0
    assert state["claude_lockdown_active"] == False

def test_lockdown_activates_at_5_failures():
    """Lockdown activates after 5 consecutive failures."""
    import claude_intel
    state = {"claude_consecutive_failures": 0, "claude_lockdown_active": False}
    for _ in range(4):
        claude_intel._track_claude_failure(state)
    assert not state["claude_lockdown_active"], "Should not lockdown at 4 failures"
    claude_intel._track_claude_failure(state)
    assert state["claude_lockdown_active"], "Should lockdown at 5 failures"
