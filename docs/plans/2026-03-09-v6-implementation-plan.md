# V6: Claude-First Architecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure the trading engine so Claude makes all entry/exit decisions, with rules providing signals and safety guardrails only.

**Architecture:** Three layers — Data Layer (screener, indicators, market intel), Claude Layer (all decisions), Safety Layer (SL, circuit breaker, position limits). Remove 45+ pre-Claude filters and allocation modifiers. Add production hardening (backup, locking, replay, kill switch).

**Tech Stack:** Python 3.13, AngelOne SmartAPI, Claude Code CLI (Max subscription), Telegram Bot API, Obsidian vault, Raspberry Pi (cron-based)

**Design Doc:** `docs/plans/2026-03-09-v6-claude-first-architecture-design.md`

---

## Task 1: Production Hardening — Portfolio Safety

**Files:**
- Modify: `paper_trade.py` (`load_portfolio`, `save_portfolio`, `close_position`)
- Modify: `smart_monitor.py` (`_load_state`, `_save_state`, new `cleanup_closed_positions`)
- Test: `tests/test_portfolio_safety.py`

**Step 1: Write failing tests**

```python
# tests/test_portfolio_safety.py
import json, os, sys, shutil
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_save_creates_backup(tmp_path):
    """save_portfolio creates .bak before overwriting."""
    pf = tmp_path / "portfolio.json"
    pf.write_text(json.dumps({"capital": 90000, "positions": [], "closed_trades": [], "stats": {}}))

    import paper_trade
    orig_file = paper_trade.PORTFOLIO_FILE
    paper_trade.PORTFOLIO_FILE = pf

    try:
        new_portfolio = {"capital": 100000, "available_capital": 100000, "positions": [], "closed_trades": [], "stats": {}}
        paper_trade.save_portfolio(new_portfolio)
        bak = pf.with_suffix(".json.bak")
        assert bak.exists(), "Backup file not created"
        assert json.loads(bak.read_text())["capital"] == 90000
        assert json.loads(pf.read_text())["capital"] == 100000
    finally:
        paper_trade.PORTFOLIO_FILE = orig_file

def test_load_recovers_from_corruption(tmp_path):
    """load_portfolio falls back to .bak if main file is corrupted."""
    pf = tmp_path / "portfolio.json"
    bak = tmp_path / "portfolio.json.bak"
    pf.write_text("{corrupted json!!!}")
    bak.write_text(json.dumps({"capital": 95000, "available_capital": 95000, "positions": [], "closed_trades": [], "stats": {}}))

    import paper_trade
    orig_file = paper_trade.PORTFOLIO_FILE
    paper_trade.PORTFOLIO_FILE = pf

    try:
        result = paper_trade.load_portfolio()
        assert result["capital"] == 95000, "Should recover from backup"
    finally:
        paper_trade.PORTFOLIO_FILE = orig_file

def test_load_returns_empty_if_both_corrupted(tmp_path):
    """load_portfolio returns empty portfolio if both files corrupted."""
    pf = tmp_path / "portfolio.json"
    bak = tmp_path / "portfolio.json.bak"
    pf.write_text("bad")
    bak.write_text("also bad")

    import paper_trade
    orig_file = paper_trade.PORTFOLIO_FILE
    paper_trade.PORTFOLIO_FILE = pf

    try:
        result = paper_trade.load_portfolio()
        assert "positions" in result
        assert result["positions"] == []
    finally:
        paper_trade.PORTFOLIO_FILE = orig_file

def test_save_atomic_write(tmp_path):
    """save_portfolio uses atomic write (tmp + rename)."""
    pf = tmp_path / "portfolio.json"
    import paper_trade
    orig_file = paper_trade.PORTFOLIO_FILE
    paper_trade.PORTFOLIO_FILE = pf

    try:
        portfolio = {"capital": 100000, "available_capital": 100000, "positions": [], "closed_trades": [], "stats": {}}
        paper_trade.save_portfolio(portfolio)
        # tmp file should not exist after successful save
        assert not pf.with_suffix(".json.tmp").exists()
        assert pf.exists()
    finally:
        paper_trade.PORTFOLIO_FILE = orig_file

def test_cleanup_closed_positions():
    """cleanup_closed_positions removes stale state entries."""
    from smart_monitor import cleanup_closed_positions

    portfolio = {
        "positions": [
            {"symbol": "MCX", "status": "open"},
            {"symbol": "INFY", "status": "open"},
        ]
    }
    state = {
        "last_check": {"MCX": "2026-03-09T12:00:00", "BPCL": "2026-03-09T11:00:00", "INFY": "2026-03-09T12:00:00"},
        "position_assessments": {"MCX": {"action": "HOLD"}, "BPCL": {"action": "EXIT"}},
    }
    cleanup_closed_positions(portfolio, state)
    assert "MCX" in state["last_check"]
    assert "INFY" in state["last_check"]
    assert "BPCL" not in state["last_check"]
    assert "BPCL" not in state["position_assessments"]
```

**Step 2: Run tests to verify they fail**

```bash
cd ~/financial-agent-india && python3 -m pytest tests/test_portfolio_safety.py -v
```

Expected: FAIL (save_portfolio doesn't create backup, load_portfolio doesn't handle corruption, cleanup_closed_positions doesn't exist)

**Step 3: Implement**

In `paper_trade.py`, replace `save_portfolio()` (lines ~541-544):

```python
def save_portfolio(portfolio: dict) -> None:
    """Save with atomic write + backup."""
    import shutil
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    content = json.dumps(portfolio, indent=2, default=str)

    # Backup current file before overwriting
    if PORTFOLIO_FILE.exists():
        backup = PORTFOLIO_FILE.with_suffix(".json.bak")
        shutil.copy2(PORTFOLIO_FILE, backup)

    # Atomic write: write to temp, then rename
    tmp = PORTFOLIO_FILE.with_suffix(".json.tmp")
    tmp.write_text(content)
    tmp.rename(PORTFOLIO_FILE)
```

Replace `load_portfolio()` (lines ~534-538):

```python
def load_portfolio() -> dict:
    """Load with corruption recovery from .bak."""
    for path in [PORTFOLIO_FILE, PORTFOLIO_FILE.with_suffix(".json.bak")]:
        if path and path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, ValueError) as e:
                logger.error("Corrupted %s: %s", path, e)
                continue
    return _empty_portfolio()
```

In `smart_monitor.py`, apply same backup pattern to `_save_state()` and `_load_state()`. Add new function:

```python
def cleanup_closed_positions(portfolio: dict, state: dict) -> None:
    """Remove state for positions that are no longer open."""
    open_symbols = {pos["symbol"] for pos in portfolio.get("positions", [])
                    if pos.get("status") == "open"}
    state["last_check"] = {k: v for k, v in state.get("last_check", {}).items()
                           if k in open_symbols}
    state["position_assessments"] = {k: v for k, v in state.get("position_assessments", {}).items()
                                      if k in open_symbols}
```

Call `cleanup_closed_positions(portfolio, state)` at end of `close_position()` in `paper_trade.py` (after trade lesson recording, line ~2908).

**Step 4: Run tests**

```bash
python3 -m pytest tests/test_portfolio_safety.py -v
```

Expected: ALL PASS

**Step 5: Commit**

```bash
git add paper_trade.py smart_monitor.py tests/test_portfolio_safety.py
git commit -m "feat(v6): add portfolio backup, corruption recovery, state cleanup"
```

---

## Task 2: Production Hardening — Cron Script

**Files:**
- Modify: `scripts/paper_trade_cron.sh`
- Modify: `config.py` (holiday calendar already exists at line 175)

**Step 1: Add flock, holiday check, log rotation, tick timing, EOD fix to cron script**

Replace `scripts/paper_trade_cron.sh` with these additions at key points:

After `cd ~/financial-agent-india` and `source venv/bin/activate`:

```bash
# --- File lock: prevent overlapping ticks ---
LOCKFILE="/tmp/paper_trade.lock"
exec 200>"$LOCKFILE"
flock -n 200 || { echo "[SKIP] Previous tick still running at $(TZ=Asia/Kolkata date)" >> "$LOG"; exit 0; }

# --- Log rotation: cap at 5MB ---
if [ -f "$LOG" ] && [ "$(stat -c%s "$LOG" 2>/dev/null || stat -f%z "$LOG" 2>/dev/null || echo 0)" -gt 5242880 ]; then
    mv "$LOG" "${LOG}.old"
    echo "--- Log rotated $(TZ=Asia/Kolkata date) ---" > "$LOG"
fi
```

After time calculation, before mode logic:

```bash
# --- Holiday check: skip NSE holidays ---
TODAY_DATE=$(TZ=Asia/Kolkata date +%Y-%m-%d)
IS_HOLIDAY=$(python3 -c "
from config import NSE_HOLIDAYS
print('yes' if '$TODAY_DATE' in [str(d) for d in NSE_HOLIDAYS.get(${TODAY_DATE:0:4}, [])] else 'no')
" 2>/dev/null || echo "no")

if [ "$IS_HOLIDAY" = "yes" ]; then
    echo "[SKIP] NSE holiday ($TODAY_DATE)" >> "$LOG"
    echo "" >> "$LOG"
    exit 0
fi
```

Wrap main execution with tick timer:

```bash
TICK_START=$(date +%s)

# ... all existing mode logic ...

TICK_END=$(date +%s)
TICK_DURATION=$((TICK_END - TICK_START))
echo "[TIMING] Tick completed in ${TICK_DURATION}s" >> "$LOG"

if [ "$TICK_DURATION" -gt 240 ]; then
    MSG="⚠️ Slow tick: ${TICK_DURATION}s (limit 300s)"
    # ... existing Telegram send pattern ...
fi
```

Fix EOD wrap timing (line ~86):

```bash
# OLD: if [ "$HOUR" -eq 15 ] && [ "$MIN" -ge 18 ] && [ "$MIN" -le 25 ]; then
# NEW:
if [ "$HOUR" -eq 15 ] && [ "$MIN" -ge 33 ] && [ "$MIN" -le 40 ]; then
```

Add decision log cleanup in EOD section:

```bash
# Clean up old decision replay files (>30 days)
find data/paper_trades/claude_decisions/ -name "*.json" -mtime +30 -delete 2>/dev/null
```

**Step 2: Update config.py holiday format**

Verify `config.py` line ~175 has `NSE_HOLIDAYS` dict accessible. Currently it's used in `is_trading_day()`. Ensure it's importable as `from config import NSE_HOLIDAYS`. If it's structured differently, adapt the holiday check above.

**Step 3: Test manually**

```bash
# Test flock
bash scripts/paper_trade_cron.sh & bash scripts/paper_trade_cron.sh
# Second should log "[SKIP] Previous tick still running"

# Test holiday check
grep "holiday" data/paper_trades/cron.log

# Test log rotation (create oversized log)
dd if=/dev/zero of=data/paper_trades/cron.log bs=1M count=6 2>/dev/null
bash scripts/paper_trade_cron.sh
ls -la data/paper_trades/cron.log*
```

**Step 4: Commit**

```bash
git add scripts/paper_trade_cron.sh config.py
git commit -m "feat(v6): add cron locking, holiday check, log rotation, tick timing, EOD fix"
```

---

## Task 3: Production Hardening — Telegram & Claude Validation

**Files:**
- Modify: `paper_trade.py` (`_telegram_send`)
- Modify: `claude_intel.py` (add `_validate_entry_response`, `_validate_exit_response`)
- Test: `tests/test_telegram_split.py`
- Test: `tests/test_claude_validation.py`

**Step 1: Write failing tests**

```python
# tests/test_telegram_split.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_short_message_not_split(monkeypatch):
    """Messages under 4000 chars sent as single message."""
    sent = []
    import paper_trade
    monkeypatch.setattr(paper_trade, "_send_single_telegram", lambda msg, **kw: sent.append(msg))
    paper_trade._telegram_send("Short message")
    assert len(sent) == 1

def test_long_message_split(monkeypatch):
    """Messages over 4000 chars split at newlines."""
    sent = []
    import paper_trade
    monkeypatch.setattr(paper_trade, "_send_single_telegram", lambda msg, **kw: sent.append(msg))
    long_msg = "\n".join([f"Line {i}: " + "x" * 50 for i in range(200)])
    assert len(long_msg) > 4000
    paper_trade._telegram_send(long_msg)
    assert len(sent) > 1
    for chunk in sent:
        assert len(chunk) <= 4100  # 4000 + header buffer
```

```python
# tests/test_claude_validation.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from claude_intel import _validate_entry_response, _validate_exit_response

def test_valid_entry_response():
    resp = {"action": "TRADE", "conviction": "high", "allocation_adj": 1.2, "reasoning": "Good setup"}
    result = _validate_entry_response(resp)
    assert result["action"] == "TRADE"
    assert result["allocation_adj"] == 1.2

def test_entry_response_missing_fields():
    resp = {"action": "TRADE"}
    result = _validate_entry_response(resp)
    assert result["conviction"] == "medium"
    assert result["allocation_adj"] == 1.0
    assert "reasoning" in result

def test_entry_response_null_allocation():
    resp = {"action": "TRADE", "allocation_adj": None}
    result = _validate_entry_response(resp)
    assert result["allocation_adj"] == 1.0

def test_entry_response_out_of_bounds_allocation():
    resp = {"action": "TRADE", "allocation_adj": 3.0}
    result = _validate_entry_response(resp)
    assert result["allocation_adj"] == 1.5

def test_entry_response_invalid_action():
    resp = {"action": "BUY", "conviction": "high"}
    result = _validate_entry_response(resp)
    assert result["action"] == "SKIP"

def test_exit_response_valid():
    resp = {"action": "HOLD", "reasoning": "Momentum still strong"}
    result = _validate_exit_response(resp)
    assert result["action"] == "HOLD"

def test_exit_response_invalid_action():
    resp = {"action": "SELL"}
    result = _validate_exit_response(resp)
    assert result["action"] == "EXIT"  # default to safe exit

def test_exit_response_missing():
    result = _validate_exit_response({})
    assert result["action"] == "EXIT"
    assert "reasoning" in result

def test_exit_response_add_action():
    resp = {"action": "ADD", "reasoning": "Strong momentum, add to winner"}
    result = _validate_exit_response(resp)
    assert result["action"] == "ADD"
```

**Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_telegram_split.py tests/test_claude_validation.py -v
```

**Step 3: Implement**

In `paper_trade.py`, refactor `_telegram_send()` (lines ~268-292):

```python
def _send_single_telegram(text: str, **kwargs) -> None:
    """Send a single Telegram message (existing implementation)."""
    # Move current _telegram_send body here
    ...

def _telegram_send(text: str, **kwargs) -> None:
    """Send Telegram message, splitting if >4000 chars."""
    MAX_LEN = 4000
    if len(text) <= MAX_LEN:
        _send_single_telegram(text, **kwargs)
        return

    chunks = []
    while text:
        if len(text) <= MAX_LEN:
            chunks.append(text)
            break
        split_at = text.rfind('\n', 0, MAX_LEN)
        if split_at == -1:
            split_at = MAX_LEN
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip('\n')

    for i, chunk in enumerate(chunks):
        header = f"({i+1}/{len(chunks)})\n" if len(chunks) > 1 else ""
        _send_single_telegram(header + chunk, **kwargs)
```

In `claude_intel.py`, add validation functions after `_parse_json()`:

```python
def _validate_entry_response(parsed: dict) -> dict:
    """Normalize and validate Claude entry response."""
    defaults = {"action": "SKIP", "conviction": "medium", "allocation_adj": 1.0, "reasoning": "No reasoning provided"}
    result = {}
    for key, default in defaults.items():
        val = parsed.get(key, default)
        result[key] = default if val is None else val
    try:
        result["allocation_adj"] = max(0.5, min(1.5, float(result["allocation_adj"])))
    except (TypeError, ValueError):
        result["allocation_adj"] = 1.0
    result["action"] = result["action"].upper() if isinstance(result["action"], str) else "SKIP"
    if result["action"] not in ("TRADE", "SKIP"):
        result["action"] = "SKIP"
    return result

def _validate_exit_response(parsed: dict) -> dict:
    """Normalize and validate Claude exit response."""
    defaults = {"action": "EXIT", "reasoning": "No reasoning provided"}
    result = {}
    for key, default in defaults.items():
        val = parsed.get(key, default)
        result[key] = default if val is None else val
    result["action"] = result["action"].upper() if isinstance(result["action"], str) else "EXIT"
    if result["action"] not in ("EXIT", "HOLD", "PARTIAL", "ADD"):
        result["action"] = "EXIT"
    return result
```

Then use these in `evaluate_entry()` and `evaluate_exit()` after `_parse_json()` calls.

**Step 4: Run tests**

```bash
python3 -m pytest tests/test_telegram_split.py tests/test_claude_validation.py -v
```

**Step 5: Commit**

```bash
git add paper_trade.py claude_intel.py tests/test_telegram_split.py tests/test_claude_validation.py
git commit -m "feat(v6): add Telegram message splitting and Claude response validation"
```

---

## Task 4: Kill Switch & Config Cleanup

**Files:**
- Modify: `config.py` (add `V6_CLAUDE_FIRST`, remove ~40 constants)
- Test: `tests/test_config_v6.py`

**Step 1: Write failing test**

```python
# tests/test_config_v6.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

def test_v6_flag_exists():
    assert hasattr(config, "V6_CLAUDE_FIRST")
    assert isinstance(config.V6_CLAUDE_FIRST, bool)

def test_removed_constants_gone():
    """V6 removes obsolete allocation/drawdown constants."""
    removed = [
        "US_CRASH_BLOCK_PCT", "US_SEVERE_CRASH_PCT", "US_MILD_RED_PCT",
        "NASDAQ_IT_CRASH_PCT", "GIFT_GAP_REDUCE_PCT", "GIFT_GAP_BLOCK_PCT",
        "FII_HEAVY_SELL_CRORES", "FII_EXTREME_SELL_CRORES",
        "SUPERTREND_DISAGREE_REDUCTION",
        "DRAWDOWN_DAILY_HALT", "DRAWDOWN_WEEKLY_REDUCE", "DRAWDOWN_MONTHLY_HALT",
        "CONSECUTIVE_LOSS_PAUSE",
    ]
    for name in removed:
        assert not hasattr(config, name), f"{name} should be removed in V6"

def test_kept_constants_exist():
    """Safety-critical constants must survive V6 cleanup."""
    kept = [
        "DAILY_LOSS_CIRCUIT_BREAKER_PCT", "CASH_RESERVE_PCT", "MAX_STRATEGY_ALLOC_PCT",
        "MAX_CONCURRENT_POSITIONS", "MAX_SAME_SECTOR", "API_DELAY",
        "HOT_CHECK_INTERVAL_MIN", "WARM_CHECK_INTERVAL_MIN", "COLD_CHECK_INTERVAL_MIN",
    ]
    for name in kept:
        assert hasattr(config, name), f"{name} must be kept in V6"
```

**Step 2: Run tests to verify failure**

```bash
python3 -m pytest tests/test_config_v6.py -v
```

**Step 3: Implement**

Add to `config.py` near the top (after imports):

```python
# ── V6 Claude-First Architecture ──────────────────────────────────────────
V6_CLAUDE_FIRST = True  # Set False to revert to V5 rule-based decisions
```

Remove the constants listed in `test_removed_constants_gone`. Check if any are referenced elsewhere — if so, wrap usage in `if not config.V6_CLAUDE_FIRST:` guard or remove the reference entirely. Key files to check:
- `global_intel.py` `compute_hard_gate()` — uses `US_CRASH_BLOCK_PCT` etc. These thresholds are hardcoded in the function body (lines 176-232), NOT from config constants. **Safe to remove from config.**
- `paper_trade.py` — uses `SUPERTREND_DISAGREE_REDUCTION`, `DRAWDOWN_*`, `CONSECUTIVE_LOSS_PAUSE`. Wrap in V6 guard or remove.

**Step 4: Run tests + existing tests**

```bash
python3 -m pytest tests/test_config_v6.py tests/test_config.py -v
```

**Step 5: Commit**

```bash
git add config.py tests/test_config_v6.py
git commit -m "feat(v6): add V6_CLAUDE_FIRST kill switch, remove 40+ obsolete constants"
```

---

## Task 5: Decision Replay Mode

**Files:**
- Modify: `claude_intel.py` (save decision logs after every eval)
- Create: `tests/test_decision_replay.py`

**Step 1: Write failing test**

```python
# tests/test_decision_replay.py
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_save_decision_log(tmp_path, monkeypatch):
    """Decision log saved after evaluate_entry."""
    import claude_intel
    monkeypatch.setattr(claude_intel, "DECISION_LOG_DIR", tmp_path)
    # Mock _call_claude to return a canned response
    monkeypatch.setattr(claude_intel, "_call_claude", lambda *a, **kw: '{"action":"TRADE","conviction":"high","allocation_adj":1.2,"reasoning":"Strong setup"}')

    candidate = {"symbol": "MCX", "direction": "bullish", "score": 7.5, "rsi": 55, "volume_ratio": 2.1}
    result = claude_intel.evaluate_entry(
        candidate, "EQ", "TRENDING_UP", 16.0, None, 0.85, 23800, {"positions": [], "capital": 100000}
    )

    logs = list(tmp_path.glob("*.json"))
    assert len(logs) == 1
    data = json.loads(logs[0].read_text())
    assert data["type"] == "entry"
    assert data["symbol"] == "MCX"
    assert "prompt_sent" in data
    assert "claude_response" in data
    assert data["parsed_action"] == "TRADE"
```

**Step 2: Run to verify failure**

```bash
python3 -m pytest tests/test_decision_replay.py -v
```

**Step 3: Implement**

In `claude_intel.py`, add at module level:

```python
DECISION_LOG_DIR = Path("data/paper_trades/claude_decisions")

def _save_decision_log(decision_type: str, symbol: str, prompt: str, response: str, parsed: dict, extra: dict = None):
    """Save Claude decision for replay/debugging."""
    try:
        DECISION_LOG_DIR.mkdir(parents=True, exist_ok=True)
        from datetime import datetime, timezone, timedelta
        IST = timezone(timedelta(hours=5, minutes=30))
        now = datetime.now(IST)
        filename = f"{now.strftime('%Y-%m-%d_%H-%M')}_{decision_type}_{symbol}.json"
        data = {
            "timestamp": now.isoformat(),
            "type": decision_type,
            "symbol": symbol,
            "prompt_sent": prompt,
            "claude_response": response,
            "parsed_action": parsed.get("action", ""),
            **(extra or {}),
        }
        (DECISION_LOG_DIR / filename).write_text(json.dumps(data, indent=2, default=str))
    except Exception as e:
        logger.debug("Decision log save failed: %s", e)
```

Call `_save_decision_log("entry", symbol, prompt, raw_response, parsed)` at the end of `evaluate_entry()` and similarly for `evaluate_exit()`.

**Step 4: Run tests**

```bash
python3 -m pytest tests/test_decision_replay.py -v
```

**Step 5: Commit**

```bash
git add claude_intel.py tests/test_decision_replay.py
git commit -m "feat(v6): add decision replay logging for all Claude evaluations"
```

---

## Task 6: Claude System Prompt Update

**Files:**
- Modify: `claude_intel.py` (replace `SYSTEM_PROMPT`, lines 17-29)
- Test: `tests/test_system_prompt.py`

**Step 1: Write test**

```python
# tests/test_system_prompt.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from claude_intel import SYSTEM_PROMPT

def test_system_prompt_has_v6_sections():
    """V6 system prompt includes all required intelligence sections."""
    required = [
        "MARKET MICROSTRUCTURE",
        "VWAP",
        "OI changes",
        "EXPIRY DAY RULES",
        "GAP HANDLING",
        "TIME AWARENESS",
        "POSITION MANAGEMENT",
        "IV AWARENESS",
        "HARD CONSTRAINTS",
        "Never add to losing positions",
    ]
    for section in required:
        assert section in SYSTEM_PROMPT, f"Missing: {section}"

def test_system_prompt_no_old_references():
    """V6 prompt shouldn't reference removed V5 concepts."""
    removed = ["15+ years", "pithy"]
    for term in removed:
        assert term not in SYSTEM_PROMPT, f"Old reference found: {term}"
```

**Step 2: Run to verify failure**

```bash
python3 -m pytest tests/test_system_prompt.py -v
```

**Step 3: Implement**

Replace `SYSTEM_PROMPT` in `claude_intel.py` (lines 17-29) with the full prompt from design doc Section 12 (the one starting with "You are a full-time F&O desk trader..."). Copy it verbatim from `docs/plans/2026-03-09-v6-claude-first-architecture-design.md`, Section 12, lines 432-497.

**Step 4: Run tests**

```bash
python3 -m pytest tests/test_system_prompt.py -v
```

**Step 5: Commit**

```bash
git add claude_intel.py tests/test_system_prompt.py
git commit -m "feat(v6): replace system prompt with full V6 Claude-First prompt"
```

---

## Task 7: Entry Thesis & Position-Level Claude Memory

**Files:**
- Modify: `paper_trade.py` (store thesis on position in open flow, pass to monitor)
- Modify: `smart_monitor.py` (add `position_assessments` to state, save/load/clear)
- Modify: `claude_intel.py` (`evaluate_entry` returns thesis, `evaluate_exit` receives it)
- Test: `tests/test_entry_thesis.py`

**Step 1: Write failing tests**

```python
# tests/test_entry_thesis.py
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_entry_thesis_schema():
    """Entry thesis has required fields."""
    thesis = {
        "reasoning": "Strong OI buildup with volume breakout",
        "conviction": "high",
        "key_conditions": {"regime": "TRENDING_UP", "vix": 16.2},
        "invalidation": "Price below 238 or regime shift",
        "expected_hold": "5-7 trading days",
        "target_scenario": "Breakout continuation toward 270",
    }
    required = ["reasoning", "conviction", "key_conditions", "invalidation", "expected_hold", "target_scenario"]
    for field in required:
        assert field in thesis, f"Missing field: {field}"

def test_position_assessment_storage():
    """Position assessments stored and retrieved from state."""
    from smart_monitor import _load_state, _save_state, save_position_assessment, get_position_assessment
    import tempfile, os
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        state_file = Path(td) / "state.json"
        import smart_monitor
        orig = smart_monitor.STATE_FILE
        smart_monitor.STATE_FILE = state_file

        try:
            state = _load_state()
            save_position_assessment(state, "MCX", "HOLD", "RSI divergence, wait", ["trailing_stop_near"])
            _save_state(state)

            state2 = _load_state()
            assessment = get_position_assessment(state2, "MCX")
            assert assessment is not None
            assert assessment["action"] == "HOLD"
            assert "trailing_stop_near" in assessment["signals_at_assessment"]
        finally:
            smart_monitor.STATE_FILE = orig

def test_position_assessment_cleared_on_close():
    """Position assessment removed when position closes."""
    from smart_monitor import cleanup_closed_positions

    state = {"position_assessments": {"MCX": {"action": "HOLD"}, "BPCL": {"action": "EXIT"}}, "last_check": {}}
    portfolio = {"positions": [{"symbol": "MCX", "status": "open"}]}
    cleanup_closed_positions(portfolio, state)
    assert "MCX" in state["position_assessments"]
    assert "BPCL" not in state["position_assessments"]
```

**Step 2: Run to verify failure**

```bash
python3 -m pytest tests/test_entry_thesis.py -v
```

**Step 3: Implement**

In `smart_monitor.py`, add:

```python
def save_position_assessment(state: dict, symbol: str, action: str, reasoning: str, signals: list[str]):
    """Save Claude's assessment for a position."""
    state.setdefault("position_assessments", {})
    state["position_assessments"][symbol] = {
        "timestamp": _now_ist().isoformat(),
        "action": action,
        "reasoning": reasoning,
        "signals_at_assessment": signals,
    }

def get_position_assessment(state: dict, symbol: str) -> dict | None:
    """Get last Claude assessment for a position."""
    return state.get("position_assessments", {}).get(symbol)
```

In `claude_intel.py` `evaluate_entry()`, after Claude returns and is parsed, build thesis dict:

```python
entry_thesis = {
    "reasoning": parsed.get("reasoning", ""),
    "conviction": parsed.get("conviction", "medium"),
    "key_conditions": {"regime": regime, "vix": vix, "pcr": pcr},
    "invalidation": "",  # Claude doesn't return this yet; leave empty for V6.1
    "expected_hold": "",
    "target_scenario": "",
}
```

Return it as a 4th element: `return (approved, reasoning, alloc_adj, entry_thesis)`.

In `paper_trade.py` `open_positions()`, after Claude approves a candidate and before appending to portfolio, store thesis on the position dict:

```python
pos["entry_thesis"] = entry_thesis
```

In `evaluate_exit()`, add entry thesis and previous assessment to the prompt if available:

```python
# Add to prompt construction
thesis = pos.get("entry_thesis", {})
if thesis:
    prompt += f"\nENTRY THESIS:\n- \"{thesis.get('reasoning', 'N/A')}\"\n- Invalidation: \"{thesis.get('invalidation', 'N/A')}\"\n"

prev = extra_context  # pass previous assessment via extra_context from monitor
if prev:
    prompt += f"\nPREVIOUS ASSESSMENT:\n{prev}\n"
```

**Step 4: Run tests**

```bash
python3 -m pytest tests/test_entry_thesis.py -v
```

**Step 5: Commit**

```bash
git add paper_trade.py claude_intel.py smart_monitor.py tests/test_entry_thesis.py
git commit -m "feat(v6): add entry thesis storage and position-level Claude memory"
```

---

## Task 8: Claude-Down Fail-Safe

**Files:**
- Modify: `claude_intel.py` (track failures, graduated escalation)
- Modify: `smart_monitor.py` (state fields for failure tracking)
- Test: `tests/test_claude_failsafe.py`

**Step 1: Write failing tests**

```python
# tests/test_claude_failsafe.py
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
    import claude_intel, smart_monitor
    monkeypatch.setattr(claude_intel, "_call_claude", lambda *a, **kw: None)

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
```

**Step 2: Run to verify failure**

```bash
python3 -m pytest tests/test_claude_failsafe.py -v
```

**Step 3: Implement**

In `claude_intel.py`:

```python
def _track_claude_failure(state: dict) -> None:
    state["claude_consecutive_failures"] = state.get("claude_consecutive_failures", 0) + 1
    if state["claude_consecutive_failures"] >= 5:
        state["claude_lockdown_active"] = True

def _track_claude_success(state: dict) -> None:
    state["claude_consecutive_failures"] = 0
    state["claude_lockdown_active"] = False
```

In `evaluate_entry()`, change the fail-open to fail-safe when V6 is enabled:

```python
# OLD (line ~241-243):
# if raw is None: return (True, "", 1.0)

# NEW:
if raw is None:
    import config
    if getattr(config, "V6_CLAUDE_FIRST", False):
        return (False, "Claude unavailable — blocking entry (fail-safe)", 0.0, {})
    return (True, "", 1.0)  # V5 legacy: fail-open
```

In `smart_monitor.py` `_load_state()` defaults, add:

```python
state.setdefault("claude_consecutive_failures", 0)
state.setdefault("claude_lockdown_active", False)
state.setdefault("position_assessments", {})
```

**Step 4: Run tests**

```bash
python3 -m pytest tests/test_claude_failsafe.py -v
```

**Step 5: Commit**

```bash
git add claude_intel.py smart_monitor.py tests/test_claude_failsafe.py
git commit -m "feat(v6): implement Claude-down fail-safe with failure tracking"
```

---

## Task 9: Entry Flow Restructure

**Files:**
- Modify: `paper_trade.py` `open_positions()` (lines ~1896-2696)
- Test: `tests/test_entry_flow_v6.py`

This is the largest task. The V6 entry flow removes ~150 lines of pre-Claude filters and replaces them with "pass everything to Claude with full context."

**Step 1: Write failing test**

```python
# tests/test_entry_flow_v6.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_v6_no_regime_gating(monkeypatch):
    """V6 doesn't block bullish candidates in TRENDING_DOWN regime."""
    import config
    monkeypatch.setattr(config, "V6_CLAUDE_FIRST", True)

    # In V5, bullish candidates were filtered out in TRENDING_DOWN
    # In V6, they pass through to Claude
    # Test: a bullish candidate in TRENDING_DOWN reaches Claude evaluation
    reached_claude = []

    import claude_intel
    orig_eval = claude_intel.evaluate_candidates
    def mock_eval(candidates, **kwargs):
        reached_claude.extend([c["symbol"] for c in candidates])
        return candidates  # approve all
    monkeypatch.setattr(claude_intel, "evaluate_candidates", mock_eval)

    # This test verifies the filter is removed, not full open_positions flow
    # Specific assertion: bullish candidate not filtered pre-Claude
    candidate = {"symbol": "TEST", "direction": "bullish", "score": 6.0, "rsi": 55, "volume_ratio": 2.0}
    # Regime = TRENDING_DOWN should NOT filter this in V6
    from paper_trade import _v6_pre_filter_candidates
    filtered = _v6_pre_filter_candidates([candidate], {"positions": []})
    assert len(filtered) == 1, "V6 should not filter by regime"
```

**Step 2: Run to verify failure**

```bash
python3 -m pytest tests/test_entry_flow_v6.py -v
```

**Step 3: Implement**

The key change is replacing the ~150-line quality filter loop (lines ~2164-2304) with a minimal safety-only filter when V6 is enabled.

In `paper_trade.py`, add a new function:

```python
def _v6_pre_filter_candidates(candidates: list[dict], portfolio: dict) -> list[dict]:
    """V6: Only filter for safety constraints, NOT market conditions.
    Claude sees everything else as context and decides."""
    filtered = []
    open_symbols = {p["symbol"] for p in portfolio.get("positions", []) if p.get("status") == "open"}

    for c in candidates:
        symbol = c.get("symbol", "")
        # Skip: already have position in this symbol
        if symbol in open_symbols:
            continue
        # Skip: sector limit (max 2 per sector)
        if check_sector_limit(symbol, portfolio.get("positions", [])):
            continue
        # Skip: on cooldown (recently stopped out)
        if check_cooldown(symbol, portfolio.get("closed_trades", []), _today_ist()):
            continue
        filtered.append(c)
    return filtered
```

In `open_positions()`, wrap the existing filter loop:

```python
if config.V6_CLAUDE_FIRST:
    quality_filtered = _v6_pre_filter_candidates(candidates, portfolio)
else:
    # ... existing V5 filter loop (lines 2164-2304) ...
    quality_filtered = [c for c in candidates if c.get("_passed_filters")]
```

Similarly, in the allocation loop, remove the 8-layer cascade when V6:

```python
if config.V6_CLAUDE_FIRST:
    # V6: Claude's allocation_adj is the only modifier
    final_alloc = base_alloc * claude_allocation_adj
    # Safety cap only
    final_alloc = min(final_alloc, portfolio["capital"] * config.MAX_STRATEGY_ALLOC_PCT)
else:
    # V5: existing cascade
    final_alloc = base_alloc * dd_mult * corr_mult * evt_mult * ...
```

Enhance the Claude prompt in `evaluate_candidates()` to include full portfolio context (already in design doc Section 5). Add regime, VIX tier, FII/DII, PCR, max pain, supertrend, all open positions with sector/direction, unrealized P&L, today's realized.

**Step 4: Run tests**

```bash
python3 -m pytest tests/test_entry_flow_v6.py -v
```

**Step 5: Verify existing tests still pass**

```bash
python3 -m pytest tests/ -v --timeout=30
```

**Step 6: Commit**

```bash
git add paper_trade.py tests/test_entry_flow_v6.py
git commit -m "feat(v6): restructure entry flow — remove pre-Claude filters, Claude decides all"
```

---

## Task 10: Exit Flow Restructure

**Files:**
- Modify: `paper_trade.py` `monitor_positions()` (lines ~2941-3573)
- Modify: `claude_intel.py` `evaluate_exit()` (enhance prompt)
- Test: `tests/test_exit_flow_v6.py`

**Step 1: Write failing test**

```python
# tests/test_exit_flow_v6.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_v6_exit_signals_computed_not_triggered():
    """V6 computes exit signals but doesn't auto-trigger (except SL)."""
    from paper_trade import _compute_exit_signals

    pos = {
        "instrument": "EQ", "direction": "bullish",
        "entry_price": 100, "target_price": 105,
        "stoploss_price": 97, "atr_at_entry": 2.0,
        "peak_price": 104, "entry_date": "2026-03-01",
    }
    signals = _compute_exit_signals(pos, current_ltp=104.5, pnl_pct=4.5)

    assert "trailing_stop" in signals
    assert "target" in signals
    assert "time_pressure" in signals
    # Signals are dicts with fired/value, not booleans
    assert isinstance(signals["trailing_stop"], dict)
    assert "fired" in signals["trailing_stop"]
    assert "distance" in signals["trailing_stop"]

def test_v6_mechanical_sl_always_fires():
    """Mechanical SL bypasses Claude — non-negotiable."""
    from paper_trade import _compute_exit_signals

    pos = {
        "instrument": "EQ", "direction": "bullish",
        "entry_price": 100, "stoploss_price": 97,
        "atr_at_entry": 2.0, "peak_price": 100,
        "target_price": 105, "entry_date": "2026-03-01",
    }
    signals = _compute_exit_signals(pos, current_ltp=96.5, pnl_pct=-3.5)
    assert signals["mechanical_sl"]["fired"] is True
```

**Step 2: Run to verify failure**

```bash
python3 -m pytest tests/test_exit_flow_v6.py -v
```

**Step 3: Implement**

Add new function to `paper_trade.py`:

```python
def _compute_exit_signals(pos: dict, current_ltp: float, pnl_pct: float) -> dict:
    """V6: Compute all exit signals without triggering any.
    Returns signal dict for Claude to evaluate."""
    signals = {}
    direction = pos.get("direction", "bullish")
    entry = pos["entry_price"]
    sl = pos.get("stoploss_price", 0)
    target = pos.get("target_price", 0)
    atr = pos.get("atr_at_entry", 0)
    peak = pos.get("peak_price", entry)

    # Mechanical SL (non-negotiable)
    sl_fired = (direction == "bullish" and current_ltp <= sl) or \
               (direction == "bearish" and current_ltp >= sl)
    signals["mechanical_sl"] = {"fired": sl_fired, "sl_price": sl, "distance_pct": round((current_ltp - sl) / entry * 100, 2)}

    # Trailing stop
    if atr > 0:
        trail_mult = config.TRAILING_SL_ATR_MULT if hasattr(config, 'TRAILING_SL_ATR_MULT') else 2.5
        if direction == "bullish":
            trail_sl = peak - trail_mult * atr
            trail_fired = current_ltp <= trail_sl
        else:
            trail_sl = peak + trail_mult * atr
            trail_fired = current_ltp >= trail_sl
        signals["trailing_stop"] = {"fired": trail_fired, "trail_price": round(trail_sl, 2), "distance": round(abs(current_ltp - trail_sl), 2)}
    else:
        signals["trailing_stop"] = {"fired": False, "distance": 999}

    # Target
    if target > 0:
        target_fired = (direction == "bullish" and current_ltp >= target) or \
                       (direction == "bearish" and current_ltp <= target)
        signals["target"] = {"fired": target_fired, "target_price": target, "distance_pct": round(abs(current_ltp - target) / entry * 100, 2)}
    else:
        signals["target"] = {"fired": False, "distance_pct": 999}

    # Time pressure
    entry_date = pos.get("entry_date", "")
    max_hold = pos.get("max_hold_date", "")
    if entry_date and max_hold:
        from datetime import datetime
        try:
            days_held = (datetime.now().date() - datetime.strptime(entry_date[:10], "%Y-%m-%d").date()).days
            max_days = (datetime.strptime(max_hold[:10], "%Y-%m-%d").date() - datetime.strptime(entry_date[:10], "%Y-%m-%d").date()).days
            pct_elapsed = days_held / max(max_days, 1) * 100
        except ValueError:
            days_held, pct_elapsed = 0, 0
        signals["time_pressure"] = {"fired": pct_elapsed >= 100, "days_held": days_held, "pct_elapsed": round(pct_elapsed, 1)}
    else:
        signals["time_pressure"] = {"fired": False, "days_held": 0, "pct_elapsed": 0}

    return signals
```

In `monitor_positions()`, when V6 is enabled:

```python
if config.V6_CLAUDE_FIRST:
    signals = _compute_exit_signals(pos, ltp, pnl_pct)

    # Mechanical SL: always fire, never ask Claude
    if signals["mechanical_sl"]["fired"]:
        reason = "stoploss"
        # ... close immediately ...
        continue

    # Any other signal fired? Ask Claude
    any_signal_fired = any(s.get("fired") for name, s in signals.items() if name != "mechanical_sl")
    if any_signal_fired:
        # Build signal summary for Claude
        signal_text = _format_signals_for_claude(signals)
        # Get previous assessment
        prev = get_position_assessment(state, pos["symbol"])
        prev_text = f"[{prev['action']}] {prev['reasoning']}" if prev else ""

        should_exit, reasoning = evaluate_exit(
            pos, signal_text, ltp, pnl, pnl_pct, portfolio,
            extra_context=prev_text
        )
        # Save assessment
        save_position_assessment(state, pos["symbol"], "EXIT" if should_exit else "HOLD", reasoning, [n for n,s in signals.items() if s.get("fired")])

        if should_exit:
            # ... close ...
    # else: no signals, skip Claude call (save latency)
else:
    # ... existing V5 exit logic ...
```

**Step 4: Run tests**

```bash
python3 -m pytest tests/test_exit_flow_v6.py -v
```

**Step 5: Commit**

```bash
git add paper_trade.py claude_intel.py tests/test_exit_flow_v6.py
git commit -m "feat(v6): restructure exit flow — rules as signals, Claude decides"
```

---

## Task 11: Market Intelligence — VWAP & OI

**Files:**
- Create: `market_intel.py` (VWAP computation, OI delta tracking, OI support/resistance)
- Test: `tests/test_market_intel.py`

**Step 1: Write failing tests**

```python
# tests/test_market_intel.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_compute_vwap():
    from market_intel import compute_vwap_from_candles
    candles = [
        {"close": 100, "high": 102, "low": 98, "volume": 1000},
        {"close": 105, "high": 106, "low": 103, "volume": 2000},
        {"close": 103, "high": 105, "low": 101, "volume": 1500},
    ]
    vwap = compute_vwap_from_candles(candles)
    assert vwap is not None
    assert 100 < vwap < 106
    # VWAP = sum(typical_price * volume) / sum(volume)
    # typical = (high + low + close) / 3
    tp1 = (102+98+100)/3 * 1000   # 100 * 1000
    tp2 = (106+103+105)/3 * 2000  # 104.67 * 2000
    tp3 = (105+101+103)/3 * 1500  # 103 * 1500
    expected = (tp1 + tp2 + tp3) / (1000 + 2000 + 1500)
    assert abs(vwap - expected) < 0.01

def test_classify_oi_change():
    from market_intel import classify_oi_change
    assert classify_oi_change(oi_change_pct=12, price_change_pct=2.0) == "long_buildup"
    assert classify_oi_change(oi_change_pct=15, price_change_pct=-1.5) == "short_buildup"
    assert classify_oi_change(oi_change_pct=-10, price_change_pct=1.0) == "short_covering"
    assert classify_oi_change(oi_change_pct=-8, price_change_pct=-2.0) == "long_unwinding"

def test_get_oi_support_resistance():
    from market_intel import get_oi_support_resistance
    option_chain = {
        "CE": [
            {"strike": 24000, "oi": 50000},
            {"strike": 24200, "oi": 120000},
            {"strike": 24400, "oi": 80000},
        ],
        "PE": [
            {"strike": 23600, "oi": 60000},
            {"strike": 23800, "oi": 150000},
            {"strike": 24000, "oi": 40000},
        ],
    }
    result = get_oi_support_resistance(option_chain)
    assert result["support_strike"] == 23800
    assert result["resistance_strike"] == 24200
```

**Step 2: Run to verify failure**

```bash
python3 -m pytest tests/test_market_intel.py -v
```

**Step 3: Implement**

```python
# market_intel.py
"""V6 Market Intelligence: VWAP, OI delta, OI support/resistance."""
import logging

logger = logging.getLogger(__name__)


def compute_vwap_from_candles(candles: list[dict]) -> float | None:
    """Compute VWAP from intraday candles.
    Each candle: {high, low, close, volume}"""
    if not candles:
        return None
    cum_tp_vol = 0.0
    cum_vol = 0
    for c in candles:
        typical_price = (c["high"] + c["low"] + c["close"]) / 3
        vol = c.get("volume", 0)
        cum_tp_vol += typical_price * vol
        cum_vol += vol
    return cum_tp_vol / cum_vol if cum_vol > 0 else None


def classify_oi_change(oi_change_pct: float, price_change_pct: float) -> str:
    """Classify OI + price change into standard patterns."""
    oi_up = oi_change_pct > 0
    price_up = price_change_pct > 0
    if oi_up and price_up:
        return "long_buildup"
    elif oi_up and not price_up:
        return "short_buildup"
    elif not oi_up and price_up:
        return "short_covering"
    else:
        return "long_unwinding"


def get_oi_support_resistance(option_chain: dict) -> dict:
    """Find max put OI (support) and max call OI (resistance) strikes."""
    ce_data = option_chain.get("CE", [])
    pe_data = option_chain.get("PE", [])

    resistance_strike = max(ce_data, key=lambda x: x.get("oi", 0))["strike"] if ce_data else None
    support_strike = max(pe_data, key=lambda x: x.get("oi", 0))["strike"] if pe_data else None

    return {
        "support_strike": support_strike,
        "resistance_strike": resistance_strike,
        "max_put_oi": max((x.get("oi", 0) for x in pe_data), default=0),
        "max_call_oi": max((x.get("oi", 0) for x in ce_data), default=0),
    }


def fetch_vwap(smart_api, token: str) -> float | None:
    """Fetch today's 5-min candles and compute VWAP."""
    try:
        from datetime import datetime, timedelta, timezone
        IST = timezone(timedelta(hours=5, minutes=30))
        now = datetime.now(IST)
        today = now.strftime("%Y-%m-%d")
        from_time = f"{today} 09:15"
        to_time = now.strftime("%Y-%m-%d %H:%M")

        import time, config
        time.sleep(config.API_DELAY)
        candle_data = smart_api.getCandleData({
            "exchange": "NSE",
            "symboltoken": token,
            "interval": "FIVE_MINUTE",
            "fromdate": from_time,
            "todate": to_time,
        })

        if candle_data and candle_data.get("data"):
            candles = [
                {"high": c[2], "low": c[3], "close": c[4], "volume": c[5]}
                for c in candle_data["data"]
            ]
            return compute_vwap_from_candles(candles)
    except Exception as e:
        logger.debug("VWAP fetch failed: %s", e)
    return None


def format_market_intel_for_claude(vwap: float | None, ltp: float,
                                    oi_change: dict | None,
                                    oi_sr: dict | None) -> str:
    """Format all market intel signals for Claude prompt."""
    lines = []
    if vwap:
        diff_pct = (ltp - vwap) / vwap * 100
        bias = "above" if diff_pct > 0 else "below"
        lines.append(f"VWAP: ₹{vwap:,.1f} | LTP: ₹{ltp:,.1f} ({diff_pct:+.1f}% {bias}) — {'bullish' if diff_pct > 0 else 'bearish'} confirmation")

    if oi_change:
        pattern = classify_oi_change(oi_change.get("oi_change_pct", 0), oi_change.get("price_change_pct", 0))
        lines.append(f"OI Delta: {oi_change.get('oi_change_pct', 0):+.1f}% → {pattern.replace('_', ' ').title()}")

    if oi_sr:
        if oi_sr.get("support_strike"):
            lines.append(f"OI Support: {oi_sr['support_strike']} PE (max put OI)")
        if oi_sr.get("resistance_strike"):
            lines.append(f"OI Resistance: {oi_sr['resistance_strike']} CE (max call OI)")

    return "\n".join(lines) if lines else ""
```

**Step 4: Run tests**

```bash
python3 -m pytest tests/test_market_intel.py -v
```

**Step 5: Commit**

```bash
git add market_intel.py tests/test_market_intel.py
git commit -m "feat(v6): add VWAP computation, OI delta tracking, OI support/resistance"
```

---

## Task 12: Market Intelligence — Time & Events

**Files:**
- Modify: `market_intel.py` (add expiry detection, gap handling, no-trade zones, IV crush)
- Test: `tests/test_time_intel.py`

**Step 1: Write failing tests**

```python
# tests/test_time_intel.py
import os, sys
from datetime import date, time, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_is_expiry_day_thursday():
    from market_intel import is_expiry_day
    # A Thursday
    assert is_expiry_day(date(2026, 3, 12)) == "weekly"

def test_is_expiry_day_last_thursday():
    from market_intel import is_expiry_day
    assert is_expiry_day(date(2026, 3, 26)) == "monthly"

def test_is_expiry_day_not_thursday():
    from market_intel import is_expiry_day
    assert is_expiry_day(date(2026, 3, 11)) is None  # Wednesday

def test_expiry_shifts_on_holiday():
    from market_intel import is_expiry_day
    # If Thursday is a holiday, expiry shifts to Wednesday
    # March 10 2026 is Maha Shivaratri (holiday) — but it's a Tuesday
    # Test with a known pattern: if we pass holidays list
    assert is_expiry_day(date(2026, 3, 12), holidays=[date(2026, 3, 12)]) is None  # Holiday
    assert is_expiry_day(date(2026, 3, 11), holidays=[date(2026, 3, 12)]) == "weekly"  # Shifted to Wed

def test_get_no_trade_zone():
    from market_intel import get_no_trade_zone
    # 9:20 AM — opening auction
    assert get_no_trade_zone(time(9, 20)) == "opening_auction"
    # 1:30 PM — post lunch
    assert get_no_trade_zone(time(13, 30)) == "post_lunch"
    # 3:20 PM — closing auction
    assert get_no_trade_zone(time(15, 20)) == "closing_auction"
    # 11:00 AM — normal trading
    assert get_no_trade_zone(time(11, 0)) is None

def test_compute_gap():
    from market_intel import compute_gap
    result = compute_gap(prev_close=23800, today_open=24100)
    assert result["gap_pct"] > 1.0
    assert result["direction"] == "up"
    result2 = compute_gap(prev_close=23800, today_open=23500)
    assert result2["direction"] == "down"

def test_is_iv_elevated():
    from market_intel import is_iv_elevated
    assert is_iv_elevated(current_vix=22.5, avg_20d_vix=15.0) is True
    assert is_iv_elevated(current_vix=15.5, avg_20d_vix=15.0) is False
```

**Step 2: Run to verify failure**

```bash
python3 -m pytest tests/test_time_intel.py -v
```

**Step 3: Implement**

Add to `market_intel.py`:

```python
from datetime import date, time, timedelta
import calendar


def is_expiry_day(d: date, holidays: list[date] | None = None) -> str | None:
    """Return 'weekly', 'monthly', or None. Handles holiday shifts."""
    holidays = holidays or []

    # Find last Thursday of this month
    last_day = calendar.monthrange(d.year, d.month)[1]
    last_thurs = date(d.year, d.month, last_day)
    while last_thurs.weekday() != 3:  # Thursday
        last_thurs -= timedelta(days=1)

    # Check if this Thursday is a holiday → shift to Wednesday
    def _effective_expiry(thurs: date) -> date:
        if thurs in holidays:
            return thurs - timedelta(days=1)
        return thurs

    effective_monthly = _effective_expiry(last_thurs)

    # All Thursdays this month (weekly expiry)
    day = date(d.year, d.month, 1)
    thursdays = []
    while day.month == d.month:
        if day.weekday() == 3:
            thursdays.append(day)
        day += timedelta(days=1)

    for thurs in thursdays:
        effective = _effective_expiry(thurs)
        if d == effective:
            return "monthly" if thurs == last_thurs else "weekly"

    return None


def get_no_trade_zone(t: time) -> str | None:
    """Check if current time is in a no-trade zone."""
    if time(9, 15) <= t <= time(9, 30):
        return "opening_auction"
    if time(13, 0) <= t <= time(14, 0):
        return "post_lunch"
    if time(15, 15) <= t <= time(15, 30):
        return "closing_auction"
    return None


def compute_gap(prev_close: float, today_open: float) -> dict:
    """Compute opening gap % and direction."""
    gap_pct = (today_open - prev_close) / prev_close * 100
    return {
        "gap_pct": round(abs(gap_pct), 2),
        "direction": "up" if gap_pct > 0 else "down",
        "signed_pct": round(gap_pct, 2),
    }


def is_iv_elevated(current_vix: float, avg_20d_vix: float) -> bool:
    """Check if IV is elevated vs recent average (>20% above)."""
    if avg_20d_vix <= 0:
        return False
    return current_vix > avg_20d_vix * 1.2


def format_time_context_for_claude(current_time: time, d: date,
                                     gap: dict | None = None,
                                     vix: float | None = None,
                                     avg_vix: float | None = None,
                                     holidays: list[date] | None = None) -> str:
    """Format time-based context for Claude prompt."""
    lines = []

    # Expiry
    expiry = is_expiry_day(d, holidays)
    if expiry:
        lines.append(f"⚠️ Today is {expiry} expiry. Gamma elevated, tighter SLs recommended.")

    # No-trade zone
    zone = get_no_trade_zone(current_time)
    if zone:
        labels = {
            "opening_auction": "Opening auction (9:15-9:30) — prices settling, avoid new entries",
            "post_lunch": "Post-lunch low volume (1:00-2:00) — reduce conviction on new signals",
            "closing_auction": "Closing auction (3:15-3:30) — no new entries",
        }
        lines.append(labels.get(zone, zone))

    # Expiry afternoon cutoff
    if expiry and current_time >= time(14, 0):
        lines.append("⛔ Expiry day after 2 PM — no new option entries (theta/gamma risk)")

    # Gap
    if gap and abs(gap.get("signed_pct", 0)) >= 1.0:
        lines.append(f"Gap {gap['direction']} {gap['gap_pct']}% — wait 15-30 min for settlement")

    # IV
    if vix and avg_vix and is_iv_elevated(vix, avg_vix):
        lines.append(f"IV elevated: VIX {vix:.1f} vs 20-day avg {avg_vix:.1f} — avoid buying options pre-event")

    return "\n".join(lines) if lines else ""
```

**Step 4: Run tests**

```bash
python3 -m pytest tests/test_time_intel.py -v
```

**Step 5: Commit**

```bash
git add market_intel.py tests/test_time_intel.py
git commit -m "feat(v6): add expiry detection, gap handling, no-trade zones, IV crush awareness"
```

---

## Task 13: Position Adding/Scaling

**Files:**
- Modify: `paper_trade.py` (handle ADD action from Claude)
- Modify: `claude_intel.py` (ADD in exit response)
- Test: `tests/test_position_adding.py`

**Step 1: Write failing test**

```python
# tests/test_position_adding.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_add_action_increases_position():
    """ADD action increases position quantity within safety limits."""
    from paper_trade import _handle_position_add

    pos = {
        "symbol": "MCX", "direction": "bullish", "instrument": "EQ",
        "entry_price": 100, "quantity": 10, "allocated": 1000,
        "atr_at_entry": 3.0, "_add_count": 0,
    }
    portfolio = {"available_capital": 50000, "capital": 100000, "positions": [pos]}

    result = _handle_position_add(pos, current_ltp=106, portfolio=portfolio)
    assert result is True
    assert pos["quantity"] > 10
    assert pos["_add_count"] == 1

def test_add_blocked_if_losing():
    """ADD blocked if position is in loss."""
    from paper_trade import _handle_position_add

    pos = {
        "symbol": "MCX", "direction": "bullish", "instrument": "EQ",
        "entry_price": 100, "quantity": 10, "allocated": 1000,
        "atr_at_entry": 3.0, "_add_count": 0,
    }
    portfolio = {"available_capital": 50000, "capital": 100000, "positions": [pos]}

    result = _handle_position_add(pos, current_ltp=98, portfolio=portfolio)
    assert result is False
    assert pos["quantity"] == 10

def test_add_blocked_if_already_added():
    """Maximum 1 add per position."""
    from paper_trade import _handle_position_add

    pos = {
        "symbol": "MCX", "direction": "bullish", "instrument": "EQ",
        "entry_price": 100, "quantity": 15, "allocated": 1500,
        "atr_at_entry": 3.0, "_add_count": 1,
    }
    portfolio = {"available_capital": 50000, "capital": 100000, "positions": [pos]}

    result = _handle_position_add(pos, current_ltp=108, portfolio=portfolio)
    assert result is False
```

**Step 2: Run to verify failure**

```bash
python3 -m pytest tests/test_position_adding.py -v
```

**Step 3: Implement**

In `paper_trade.py`:

```python
def _handle_position_add(pos: dict, current_ltp: float, portfolio: dict) -> bool:
    """Add to a winning position. Returns True if add executed."""
    # Safety: never add to losers
    entry = pos["entry_price"]
    direction = pos.get("direction", "bullish")
    if direction == "bullish" and current_ltp <= entry:
        return False
    if direction == "bearish" and current_ltp >= entry:
        return False

    # Max 1 add per position
    if pos.get("_add_count", 0) >= 1:
        return False

    # Must be at least 1x ATR from entry
    atr = pos.get("atr_at_entry", 0)
    if atr > 0:
        if direction == "bullish" and (current_ltp - entry) < atr:
            return False
        if direction == "bearish" and (entry - current_ltp) < atr:
            return False

    # Add 50% of original quantity
    add_qty = max(1, pos["quantity"] // 2)
    add_cost = add_qty * current_ltp

    if portfolio.get("available_capital", 0) < add_cost:
        return False

    # Safety cap
    import config
    max_alloc = portfolio["capital"] * config.MAX_STRATEGY_ALLOC_PCT
    if (pos["allocated"] + add_cost) > max_alloc:
        return False

    # Execute add
    pos["quantity"] += add_qty
    pos["allocated"] = pos.get("allocated", 0) + add_cost
    pos["_add_count"] = pos.get("_add_count", 0) + 1
    portfolio["available_capital"] -= add_cost

    return True
```

In `monitor_positions()` V6 path, after Claude returns ADD:

```python
if parsed_action == "ADD":
    added = _handle_position_add(pos, ltp, portfolio)
    if added:
        _telegram_send(f"➕ Added to {pos['symbol']} at ₹{ltp:,.1f}")
```

**Step 4: Run tests**

```bash
python3 -m pytest tests/test_position_adding.py -v
```

**Step 5: Commit**

```bash
git add paper_trade.py tests/test_position_adding.py
git commit -m "feat(v6): add position adding/scaling (ADD action from Claude)"
```

---

## Task 14: Daily Journal → Obsidian

**Files:**
- Modify: `smart_monitor.py` (new `write_daily_journal()`, `read_recent_journal()`)
- Modify: `scripts/paper_trade_cron.sh` (call journal writer in EOD)
- Test: `tests/test_obsidian_journal.py`

**Step 1: Write failing test**

```python
# tests/test_obsidian_journal.py
import os, sys, json
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_format_daily_journal():
    from smart_monitor import format_daily_journal

    portfolio = {
        "positions": [
            {"symbol": "MCX", "direction": "bullish", "instrument": "EQ",
             "entry_price": 2557, "entry_date": "2026-03-09", "status": "open",
             "entry_thesis": {"reasoning": "Strong OI buildup with volume breakout"},
             "_last_ltp": 2590, "allocated": 5000},
        ],
        "closed_trades": [
            {"symbol": "BPCL", "direction": "bearish", "instrument": "PE",
             "entry_price": 42.50, "exit_price": 38.00, "exit_date": "2026-03-09",
             "pnl": -380, "pnl_pct": -3.2, "exit_reason": "stoploss"},
        ],
        "stats": {"total_pnl": -380},
    }
    context = {"regime": "TRENDING_UP", "vix": 16.2, "fii_net": -2300, "pcr": 0.85}

    md = format_daily_journal(portfolio, context, "2026-03-09")
    assert "## Trading Journal" in md
    assert "MCX" in md
    assert "BPCL" in md
    assert "TRENDING_UP" in md

def test_read_recent_journal(tmp_path):
    from smart_monitor import read_recent_journal

    # Create fake daily notes
    daily = tmp_path / "daily"
    daily.mkdir()
    (daily / "2026-03-07.md").write_text("# March 7\n## Trading Journal\nBought MCX at 2500\n")
    (daily / "2026-03-08.md").write_text("# March 8\n## Trading Journal\nSold BPCL at loss\n")
    (daily / "2026-03-09.md").write_text("# March 9\nNo trading section\n")

    result = read_recent_journal(vault_dir=tmp_path, days=3, today="2026-03-09")
    assert len(result) >= 1
    assert "MCX" in result[0] or "BPCL" in result[0]  # At least one journal found
```

**Step 2: Run to verify failure**

```bash
python3 -m pytest tests/test_obsidian_journal.py -v
```

**Step 3: Implement**

In `smart_monitor.py`:

```python
VAULT_DIR = Path.home() / "Documents" / "Obsidian"

def format_daily_journal(portfolio: dict, context: dict, today: str) -> str:
    """Format trading journal entry for Obsidian daily note."""
    lines = ["\n## Trading Journal\n"]

    # Entries (today's new positions)
    entries = [p for p in portfolio.get("positions", [])
               if p.get("entry_date", "")[:10] == today and p.get("status") == "open"]
    if entries:
        lines.append("### Entries")
        for p in entries:
            thesis = p.get("entry_thesis", {}).get("reasoning", "N/A")
            lines.append(f"- **{p['symbol']}** ({p.get('instrument','')}, {p.get('direction','')}) @ ₹{p['entry_price']:,.1f} — \"{thesis}\"")
        lines.append("")

    # Exits (today's closed trades)
    exits = [t for t in portfolio.get("closed_trades", [])
             if t.get("exit_date", "")[:10] == today]
    if exits:
        lines.append("### Exits")
        for t in exits:
            lines.append(f"- **{t['symbol']}** ({t.get('instrument','')}) @ ₹{t.get('exit_price',0):,.1f} — {t.get('exit_reason','')}, P&L: ₹{t.get('pnl',0):,.0f} ({t.get('pnl_pct',0):+.1f}%)")
        lines.append("")

    # Portfolio summary
    open_pos = [p for p in portfolio.get("positions", []) if p.get("status") == "open"]
    total_realized = sum(t.get("pnl", 0) for t in exits)
    lines.append("### Portfolio")
    lines.append(f"- Realized today: ₹{total_realized:,.0f} | Open: {len(open_pos)} positions")
    lines.append("")

    # Market context
    if context:
        lines.append("### Market Context")
        parts = []
        if context.get("regime"): parts.append(f"Regime: {context['regime']}")
        if context.get("vix"): parts.append(f"VIX: {context['vix']:.1f}")
        if context.get("fii_net"): parts.append(f"FII: {context['fii_net']:,.0f}cr")
        if context.get("pcr"): parts.append(f"PCR: {context['pcr']:.2f}")
        lines.append(f"- {' | '.join(parts)}")

    return "\n".join(lines)


def write_daily_journal(portfolio: dict, context: dict, today: str = None) -> str | None:
    """Append trading journal to Obsidian daily note."""
    if today is None:
        today = _now_ist().strftime("%Y-%m-%d")

    journal_md = format_daily_journal(portfolio, context, today)

    daily_file = VAULT_DIR / "daily" / f"{today}.md"
    try:
        if daily_file.exists():
            content = daily_file.read_text()
            if "## Trading Journal" not in content:
                # Append journal section
                with open(daily_file, "a") as f:
                    f.write(journal_md)
        else:
            # Create daily note with journal
            daily_file.parent.mkdir(parents=True, exist_ok=True)
            daily_file.write_text(f"# {today}\n{journal_md}")
        return str(daily_file)
    except Exception as e:
        logger.error("Failed to write daily journal: %s", e)
        return None


def read_recent_journal(vault_dir: Path = None, days: int = 3, today: str = None) -> list[str]:
    """Read recent trading journal entries from Obsidian daily notes."""
    vault = vault_dir or VAULT_DIR
    if today is None:
        today = _now_ist().strftime("%Y-%m-%d")

    from datetime import datetime, timedelta
    journals = []
    d = datetime.strptime(today, "%Y-%m-%d").date()

    for i in range(1, days + 1):
        check_date = d - timedelta(days=i)
        daily_file = vault / "daily" / f"{check_date}.md"
        if daily_file.exists():
            content = daily_file.read_text()
            if "## Trading Journal" in content:
                # Extract journal section
                idx = content.index("## Trading Journal")
                # Find next ## or end of file
                rest = content[idx:]
                next_section = rest.find("\n## ", 3)
                if next_section > 0:
                    journals.append(rest[:next_section])
                else:
                    journals.append(rest)
    return journals
```

Update `scripts/paper_trade_cron.sh` EOD section to call journal writer:

```bash
# In EOD wrap section (after the existing EOD logic):
python3 -c "
from paper_trade import load_portfolio
from smart_monitor import write_daily_journal
portfolio = load_portfolio()
context = {}  # Context already captured during the day
path = write_daily_journal(portfolio, context)
if path:
    print(f'Journal written: {path}')
" >> "$LOG" 2>&1
```

**Step 4: Run tests**

```bash
python3 -m pytest tests/test_obsidian_journal.py -v
```

**Step 5: Commit**

```bash
git add smart_monitor.py scripts/paper_trade_cron.sh tests/test_obsidian_journal.py
git commit -m "feat(v6): add daily trading journal to Obsidian vault"
```

---

## Task 15: V5 → V6 Migration Script

**Files:**
- Create: `scripts/migrate_v5_to_v6.py`
- Test: `tests/test_migration.py`

**Step 1: Write failing test**

```python
# tests/test_migration.py
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

def test_migrate_adds_entry_thesis(tmp_path):
    from migrate_v5_to_v6 import migrate_portfolio

    old_portfolio = {
        "capital": 100000,
        "available_capital": 85000,
        "positions": [
            {"symbol": "MCX", "status": "open", "entry_price": 2557,
             "entry_date": "2026-03-07", "direction": "bullish", "instrument": "EQ"},
        ],
        "closed_trades": [],
        "stats": {"total_pnl": 0},
    }

    result = migrate_portfolio(old_portfolio)
    pos = result["positions"][0]
    assert "entry_thesis" in pos
    assert pos["entry_thesis"]["conviction"] == "medium"
    assert result.get("schema_version") == 6

def test_migrate_preserves_existing_thesis(tmp_path):
    from migrate_v5_to_v6 import migrate_portfolio

    old_portfolio = {
        "capital": 100000,
        "available_capital": 85000,
        "positions": [
            {"symbol": "MCX", "status": "open", "entry_price": 2557,
             "entry_thesis": {"reasoning": "Original thesis", "conviction": "high",
                              "key_conditions": {}, "invalidation": "", "expected_hold": "", "target_scenario": ""}},
        ],
        "closed_trades": [],
        "stats": {"total_pnl": 0},
    }

    result = migrate_portfolio(old_portfolio)
    assert result["positions"][0]["entry_thesis"]["reasoning"] == "Original thesis"

def test_migrate_state(tmp_path):
    from migrate_v5_to_v6 import migrate_state

    old_state = {
        "last_check": {"MCX": "2026-03-09T12:00:00"},
        "circuit_breaker_active": False,
    }

    result = migrate_state(old_state)
    assert "position_assessments" in result
    assert "claude_consecutive_failures" in result
    assert result.get("schema_version") == 6
```

**Step 2: Run to verify failure**

```bash
python3 -m pytest tests/test_migration.py -v
```

**Step 3: Implement**

```python
# scripts/migrate_v5_to_v6.py
"""Migrate portfolio and state files from V5 to V6 schema."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def migrate_portfolio(portfolio: dict) -> dict:
    """Add V6 fields to existing portfolio."""
    for pos in portfolio.get("positions", []):
        if "entry_thesis" not in pos:
            pos["entry_thesis"] = {
                "reasoning": f"Pre-V6 position opened on {pos.get('entry_date', 'unknown')}",
                "conviction": "medium",
                "key_conditions": {},
                "invalidation": "Manual review required",
                "expected_hold": "unknown",
                "target_scenario": "unknown",
            }
        pos.setdefault("_add_count", 0)

    portfolio["schema_version"] = 6
    return portfolio


def migrate_state(state: dict) -> dict:
    """Add V6 fields to existing monitor state."""
    state.setdefault("position_assessments", {})
    state.setdefault("claude_consecutive_failures", 0)
    state.setdefault("claude_lockdown_active", False)
    state["schema_version"] = 6
    return state


def main():
    from paper_trade import load_portfolio, save_portfolio, PORTFOLIO_FILE
    from smart_monitor import _load_state, _save_state, STATE_FILE

    print(f"Migrating portfolio: {PORTFOLIO_FILE}")
    portfolio = load_portfolio()
    portfolio = migrate_portfolio(portfolio)
    save_portfolio(portfolio)
    print(f"  {len(portfolio.get('positions', []))} positions updated")

    print(f"Migrating state: {STATE_FILE}")
    state = _load_state()
    state = migrate_state(state)
    _save_state(state)
    print(f"  State migrated to schema v6")

    print("Migration complete.")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests**

```bash
python3 -m pytest tests/test_migration.py -v
```

**Step 5: Commit**

```bash
git add scripts/migrate_v5_to_v6.py tests/test_migration.py
git commit -m "feat(v6): add V5-to-V6 migration script for portfolio and state files"
```

---

## Task 16: Integration — Wire Market Intel into Claude Prompts

**Files:**
- Modify: `claude_intel.py` (`evaluate_entry`, `evaluate_exit` — add market intel context)
- Modify: `paper_trade.py` (`open_positions`, `monitor_positions` — fetch and pass intel)

**Step 1: Implement wiring**

In `paper_trade.py` `open_positions()`, after fetching Nifty LTP and before Claude evaluation, when V6 is enabled:

```python
if config.V6_CLAUDE_FIRST:
    # Fetch VWAP for Nifty
    from market_intel import fetch_vwap, format_market_intel_for_claude, format_time_context_for_claude
    from market_intel import compute_gap, get_oi_support_resistance
    from datetime import datetime, timezone, timedelta

    IST = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(IST)

    nifty_vwap = fetch_vwap(smart_api, NIFTY_TOKEN)

    # Gap detection (compare today open to yesterday close)
    gap_info = None
    if nifty_candles and len(nifty_candles) >= 2:
        prev_close = nifty_candles[-2][4]  # yesterday's close
        today_open = nifty_candles[-1][1]  # today's open
        gap_info = compute_gap(prev_close, today_open)

    # OI support/resistance from option chain (already fetched for PCR)
    oi_sr = None
    if option_chain_data:
        oi_sr = get_oi_support_resistance(option_chain_data)

    # Time context
    time_context = format_time_context_for_claude(
        now.time(), now.date(), gap=gap_info, vix=vix_ltp,
        avg_vix=None,  # TODO: compute 20-day avg VIX
    )

    # Market intel context
    intel_context = format_market_intel_for_claude(nifty_vwap, nifty_ltp, None, oi_sr)

    extra_context = f"{intel_context}\n{time_context}".strip()
```

Pass `extra_context` to `evaluate_candidates()` via a new parameter, or append to the prompt.

In `monitor_positions()` V6 path, before Claude exit eval:

```python
# Fetch VWAP for position's underlying
vwap = fetch_vwap(smart_api, pos_token) if pos.get("instrument") == "EQ" else None

# OI change for position
oi_at_entry = pos.get("oi_at_entry")
if oi_at_entry:
    current_oi = ...  # fetch from option chain
    oi_change = {"oi_change_pct": (current_oi - oi_at_entry) / oi_at_entry * 100, "price_change_pct": pnl_pct}
else:
    oi_change = None

intel_text = format_market_intel_for_claude(vwap, ltp, oi_change, None)
```

**Step 2: Smoke test on Pi**

```bash
ssh pi@homepi.local
cd ~/financial-agent-india
source venv/bin/activate
python3 -c "
from paper_trade import load_portfolio
from smart_monitor import _load_state
p = load_portfolio()
s = _load_state()
print(f'Schema: {p.get(\"schema_version\", \"v5\")}')
print(f'Positions: {len([pos for pos in p.get(\"positions\", []) if pos.get(\"status\") == \"open\"])}')
print(f'State keys: {list(s.keys())}')
"
```

**Step 3: Commit**

```bash
git add paper_trade.py claude_intel.py
git commit -m "feat(v6): wire market intel (VWAP, OI, time context) into Claude prompts"
```

---

## Task 17: Final Integration Test & Deploy

**Step 1: Run all tests**

```bash
python3 -m pytest tests/ -v --timeout=30 2>&1 | tail -20
```

All tests must pass.

**Step 2: Compile check**

```bash
python3 -m py_compile paper_trade.py
python3 -m py_compile claude_intel.py
python3 -m py_compile smart_monitor.py
python3 -m py_compile config.py
python3 -m py_compile market_intel.py
python3 -m py_compile scripts/migrate_v5_to_v6.py
```

**Step 3: Run migration on Pi**

```bash
scp paper_trade.py claude_intel.py smart_monitor.py config.py market_intel.py pi@homepi.local:~/financial-agent-india/
scp scripts/paper_trade_cron.sh pi@homepi.local:~/financial-agent-india/scripts/
scp scripts/migrate_v5_to_v6.py pi@homepi.local:~/financial-agent-india/scripts/

ssh pi@homepi.local "cd ~/financial-agent-india && source venv/bin/activate && python3 scripts/migrate_v5_to_v6.py"
```

**Step 4: Smoke test on Pi**

```bash
ssh pi@homepi.local "cd ~/financial-agent-india && source venv/bin/activate && python3 paper_trade.py monitor"
```

Should run without errors, log timing, and exit cleanly.

**Step 5: Verify cron**

```bash
ssh pi@homepi.local "tail -20 ~/financial-agent-india/data/paper_trades/cron.log"
```

Look for: `[TIMING]` entries, no `[SKIP] Previous tick still running`, clean monitor output.

**Step 6: Commit & tag**

```bash
git add -A
git commit -m "feat(v6): complete Claude-First Architecture — 17 tasks implemented"
git tag v6.0.0
```

---

## Dependency Graph

```
Task 1 (portfolio safety) ──┐
Task 2 (cron hardening)  ───┤
Task 3 (telegram/validation)┤
                             ├──→ Task 4 (kill switch + config) ──→ Task 9 (entry restructure)
                             │                                  ──→ Task 10 (exit restructure)
Task 5 (decision replay) ───┤
Task 6 (system prompt) ─────┤
Task 7 (thesis + memory) ───┤──→ Task 10 (exit restructure)
Task 8 (fail-safe) ─────────┘
Task 11 (VWAP/OI) ─────────────→ Task 16 (wire into prompts)
Task 12 (time/events) ─────────→ Task 16
Task 13 (position adding) ─────→ Task 16
Task 14 (obsidian journal) ───┐
Task 15 (migration) ──────────┤──→ Task 17 (integration test + deploy)
Task 16 (wire intel) ─────────┘
```

Tasks 1-8 and 11-14 can run in parallel. Tasks 9, 10, 16 depend on earlier tasks. Task 17 is the final integration.
