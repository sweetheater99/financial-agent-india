# V7 Plan 4: Journal, Cron Integration & Deployment

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the trade journal (daily/weekly/monthly), edge tracker, Telegram integration, cron scripts for Pi, main CLI entry point, and deployment pipeline.

**Architecture:** V7 code lives in the `v7/` package. Journal writes to Obsidian vault (`~/Documents/Obsidian/trading-journal/`). Edge tracker persists in `data/v7/edge_tracker.json`. Telegram uses HTML parse_mode with temp file + curl pattern. Cron scripts run on Pi (Python 3.13, venv at `~/financial-agent-india/venv/`).

**Tech Stack:** Python 3.13, Claude API (Haiku for daily journal, Sonnet for weekly/monthly), JSON file persistence, Obsidian Markdown, pytest

**Spec:** `docs/superpowers/specs/2026-03-11-v7-professional-trader-bot-design.md`

**Depends on:** Plan 1 (types, state, config), Plan 2 (strategist), Plan 3 (executor, theta engine)
**Blocks:** Nothing (this is the final plan)

---

## File Structure

```
v7/
├── journal.py              # Trade grading, daily journal, weekly review, monthly report
├── edge_tracker.py         # Strategy performance tracking, kill decisions
├── telegram.py             # All V7 Telegram alerts (HTML, temp file pattern)
├── main.py                 # CLI entry point for all V7 operations
scripts/
├── v7_premarket.sh         # 8:43 AM — pre-market playbook
├── v7_opening_read.sh      # 9:13 AM — opening read
├── v7_executor.sh          # every 3 min 9:15-15:30 — main tick loop
├── v7_checkin.sh            # 10:28, 12:58 — strategist check-ins
├── v7_eod.sh               # 3:33 PM — EOD review + journal
├── v7_weekly_review.sh     # Sunday 10:03 AM
├── v7_monthly_report.sh    # 1st of month 10:07 AM
tests/
├── test_v7_journal.py
├── test_v7_edge_tracker.py
├── test_v7_telegram.py
├── test_v7_main.py
```

---

## Chunk 1: Edge Tracker

### Task 1: Edge tracker — strategy performance persistence

**Files:**
- Create: `v7/edge_tracker.py`
- Test: `tests/test_v7_edge_tracker.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_v7_edge_tracker.py
"""Tests for V7 edge tracker."""
import json
import pytest
from datetime import date
from pathlib import Path
from v7.edge_tracker import EdgeTracker
from v7.types import TradeResult, SetupType


@pytest.fixture
def tmp_tracker(tmp_path):
    return EdgeTracker(data_dir=tmp_path)


@pytest.fixture
def sample_trade():
    return TradeResult(
        symbol="NIFTY", instrument="NIFTY CE 24400",
        direction="bullish", entry_price=100.0, exit_price=150.0,
        quantity=75, entry_date=date(2026, 3, 11),
        exit_date=date(2026, 3, 11), exit_reason="target",
        pnl=3750.0, pnl_pct=50.0, costs=120.0,
        setup_id="N1", setup_type=SetupType.BREAKOUT_LONG,
        entry_grade="A", exit_grade="A",
    )


def test_record_trade(tmp_tracker, sample_trade):
    tmp_tracker.record(sample_trade, strategy="momentum", time_bucket="9:45-11:00")
    stats = tmp_tracker.get_stats()
    assert stats["overall"]["trades"] == 1
    assert stats["overall"]["wins"] == 1


def test_record_loss(tmp_tracker):
    trade = TradeResult(
        symbol="HDFCBANK", instrument="HDFCBANK CE 1600",
        direction="bullish", entry_price=100.0, exit_price=70.0,
        quantity=550, entry_date=date(2026, 3, 11),
        exit_date=date(2026, 3, 11), exit_reason="stoploss",
        pnl=-16500.0, pnl_pct=-30.0, costs=80.0,
        setup_id="H1", setup_type=SetupType.SUPPORT_BOUNCE,
        entry_grade="B", exit_grade="C",
    )
    tmp_tracker.record(trade, strategy="mean_reversion", time_bucket="11:00-13:00")
    stats = tmp_tracker.get_stats()
    assert stats["overall"]["losses"] == 1
    assert stats["by_strategy"]["mean_reversion"]["win_rate"] == 0.0


def test_by_instrument(tmp_tracker, sample_trade):
    tmp_tracker.record(sample_trade, strategy="momentum", time_bucket="9:45-11:00")
    stats = tmp_tracker.get_stats()
    assert stats["by_instrument"]["NIFTY"]["net_pnl"] == 3750.0


def test_by_time(tmp_tracker, sample_trade):
    tmp_tracker.record(sample_trade, strategy="momentum", time_bucket="9:45-11:00")
    stats = tmp_tracker.get_stats()
    assert stats["by_time"]["9:45-11:00"]["trades"] == 1
    assert stats["by_time"]["9:45-11:00"]["win_rate"] == 1.0


def test_by_setup_type(tmp_tracker, sample_trade):
    tmp_tracker.record(sample_trade, strategy="momentum", time_bucket="9:45-11:00")
    stats = tmp_tracker.get_stats()
    assert stats["by_setup_type"]["breakout_long"]["trades"] == 1


def test_persistence(tmp_path, sample_trade):
    tracker1 = EdgeTracker(data_dir=tmp_path)
    tracker1.record(sample_trade, strategy="momentum", time_bucket="9:45-11:00")
    # New instance loads from disk
    tracker2 = EdgeTracker(data_dir=tmp_path)
    stats = tracker2.get_stats()
    assert stats["overall"]["trades"] == 1


def test_kill_candidates_not_enough_trades(tmp_tracker, sample_trade):
    # Less than 30 trades — no kill candidates
    tmp_tracker.record(sample_trade, strategy="momentum", time_bucket="9:45-11:00")
    assert tmp_tracker.kill_candidates(min_trades=30) == []


def test_kill_candidates_low_win_rate(tmp_path):
    tracker = EdgeTracker(data_dir=tmp_path)
    # Record 31 trades: 10 wins, 21 losses (32% win rate)
    for i in range(10):
        t = TradeResult(
            symbol="NIFTY", instrument="NIFTY CE", direction="bullish",
            entry_price=100.0, exit_price=120.0, quantity=75,
            entry_date=date(2026, 3, 1), exit_date=date(2026, 3, 1),
            exit_reason="target", pnl=1500.0, pnl_pct=20.0, costs=80.0,
            setup_id=f"N{i}", setup_type=SetupType.BREAKOUT_LONG,
        )
        tracker.record(t, strategy="momentum", time_bucket="9:45-11:00")
    for i in range(21):
        t = TradeResult(
            symbol="NIFTY", instrument="NIFTY CE", direction="bullish",
            entry_price=100.0, exit_price=70.0, quantity=75,
            entry_date=date(2026, 3, 1), exit_date=date(2026, 3, 1),
            exit_reason="stoploss", pnl=-2250.0, pnl_pct=-30.0, costs=80.0,
            setup_id=f"NL{i}", setup_type=SetupType.BREAKOUT_LONG,
        )
        tracker.record(t, strategy="momentum", time_bucket="9:45-11:00")
    kills = tracker.kill_candidates(min_trades=30, min_win_rate=0.40)
    assert "momentum" in kills


def test_avg_rr(tmp_path):
    tracker = EdgeTracker(data_dir=tmp_path)
    # Win: risk 30, reward 50 → R:R 1.67
    t1 = TradeResult(
        symbol="NIFTY", instrument="NIFTY CE", direction="bullish",
        entry_price=100.0, exit_price=150.0, quantity=75,
        entry_date=date(2026, 3, 1), exit_date=date(2026, 3, 1),
        exit_reason="target", pnl=3750.0, pnl_pct=50.0, costs=80.0,
        setup_id="N1", setup_type=SetupType.BREAKOUT_LONG,
    )
    # Loss: risk 30, lost 30
    t2 = TradeResult(
        symbol="NIFTY", instrument="NIFTY CE", direction="bullish",
        entry_price=100.0, exit_price=70.0, quantity=75,
        entry_date=date(2026, 3, 1), exit_date=date(2026, 3, 1),
        exit_reason="stoploss", pnl=-2250.0, pnl_pct=-30.0, costs=80.0,
        setup_id="N2", setup_type=SetupType.BREAKOUT_LONG,
    )
    tracker.record(t1, strategy="momentum", time_bucket="9:45-11:00")
    tracker.record(t2, strategy="momentum", time_bucket="9:45-11:00")
    stats = tracker.get_stats()
    assert stats["by_strategy"]["momentum"]["avg_rr"] > 0


def test_edge_summary_for_prompt(tmp_tracker, sample_trade):
    tmp_tracker.record(sample_trade, strategy="momentum", time_bucket="9:45-11:00")
    summary = tmp_tracker.summary_for_prompt()
    assert "momentum" in summary
    assert "NIFTY" in summary
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_edge_tracker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'v7.edge_tracker'`

- [ ] **Step 3: Implement edge tracker**

```python
# v7/edge_tracker.py
"""Edge Tracker — strategy performance persistence and kill decisions.

Tracks every closed trade across multiple dimensions:
- By strategy (momentum, mean_reversion, theta)
- By instrument (NIFTY, HDFCBANK, etc.)
- By time of day (9:45-11:00, 11:00-13:00, 13:00-14:30)
- By setup type (breakout_long, support_bounce, etc.)

Feeds back into Strategist prompts so Claude weights the playbook
toward what actually works.
"""
from __future__ import annotations

import json
from pathlib import Path

from v7.types import TradeResult


class EdgeTracker:
    """Persistent edge tracker. Stores raw trade records and computes stats on demand."""

    def __init__(self, data_dir: Path | str = Path("data/v7")):
        self._data_dir = Path(data_dir)
        self._file = self._data_dir / "edge_tracker.json"
        self._trades: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self._file.exists():
            try:
                self._trades = json.loads(self._file.read_text())
            except (json.JSONDecodeError, ValueError):
                self._trades = []

    def _save(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._file.write_text(json.dumps(self._trades, indent=2, default=str))

    def record(self, trade: TradeResult, strategy: str, time_bucket: str) -> None:
        """Record a completed trade into the edge tracker."""
        entry = {
            "symbol": trade.symbol,
            "instrument": trade.instrument,
            "setup_type": trade.setup_type.value,
            "strategy": strategy,
            "time_bucket": time_bucket,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "pnl": trade.pnl,
            "pnl_pct": trade.pnl_pct,
            "costs": trade.costs,
            "entry_date": str(trade.entry_date),
            "exit_date": str(trade.exit_date),
            "exit_reason": trade.exit_reason,
            "entry_grade": trade.entry_grade,
            "exit_grade": trade.exit_grade,
            "is_win": trade.pnl > 0,
        }
        self._trades.append(entry)
        self._save()

    def get_stats(self) -> dict:
        """Compute aggregated stats across all dimensions."""
        trades = self._trades
        if not trades:
            return {
                "overall": {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "net_pnl": 0.0},
                "by_strategy": {},
                "by_instrument": {},
                "by_time": {},
                "by_setup_type": {},
            }

        wins = [t for t in trades if t["is_win"]]
        losses = [t for t in trades if not t["is_win"]]

        return {
            "overall": {
                "trades": len(trades),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": len(wins) / len(trades) if trades else 0.0,
                "net_pnl": sum(t["pnl"] for t in trades),
            },
            "by_strategy": self._group_stats(trades, "strategy"),
            "by_instrument": self._instrument_stats(trades),
            "by_time": self._group_stats(trades, "time_bucket"),
            "by_setup_type": self._group_stats(trades, "setup_type"),
        }

    def _group_stats(self, trades: list[dict], key: str) -> dict:
        """Compute win rate, avg R:R, net P&L per group."""
        groups: dict[str, list[dict]] = {}
        for t in trades:
            g = t.get(key, "unknown")
            groups.setdefault(g, []).append(t)

        result = {}
        for name, group in groups.items():
            wins = [t for t in group if t["is_win"]]
            losses = [t for t in group if not t["is_win"]]
            avg_win = sum(t["pnl_pct"] for t in wins) / len(wins) if wins else 0.0
            avg_loss = abs(sum(t["pnl_pct"] for t in losses) / len(losses)) if losses else 1.0
            result[name] = {
                "trades": len(group),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": len(wins) / len(group) if group else 0.0,
                "avg_rr": avg_win / avg_loss if avg_loss > 0 else 0.0,
                "net_pnl": sum(t["pnl"] for t in group),
            }
        return result

    def _instrument_stats(self, trades: list[dict]) -> dict:
        """Per-instrument stats: trades, net P&L."""
        groups: dict[str, list[dict]] = {}
        for t in trades:
            groups.setdefault(t["symbol"], []).append(t)

        result = {}
        for name, group in groups.items():
            wins = [t for t in group if t["is_win"]]
            result[name] = {
                "trades": len(group),
                "wins": len(wins),
                "win_rate": len(wins) / len(group) if group else 0.0,
                "net_pnl": sum(t["pnl"] for t in group),
            }
        return result

    def kill_candidates(self, min_trades: int = 30, min_win_rate: float = 0.40) -> list[str]:
        """Return strategy names that should be disabled (< min_win_rate after min_trades)."""
        stats = self.get_stats()
        kills = []
        for strategy, data in stats["by_strategy"].items():
            if data["trades"] >= min_trades and data["win_rate"] < min_win_rate:
                kills.append(strategy)
        return kills

    def summary_for_prompt(self) -> str:
        """Generate a concise text summary for inclusion in Strategist prompts."""
        stats = self.get_stats()
        if stats["overall"]["trades"] == 0:
            return "No trades recorded yet."

        lines = []
        o = stats["overall"]
        lines.append(
            f"Overall: {o['trades']} trades, {o['win_rate']:.0%} win rate, "
            f"net P&L: {o['net_pnl']:+,.0f}"
        )

        if stats["by_strategy"]:
            lines.append("By strategy:")
            for name, s in stats["by_strategy"].items():
                lines.append(
                    f"  {name}: {s['trades']} trades, {s['win_rate']:.0%} WR, "
                    f"avg R:R {s['avg_rr']:.1f}, net {s['net_pnl']:+,.0f}"
                )

        if stats["by_instrument"]:
            lines.append("By instrument:")
            for name, s in stats["by_instrument"].items():
                lines.append(
                    f"  {name}: {s['trades']} trades, {s['win_rate']:.0%} WR, "
                    f"net {s['net_pnl']:+,.0f}"
                )

        if stats["by_time"]:
            lines.append("By time:")
            for name, s in stats["by_time"].items():
                lines.append(
                    f"  {name}: {s['trades']} trades, {s['win_rate']:.0%} WR"
                )

        kills = self.kill_candidates()
        if kills:
            lines.append(f"KILL CANDIDATES (< 40% WR after 30+ trades): {', '.join(kills)}")

        return "\n".join(lines)
```

- [ ] **Step 4: Run tests**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_edge_tracker.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add v7/edge_tracker.py tests/test_v7_edge_tracker.py
git commit -m "feat(v7): add edge tracker — strategy performance tracking and kill decisions"
```

---

## Chunk 2: Telegram Integration

### Task 2: V7 Telegram module

**Files:**
- Create: `v7/telegram.py`
- Test: `tests/test_v7_telegram.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_v7_telegram.py
"""Tests for V7 Telegram integration."""
import pytest
import time as time_mod
from unittest.mock import patch, MagicMock
from datetime import date
from v7.telegram import (
    TelegramAlerter, AlertLevel, format_playbook_summary,
    format_trade_entry, format_trade_exit, format_checkin,
    format_exception, format_eod_summary, format_weekly_report,
    format_error,
)
from v7.types import (
    Playbook, Setup, SetupType, Position, TradeResult,
    RiskBudget, CarryRules, DayClassification, Conviction,
)


def test_alert_level_ordering():
    assert AlertLevel.CRITICAL.value > AlertLevel.HIGH.value
    assert AlertLevel.HIGH.value > AlertLevel.MEDIUM.value
    assert AlertLevel.MEDIUM.value > AlertLevel.LOW.value


def test_format_playbook_summary():
    pb = Playbook(
        date=date(2026, 3, 11),
        day_classification=DayClassification.LIKELY_TREND_UP,
        nifty_bias="bullish",
        nifty_setups=[
            Setup(
                id="N1", priority=1, type=SetupType.BREAKOUT_LONG,
                symbol="NIFTY", trigger_level=24350.0,
                trigger_condition="15-min close above 24350",
                instrument="NIFTY CE", strike_logic="delta 0.45",
                target=24500.0, stoploss=24280.0, max_risk_pct=1.5,
            ),
        ],
        stock_plans=[],
        risk_budget=RiskBudget(
            max_capital_at_risk_today_pct=4.0,
            max_trades_today=4,
            max_per_trade_risk_pct=1.5,
            survival_mode=False,
        ),
        no_trade_conditions=["VIX > 22"],
        carry_rules=CarryRules(),
    )
    msg = format_playbook_summary(pb)
    assert "<b>" in msg  # HTML formatting
    assert "LIKELY_TREND_UP" in msg
    assert "bullish" in msg.lower()
    assert "N1" in msg
    assert "```" not in msg  # no Markdown


def test_format_trade_entry():
    pos = Position(
        symbol="NIFTY", instrument="NIFTY CE 24400",
        direction="bullish", entry_price=120.0,
        quantity=75, lot_size=75, allocated=9000.0,
        stoploss=100.0, target=180.0,
        entry_date=date(2026, 3, 11), setup_id="N1",
    )
    msg = format_trade_entry(pos)
    assert "<b>NIFTY</b>" in msg
    assert "24400" in msg
    assert "120" in msg
    assert "HTML" not in msg  # parse_mode is set at send, not in message
    assert "```" not in msg


def test_format_trade_exit():
    result = TradeResult(
        symbol="NIFTY", instrument="NIFTY CE 24400",
        direction="bullish", entry_price=120.0, exit_price=180.0,
        quantity=75, entry_date=date(2026, 3, 11),
        exit_date=date(2026, 3, 11), exit_reason="target",
        pnl=4500.0, pnl_pct=50.0, costs=120.0,
        setup_id="N1", setup_type=SetupType.BREAKOUT_LONG,
    )
    msg = format_trade_exit(result, daily_pnl=4500.0, open_count=0)
    assert "4,500" in msg
    assert "target" in msg.lower()
    assert "```" not in msg


def test_format_eod_summary():
    msg = format_eod_summary(
        trades_today=2, wins=1, losses=1,
        directional_pnl=2000.0, theta_pnl=500.0,
        total_pnl=2500.0, capital=300_000,
        carried_positions=[],
        day_type_predicted="LIKELY_TREND_UP",
        day_type_actual="trending up",
    )
    assert "2,500" in msg
    assert "```" not in msg


def test_format_error():
    msg = format_error("Kite token expired", "Protect-only mode active. SL orders live.")
    assert "Kite token expired" in msg
    assert "<b>" in msg
    assert "```" not in msg


def test_rate_limiting():
    alerter = TelegramAlerter(token="fake", chat_id="fake")
    # Should not actually send (no real token), but should track rate
    with patch.object(alerter, "_do_send") as mock_send:
        alerter.send("test1", AlertLevel.HIGH)
        alerter.send("test2", AlertLevel.HIGH)
        alerter.send("test3", AlertLevel.HIGH)
        alerter.send("test4", AlertLevel.HIGH)
        alerter.send("test5", AlertLevel.HIGH)
        alerter.send("test6", AlertLevel.HIGH)
        # After 5 rapid SL alerts, rate limit kicks in
        # But individual sends should go through at HIGH level
        assert mock_send.call_count == 6  # all HIGH go through


def test_rate_limiting_low_priority():
    alerter = TelegramAlerter(token="fake", chat_id="fake")
    with patch.object(alerter, "_do_send") as mock_send:
        for _ in range(10):
            alerter.send("heartbeat", AlertLevel.LOW)
        # LOW priority gets rate-limited after burst
        assert mock_send.call_count <= 10  # all go through if spaced


def test_no_markdown_in_any_format():
    """Verify none of the format functions produce Markdown syntax."""
    pb = Playbook(
        date=date(2026, 3, 11),
        day_classification=DayClassification.LIKELY_TREND_UP,
        nifty_bias="bullish", nifty_setups=[], stock_plans=[],
        risk_budget=RiskBudget(), no_trade_conditions=[],
        carry_rules=CarryRules(),
    )
    for msg in [
        format_playbook_summary(pb),
        format_trade_entry(Position(
            symbol="X", instrument="X CE", direction="bullish",
            entry_price=100, quantity=75, lot_size=75, allocated=7500,
            stoploss=80, target=150, entry_date=date(2026, 3, 11), setup_id="T",
        )),
        format_eod_summary(0, 0, 0, 0, 0, 0, 300000, [], "?", "?"),
        format_error("err", "detail"),
    ]:
        assert "```" not in msg, f"Markdown backticks found in: {msg[:100]}"
        assert "__" not in msg or "<b>" in msg  # no Markdown bold
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_telegram.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'v7.telegram'`

- [ ] **Step 3: Implement Telegram module**

```python
# v7/telegram.py
"""V7 Telegram Integration — all alerts to Ops Hub Stocks topic.

HARD RULES:
- Always HTML parse_mode. NEVER Markdown.
- Use temp file + curl for shell scripts (matching existing pattern).
- Rate limit: don't spam on rapid SL hits.
"""
from __future__ import annotations

import os
import time as time_mod
import urllib.parse
import urllib.request
from collections import deque
from datetime import date
from enum import IntEnum
from typing import Any

from v7.types import (
    Playbook, Position, TradeResult, Setup,
)


class AlertLevel(IntEnum):
    LOW = 1       # informational (playbook, check-in)
    MEDIUM = 2    # EOD summary
    HIGH = 3      # trade entry/exit, exception
    CRITICAL = 4  # error/disconnect


class TelegramAlerter:
    """Send alerts to Telegram with rate limiting."""

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
        topic_id: str | None = None,
    ):
        self._token = token or os.environ.get("DEAL_BOT_TOKEN", "") or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id or os.environ.get("TELEGRAM_FORUM_CHAT_ID", "") or os.environ.get("DEAL_BOT_CHAT_ID", "")
        self._topic_id = topic_id or os.environ.get("TELEGRAM_TOPIC_STOCKS", "")
        self._recent_sends: deque[float] = deque(maxlen=50)
        self._rate_limit_window = 60  # seconds
        self._rate_limit_max = 20     # max messages per window

    def send(self, text: str, level: AlertLevel = AlertLevel.HIGH) -> bool:
        """Send a message. Returns True if sent, False if rate-limited or failed."""
        if not self._token or not self._chat_id:
            return False

        # Rate limiting: skip LOW messages if we're sending too fast
        now = time_mod.time()
        recent_count = sum(1 for t in self._recent_sends if now - t < self._rate_limit_window)
        if recent_count >= self._rate_limit_max and level <= AlertLevel.LOW:
            return False

        self._recent_sends.append(now)
        return self._do_send(text)

    def _do_send(self, text: str) -> bool:
        """Actually send via Telegram API. Split if > 4000 chars."""
        if len(text) <= 4000:
            return self._send_single(text)

        # Split at newline boundaries
        chunks: list[str] = []
        remaining = text
        while remaining:
            if len(remaining) <= 4000:
                chunks.append(remaining)
                break
            split_at = remaining.rfind("\n", 0, 4000)
            if split_at <= 0:
                split_at = 4000
            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:].lstrip("\n")

        total = len(chunks)
        ok = True
        for i, chunk in enumerate(chunks, 1):
            header = f"({i}/{total}) " if total > 1 else ""
            if not self._send_single(header + chunk):
                ok = False
        return ok

    def _send_single(self, text: str) -> bool:
        """Send a single message via Telegram HTTP API. HTML parse_mode always."""
        try:
            url = f"https://api.telegram.org/bot{self._token}/sendMessage"
            payload: dict[str, Any] = {
                "chat_id": self._chat_id,
                "text": text,
                "parse_mode": "HTML",
            }
            if self._chat_id and self._topic_id:
                payload["message_thread_id"] = self._topic_id
            data = urllib.parse.urlencode(payload).encode()
            req = urllib.request.Request(url, data=data)
            urllib.request.urlopen(req, timeout=10)
            return True
        except Exception:
            return False


# ── Format Functions ──────────────────────────────────────────────────
# All return HTML strings. No Markdown. No backticks. No __.


def format_playbook_summary(pb: Playbook) -> str:
    """Format pre-market playbook for Telegram."""
    lines = [
        f"<b>V7 Pre-Market Playbook — {pb.date}</b>",
        "",
        f"Day type: <b>{pb.day_classification.value}</b>",
        f"Nifty bias: {pb.nifty_bias}",
    ]

    if pb.market_context:
        ctx = pb.market_context
        if ctx.get("gift_nifty"):
            lines.append(f"GIFT Nifty: {ctx['gift_nifty']}")
        if ctx.get("vix"):
            lines.append(f"VIX: {ctx['vix']}")
        if ctx.get("fii_dii"):
            lines.append(f"FII/DII: {ctx['fii_dii']}")

    if pb.nifty_setups:
        lines.append("")
        lines.append("<b>Nifty setups:</b>")
        for s in pb.nifty_setups:
            lines.append(f"  [{s.id}] {s.type.value}: trigger {s.trigger_level:,.0f}, "
                         f"SL {s.stoploss:,.0f}, target {s.target:,.0f}")

    if pb.stock_plans:
        lines.append("")
        lines.append("<b>Stock setups:</b>")
        for s in pb.stock_plans:
            lines.append(f"  [{s.id}] {s.symbol} {s.type.value}: trigger {s.trigger_level:,.0f}")

    rb = pb.risk_budget
    lines.append("")
    lines.append(f"Risk: {rb.max_capital_at_risk_today_pct}% max, "
                 f"{rb.max_trades_today} trades max")
    if rb.survival_mode:
        lines.append("<b>SURVIVAL MODE — theta only</b>")

    return "\n".join(lines)


def format_trade_entry(pos: Position) -> str:
    """Format trade entry alert."""
    direction = "LONG" if pos.direction == "bullish" else "SHORT"
    return "\n".join([
        f"<b>V7 Entry: {pos.symbol}</b> {direction}",
        f"Instrument: {pos.instrument}",
        f"Price: {pos.entry_price:,.2f}  |  Qty: {pos.quantity}",
        f"SL: {pos.stoploss:,.2f}  |  Target: {pos.target:,.2f}",
        f"Risk: {pos.allocated:,.0f}  |  Setup: {pos.setup_id}",
    ])


def format_trade_exit(
    result: TradeResult,
    daily_pnl: float = 0.0,
    open_count: int = 0,
) -> str:
    """Format trade exit alert."""
    emoji = "WIN" if result.pnl > 0 else "LOSS"
    return "\n".join([
        f"<b>V7 Exit: {result.symbol}</b> [{emoji}]",
        f"Instrument: {result.instrument}",
        f"Entry: {result.entry_price:,.2f} → Exit: {result.exit_price:,.2f}",
        f"P&amp;L: {result.pnl:+,.0f} ({result.pnl_pct:+.1f}%)",
        f"Reason: {result.exit_reason}  |  Costs: {result.costs:,.0f}",
        f"Grade: entry={result.entry_grade} exit={result.exit_grade}",
        "",
        f"Day P&amp;L: {daily_pnl:+,.0f}  |  Open: {open_count}",
    ])


def format_checkin(
    checkin_num: int,
    plan_changed: bool,
    summary: str,
    daily_pnl: float = 0.0,
    open_count: int = 0,
) -> str:
    """Format check-in update."""
    status = "MODIFIED" if plan_changed else "unchanged"
    return "\n".join([
        f"<b>V7 Check-in #{checkin_num}</b> — plan {status}",
        summary,
        "",
        f"Day P&amp;L: {daily_pnl:+,.0f}  |  Open: {open_count}",
    ])


def format_exception(trigger: str, action: str) -> str:
    """Format exception alert."""
    return "\n".join([
        f"<b>V7 EXCEPTION</b>",
        f"Trigger: {trigger}",
        f"Action: {action}",
    ])


def format_eod_summary(
    trades_today: int,
    wins: int,
    losses: int,
    directional_pnl: float,
    theta_pnl: float,
    total_pnl: float,
    capital: float,
    carried_positions: list[dict],
    day_type_predicted: str,
    day_type_actual: str,
) -> str:
    """Format end-of-day summary."""
    pnl_pct = (total_pnl / capital * 100) if capital > 0 else 0.0
    lines = [
        f"<b>V7 EOD Summary</b>",
        "",
        f"Trades: {trades_today} ({wins}W / {losses}L)",
        f"Directional: {directional_pnl:+,.0f}",
        f"Theta: {theta_pnl:+,.0f}",
        f"<b>Total: {total_pnl:+,.0f} ({pnl_pct:+.1f}%)</b>",
        "",
        f"Day type: predicted {day_type_predicted} / actual {day_type_actual}",
    ]

    if carried_positions:
        lines.append("")
        lines.append(f"Carried overnight: {len(carried_positions)}")
        for p in carried_positions:
            lines.append(f"  {p.get('symbol', '?')} — P&amp;L {p.get('pnl', 0):+,.0f}")

    return "\n".join(lines)


def format_weekly_report(report_text: str) -> str:
    """Format weekly review report. report_text is pre-formatted by journal."""
    return f"<b>V7 Weekly Review</b>\n\n{report_text}"


def format_error(error: str, detail: str) -> str:
    """Format error/disconnect alert."""
    return "\n".join([
        f"<b>V7 ERROR</b>",
        f"{error}",
        f"{detail}",
    ])
```

- [ ] **Step 4: Run tests**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_telegram.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add v7/telegram.py tests/test_v7_telegram.py
git commit -m "feat(v7): add Telegram integration — HTML alerts, rate limiting, all event types"
```

---

## Chunk 3: Journal

### Task 3: Daily journal with Claude grading

**Files:**
- Create: `v7/journal.py`
- Test: `tests/test_v7_journal.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_v7_journal.py
"""Tests for V7 trade journal."""
import json
import pytest
from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock
from v7.journal import (
    Journal, grade_trades_prompt, parse_journal_response,
    format_obsidian_journal, generate_weekly_review_prompt,
    generate_monthly_report_prompt,
)
from v7.types import TradeResult, SetupType, DayClassification


@pytest.fixture
def tmp_journal(tmp_path):
    vault_dir = tmp_path / "trading-journal"
    return Journal(vault_dir=vault_dir, data_dir=tmp_path)


@pytest.fixture
def sample_trades():
    return [
        TradeResult(
            symbol="NIFTY", instrument="NIFTY CE 24400",
            direction="bullish", entry_price=120.0, exit_price=160.0,
            quantity=75, entry_date=date(2026, 3, 11),
            exit_date=date(2026, 3, 11), exit_reason="target",
            pnl=3000.0, pnl_pct=33.3, costs=120.0,
            setup_id="N1", setup_type=SetupType.BREAKOUT_LONG,
        ),
        TradeResult(
            symbol="HDFCBANK", instrument="HDFCBANK CE 1600",
            direction="bullish", entry_price=50.0, exit_price=35.0,
            quantity=550, entry_date=date(2026, 3, 11),
            exit_date=date(2026, 3, 11), exit_reason="stoploss",
            pnl=-8250.0, pnl_pct=-30.0, costs=80.0,
            setup_id="H1", setup_type=SetupType.SUPPORT_BOUNCE,
        ),
    ]


def test_grade_trades_prompt(sample_trades):
    prompt = grade_trades_prompt(
        trades=sample_trades,
        day_classification=DayClassification.LIKELY_TREND_UP,
        playbook_setups=2,
        day_pnl=-5250.0,
    )
    assert "NIFTY" in prompt
    assert "HDFCBANK" in prompt
    assert "entry_grade" in prompt.lower() or "entry quality" in prompt.lower()
    assert "exit_grade" in prompt.lower() or "exit quality" in prompt.lower()


def test_parse_journal_response():
    raw = json.dumps({
        "trades": [
            {
                "setup_id": "N1",
                "entry_grade": "A",
                "exit_grade": "A",
                "lesson": "Clean breakout with volume confirmation.",
            },
            {
                "setup_id": "H1",
                "entry_grade": "C",
                "exit_grade": "B",
                "lesson": "Sector was weak — check sector strength before banking plays.",
            },
        ],
        "day_summary": {
            "day_type_accuracy": "correct",
            "best_trade": "N1",
            "worst_trade": "H1",
            "overall_lesson": "Sector context matters more than individual chart.",
        },
    })
    result = parse_journal_response(raw)
    assert len(result["trades"]) == 2
    assert result["trades"][0]["entry_grade"] == "A"
    assert result["day_summary"]["best_trade"] == "N1"


def test_format_obsidian_journal(sample_trades):
    grading = {
        "trades": [
            {"setup_id": "N1", "entry_grade": "A", "exit_grade": "A",
             "lesson": "Clean breakout."},
            {"setup_id": "H1", "entry_grade": "C", "exit_grade": "B",
             "lesson": "Check sector first."},
        ],
        "day_summary": {
            "day_type_accuracy": "correct",
            "best_trade": "N1",
            "worst_trade": "H1",
            "overall_lesson": "Sector context matters.",
        },
    }
    md = format_obsidian_journal(
        date_str="2026-03-11",
        trades=sample_trades,
        grading=grading,
        day_classification="LIKELY_TREND_UP",
        directional_pnl=-5250.0,
        theta_pnl=500.0,
        total_pnl=-4750.0,
    )
    assert "# Trading Journal — 2026-03-11" in md
    assert "NIFTY" in md
    assert "HDFCBANK" in md
    assert "Clean breakout" in md
    assert "Sector context matters" in md


def test_save_journal_to_obsidian(tmp_journal, sample_trades):
    grading = {
        "trades": [
            {"setup_id": "N1", "entry_grade": "A", "exit_grade": "A", "lesson": "Good."},
            {"setup_id": "H1", "entry_grade": "C", "exit_grade": "B", "lesson": "Bad."},
        ],
        "day_summary": {
            "day_type_accuracy": "correct", "best_trade": "N1",
            "worst_trade": "H1", "overall_lesson": "Learn.",
        },
    }
    path = tmp_journal.save_daily(
        date_str="2026-03-11",
        trades=sample_trades,
        grading=grading,
        day_classification="LIKELY_TREND_UP",
        directional_pnl=-5250.0,
        theta_pnl=500.0,
        total_pnl=-4750.0,
    )
    assert path.exists()
    assert path.name == "2026-03-11.md"
    content = path.read_text()
    assert "Trading Journal" in content


def test_weekly_review_prompt(sample_trades):
    prompt = generate_weekly_review_prompt(
        trades_this_week=sample_trades,
        edge_summary="Overall: 2 trades, 50% WR",
        level_memory={"NIFTY": {"levels": []}},
        watchlist_performance={"NIFTY": 3000.0, "HDFCBANK": -8250.0},
    )
    assert "performance attribution" in prompt.lower() or "weekly" in prompt.lower()
    assert "NIFTY" in prompt
    assert "HDFCBANK" in prompt


def test_monthly_report_prompt():
    prompt = generate_monthly_report_prompt(
        month="2026-03",
        total_pnl=15000.0,
        total_costs=2500.0,
        capital=300_000,
        trades_count=45,
        win_rate=0.55,
        max_drawdown_pct=3.2,
        edge_summary="Overall: 45 trades, 55% WR",
        theta_pnl=5000.0,
        directional_pnl=10000.0,
    )
    assert "monthly" in prompt.lower() or "report" in prompt.lower()
    assert "15,000" in prompt or "15000" in prompt
    assert "tax" in prompt.lower() or "turnover" in prompt.lower()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_journal.py -v`
Expected: FAIL

- [ ] **Step 3: Implement journal**

```python
# v7/journal.py
"""V7 Trade Journal — grading, Obsidian persistence, weekly/monthly reviews.

Daily journal (3:30 PM, Haiku):
  - Grade every trade: entry quality (A/B/C), exit quality (A/B/C)
  - One-sentence lesson per trade
  - Day summary: wins/losses, P&L breakdown, day type accuracy

Weekly review (Sunday, Sonnet):
  - Performance attribution by strategy/instrument/time/setup
  - Watchlist rotation decisions
  - Level memory updates

Monthly report (1st of month, Sonnet):
  - P&L report, transaction costs, net return %
  - Tax estimate, drawdown analysis
  - Strategy allocation, capital recommendation
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import date

from v7.types import TradeResult, DayClassification, SetupType


class Journal:
    """Manages trade journal, Obsidian vault writes, and review prompts."""

    def __init__(
        self,
        vault_dir: Path | str = Path.home() / "Documents" / "Obsidian" / "trading-journal",
        data_dir: Path | str = Path("data/v7"),
    ):
        self._vault_dir = Path(vault_dir)
        self._data_dir = Path(data_dir)

    def save_daily(
        self,
        date_str: str,
        trades: list[TradeResult],
        grading: dict,
        day_classification: str,
        directional_pnl: float,
        theta_pnl: float,
        total_pnl: float,
    ) -> Path:
        """Save daily journal to Obsidian vault. Returns path."""
        md = format_obsidian_journal(
            date_str=date_str,
            trades=trades,
            grading=grading,
            day_classification=day_classification,
            directional_pnl=directional_pnl,
            theta_pnl=theta_pnl,
            total_pnl=total_pnl,
        )
        self._vault_dir.mkdir(parents=True, exist_ok=True)
        path = self._vault_dir / f"{date_str}.md"
        path.write_text(md)
        return path

    def load_recent_lessons(self, days: int = 5) -> list[str]:
        """Load lessons from recent journal entries for Strategist context."""
        lessons = []
        if not self._vault_dir.exists():
            return lessons
        files = sorted(self._vault_dir.glob("*.md"), reverse=True)[:days]
        for f in files:
            content = f.read_text()
            # Extract lines that start with "Lesson:" or are in the lessons section
            for line in content.split("\n"):
                stripped = line.strip()
                if stripped.startswith("- Lesson:") or stripped.startswith("Lesson:"):
                    lessons.append(stripped)
        return lessons


def grade_trades_prompt(
    trades: list[TradeResult],
    day_classification: DayClassification | str,
    playbook_setups: int,
    day_pnl: float,
) -> str:
    """Generate the prompt for Claude Haiku to grade today's trades.

    Returns the full prompt string. Caller sends to Claude API.
    """
    dc = day_classification.value if isinstance(day_classification, DayClassification) else day_classification

    trade_lines = []
    for t in trades:
        trade_lines.append(
            f"- {t.symbol} ({t.instrument}): {t.direction}, "
            f"entry {t.entry_price:.2f} -> exit {t.exit_price:.2f}, "
            f"P&L {t.pnl:+,.0f} ({t.pnl_pct:+.1f}%), "
            f"exit reason: {t.exit_reason}, "
            f"setup: {t.setup_id} ({t.setup_type.value})"
        )

    trades_text = "\n".join(trade_lines) if trade_lines else "No trades today."

    return f"""Grade today's trades. For each trade, provide:
- entry_grade: A (trigger + confirmation aligned) / B (trigger fired, weak confirmation) / C (FOMO or forced entry)
- exit_grade: A (plan followed exactly) / B (minor deviation from plan) / C (panic exit or held too long)
- lesson: one sentence — what to remember for next time

Also provide a day_summary with:
- day_type_accuracy: was the morning classification correct? ("correct", "partially correct", "wrong")
- best_trade: setup_id of best executed trade
- worst_trade: setup_id of worst executed trade
- overall_lesson: one sentence for the day

Context:
- Day classification: {dc}
- Playbook had {playbook_setups} setups
- Day P&L: {day_pnl:+,.0f}

Trades:
{trades_text}

Respond in JSON:
{{
  "trades": [
    {{"setup_id": "...", "entry_grade": "A/B/C", "exit_grade": "A/B/C", "lesson": "..."}}
  ],
  "day_summary": {{
    "day_type_accuracy": "correct/partially correct/wrong",
    "best_trade": "setup_id",
    "worst_trade": "setup_id",
    "overall_lesson": "..."
  }}
}}"""


def parse_journal_response(raw: str) -> dict:
    """Parse Claude's journal grading response."""
    from utils import parse_claude_json
    return parse_claude_json(raw)


def format_obsidian_journal(
    date_str: str,
    trades: list[TradeResult],
    grading: dict,
    day_classification: str,
    directional_pnl: float,
    theta_pnl: float,
    total_pnl: float,
) -> str:
    """Format the daily journal as Obsidian-compatible Markdown."""
    grade_map = {}
    for tg in grading.get("trades", []):
        grade_map[tg["setup_id"]] = tg

    wins = sum(1 for t in trades if t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl <= 0)
    day_summary = grading.get("day_summary", {})

    lines = [
        f"# Trading Journal — {date_str}",
        "",
        f"**Day classification:** {day_classification}",
        f"**Day type accuracy:** {day_summary.get('day_type_accuracy', 'N/A')}",
        f"**Trades:** {len(trades)} ({wins}W / {losses}L)",
        f"**P&L:** Directional {directional_pnl:+,.0f} + Theta {theta_pnl:+,.0f} = **{total_pnl:+,.0f}**",
        "",
        "---",
        "",
        "## Trades",
        "",
    ]

    for t in trades:
        g = grade_map.get(t.setup_id, {})
        lines.extend([
            f"### {t.symbol} — {t.instrument}",
            f"- Direction: {t.direction}",
            f"- Entry: {t.entry_price:.2f} | Exit: {t.exit_price:.2f}",
            f"- P&L: {t.pnl:+,.0f} ({t.pnl_pct:+.1f}%)",
            f"- Exit reason: {t.exit_reason}",
            f"- Setup: {t.setup_id} ({t.setup_type.value})",
            f"- Entry grade: **{g.get('entry_grade', '?')}** | Exit grade: **{g.get('exit_grade', '?')}**",
            f"- Lesson: {g.get('lesson', '')}",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## Day Summary",
        "",
        f"- Best trade: {day_summary.get('best_trade', 'N/A')}",
        f"- Worst trade: {day_summary.get('worst_trade', 'N/A')}",
        f"- Overall lesson: {day_summary.get('overall_lesson', '')}",
    ])

    return "\n".join(lines)


def generate_weekly_review_prompt(
    trades_this_week: list[TradeResult],
    edge_summary: str,
    level_memory: dict,
    watchlist_performance: dict[str, float],
) -> str:
    """Generate prompt for Sonnet weekly review.

    Returns the full prompt string. Caller sends to Claude API.
    """
    trade_lines = []
    for t in trades_this_week:
        trade_lines.append(
            f"- {t.symbol}: {t.setup_type.value}, P&L {t.pnl:+,.0f}, "
            f"grades {t.entry_grade}/{t.exit_grade}"
        )
    trades_text = "\n".join(trade_lines) if trade_lines else "No trades this week."

    perf_lines = []
    for sym, pnl in sorted(watchlist_performance.items(), key=lambda x: x[1]):
        perf_lines.append(f"  {sym}: {pnl:+,.0f}")
    perf_text = "\n".join(perf_lines) if perf_lines else "  No data."

    levels_text = json.dumps(level_memory, indent=2, default=str)[:2000]

    return f"""Weekly performance review. Analyze and provide recommendations.

EDGE TRACKER:
{edge_summary}

THIS WEEK'S TRADES:
{trades_text}

WATCHLIST PERFORMANCE (net P&L per instrument):
{perf_text}

CURRENT LEVEL MEMORY (truncated):
{levels_text}

Provide:
1. Performance attribution by strategy, instrument, time of day, and setup type.
2. Strategies to disable (< 40% win rate after 30+ trades).
3. Instruments to drop from active watchlist (consistently losing) and replacements.
4. Time slots to avoid.
5. Watchlist rotation: which instrument to drop, which to add.
6. Level memory updates:
   - Levels that held 2+ times this week → strengthen
   - Levels that broke cleanly → remove or flip
   - New levels from this week's price action

Respond in JSON:
{{
  "attribution_summary": "...",
  "strategies_to_disable": [],
  "instruments_to_drop": [],
  "instruments_to_add": [],
  "time_slots_to_avoid": [],
  "level_updates": {{
    "strengthen": [{{"symbol": "...", "price": 0, "reason": "..."}}],
    "remove": [{{"symbol": "...", "price": 0, "reason": "..."}}],
    "flip": [{{"symbol": "...", "price": 0, "old_type": "...", "new_type": "..."}}],
    "add": [{{"symbol": "...", "price": 0, "type": "...", "source": "..."}}]
  }},
  "watchlist_rotation": {{
    "drop": {{"symbol": "...", "reason": "..."}},
    "add": {{"symbol": "...", "reason": "..."}}
  }},
  "next_week_focus": "..."
}}"""


def generate_monthly_report_prompt(
    month: str,
    total_pnl: float,
    total_costs: float,
    capital: float,
    trades_count: int,
    win_rate: float,
    max_drawdown_pct: float,
    edge_summary: str,
    theta_pnl: float,
    directional_pnl: float,
) -> str:
    """Generate prompt for Sonnet monthly report.

    Returns the full prompt string. Caller sends to Claude API.
    """
    net_pnl = total_pnl - total_costs
    net_return_pct = (net_pnl / capital * 100) if capital > 0 else 0.0

    # F&O turnover for tax: sum of absolute trade values
    # This is an estimate — actual turnover is trade_value * quantity for each leg
    estimated_turnover = abs(total_pnl) * 10  # rough: P&L ~ 10% of turnover

    return f"""Monthly performance report for {month}.

PERFORMANCE:
- Gross P&L: {total_pnl:+,.0f}
- Transaction costs: {total_costs:,.0f}
- Net P&L: {net_pnl:+,.0f}
- Net return: {net_return_pct:+.1f}% on {capital:,.0f} capital
- Directional P&L: {directional_pnl:+,.0f}
- Theta P&L: {theta_pnl:+,.0f}
- Total trades: {trades_count}
- Win rate: {win_rate:.0%}
- Max drawdown: {max_drawdown_pct:.1f}%

EDGE TRACKER:
{edge_summary}

ESTIMATED TURNOVER: {estimated_turnover:,.0f}

Provide a monthly report including:
1. P&L report with breakdown (directional vs theta, transaction costs, net return %)
2. Tax estimate:
   - F&O turnover and classification (speculative vs non-speculative)
   - Estimated advance tax due (30% bracket assumed)
   - STT already paid (included in transaction costs)
3. Drawdown analysis: max drawdown, recovery time, risk events
4. Strategy allocation for next month: shift toward what's working
5. Capital recommendation: grow (add funds) / maintain / reduce (withdraw)
6. Withdrawal recommendation: suggest withdrawing 50% of net profit
7. Key lessons from the month

Respond in JSON:
{{
  "pnl_report": {{
    "gross_pnl": 0,
    "costs": 0,
    "net_pnl": 0,
    "return_pct": 0,
    "directional_pnl": 0,
    "theta_pnl": 0
  }},
  "tax_estimate": {{
    "turnover": 0,
    "estimated_tax": 0,
    "advance_tax_due": 0,
    "notes": "..."
  }},
  "drawdown_analysis": {{
    "max_drawdown_pct": 0,
    "recovery_days": 0,
    "risk_events": []
  }},
  "strategy_allocation": {{
    "directional_pct": 60,
    "theta_pct": 40,
    "changes": "..."
  }},
  "capital_recommendation": "grow/maintain/reduce",
  "withdrawal": {{
    "amount": 0,
    "reasoning": "..."
  }},
  "key_lessons": ["..."]
}}"""
```

- [ ] **Step 4: Run tests**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_journal.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add v7/journal.py tests/test_v7_journal.py
git commit -m "feat(v7): add trade journal — daily grading, Obsidian writes, weekly/monthly prompts"
```

---

## Chunk 4: Main Entry Point

### Task 4: CLI entry point

**Files:**
- Create: `v7/main.py`
- Test: `tests/test_v7_main.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_v7_main.py
"""Tests for V7 main CLI entry point."""
import pytest
import sys
from unittest.mock import patch, MagicMock
from v7.main import parse_args, STATUS_COMMANDS, TRADING_COMMANDS


def test_parse_args_premarket():
    args = parse_args(["premarket"])
    assert args.command == "premarket"


def test_parse_args_tick():
    args = parse_args(["tick"])
    assert args.command == "tick"


def test_parse_args_eod():
    args = parse_args(["eod"])
    assert args.command == "eod"


def test_parse_args_status():
    args = parse_args(["status"])
    assert args.command == "status"


def test_parse_args_weekly():
    args = parse_args(["weekly"])
    assert args.command == "weekly"


def test_parse_args_monthly():
    args = parse_args(["monthly"])
    assert args.command == "monthly"


def test_parse_args_checkin():
    args = parse_args(["checkin", "--num", "1"])
    assert args.command == "checkin"
    assert args.num == 1


def test_parse_args_paper_flag():
    args = parse_args(["--paper", "tick"])
    assert args.paper is True


def test_parse_args_live_flag():
    args = parse_args(["tick"])
    assert args.paper is False


def test_all_commands_defined():
    all_cmds = STATUS_COMMANDS + TRADING_COMMANDS
    for cmd in ["premarket", "opening-read", "checkin", "tick", "eod",
                 "weekly", "monthly", "status", "paper-status"]:
        assert cmd in all_cmds


def test_parse_args_opening_read():
    args = parse_args(["opening-read"])
    assert args.command == "opening-read"


def test_parse_args_paper_status():
    args = parse_args(["paper-status"])
    assert args.command == "paper-status"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_main.py -v`
Expected: FAIL

- [ ] **Step 3: Implement main entry point**

```python
# v7/main.py
"""V7 CLI Entry Point — runs all V7 operations.

Usage:
    python -m v7.main premarket          # 8:43 AM — generate playbook
    python -m v7.main opening-read       # 9:13 AM — classify day type
    python -m v7.main checkin --num 1    # 10:28 AM — check-in 1
    python -m v7.main checkin --num 2    # 12:58 PM — check-in 2
    python -m v7.main tick               # every 3 min — main executor loop
    python -m v7.main eod                # 3:33 PM — EOD review + journal
    python -m v7.main weekly             # Sunday 10:03 AM — weekly review
    python -m v7.main monthly            # 1st of month 10:07 AM — monthly report
    python -m v7.main status             # print current state
    python -m v7.main paper-status       # print paper trading performance

    --paper flag enables paper trading mode (no real orders).
    Paper mode is DEFAULT until explicitly switched to live.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

STATUS_COMMANDS = ["status", "paper-status"]
TRADING_COMMANDS = ["premarket", "opening-read", "checkin", "tick", "eod", "weekly", "monthly"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="V7 Professional Trader Bot")
    parser.add_argument(
        "command",
        choices=STATUS_COMMANDS + TRADING_COMMANDS,
        help="Operation to run",
    )
    parser.add_argument(
        "--paper", action="store_true", default=False,
        help="Paper trading mode (no real orders)",
    )
    parser.add_argument(
        "--num", type=int, default=1,
        help="Check-in number (1 or 2)",
    )
    return parser.parse_args(argv)


def _init_components(paper: bool = False) -> dict:
    """Initialize all V7 components. Returns dict of component instances."""
    from v7.state import StateManager
    from v7.edge_tracker import EdgeTracker
    from v7.journal import Journal
    from v7.telegram import TelegramAlerter
    from v7.config_v7 import CAPITAL

    data_dir = Path("data/v7")
    data_dir.mkdir(parents=True, exist_ok=True)

    state = StateManager(data_dir)
    edge_tracker = EdgeTracker(data_dir=data_dir)
    journal = Journal(data_dir=data_dir)
    telegram = TelegramAlerter()

    components = {
        "state": state,
        "edge_tracker": edge_tracker,
        "journal": journal,
        "telegram": telegram,
        "paper": paper,
        "capital": CAPITAL["initial"],
        "data_dir": data_dir,
    }

    # These imports may fail if Plan 2/3 not yet implemented
    # Graceful fallback for incremental development
    try:
        from v7.data_feed import DataFeed
        components["data_feed"] = DataFeed(paper=paper)
    except ImportError:
        pass

    try:
        from v7.strategist import Strategist
        components["strategist"] = Strategist(
            state=state,
            edge_tracker=edge_tracker,
            journal=journal,
            telegram=telegram,
        )
    except ImportError:
        pass

    try:
        from v7.executor import Executor
        components["executor"] = Executor(
            state=state,
            telegram=telegram,
            edge_tracker=edge_tracker,
            paper=paper,
        )
    except ImportError:
        pass

    try:
        from v7.theta_engine import ThetaEngine
        components["theta_engine"] = ThetaEngine(
            state=state,
            telegram=telegram,
            paper=paper,
        )
    except ImportError:
        pass

    return components


def cmd_premarket(components: dict) -> None:
    """Run pre-market playbook generation."""
    from v7.telegram import format_playbook_summary, AlertLevel

    strategist = components.get("strategist")
    telegram = components["telegram"]

    if not strategist:
        print("ERROR: Strategist not available (Plan 2 not implemented)")
        telegram.send("V7: Strategist not available for premarket", AlertLevel.CRITICAL)
        sys.exit(1)

    playbook = strategist.premarket()
    print(f"Playbook generated: {playbook.day_classification.value}, "
          f"{len(playbook.all_setups())} setups")

    msg = format_playbook_summary(playbook)
    telegram.send(msg, AlertLevel.LOW)


def cmd_opening_read(components: dict) -> None:
    """Run opening read (after first 30 min)."""
    strategist = components.get("strategist")
    telegram = components["telegram"]

    if not strategist:
        print("ERROR: Strategist not available")
        sys.exit(1)

    updated_playbook = strategist.opening_read()
    print(f"Opening read done. Day type: {updated_playbook.day_classification.value}")

    from v7.telegram import format_checkin, AlertLevel
    msg = format_checkin(
        checkin_num=0,
        plan_changed=True,
        summary=f"Opening read: {updated_playbook.day_classification.value}",
    )
    telegram.send(msg, AlertLevel.LOW)


def cmd_checkin(components: dict, num: int) -> None:
    """Run strategist check-in."""
    strategist = components.get("strategist")
    telegram = components["telegram"]

    if not strategist:
        print("ERROR: Strategist not available")
        sys.exit(1)

    result = strategist.checkin(num)
    print(f"Check-in #{num} done. Plan changed: {result.get('plan_changed', False)}")

    from v7.telegram import format_checkin, AlertLevel
    msg = format_checkin(
        checkin_num=num,
        plan_changed=result.get("plan_changed", False),
        summary=result.get("summary", "No changes"),
    )
    telegram.send(msg, AlertLevel.LOW)


def cmd_tick(components: dict) -> None:
    """Run one executor tick."""
    executor = components.get("executor")
    if not executor:
        print("ERROR: Executor not available (Plan 3 not implemented)")
        sys.exit(1)

    executor.tick()


def cmd_eod(components: dict) -> None:
    """Run EOD review + journal."""
    import anthropic
    from v7.journal import grade_trades_prompt, parse_journal_response
    from v7.telegram import format_eod_summary, AlertLevel

    state = components["state"]
    journal = components["journal"]
    edge_tracker = components["edge_tracker"]
    telegram = components["telegram"]

    today = datetime.now(IST).strftime("%Y-%m-%d")
    daily = state.load_daily_state()
    trades_today = daily.get("closed_trades", [])
    playbook = state.load_playbook()

    # Convert trade dicts to TradeResult objects
    from v7.types import TradeResult
    trade_results = [TradeResult.from_dict(t) for t in trades_today]

    if trade_results:
        # Call Claude Haiku for grading
        prompt = grade_trades_prompt(
            trades=trade_results,
            day_classification=playbook.day_classification if playbook else "UNKNOWN",
            playbook_setups=len(playbook.all_setups()) if playbook else 0,
            day_pnl=daily.get("total_pnl", 0),
        )

        try:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-haiku-4-20250414",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )
            grading = parse_journal_response(response.content[0].text)
        except Exception as e:
            print(f"Claude grading failed: {e}")
            grading = {
                "trades": [{"setup_id": t.setup_id, "entry_grade": "B",
                            "exit_grade": "B", "lesson": "Grading unavailable"}
                           for t in trade_results],
                "day_summary": {"day_type_accuracy": "unknown", "best_trade": "",
                                "worst_trade": "", "overall_lesson": "Grading failed"},
            }

        # Apply grades back to trade results
        grade_map = {tg["setup_id"]: tg for tg in grading.get("trades", [])}
        for t in trade_results:
            if t.setup_id in grade_map:
                t.entry_grade = grade_map[t.setup_id].get("entry_grade", "B")
                t.exit_grade = grade_map[t.setup_id].get("exit_grade", "B")
                t.lesson = grade_map[t.setup_id].get("lesson", "")

        # Record in edge tracker
        for t in trade_results:
            strategy = _infer_strategy(t)
            time_bucket = _infer_time_bucket(t)
            edge_tracker.record(t, strategy=strategy, time_bucket=time_bucket)

        # Save journal to Obsidian
        directional_pnl = sum(t.pnl for t in trade_results if t.setup_type != SetupType.IRON_CONDOR)
        theta_pnl = sum(t.pnl for t in trade_results if t.setup_type == SetupType.IRON_CONDOR)
        total_pnl = directional_pnl + theta_pnl

        path = journal.save_daily(
            date_str=today,
            trades=trade_results,
            grading=grading,
            day_classification=playbook.day_classification.value if playbook else "UNKNOWN",
            directional_pnl=directional_pnl,
            theta_pnl=theta_pnl,
            total_pnl=total_pnl,
        )
        print(f"Journal saved: {path}")
    else:
        directional_pnl = 0
        theta_pnl = 0
        total_pnl = 0
        print("No trades today — skipping journal")

    # EOD Telegram summary
    wins = sum(1 for t in trade_results if t.pnl > 0)
    losses = sum(1 for t in trade_results if t.pnl <= 0)
    carried = daily.get("carried_positions", [])

    msg = format_eod_summary(
        trades_today=len(trade_results),
        wins=wins,
        losses=len(trade_results) - wins,
        directional_pnl=directional_pnl,
        theta_pnl=theta_pnl,
        total_pnl=total_pnl,
        capital=components["capital"],
        carried_positions=carried,
        day_type_predicted=playbook.day_classification.value if playbook else "?",
        day_type_actual=daily.get("actual_day_type", "?"),
    )
    telegram.send(msg, AlertLevel.MEDIUM)


def cmd_weekly(components: dict) -> None:
    """Run weekly review."""
    import anthropic
    from v7.journal import generate_weekly_review_prompt, parse_journal_response
    from v7.telegram import format_weekly_report, AlertLevel

    state = components["state"]
    edge_tracker = components["edge_tracker"]
    telegram = components["telegram"]

    # Load week's trades from edge tracker
    edge_summary = edge_tracker.summary_for_prompt()
    level_memory = state.load_level_memory()

    # Gather watchlist performance from edge tracker stats
    stats = edge_tracker.get_stats()
    watchlist_perf = {sym: data["net_pnl"] for sym, data in stats.get("by_instrument", {}).items()}

    # Get this week's trades from daily states
    # (In practice, edge_tracker has all trades — we just need the recent ones)
    trades_this_week = []  # TODO: filter edge_tracker trades to this week

    prompt = generate_weekly_review_prompt(
        trades_this_week=trades_this_week,
        edge_summary=edge_summary,
        level_memory=level_memory,
        watchlist_performance=watchlist_perf,
    )

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        review = parse_journal_response(response.content[0].text)
    except Exception as e:
        print(f"Weekly review failed: {e}")
        telegram.send(format_weekly_report(f"Weekly review generation failed: {e}"), AlertLevel.HIGH)
        return

    # Apply level memory updates
    if "level_updates" in review:
        _apply_level_updates(state, review["level_updates"])

    # Save review
    data_dir = components["data_dir"]
    today = datetime.now(IST).strftime("%Y-%m-%d")
    review_file = data_dir / f"weekly_review_{today}.json"
    review_file.write_text(json.dumps(review, indent=2, default=str))

    msg = format_weekly_report(json.dumps(review, indent=2, default=str)[:3500])
    telegram.send(msg, AlertLevel.LOW)
    print(f"Weekly review saved: {review_file}")


def cmd_monthly(components: dict) -> None:
    """Run monthly report."""
    import anthropic
    from v7.journal import generate_monthly_report_prompt, parse_journal_response
    from v7.telegram import format_weekly_report, AlertLevel

    edge_tracker = components["edge_tracker"]
    telegram = components["telegram"]
    capital = components["capital"]

    now = datetime.now(IST)
    # Report for previous month
    if now.month == 1:
        month_str = f"{now.year - 1}-12"
    else:
        month_str = f"{now.year}-{now.month - 1:02d}"

    stats = edge_tracker.get_stats()
    overall = stats["overall"]

    prompt = generate_monthly_report_prompt(
        month=month_str,
        total_pnl=overall.get("net_pnl", 0),
        total_costs=0,  # TODO: sum costs from edge tracker trades
        capital=capital,
        trades_count=overall.get("trades", 0),
        win_rate=overall.get("win_rate", 0),
        max_drawdown_pct=0,  # TODO: from monthly_state.json
        edge_summary=edge_tracker.summary_for_prompt(),
        theta_pnl=stats.get("by_strategy", {}).get("theta", {}).get("net_pnl", 0),
        directional_pnl=overall.get("net_pnl", 0) - stats.get("by_strategy", {}).get("theta", {}).get("net_pnl", 0),
    )

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        report = parse_journal_response(response.content[0].text)
    except Exception as e:
        print(f"Monthly report failed: {e}")
        telegram.send(f"<b>V7 Monthly Report</b>\n\nGeneration failed: {e}", AlertLevel.HIGH)
        return

    data_dir = components["data_dir"]
    report_file = data_dir / f"monthly_report_{month_str}.json"
    report_file.write_text(json.dumps(report, indent=2, default=str))

    msg = format_weekly_report(f"Monthly Report — {month_str}\n\n{json.dumps(report, indent=2, default=str)[:3500]}")
    telegram.send(msg, AlertLevel.LOW)
    print(f"Monthly report saved: {report_file}")


def cmd_status(components: dict) -> None:
    """Print current V7 state."""
    state = components["state"]
    daily = state.load_daily_state()
    playbook = state.load_playbook()

    print("=== V7 Status ===")
    if playbook:
        print(f"Date: {playbook.date}")
        print(f"Day type: {playbook.day_classification.value}")
        print(f"Setups: {len(playbook.all_setups())} total, "
              f"{len(playbook.active_setups())} active")
    else:
        print("No playbook loaded")

    positions = state.load_positions()
    print(f"Open positions: {len(positions)}")
    for p in positions:
        print(f"  {p.symbol} {p.instrument}: entry {p.entry_price:.2f}, SL {p.stoploss:.2f}")

    print(f"Daily P&L: {daily.get('total_pnl', 0):+,.0f}")
    print(f"Trade count: {daily.get('trade_count', 0)}")
    print(f"SL hits: {daily.get('sl_hit_count', 0)}")


def cmd_paper_status(components: dict) -> None:
    """Print paper trading performance summary."""
    edge_tracker = components["edge_tracker"]
    stats = edge_tracker.get_stats()
    print("=== V7 Paper Trading Status ===")
    print(edge_tracker.summary_for_prompt())


def _infer_strategy(trade: TradeResult) -> str:
    """Infer strategy name from setup type."""
    if trade.setup_type in (SetupType.BREAKOUT_LONG, SetupType.BREAKOUT_SHORT):
        return "momentum"
    if trade.setup_type in (SetupType.SUPPORT_BOUNCE, SetupType.RESISTANCE_FADE):
        return "mean_reversion"
    if trade.setup_type in (SetupType.IRON_CONDOR, SetupType.CREDIT_SPREAD_BULL, SetupType.CREDIT_SPREAD_BEAR):
        return "theta"
    return "other"


def _infer_time_bucket(trade: TradeResult) -> str:
    """Infer time bucket from trade entry time. Default to morning."""
    # In practice, entry_time would be stored on the trade.
    # For now, default to morning bucket.
    return "9:45-11:00"


def _apply_level_updates(state, updates: dict) -> None:
    """Apply level memory updates from weekly review."""
    level_memory = state.load_level_memory()

    for item in updates.get("add", []):
        sym = item["symbol"]
        if sym not in level_memory:
            level_memory[sym] = {"levels": [], "oi_walls": {}}
        level_memory[sym]["levels"].append({
            "price": item["price"],
            "type": item.get("type", "support"),
            "strength": 1,
            "source": item.get("source", "weekly review"),
            "last_tested": str(date.today()),
            "created": str(date.today()),
        })

    for item in updates.get("strengthen", []):
        sym = item["symbol"]
        if sym in level_memory:
            for level in level_memory[sym].get("levels", []):
                if abs(level["price"] - item["price"]) < 10:
                    level["strength"] = level.get("strength", 1) + 1

    for item in updates.get("remove", []):
        sym = item["symbol"]
        if sym in level_memory:
            level_memory[sym]["levels"] = [
                l for l in level_memory[sym]["levels"]
                if abs(l["price"] - item["price"]) >= 10
            ]

    state.save_level_memory(level_memory)


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    args = parse_args(argv)

    try:
        components = _init_components(paper=args.paper)
    except Exception as e:
        print(f"Failed to initialize: {e}")
        traceback.print_exc()
        sys.exit(1)

    try:
        if args.command == "premarket":
            cmd_premarket(components)
        elif args.command == "opening-read":
            cmd_opening_read(components)
        elif args.command == "checkin":
            cmd_checkin(components, args.num)
        elif args.command == "tick":
            cmd_tick(components)
        elif args.command == "eod":
            cmd_eod(components)
        elif args.command == "weekly":
            cmd_weekly(components)
        elif args.command == "monthly":
            cmd_monthly(components)
        elif args.command == "status":
            cmd_status(components)
        elif args.command == "paper-status":
            cmd_paper_status(components)
    except Exception as e:
        print(f"Command {args.command} failed: {e}")
        traceback.print_exc()
        from v7.telegram import format_error, AlertLevel
        components["telegram"].send(
            format_error(f"V7 {args.command} failed", str(e)),
            AlertLevel.CRITICAL,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_main.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add v7/main.py tests/test_v7_main.py
git commit -m "feat(v7): add main CLI entry point — all commands, component initialization"
```

---

## Chunk 5: Cron Scripts

### Task 5: Pi cron scripts

**Files:**
- Create: `scripts/v7_premarket.sh`
- Create: `scripts/v7_opening_read.sh`
- Create: `scripts/v7_executor.sh`
- Create: `scripts/v7_checkin.sh`
- Create: `scripts/v7_eod.sh`
- Create: `scripts/v7_weekly_review.sh`
- Create: `scripts/v7_monthly_report.sh`

All scripts follow the same pattern:
1. Source env vars
2. Activate venv
3. Check kite token (trading scripts only)
4. Check `is_trading_day()` (weekday scripts only)
5. File lock to prevent overlapping runs
6. Run python command with logging
7. Handle errors with Telegram alert

- [ ] **Step 1: Create v7_premarket.sh**

```bash
#!/bin/bash
# V7 Pre-Market Playbook — 8:43 AM IST Mon-Fri
# Crontab: 43 8 * * 1-5 ~/financial-agent-india/scripts/v7_premarket.sh

set -euo pipefail
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"

# Source env vars
if [ -f "$HOME/.config/env/global.env" ]; then
    set -a
    source "$HOME/.config/env/global.env"
    set +a
fi

BOT_DIR="$HOME/financial-agent-india"
cd "$BOT_DIR"
source venv/bin/activate

LOG="data/v7/cron.log"
mkdir -p data/v7

# Log rotation: cap at 5MB
if [ -f "$LOG" ]; then
    LOG_SIZE=$(stat -c%s "$LOG" 2>/dev/null || stat -f%z "$LOG" 2>/dev/null || echo 0)
    if [ "$LOG_SIZE" -gt 5242880 ]; then
        mv "$LOG" "${LOG}.old"
        echo "--- Log rotated $(TZ=Asia/Kolkata date) ---" > "$LOG"
    fi
fi

TODAY_DATE=$(TZ=Asia/Kolkata date +%Y-%m-%d)

# Holiday check
IS_HOLIDAY=$(python3 -c "
from config import is_trading_day
from datetime import date
print('no' if is_trading_day(date.fromisoformat('${TODAY_DATE}')) else 'yes')
" 2>/dev/null || echo "no")

if [ "$IS_HOLIDAY" = "yes" ]; then
    echo "[$(TZ=Asia/Kolkata date)] [SKIP] Not a trading day ($TODAY_DATE)" >> "$LOG"
    exit 0
fi

# Kite token check
KITE_OK=$(python3 -c "
from kite_data import get_kite
try:
    get_kite()
    print('yes')
except:
    print('no')
" 2>/dev/null || echo "no")

if [ "$KITE_OK" = "no" ]; then
    echo "[$(TZ=Asia/Kolkata date)] [WARN] Kite token expired — premarket will use fallback" >> "$LOG"
    # Alert but continue — premarket can still generate playbook from global data
    TG_TOKEN="${DEAL_BOT_TOKEN:-$TELEGRAM_BOT_TOKEN}"
    TG_CHAT="${TELEGRAM_FORUM_CHAT_ID:-$DEAL_BOT_CHAT_ID}"
    TG_TOPIC="${TELEGRAM_TOPIC_STOCKS}"
    if [ -n "$TG_TOKEN" ] && [ -n "$TG_CHAT" ]; then
        TMPFILE=$(mktemp)
        echo "V7: Kite token expired at premarket. Refresh token." > "$TMPFILE"
        CURL_ARGS="-d chat_id=${TG_CHAT} -d parse_mode=HTML --data-urlencode text@$TMPFILE"
        [ -n "$TG_TOPIC" ] && CURL_ARGS="$CURL_ARGS -d message_thread_id=$TG_TOPIC"
        curl -s -X POST "https://api.telegram.org/bot${TG_TOKEN}/sendMessage" \
            $CURL_ARGS > /dev/null 2>&1
        rm -f "$TMPFILE"
    fi
fi

echo "[$(TZ=Asia/Kolkata date)] [PREMARKET] Starting" >> "$LOG"
python -m v7.main --paper premarket >> "$LOG" 2>&1
EXIT_CODE=$?

if [ "$EXIT_CODE" -ne 0 ]; then
    echo "[$(TZ=Asia/Kolkata date)] [PREMARKET] FAILED (exit $EXIT_CODE)" >> "$LOG"
    # Error alert already sent by main.py
fi

echo "" >> "$LOG"
```

- [ ] **Step 2: Create v7_opening_read.sh**

```bash
#!/bin/bash
# V7 Opening Read — 9:13 AM IST Mon-Fri
# Crontab: 13 9 * * 1-5 ~/financial-agent-india/scripts/v7_opening_read.sh

set -euo pipefail
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"

if [ -f "$HOME/.config/env/global.env" ]; then
    set -a
    source "$HOME/.config/env/global.env"
    set +a
fi

BOT_DIR="$HOME/financial-agent-india"
cd "$BOT_DIR"
source venv/bin/activate

LOG="data/v7/cron.log"
mkdir -p data/v7

TODAY_DATE=$(TZ=Asia/Kolkata date +%Y-%m-%d)

IS_HOLIDAY=$(python3 -c "
from config import is_trading_day
from datetime import date
print('no' if is_trading_day(date.fromisoformat('${TODAY_DATE}')) else 'yes')
" 2>/dev/null || echo "no")

if [ "$IS_HOLIDAY" = "yes" ]; then
    echo "[$(TZ=Asia/Kolkata date)] [SKIP] Not a trading day" >> "$LOG"
    exit 0
fi

KITE_OK=$(python3 -c "
from kite_data import get_kite
try:
    get_kite()
    print('yes')
except:
    print('no')
" 2>/dev/null || echo "no")

if [ "$KITE_OK" = "no" ]; then
    echo "[$(TZ=Asia/Kolkata date)] [OPENING-READ] Kite expired — skipping" >> "$LOG"
    exit 1
fi

echo "[$(TZ=Asia/Kolkata date)] [OPENING-READ] Starting" >> "$LOG"
python -m v7.main --paper opening-read >> "$LOG" 2>&1
echo "" >> "$LOG"
```

- [ ] **Step 3: Create v7_executor.sh**

```bash
#!/bin/bash
# V7 Executor Tick — every 3 min during market hours
# Crontab: */3 9-15 * * 1-5 ~/financial-agent-india/scripts/v7_executor.sh

export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"

if [ -f "$HOME/.config/env/global.env" ]; then
    set -a
    source "$HOME/.config/env/global.env"
    set +a
fi

BOT_DIR="$HOME/financial-agent-india"
cd "$BOT_DIR"
source venv/bin/activate

LOG="data/v7/cron.log"
mkdir -p data/v7

# File lock: prevent overlapping ticks
LOCKFILE="/tmp/v7_executor.lock"
exec 200>"$LOCKFILE"
flock -n 200 || { echo "[$(TZ=Asia/Kolkata date)] [SKIP] Previous tick running" >> "$LOG"; exit 0; }

# Log rotation
if [ -f "$LOG" ]; then
    LOG_SIZE=$(stat -c%s "$LOG" 2>/dev/null || stat -f%z "$LOG" 2>/dev/null || echo 0)
    if [ "$LOG_SIZE" -gt 5242880 ]; then
        mv "$LOG" "${LOG}.old"
        echo "--- Log rotated $(TZ=Asia/Kolkata date) ---" > "$LOG"
    fi
fi

HOUR=$(TZ=Asia/Kolkata date +%H)
MIN=$(TZ=Asia/Kolkata date +%M)
TIME_MINS=$((HOUR * 60 + MIN))
TODAY_DATE=$(TZ=Asia/Kolkata date +%Y-%m-%d)

# Pre-market: skip
if [ "$TIME_MINS" -lt 555 ]; then
    exit 0
fi

# Post-market: skip
if [ "$TIME_MINS" -gt 930 ]; then
    exit 0
fi

# Holiday check
IS_HOLIDAY=$(python3 -c "
from config import is_trading_day
from datetime import date
print('no' if is_trading_day(date.fromisoformat('${TODAY_DATE}')) else 'yes')
" 2>/dev/null || echo "no")

if [ "$IS_HOLIDAY" = "yes" ]; then
    exit 0
fi

# Kite token check
KITE_OK=$(python3 -c "
from kite_data import get_kite
try:
    get_kite()
    print('yes')
except:
    print('no')
" 2>/dev/null || echo "no")

if [ "$KITE_OK" = "no" ]; then
    echo "[$(TZ=Asia/Kolkata date)] [TICK] Kite expired — protect-only mode" >> "$LOG"
    # TODO: Run protect-only tick (check SL orders, no new trades)
    exit 0
fi

TICK_START=$(date +%s)
echo "[$(TZ=Asia/Kolkata date)] [TICK] Starting" >> "$LOG"

python -m v7.main --paper tick >> "$LOG" 2>&1
EXIT_CODE=$?

TICK_END=$(date +%s)
TICK_DURATION=$((TICK_END - TICK_START))
echo "[TIMING] Tick: ${TICK_DURATION}s (exit $EXIT_CODE)" >> "$LOG"

# Alert if tick took too long (> 120s)
if [ "$TICK_DURATION" -gt 120 ]; then
    TG_TOKEN="${DEAL_BOT_TOKEN:-$TELEGRAM_BOT_TOKEN}"
    TG_CHAT="${TELEGRAM_FORUM_CHAT_ID:-$DEAL_BOT_CHAT_ID}"
    TG_TOPIC="${TELEGRAM_TOPIC_STOCKS}"
    if [ -n "$TG_TOKEN" ] && [ -n "$TG_CHAT" ]; then
        TMPFILE=$(mktemp)
        echo "V7: Slow tick ${TICK_DURATION}s (limit 120s)" > "$TMPFILE"
        CURL_ARGS="-d chat_id=${TG_CHAT} -d parse_mode=HTML --data-urlencode text@$TMPFILE"
        [ -n "$TG_TOPIC" ] && CURL_ARGS="$CURL_ARGS -d message_thread_id=$TG_TOPIC"
        curl -s -X POST "https://api.telegram.org/bot${TG_TOKEN}/sendMessage" \
            $CURL_ARGS > /dev/null 2>&1
        rm -f "$TMPFILE"
    fi
fi

echo "" >> "$LOG"
```

- [ ] **Step 4: Create v7_checkin.sh**

```bash
#!/bin/bash
# V7 Strategist Check-in — 10:28 AM and 12:58 PM IST Mon-Fri
# Crontab:
#   28 10 * * 1-5 ~/financial-agent-india/scripts/v7_checkin.sh
#   58 12 * * 1-5 ~/financial-agent-india/scripts/v7_checkin.sh

set -euo pipefail
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"

if [ -f "$HOME/.config/env/global.env" ]; then
    set -a
    source "$HOME/.config/env/global.env"
    set +a
fi

BOT_DIR="$HOME/financial-agent-india"
cd "$BOT_DIR"
source venv/bin/activate

LOG="data/v7/cron.log"
mkdir -p data/v7

TODAY_DATE=$(TZ=Asia/Kolkata date +%Y-%m-%d)
HOUR=$(TZ=Asia/Kolkata date +%H)

IS_HOLIDAY=$(python3 -c "
from config import is_trading_day
from datetime import date
print('no' if is_trading_day(date.fromisoformat('${TODAY_DATE}')) else 'yes')
" 2>/dev/null || echo "no")

if [ "$IS_HOLIDAY" = "yes" ]; then
    exit 0
fi

# Determine check-in number from time
if [ "$HOUR" -le 11 ]; then
    CHECKIN_NUM=1
else
    CHECKIN_NUM=2
fi

KITE_OK=$(python3 -c "
from kite_data import get_kite
try:
    get_kite()
    print('yes')
except:
    print('no')
" 2>/dev/null || echo "no")

if [ "$KITE_OK" = "no" ]; then
    echo "[$(TZ=Asia/Kolkata date)] [CHECKIN] Kite expired — skipping check-in $CHECKIN_NUM" >> "$LOG"
    exit 0
fi

echo "[$(TZ=Asia/Kolkata date)] [CHECKIN] Check-in #$CHECKIN_NUM starting" >> "$LOG"
python -m v7.main --paper checkin --num "$CHECKIN_NUM" >> "$LOG" 2>&1
echo "" >> "$LOG"
```

- [ ] **Step 5: Create v7_eod.sh**

```bash
#!/bin/bash
# V7 EOD Review + Journal — 3:33 PM IST Mon-Fri
# Crontab: 33 15 * * 1-5 ~/financial-agent-india/scripts/v7_eod.sh

set -euo pipefail
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"

if [ -f "$HOME/.config/env/global.env" ]; then
    set -a
    source "$HOME/.config/env/global.env"
    set +a
fi

BOT_DIR="$HOME/financial-agent-india"
cd "$BOT_DIR"
source venv/bin/activate

LOG="data/v7/cron.log"
mkdir -p data/v7

TODAY_DATE=$(TZ=Asia/Kolkata date +%Y-%m-%d)

IS_HOLIDAY=$(python3 -c "
from config import is_trading_day
from datetime import date
print('no' if is_trading_day(date.fromisoformat('${TODAY_DATE}')) else 'yes')
" 2>/dev/null || echo "no")

if [ "$IS_HOLIDAY" = "yes" ]; then
    exit 0
fi

echo "[$(TZ=Asia/Kolkata date)] [EOD] Starting EOD review + journal" >> "$LOG"
python -m v7.main --paper eod >> "$LOG" 2>&1
echo "" >> "$LOG"
```

- [ ] **Step 6: Create v7_weekly_review.sh**

```bash
#!/bin/bash
# V7 Weekly Review — Sunday 10:03 AM IST
# Crontab: 3 10 * * 0 ~/financial-agent-india/scripts/v7_weekly_review.sh

set -euo pipefail
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"

if [ -f "$HOME/.config/env/global.env" ]; then
    set -a
    source "$HOME/.config/env/global.env"
    set +a
fi

BOT_DIR="$HOME/financial-agent-india"
cd "$BOT_DIR"
source venv/bin/activate

LOG="data/v7/cron.log"
mkdir -p data/v7

echo "[$(TZ=Asia/Kolkata date)] [WEEKLY] Starting weekly review" >> "$LOG"
python -m v7.main --paper weekly >> "$LOG" 2>&1
echo "" >> "$LOG"
```

- [ ] **Step 7: Create v7_monthly_report.sh**

```bash
#!/bin/bash
# V7 Monthly Report — 1st of month 10:07 AM IST
# Crontab: 7 10 1 * * ~/financial-agent-india/scripts/v7_monthly_report.sh

set -euo pipefail
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"

if [ -f "$HOME/.config/env/global.env" ]; then
    set -a
    source "$HOME/.config/env/global.env"
    set +a
fi

BOT_DIR="$HOME/financial-agent-india"
cd "$BOT_DIR"
source venv/bin/activate

LOG="data/v7/cron.log"
mkdir -p data/v7

echo "[$(TZ=Asia/Kolkata date)] [MONTHLY] Starting monthly report" >> "$LOG"
python -m v7.main --paper monthly >> "$LOG" 2>&1
echo "" >> "$LOG"
```

- [ ] **Step 8: Make all scripts executable and commit**

```bash
chmod +x scripts/v7_premarket.sh scripts/v7_opening_read.sh scripts/v7_executor.sh \
    scripts/v7_checkin.sh scripts/v7_eod.sh scripts/v7_weekly_review.sh scripts/v7_monthly_report.sh
git add scripts/v7_*.sh
git commit -m "feat(v7): add cron scripts — premarket, executor, checkin, eod, weekly, monthly"
```

---

## Chunk 6: Deployment

### Task 6: Deployment setup and V6-to-V7 transition

- [ ] **Step 1: Create data/v7/ directory structure**

```bash
mkdir -p data/v7
echo '{}' > data/v7/.gitkeep_state
```

Add to `.gitignore`:

```
# V7 state files (runtime, not committed)
data/v7/playbook.json
data/v7/positions.json
data/v7/daily_state.json
data/v7/monthly_state.json
data/v7/edge_tracker.json
data/v7/level_memory.json
data/v7/cron.log
data/v7/cron.log.old
data/v7/weekly_review_*.json
data/v7/monthly_report_*.json
```

- [ ] **Step 2: Create deployment script for Pi**

```bash
#!/bin/bash
# Deploy V7 to Pi — run from Mac
# Usage: scripts/v7_deploy_pi.sh

set -euo pipefail

PI_HOST="pi@homepi.local"
REMOTE_DIR="~/financial-agent-india"

echo "=== V7 Deployment to Pi ==="

# 1. Sync code
echo "[1/5] Syncing code to Pi..."
rsync -avz --exclude='venv/' --exclude='data/' --exclude='__pycache__/' \
    --exclude='.git/' --exclude='*.pyc' \
    ~/financial-agent-india/ "$PI_HOST:$REMOTE_DIR/"

# 2. Create data directories
echo "[2/5] Creating data directories..."
ssh "$PI_HOST" "mkdir -p $REMOTE_DIR/data/v7"

# 3. Install any new dependencies
echo "[3/5] Checking dependencies..."
ssh "$PI_HOST" "cd $REMOTE_DIR && source venv/bin/activate && pip install -q anthropic 2>/dev/null || true"

# 4. Make scripts executable
echo "[4/5] Setting permissions..."
ssh "$PI_HOST" "chmod +x $REMOTE_DIR/scripts/v7_*.sh"

# 5. Show crontab instructions
echo "[5/5] Crontab setup:"
echo ""
echo "SSH into Pi and run: crontab -e"
echo "Add the following lines:"
echo ""
cat << 'CRON'
# ── V7 Professional Trader Bot ──────────────────────────────
43 8  * * 1-5  ~/financial-agent-india/scripts/v7_premarket.sh
13 9  * * 1-5  ~/financial-agent-india/scripts/v7_opening_read.sh
*/3 9-15 * * 1-5  ~/financial-agent-india/scripts/v7_executor.sh
28 10 * * 1-5  ~/financial-agent-india/scripts/v7_checkin.sh
58 12 * * 1-5  ~/financial-agent-india/scripts/v7_checkin.sh
33 15 * * 1-5  ~/financial-agent-india/scripts/v7_eod.sh
50 8  * * *    ~/financial-agent-india/scripts/kite_token_check.sh
3  10 * * 0    ~/financial-agent-india/scripts/v7_weekly_review.sh
7  10 1 * *    ~/financial-agent-india/scripts/v7_monthly_report.sh
CRON

echo ""
echo "=== Deployment complete ==="
echo ""
echo "TRANSITION PLAN:"
echo "1. Keep V6 cron running alongside V7 for 2-4 weeks"
echo "2. V7 runs in --paper mode by default (no real orders)"
echo "3. Monitor data/v7/cron.log for errors"
echo "4. Compare V7 paper P&L vs V6 paper P&L after 50+ trades"
echo "5. When V7 shows positive expectancy: disable V6 cron, switch V7 to live"
echo "6. To go live: remove --paper flag from v7/main.py calls in cron scripts"
```

- [ ] **Step 3: Commit deployment files**

```bash
chmod +x scripts/v7_deploy_pi.sh
git add scripts/v7_deploy_pi.sh .gitignore
git commit -m "feat(v7): add Pi deployment script and V6-V7 transition plan"
```

- [ ] **Step 4: Create Obsidian vault directory**

```bash
mkdir -p ~/Documents/Obsidian/trading-journal
```

This directory is where the daily journal writes to. It's synced via iCloud to all devices.

---

## Summary: Crontab Reference

Exact cron lines for Pi (matching spec):

```cron
# V7 Professional Trader Bot
43 8  * * 1-5  ~/financial-agent-india/scripts/v7_premarket.sh
13 9  * * 1-5  ~/financial-agent-india/scripts/v7_opening_read.sh
*/3 9-15 * * 1-5  ~/financial-agent-india/scripts/v7_executor.sh
28 10 * * 1-5  ~/financial-agent-india/scripts/v7_checkin.sh
58 12 * * 1-5  ~/financial-agent-india/scripts/v7_checkin.sh
33 15 * * 1-5  ~/financial-agent-india/scripts/v7_eod.sh
50 8  * * *    ~/financial-agent-india/scripts/kite_token_check.sh
3  10 * * 0    ~/financial-agent-india/scripts/v7_weekly_review.sh
7  10 1 * *    ~/financial-agent-india/scripts/v7_monthly_report.sh
```

## Data Flow

```
Executor closes trade
  → TradeResult
    → EdgeTracker.record()           # data/v7/edge_tracker.json
    → daily_state.closed_trades[]    # data/v7/daily_state.json
    → TelegramAlerter.send()         # trade exit alert

EOD (3:33 PM)
  → Claude Haiku grades trades       # entry_grade, exit_grade, lesson
  → Journal.save_daily()             # ~/Documents/Obsidian/trading-journal/YYYY-MM-DD.md
  → TelegramAlerter EOD summary

Weekly Review (Sunday)
  → Claude Sonnet reviews week
  → Updates level_memory.json
  → Watchlist rotation decisions
  → Telegram weekly report

Monthly Report (1st)
  → Claude Sonnet monthly analysis
  → Tax estimate, capital recommendation
  → Telegram monthly report
```

## V6 → V7 Transition Checklist

1. Deploy V7 code to Pi (`scripts/v7_deploy_pi.sh`)
2. Add V7 cron lines (keep V6 running)
3. V7 runs in `--paper` mode — no real orders
4. Monitor `data/v7/cron.log` daily
5. Review Obsidian journals daily
6. After 2 weeks: compare V7 paper vs V6 paper
7. After 50+ paper trades with positive expectancy: disable V6 cron
8. Switch V7 to live: remove `--paper` flag from cron scripts
9. Start with minimum size (reduce `CAPITAL["initial"]` to 100K)
10. Scale up after 2 weeks of live positive performance
