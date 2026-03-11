# V7 Plan 2: Strategist, Risk Engine & Level Memory

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Claude-powered Strategist that generates and adapts playbooks, the pure-rules Risk Engine that gates every trade, and the persistent Level Memory that tracks key price levels across sessions.

**Architecture:** V7 code lives in `v7/` package. These modules import shared types from `v7/types.py`, persist state via `StateManager` from `v7/state.py`, and read config from `v7/config_v7.py` (all from Plan 1). The Strategist calls Claude via `ClaudeCLIClient` from `config.py`. The Risk Engine and Level Memory are pure Python with no external API calls.

**Tech Stack:** Python 3.13, Claude API (via ClaudeCLIClient or Anthropic SDK), JSON file persistence, pytest

**Spec:** `docs/superpowers/specs/2026-03-11-v7-professional-trader-bot-design.md`

**Depends on:** Plan 1 (Core Infrastructure — types, state, config, data feed)
**Blocks:** Plan 3 (Executor), Plan 4 (Journal & Edge Tracker)

---

## File Structure

```
v7/
├── strategist.py           # Claude-powered playbook generation & adaptation
├── risk_engine.py           # Pure-rules risk gating, sizing, limits
├── level_memory.py          # Persistent key levels & OI walls
tests/
├── test_v7_strategist.py
├── test_v7_risk_engine.py
├── test_v7_level_memory.py
```

---

## Chunk 1: Level Memory

Level Memory has no external dependencies beyond Plan 1 types and StateManager, so it ships first. Both Strategist and Risk Engine read from it.

### Task 1: Level Memory

**Files:**
- Create: `v7/level_memory.py`
- Test: `tests/test_v7_level_memory.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_v7_level_memory.py
"""Tests for V7 Level Memory — persistent key levels and OI walls."""
import json
import pytest
from datetime import date
from pathlib import Path
from v7.level_memory import LevelMemory, Level


@pytest.fixture
def tmp_state_dir(tmp_path):
    return tmp_path / "v7_state"


@pytest.fixture
def lm(tmp_state_dir):
    return LevelMemory(state_dir=tmp_state_dir)


# ── Level CRUD ──────────────────────────────────────────────────────────


def test_add_level(lm):
    lm.add_level(
        symbol="NIFTY", price=24000.0, level_type="support",
        source="tested 3x in last 5 sessions",
    )
    levels = lm.get_levels("NIFTY")
    assert len(levels) == 1
    assert levels[0].price == 24000.0
    assert levels[0].level_type == "support"
    assert levels[0].strength == 1


def test_add_level_deduplicates_nearby(lm):
    """Adding a level within 0.1% of existing level strengthens it instead."""
    lm.add_level("NIFTY", 24000.0, "support", "source1")
    lm.add_level("NIFTY", 24010.0, "support", "source2")  # within 0.1% of 24000
    levels = lm.get_levels("NIFTY")
    assert len(levels) == 1
    assert levels[0].strength == 2  # strengthened


def test_add_level_different_price_creates_new(lm):
    lm.add_level("NIFTY", 24000.0, "support", "source1")
    lm.add_level("NIFTY", 24500.0, "resistance", "source2")
    levels = lm.get_levels("NIFTY")
    assert len(levels) == 2


def test_get_levels_empty_symbol(lm):
    assert lm.get_levels("UNKNOWN") == []


# ── Level Maintenance ───────────────────────────────────────────────────


def test_strengthen_on_retest(lm):
    lm.add_level("NIFTY", 24000.0, "support", "initial")
    lm.retest_level("NIFTY", 24000.0, held=True)
    levels = lm.get_levels("NIFTY")
    assert levels[0].strength == 2
    assert levels[0].last_tested == str(date.today())


def test_weaken_on_break(lm):
    lm.add_level("NIFTY", 24000.0, "support", "initial")
    lm.retest_level("NIFTY", 24000.0, held=True)  # strength=2
    lm.retest_level("NIFTY", 24000.0, held=False)  # strength=1
    levels = lm.get_levels("NIFTY")
    assert levels[0].strength == 1


def test_remove_on_strength_zero(lm):
    lm.add_level("NIFTY", 24000.0, "support", "initial")  # strength=1
    lm.retest_level("NIFTY", 24000.0, held=False)  # strength=0 → removed
    levels = lm.get_levels("NIFTY")
    assert len(levels) == 0


def test_flip_level(lm):
    lm.add_level("NIFTY", 24000.0, "resistance", "OI wall")
    lm.flip_level("NIFTY", 24000.0)
    levels = lm.get_levels("NIFTY")
    assert levels[0].level_type == "support"
    assert levels[0].strength == 1  # reset to 1 on flip


def test_flip_support_to_resistance(lm):
    lm.add_level("NIFTY", 24000.0, "support", "tested")
    lm.flip_level("NIFTY", 24000.0)
    levels = lm.get_levels("NIFTY")
    assert levels[0].level_type == "resistance"


# ── Staleness ───────────────────────────────────────────────────────────


def test_remove_stale_levels(lm):
    lm.add_level("NIFTY", 24000.0, "support", "old level")
    # Manually set last_tested to 15 days ago
    lm._data["NIFTY"]["levels"][0]["last_tested"] = "2026-02-20"
    lm.remove_stale(max_age_days=10, today=date(2026, 3, 11))
    assert len(lm.get_levels("NIFTY")) == 0


def test_keep_fresh_levels(lm):
    lm.add_level("NIFTY", 24000.0, "support", "recent level")
    lm.remove_stale(max_age_days=10, today=date.today())
    assert len(lm.get_levels("NIFTY")) == 1


# ── OI Walls ────────────────────────────────────────────────────────────


def test_update_oi_walls(lm):
    lm.update_oi_walls("NIFTY", call_max_oi_strike=24500, put_max_oi_strike=24000, pcr=1.1)
    walls = lm.get_oi_walls("NIFTY")
    assert walls["call_max_oi_strike"] == 24500
    assert walls["put_max_oi_strike"] == 24000
    assert walls["pcr"] == 1.1


def test_get_oi_walls_empty(lm):
    walls = lm.get_oi_walls("UNKNOWN")
    assert walls == {}


# ── Persistence ─────────────────────────────────────────────────────────


def test_save_and_reload(tmp_state_dir):
    lm1 = LevelMemory(state_dir=tmp_state_dir)
    lm1.add_level("NIFTY", 24000.0, "support", "tested")
    lm1.update_oi_walls("NIFTY", 24500, 24000, 1.1)
    lm1.save()

    lm2 = LevelMemory(state_dir=tmp_state_dir)
    levels = lm2.get_levels("NIFTY")
    assert len(levels) == 1
    assert levels[0].price == 24000.0
    walls = lm2.get_oi_walls("NIFTY")
    assert walls["call_max_oi_strike"] == 24500


def test_to_strategist_context(lm):
    lm.add_level("NIFTY", 24000.0, "support", "tested 3x")
    lm.add_level("NIFTY", 24500.0, "resistance", "OI wall")
    lm.update_oi_walls("NIFTY", 24500, 24000, 1.1)
    ctx = lm.to_strategist_context(["NIFTY"])
    assert "NIFTY" in ctx
    assert len(ctx["NIFTY"]["levels"]) == 2
    assert "oi_walls" in ctx["NIFTY"]


# ── Level dataclass ─────────────────────────────────────────────────────


def test_level_to_dict_roundtrip():
    lv = Level(
        price=24000.0, level_type="support", strength=3,
        source="tested 3x", last_tested="2026-03-10", created="2026-03-05",
    )
    d = lv.to_dict()
    lv2 = Level.from_dict(d)
    assert lv2.price == lv.price
    assert lv2.strength == lv.strength
    assert lv2.level_type == lv.level_type


def test_level_near():
    lv = Level(price=24000.0, level_type="support", strength=1,
               source="x", last_tested="2026-03-10", created="2026-03-10")
    assert lv.is_near(24020.0, threshold_pct=0.1) is True   # 0.08% away
    assert lv.is_near(24100.0, threshold_pct=0.1) is False  # 0.42% away
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_level_memory.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'v7.level_memory'`

- [ ] **Step 3: Implement Level Memory**

```python
# v7/level_memory.py
"""Persistent key level tracking for V7.

Stores support/resistance levels per symbol with strength scoring.
Levels are strengthened on retests that hold, weakened on breaks,
flipped on clean breaks (resistance → support and vice versa),
and pruned when stale (not tested for N sessions).

Backed by data/v7/level_memory.json via StateManager.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


@dataclass
class Level:
    """A single key price level."""
    price: float
    level_type: str        # "support" or "resistance"
    strength: int          # starts at 1, +1 on retest hold, -1 on break, removed at 0
    source: str            # human-readable origin
    last_tested: str       # ISO date
    created: str           # ISO date

    def to_dict(self) -> dict:
        return {
            "price": self.price,
            "type": self.level_type,
            "strength": self.strength,
            "source": self.source,
            "last_tested": self.last_tested,
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Level:
        return cls(
            price=d["price"],
            level_type=d["type"],
            strength=d["strength"],
            source=d["source"],
            last_tested=d["last_tested"],
            created=d.get("created", d["last_tested"]),
        )

    def is_near(self, price: float, threshold_pct: float = 0.1) -> bool:
        """Check if price is within threshold_pct% of this level."""
        if self.price == 0:
            return False
        return abs(price - self.price) / self.price * 100 <= threshold_pct


class LevelMemory:
    """Persistent store for key levels and OI walls.

    Data structure:
    {
      "NIFTY": {
        "levels": [Level.to_dict(), ...],
        "oi_walls": {"call_max_oi_strike": 24500, "put_max_oi_strike": 24000, "pcr": 1.1}
      }
    }
    """

    def __init__(self, state_dir: str | Path):
        self._dir = Path(state_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "level_memory.json"
        self._data: dict = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            with open(self._path) as f:
                self._data = json.load(f)
        else:
            self._data = {}

    def save(self) -> None:
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

    # ── Level CRUD ──────────────────────────────────────────────────

    def _ensure_symbol(self, symbol: str) -> None:
        if symbol not in self._data:
            self._data[symbol] = {"levels": [], "oi_walls": {}}

    def add_level(
        self, symbol: str, price: float, level_type: str, source: str,
        today: date | None = None,
    ) -> None:
        """Add a new level or strengthen existing nearby level."""
        today_str = str(today or date.today())
        self._ensure_symbol(symbol)

        # Check for nearby existing level (within 0.1%)
        for lv_dict in self._data[symbol]["levels"]:
            lv = Level.from_dict(lv_dict)
            if lv.is_near(price) and lv.level_type == level_type:
                lv_dict["strength"] += 1
                lv_dict["last_tested"] = today_str
                lv_dict["source"] = f"{lv_dict['source']}; {source}"
                self.save()
                return

        # New level
        new_level = Level(
            price=price, level_type=level_type, strength=1,
            source=source, last_tested=today_str, created=today_str,
        )
        self._data[symbol]["levels"].append(new_level.to_dict())
        self.save()

    def get_levels(self, symbol: str) -> list[Level]:
        """Get all levels for a symbol, sorted by price."""
        if symbol not in self._data:
            return []
        levels = [Level.from_dict(d) for d in self._data[symbol]["levels"]]
        return sorted(levels, key=lambda lv: lv.price)

    def retest_level(
        self, symbol: str, price: float, held: bool,
        today: date | None = None,
    ) -> None:
        """Update level after a retest. Strengthen if held, weaken if broken."""
        today_str = str(today or date.today())
        if symbol not in self._data:
            return

        to_remove = []
        for i, lv_dict in enumerate(self._data[symbol]["levels"]):
            lv = Level.from_dict(lv_dict)
            if lv.is_near(price):
                if held:
                    lv_dict["strength"] += 1
                    lv_dict["last_tested"] = today_str
                else:
                    lv_dict["strength"] -= 1
                    lv_dict["last_tested"] = today_str
                    if lv_dict["strength"] <= 0:
                        to_remove.append(i)

        for i in reversed(to_remove):
            self._data[symbol]["levels"].pop(i)
        self.save()

    def flip_level(self, symbol: str, price: float, today: date | None = None) -> None:
        """Flip a level: resistance → support or support → resistance. Resets strength to 1."""
        today_str = str(today or date.today())
        if symbol not in self._data:
            return

        for lv_dict in self._data[symbol]["levels"]:
            lv = Level.from_dict(lv_dict)
            if lv.is_near(price):
                if lv_dict["type"] == "support":
                    lv_dict["type"] = "resistance"
                else:
                    lv_dict["type"] = "support"
                lv_dict["strength"] = 1
                lv_dict["last_tested"] = today_str
                break
        self.save()

    def remove_stale(self, max_age_days: int = 10, today: date | None = None) -> int:
        """Remove levels not tested within max_age_days. Returns count removed."""
        today_dt = today or date.today()
        removed = 0

        for symbol in list(self._data.keys()):
            to_remove = []
            for i, lv_dict in enumerate(self._data[symbol]["levels"]):
                last_tested = date.fromisoformat(lv_dict["last_tested"])
                age = (today_dt - last_tested).days
                if age > max_age_days:
                    to_remove.append(i)

            for i in reversed(to_remove):
                self._data[symbol]["levels"].pop(i)
                removed += 1

        if removed:
            self.save()
        return removed

    # ── OI Walls ────────────────────────────────────────────────────

    def update_oi_walls(
        self, symbol: str,
        call_max_oi_strike: float, put_max_oi_strike: float, pcr: float,
    ) -> None:
        """Update OI wall data for a symbol."""
        self._ensure_symbol(symbol)
        self._data[symbol]["oi_walls"] = {
            "call_max_oi_strike": call_max_oi_strike,
            "put_max_oi_strike": put_max_oi_strike,
            "pcr": pcr,
        }
        self.save()

    def get_oi_walls(self, symbol: str) -> dict:
        if symbol not in self._data:
            return {}
        return self._data[symbol].get("oi_walls", {})

    # ── Strategist Interface ────────────────────────────────────────

    def to_strategist_context(self, symbols: list[str]) -> dict:
        """Return level data formatted for Strategist prompt injection."""
        ctx = {}
        for sym in symbols:
            if sym not in self._data:
                continue
            ctx[sym] = {
                "levels": [Level.from_dict(d).to_dict() for d in self._data[sym]["levels"]],
                "oi_walls": self._data[sym].get("oi_walls", {}),
            }
        return ctx

    def bulk_update(self, levels_dict: dict) -> None:
        """Replace all level data from a dict (e.g. from Strategist or weekly review).

        Merges with existing — new levels are added, existing levels with
        matching prices are updated.
        """
        for symbol, sym_data in levels_dict.items():
            self._ensure_symbol(symbol)
            for lv_dict in sym_data.get("levels", []):
                # Check if level already exists
                exists = False
                for existing in self._data[symbol]["levels"]:
                    existing_lv = Level.from_dict(existing)
                    if existing_lv.is_near(lv_dict["price"]):
                        existing["strength"] = max(existing["strength"], lv_dict.get("strength", 1))
                        existing["last_tested"] = lv_dict.get("last_tested", str(date.today()))
                        exists = True
                        break
                if not exists:
                    self._data[symbol]["levels"].append(lv_dict)

            if "oi_walls" in sym_data:
                self._data[symbol]["oi_walls"] = sym_data["oi_walls"]

        self.save()
```

- [ ] **Step 4: Run tests**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_level_memory.py -v`
Expected: All PASS (17 tests)

- [ ] **Step 5: Commit**

```bash
git add v7/level_memory.py tests/test_v7_level_memory.py
git commit -m "feat(v7): add Level Memory — persistent key levels and OI walls"
```

---

## Chunk 2: Risk Engine

Pure rules, no Claude. Gates every trade entry with sizing, daily limits, monthly pacing, survival mode, chop detection, F&O ban list, and brokerage optimization.

### Task 2: Risk Engine core — sizing and daily limits

**Files:**
- Create: `v7/risk_engine.py`
- Test: `tests/test_v7_risk_engine.py`

- [ ] **Step 1: Write failing tests for sizing and daily limits**

```python
# tests/test_v7_risk_engine.py
"""Tests for V7 Risk Engine — pure rules, no Claude."""
import json
import pytest
from datetime import date, time
from pathlib import Path
from v7.risk_engine import RiskEngine
from v7.types import (
    Conviction, Position, RiskBudget, PacingStatus, Setup, SetupType,
)


@pytest.fixture
def tmp_state_dir(tmp_path):
    return tmp_path / "v7_state"


@pytest.fixture
def engine(tmp_state_dir):
    return RiskEngine(capital=300_000, state_dir=tmp_state_dir)


# ── Per-Trade Sizing ────────────────────────────────────────────────────


def test_risk_amount_high_conviction(engine):
    amount = engine.risk_amount_for_conviction(Conviction.HIGH)
    assert amount == 6000.0  # 2% of 300k


def test_risk_amount_medium_conviction(engine):
    amount = engine.risk_amount_for_conviction(Conviction.MEDIUM)
    assert amount == 4500.0  # 1.5% of 300k


def test_risk_amount_low_conviction(engine):
    amount = engine.risk_amount_for_conviction(Conviction.LOW)
    assert amount == 2250.0  # 0.75% of 300k


def test_risk_amount_ahead_pacing_reduces_25pct(engine):
    engine.set_pacing(PacingStatus.AHEAD)
    amount = engine.risk_amount_for_conviction(Conviction.HIGH)
    assert amount == 4500.0  # 2% * 0.75 = 1.5% of 300k


def test_risk_amount_behind_pacing_no_increase(engine):
    engine.set_pacing(PacingStatus.BEHIND)
    amount = engine.risk_amount_for_conviction(Conviction.HIGH)
    assert amount == 6000.0  # no change — don't revenge trade


def test_lot_calculation(engine):
    lots = engine.calculate_lots(
        risk_amount=4500.0, premium=60.0, lot_size=75,
    )
    assert lots == 1  # 4500 / (60 * 75) = 1.0


def test_lot_calculation_rounds_down(engine):
    lots = engine.calculate_lots(
        risk_amount=4500.0, premium=25.0, lot_size=75,
    )
    assert lots == 2  # 4500 / (25 * 75) = 2.4 → 2


def test_lot_calculation_zero_if_too_expensive(engine):
    lots = engine.calculate_lots(
        risk_amount=4500.0, premium=100.0, lot_size=75,
    )
    assert lots == 0  # 4500 / (100 * 75) = 0.6 → 0


# ── Daily Limits ────────────────────────────────────────────────────────


def test_daily_loss_cap_blocks(engine):
    engine.record_daily_pnl(-6500.0)  # > 2% of 300k = 6000
    allowed, reason = engine.can_open_trade()
    assert allowed is False
    assert "daily loss" in reason.lower()


def test_daily_loss_under_cap_allows(engine):
    engine.record_daily_pnl(-5000.0)  # under 2%
    allowed, _ = engine.can_open_trade()
    assert allowed is True


def test_consecutive_sl_hits_blocks(engine):
    engine.record_sl_hit()
    engine.record_sl_hit()
    engine.record_sl_hit()  # 3rd SL hit
    allowed, reason = engine.can_open_trade()
    assert allowed is False
    assert "sl" in reason.lower() or "stop" in reason.lower()


def test_max_trades_blocks(engine):
    for _ in range(4):
        engine.record_trade_opened()
    allowed, reason = engine.can_open_trade()
    assert allowed is False
    assert "max trades" in reason.lower()


def test_margin_utilization_blocks(engine):
    engine.set_margin_used(0.75)  # 75% > 70% threshold
    allowed, reason = engine.can_open_trade()
    assert allowed is False
    assert "margin" in reason.lower()


def test_margin_under_threshold_allows(engine):
    engine.set_margin_used(0.60)
    allowed, _ = engine.can_open_trade()
    assert allowed is True


# ── Concurrent Risk Budget ──────────────────────────────────────────────


def test_concurrent_risk_allows(engine):
    # 4% of 300k = 12000 max concurrent risk
    can = engine.can_allocate_risk(4500.0, current_risk=0.0)
    assert can is True


def test_concurrent_risk_blocks_when_full(engine):
    can = engine.can_allocate_risk(4500.0, current_risk=10000.0)
    assert can is False  # 10000 + 4500 > 12000


def test_concurrent_risk_allows_after_close(engine):
    # Simulate: was at 9000, one trade closed freeing 4500, now at 4500
    can = engine.can_allocate_risk(4500.0, current_risk=4500.0)
    assert can is True  # 4500 + 4500 = 9000 < 12000


# ── Correlation Check ───────────────────────────────────────────────────


def test_correlation_blocks_third_same_direction(engine):
    positions = [
        Position(
            symbol="NIFTY", instrument="NIFTY CE", direction="bullish",
            entry_price=100, quantity=75, lot_size=75, allocated=7500,
            stoploss=80, target=150, entry_date=date(2026, 3, 11), setup_id="N1",
        ),
        Position(
            symbol="HDFCBANK", instrument="HDFCBANK CE", direction="bullish",
            entry_price=50, quantity=550, lot_size=550, allocated=27500,
            stoploss=40, target=70, entry_date=date(2026, 3, 11), setup_id="H1",
        ),
    ]
    blocked, reason = engine.check_correlation(positions, new_direction="bullish")
    assert blocked is True
    assert "correlation" in reason.lower()


def test_correlation_allows_different_direction(engine):
    positions = [
        Position(
            symbol="NIFTY", instrument="NIFTY CE", direction="bullish",
            entry_price=100, quantity=75, lot_size=75, allocated=7500,
            stoploss=80, target=150, entry_date=date(2026, 3, 11), setup_id="N1",
        ),
        Position(
            symbol="HDFCBANK", instrument="HDFCBANK CE", direction="bullish",
            entry_price=50, quantity=550, lot_size=550, allocated=27500,
            stoploss=40, target=70, entry_date=date(2026, 3, 11), setup_id="H1",
        ),
    ]
    blocked, _ = engine.check_correlation(positions, new_direction="bearish")
    assert blocked is False


# ── Daily Reset ─────────────────────────────────────────────────────────


def test_daily_reset(engine):
    engine.record_daily_pnl(-5000)
    engine.record_sl_hit()
    engine.record_sl_hit()
    engine.record_trade_opened()
    engine.reset_daily()
    allowed, _ = engine.can_open_trade()
    assert allowed is True
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_risk_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'v7.risk_engine'`

- [ ] **Step 3: Implement Risk Engine core**

```python
# v7/risk_engine.py
"""Risk Engine for V7 — pure rules, no Claude.

Gates every trade with:
- Per-trade sizing (conviction-based)
- Daily limits (loss cap, SL hits, trade count, margin)
- Concurrent risk budget (not cumulative)
- Monthly pacing and survival mode
- Correlation check
- Chop detection
- F&O ban list
- Brokerage optimization

All state is file-backed via JSON in state_dir.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from v7.types import Conviction, PacingStatus, Position

log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# ── Constants (from spec) ───────────────────────────────────────────────

DAILY_LOSS_CAP_PCT = 2.0           # block all new entries
MAX_CONSECUTIVE_SL = 3             # block + exception call
MAX_TRADES_PER_DAY = 4
MARGIN_UTILIZATION_BLOCK = 0.70    # 70%
MAX_CONCURRENT_RISK_PCT = 4.0      # concurrent, not cumulative
MAX_SAME_DIRECTION = 2             # correlation: max 2 same-direction
SURVIVAL_THRESHOLD_PCT = 5.0       # MTD drawdown → theta only
FULL_STOP_THRESHOLD_PCT = 8.0      # MTD drawdown → no trading
DRAWDOWN_REDUCE_PCT = 3.0          # MTD drawdown → reduce size 25%
MIN_TRADE_VALUE = 2000.0           # brokerage optimization
BROKERAGE_PER_ORDER = 20.0         # Zerodha flat fee

CONVICTION_RISK = {
    Conviction.HIGH: 2.0,          # A+ setup
    Conviction.MEDIUM: 1.5,        # B setup
    Conviction.LOW: 0.75,          # C setup
}


class RiskEngine:
    """Pure-rules risk gating engine.

    All state persists to state_dir/risk_state.json.
    Reloads on init — survives Pi reboots.
    """

    def __init__(self, capital: float, state_dir: str | Path):
        self._capital = capital
        self._dir = Path(state_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self._dir / "risk_state.json"

        # Daily counters (reset each morning)
        self._daily_pnl = 0.0
        self._sl_hits_today = 0
        self._trades_today = 0
        self._margin_used_pct = 0.0

        # Monthly state
        self._pacing = PacingStatus.ON_TRACK
        self._mtd_pnl = 0.0
        self._survival_mode = False
        self._full_stop = False

        # Chop detection
        self._whipsaw_count = 0
        self._opening_range_pct = 0.0
        self._first_hour_volume_ratio = 1.0

        # F&O ban list
        self._fo_ban_list: list[str] = []

        self._load_state()

    # ── State Persistence ───────────────────────────────────────────

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            with open(self._state_path) as f:
                data = json.load(f)
            today_str = str(date.today())
            if data.get("date") == today_str:
                self._daily_pnl = data.get("daily_pnl", 0.0)
                self._sl_hits_today = data.get("sl_hits_today", 0)
                self._trades_today = data.get("trades_today", 0)
                self._margin_used_pct = data.get("margin_used_pct", 0.0)
            # Monthly state persists across days
            self._mtd_pnl = data.get("mtd_pnl", 0.0)
            self._pacing = PacingStatus(data.get("pacing", "on_track"))
            self._survival_mode = data.get("survival_mode", False)
            self._full_stop = data.get("full_stop", False)
            self._fo_ban_list = data.get("fo_ban_list", [])
        except (json.JSONDecodeError, KeyError):
            log.warning("Risk state file corrupt — starting fresh")

    def _save_state(self) -> None:
        data = {
            "date": str(date.today()),
            "daily_pnl": self._daily_pnl,
            "sl_hits_today": self._sl_hits_today,
            "trades_today": self._trades_today,
            "margin_used_pct": self._margin_used_pct,
            "mtd_pnl": self._mtd_pnl,
            "pacing": self._pacing.value,
            "survival_mode": self._survival_mode,
            "full_stop": self._full_stop,
            "fo_ban_list": self._fo_ban_list,
        }
        with open(self._state_path, "w") as f:
            json.dump(data, f, indent=2)

    # ── Per-Trade Sizing ────────────────────────────────────────────

    def risk_amount_for_conviction(self, conviction: Conviction) -> float:
        """Calculate max risk in INR for a given conviction level."""
        base_pct = CONVICTION_RISK[conviction]
        if self._pacing == PacingStatus.AHEAD:
            base_pct *= 0.75  # protect gains in 2nd half
        if self._mtd_pnl < 0 and abs(self._mtd_pnl) / self._capital * 100 >= DRAWDOWN_REDUCE_PCT:
            base_pct *= 0.75  # reduce on 3%+ MTD drawdown
        return self._capital * (base_pct / 100)

    def calculate_lots(self, risk_amount: float, premium: float, lot_size: int) -> int:
        """Calculate number of lots affordable within risk budget."""
        if premium <= 0 or lot_size <= 0:
            return 0
        cost_per_lot = premium * lot_size
        return math.floor(risk_amount / cost_per_lot)

    # ── Daily Limits ────────────────────────────────────────────────

    def can_open_trade(self) -> tuple[bool, str]:
        """Master check: can we open a new trade right now?

        Returns (allowed, reason).
        """
        if self._full_stop:
            return False, "Full stop: MTD drawdown > 8%. No trading rest of month."

        if self._survival_mode:
            return False, "Survival mode: MTD drawdown > 5%. Theta only — no directional trades."

        # Daily loss cap
        daily_loss_limit = self._capital * (DAILY_LOSS_CAP_PCT / 100)
        if self._daily_pnl < 0 and abs(self._daily_pnl) >= daily_loss_limit:
            return False, f"Daily loss cap: ₹{abs(self._daily_pnl):.0f} exceeds {DAILY_LOSS_CAP_PCT}% (₹{daily_loss_limit:.0f})"

        # Consecutive SL hits
        if self._sl_hits_today >= MAX_CONSECUTIVE_SL:
            return False, f"3 SL hits today. No more trades."

        # Max trades per day
        if self._trades_today >= MAX_TRADES_PER_DAY:
            return False, f"Max trades ({MAX_TRADES_PER_DAY}) reached for today."

        # Margin utilization
        if self._margin_used_pct >= MARGIN_UTILIZATION_BLOCK:
            return False, f"Margin utilization {self._margin_used_pct*100:.0f}% exceeds {MARGIN_UTILIZATION_BLOCK*100:.0f}% threshold."

        return True, "OK"

    def can_open_theta(self) -> tuple[bool, str]:
        """Check if theta trades are allowed (separate from directional)."""
        if self._full_stop:
            return False, "Full stop: no trading rest of month."

        if self._margin_used_pct >= MARGIN_UTILIZATION_BLOCK:
            return False, f"Margin too high for theta: {self._margin_used_pct*100:.0f}%"

        return True, "OK"

    # ── Concurrent Risk Budget ──────────────────────────────────────

    def can_allocate_risk(self, new_risk: float, current_risk: float) -> bool:
        """Check if new_risk can be added to current concurrent risk.

        Risk budget is CONCURRENT not cumulative — when a trade closes,
        its risk is freed.
        """
        max_risk = self._capital * (MAX_CONCURRENT_RISK_PCT / 100)
        return (current_risk + new_risk) <= max_risk

    # ── Correlation Check ───────────────────────────────────────────

    def check_correlation(
        self, open_positions: list[Position], new_direction: str,
    ) -> tuple[bool, str]:
        """Block if 2+ positions already open in the same direction."""
        same_dir = sum(1 for p in open_positions if p.direction == new_direction)
        if same_dir >= MAX_SAME_DIRECTION:
            return True, f"Correlation block: {same_dir} positions already {new_direction}. Max {MAX_SAME_DIRECTION}."
        return False, ""

    # ── Monthly Pacing ──────────────────────────────────────────────

    def set_pacing(self, status: PacingStatus) -> None:
        self._pacing = status
        self._save_state()

    def update_mtd_pnl(self, mtd_pnl: float) -> None:
        """Update month-to-date P&L and evaluate survival/full-stop modes."""
        self._mtd_pnl = mtd_pnl
        mtd_dd_pct = abs(mtd_pnl) / self._capital * 100 if mtd_pnl < 0 else 0

        if mtd_dd_pct >= FULL_STOP_THRESHOLD_PCT:
            self._full_stop = True
            self._survival_mode = True
            log.critical(f"FULL STOP: MTD drawdown {mtd_dd_pct:.1f}% > {FULL_STOP_THRESHOLD_PCT}%")
        elif mtd_dd_pct >= SURVIVAL_THRESHOLD_PCT:
            self._survival_mode = True
            self._full_stop = False
            log.warning(f"SURVIVAL MODE: MTD drawdown {mtd_dd_pct:.1f}% > {SURVIVAL_THRESHOLD_PCT}%")
        elif mtd_dd_pct < DRAWDOWN_REDUCE_PCT:
            # Recovery: exit survival if drawdown recovers below 3%
            self._survival_mode = False
            self._full_stop = False

        self._save_state()

    @property
    def survival_mode(self) -> bool:
        return self._survival_mode

    @property
    def full_stop(self) -> bool:
        return self._full_stop

    # ── F&O Ban List ────────────────────────────────────────────────

    def update_fo_ban_list(self, symbols: list[str]) -> None:
        """Update the F&O ban list (fetched daily from NSE)."""
        self._fo_ban_list = symbols
        self._save_state()

    def is_fo_banned(self, symbol: str) -> bool:
        """Check if symbol is in F&O ban list."""
        return symbol.upper() in [s.upper() for s in self._fo_ban_list]

    # ── Chop Detection ──────────────────────────────────────────────

    def update_chop_signals(
        self, whipsaw_count: int, opening_range_pct: float,
        first_hour_volume_ratio: float,
    ) -> None:
        """Update chop detection signals from market data."""
        self._whipsaw_count = whipsaw_count
        self._opening_range_pct = opening_range_pct
        self._first_hour_volume_ratio = first_hour_volume_ratio

    def is_choppy(self) -> tuple[bool, str]:
        """Detect chop conditions. Returns (is_choppy, reason)."""
        reasons = []

        if self._whipsaw_count >= 3:
            reasons.append(f"{self._whipsaw_count} whipsaws in first hour")

        if 0 < self._opening_range_pct < 0.3:
            reasons.append(f"Narrow opening range: {self._opening_range_pct:.2f}%")

        if 0 < self._first_hour_volume_ratio < 0.5:
            reasons.append(f"Low volume: {self._first_hour_volume_ratio:.0%} of 20-day avg")

        if reasons:
            return True, "; ".join(reasons)
        return False, ""

    # ── Brokerage Optimization ──────────────────────────────────────

    def check_min_trade_value(self, trade_value: float) -> tuple[bool, str]:
        """Check if trade value is large enough for brokerage to be < 1%."""
        if trade_value < MIN_TRADE_VALUE:
            brokerage_pct = (BROKERAGE_PER_ORDER / trade_value * 100) if trade_value > 0 else 100
            return False, f"Trade value ₹{trade_value:.0f} too small. Brokerage would be {brokerage_pct:.1f}%. Min: ₹{MIN_TRADE_VALUE:.0f}"
        return True, ""

    # ── Recording Methods ───────────────────────────────────────────

    def record_daily_pnl(self, pnl: float) -> None:
        """Set daily P&L (absolute, not delta)."""
        self._daily_pnl = pnl
        self._save_state()

    def record_sl_hit(self) -> None:
        self._sl_hits_today += 1
        self._save_state()

    def record_trade_opened(self) -> None:
        self._trades_today += 1
        self._save_state()

    def set_margin_used(self, pct: float) -> None:
        """Set current margin utilization as a fraction (0.0 to 1.0)."""
        self._margin_used_pct = pct
        self._save_state()

    def reset_daily(self) -> None:
        """Reset daily counters. Called at start of each trading day."""
        self._daily_pnl = 0.0
        self._sl_hits_today = 0
        self._trades_today = 0
        self._margin_used_pct = 0.0
        self._whipsaw_count = 0
        self._opening_range_pct = 0.0
        self._first_hour_volume_ratio = 1.0
        self._save_state()

    def reset_monthly(self) -> None:
        """Reset monthly state. Called at start of each month."""
        self._mtd_pnl = 0.0
        self._pacing = PacingStatus.ON_TRACK
        self._survival_mode = False
        self._full_stop = False
        self._save_state()

    # ── Master Pre-Trade Check ──────────────────────────────────────

    def pre_trade_check(
        self, symbol: str, conviction: Conviction, direction: str,
        trade_value: float, open_positions: list[Position],
        current_risk: float,
    ) -> tuple[bool, str, float]:
        """Run all risk checks before opening a trade.

        Returns (allowed, reason, risk_amount).
        risk_amount is 0 if blocked.
        """
        # 1. Can open at all?
        allowed, reason = self.can_open_trade()
        if not allowed:
            return False, reason, 0.0

        # 2. F&O ban list
        if self.is_fo_banned(symbol):
            return False, f"{symbol} is in F&O ban list.", 0.0

        # 3. Brokerage min value
        ok, reason = self.check_min_trade_value(trade_value)
        if not ok:
            return False, reason, 0.0

        # 4. Correlation
        blocked, reason = self.check_correlation(open_positions, direction)
        if blocked:
            return False, reason, 0.0

        # 5. Risk budget
        risk_amount = self.risk_amount_for_conviction(conviction)
        if not self.can_allocate_risk(risk_amount, current_risk):
            max_risk = self._capital * (MAX_CONCURRENT_RISK_PCT / 100)
            return False, f"Risk budget full: current ₹{current_risk:.0f} + new ₹{risk_amount:.0f} > max ₹{max_risk:.0f}", 0.0

        # 6. Chop detection (warning, not blocking — Strategist decides)
        choppy, chop_reason = self.is_choppy()
        if choppy:
            log.warning(f"Chop detected: {chop_reason}")

        return True, "OK", risk_amount

    # ── State Summary (for Strategist prompt) ───────────────────────

    def get_state_summary(self) -> dict:
        """Return risk state for Strategist prompt injection."""
        return {
            "daily_pnl": self._daily_pnl,
            "sl_hits_today": self._sl_hits_today,
            "trades_today": self._trades_today,
            "margin_used_pct": self._margin_used_pct,
            "mtd_pnl": self._mtd_pnl,
            "mtd_pnl_pct": (self._mtd_pnl / self._capital * 100) if self._capital > 0 else 0,
            "pacing": self._pacing.value,
            "survival_mode": self._survival_mode,
            "full_stop": self._full_stop,
            "fo_ban_list": self._fo_ban_list,
        }
```

- [ ] **Step 4: Run tests**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_risk_engine.py -v`
Expected: All PASS (20 tests)

- [ ] **Step 5: Commit**

```bash
git add v7/risk_engine.py tests/test_v7_risk_engine.py
git commit -m "feat(v7): add Risk Engine — sizing, daily limits, pacing, chop, F&O ban"
```

---

### Task 3: Risk Engine — survival mode, monthly pacing, persistence tests

- [ ] **Step 1: Add tests for survival mode, monthly pacing, persistence**

Append to `tests/test_v7_risk_engine.py`:

```python
# ── Survival Mode ───────────────────────────────────────────────────────


def test_survival_mode_activates_at_5pct(engine):
    engine.update_mtd_pnl(-15000.0)  # 5% of 300k
    assert engine.survival_mode is True
    allowed, reason = engine.can_open_trade()
    assert allowed is False
    assert "survival" in reason.lower()


def test_survival_allows_theta(engine):
    engine.update_mtd_pnl(-15000.0)
    allowed, _ = engine.can_open_theta()
    assert allowed is True


def test_full_stop_at_8pct(engine):
    engine.update_mtd_pnl(-24000.0)  # 8% of 300k
    assert engine.full_stop is True
    allowed, reason = engine.can_open_trade()
    assert allowed is False
    assert "full stop" in reason.lower()


def test_full_stop_blocks_theta(engine):
    engine.update_mtd_pnl(-24000.0)
    allowed, reason = engine.can_open_theta()
    assert allowed is False


def test_recovery_exits_survival(engine):
    engine.update_mtd_pnl(-15000.0)  # enter survival
    assert engine.survival_mode is True
    engine.update_mtd_pnl(-5000.0)   # recover below 3%
    assert engine.survival_mode is False


def test_drawdown_reduces_sizing(engine):
    engine.update_mtd_pnl(-10000.0)  # 3.3% → reduce 25%
    amount = engine.risk_amount_for_conviction(Conviction.HIGH)
    assert amount == 4500.0  # 2% * 0.75 = 1.5% of 300k


# ── F&O Ban List ────────────────────────────────────────────────────────


def test_fo_ban_blocks(engine):
    engine.update_fo_ban_list(["DELTACORP", "IBULHSGFIN"])
    assert engine.is_fo_banned("DELTACORP") is True
    assert engine.is_fo_banned("deltacorp") is True  # case insensitive
    assert engine.is_fo_banned("NIFTY") is False


def test_pre_trade_check_blocks_banned(engine):
    engine.update_fo_ban_list(["DELTACORP"])
    allowed, reason, _ = engine.pre_trade_check(
        symbol="DELTACORP", conviction=Conviction.MEDIUM,
        direction="bullish", trade_value=5000.0,
        open_positions=[], current_risk=0.0,
    )
    assert allowed is False
    assert "ban" in reason.lower()


# ── Chop Detection ──────────────────────────────────────────────────────


def test_chop_detected_whipsaws(engine):
    engine.update_chop_signals(whipsaw_count=3, opening_range_pct=0.5, first_hour_volume_ratio=0.8)
    choppy, reason = engine.is_choppy()
    assert choppy is True
    assert "whipsaw" in reason.lower()


def test_chop_detected_narrow_range(engine):
    engine.update_chop_signals(whipsaw_count=0, opening_range_pct=0.2, first_hour_volume_ratio=0.8)
    choppy, reason = engine.is_choppy()
    assert choppy is True
    assert "narrow" in reason.lower()


def test_chop_detected_low_volume(engine):
    engine.update_chop_signals(whipsaw_count=0, opening_range_pct=0.5, first_hour_volume_ratio=0.4)
    choppy, reason = engine.is_choppy()
    assert choppy is True
    assert "volume" in reason.lower()


def test_no_chop_normal_market(engine):
    engine.update_chop_signals(whipsaw_count=1, opening_range_pct=0.8, first_hour_volume_ratio=1.2)
    choppy, _ = engine.is_choppy()
    assert choppy is False


# ── Brokerage Optimization ──────────────────────────────────────────────


def test_min_trade_value_blocks_small(engine):
    ok, reason = engine.check_min_trade_value(1500.0)
    assert ok is False
    assert "small" in reason.lower()


def test_min_trade_value_allows_large(engine):
    ok, _ = engine.check_min_trade_value(5000.0)
    assert ok is True


# ── Pre-Trade Master Check ──────────────────────────────────────────────


def test_pre_trade_check_passes_clean(engine):
    allowed, reason, risk = engine.pre_trade_check(
        symbol="NIFTY", conviction=Conviction.MEDIUM,
        direction="bullish", trade_value=5000.0,
        open_positions=[], current_risk=0.0,
    )
    assert allowed is True
    assert risk == 4500.0


def test_pre_trade_check_blocks_risk_full(engine):
    allowed, reason, risk = engine.pre_trade_check(
        symbol="NIFTY", conviction=Conviction.HIGH,
        direction="bullish", trade_value=5000.0,
        open_positions=[], current_risk=11000.0,
    )
    assert allowed is False
    assert "risk budget" in reason.lower()


# ── Persistence ─────────────────────────────────────────────────────────


def test_state_persists_across_instances(tmp_state_dir):
    e1 = RiskEngine(capital=300_000, state_dir=tmp_state_dir)
    e1.record_daily_pnl(-5000.0)
    e1.record_sl_hit()
    e1.record_sl_hit()
    e1.record_trade_opened()
    e1.update_mtd_pnl(-10000.0)
    e1.update_fo_ban_list(["DELTACORP"])

    e2 = RiskEngine(capital=300_000, state_dir=tmp_state_dir)
    assert e2._daily_pnl == -5000.0
    assert e2._sl_hits_today == 2
    assert e2._trades_today == 1
    assert e2.is_fo_banned("DELTACORP") is True


def test_monthly_reset(engine):
    engine.update_mtd_pnl(-20000.0)
    assert engine.survival_mode is True
    engine.reset_monthly()
    assert engine.survival_mode is False
    assert engine.full_stop is False
    allowed, _ = engine.can_open_trade()
    assert allowed is True


# ── State Summary ───────────────────────────────────────────────────────


def test_get_state_summary(engine):
    engine.record_daily_pnl(-2000.0)
    engine.record_sl_hit()
    engine.update_mtd_pnl(-5000.0)
    summary = engine.get_state_summary()
    assert summary["daily_pnl"] == -2000.0
    assert summary["sl_hits_today"] == 1
    assert summary["mtd_pnl"] == -5000.0
    assert "pacing" in summary
    assert "fo_ban_list" in summary
```

- [ ] **Step 2: Run tests**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_risk_engine.py -v`
Expected: All PASS (37 tests)

- [ ] **Step 3: Commit**

```bash
git add tests/test_v7_risk_engine.py
git commit -m "test(v7): add Risk Engine tests — survival, pacing, ban list, chop, persistence"
```

---

## Chunk 3: Strategist

Claude-powered component that generates and adapts playbooks throughout the trading day. Uses `ClaudeCLIClient` from `config.py` (which wraps either the Anthropic SDK or the `claude` CLI depending on whether `ANTHROPIC_API_KEY` is set).

### Task 4: Strategist — prompt building and playbook parsing

- [ ] **Step 1: Write failing tests for prompt building and response parsing**

```python
# tests/test_v7_strategist.py
"""Tests for V7 Strategist — Claude-powered playbook generation."""
import json
import pytest
from datetime import date, time
from pathlib import Path
from unittest.mock import MagicMock, patch
from v7.strategist import Strategist, build_premarket_prompt, parse_playbook_response
from v7.types import (
    Playbook, Setup, SetupType, DayClassification, Conviction,
    RiskBudget, PacingStatus, CarryRules,
)


# ── Prompt Building ─────────────────────────────────────────────────────


def test_build_premarket_prompt_contains_market_data():
    prompt = build_premarket_prompt(
        us_close={"sp500": "+0.3%", "nasdaq": "+0.5%", "dow": "+0.2%"},
        gift_nifty="24250 (+0.2%)",
        prev_vix=17.8,
        fii_dii="FII -1200cr, DII +800cr",
        events_today=[],
        events_this_week=["RBI policy Thu"],
        level_memory={"NIFTY": {"levels": [{"price": 24000, "type": "support", "strength": 3}], "oi_walls": {}}},
        edge_tracker={"overall_win_rate": 0.55, "by_strategy": {}},
        risk_state={"mtd_pnl_pct": 2.1, "pacing": "on_track", "survival_mode": False},
        fo_ban_list=["DELTACORP"],
        recent_lessons=["HDFCBANK breakout failed — check sector first"],
    )
    assert "24250" in prompt
    assert "RBI" in prompt
    assert "DELTACORP" in prompt
    assert "24000" in prompt
    assert "support" in prompt.lower()
    assert "sector first" in prompt


def test_build_premarket_prompt_includes_risk_state():
    prompt = build_premarket_prompt(
        us_close={}, gift_nifty="", prev_vix=17.0,
        fii_dii="", events_today=[], events_this_week=[],
        level_memory={}, edge_tracker={},
        risk_state={"mtd_pnl_pct": -3.5, "pacing": "behind", "survival_mode": False},
        fo_ban_list=[], recent_lessons=[],
    )
    assert "-3.5" in prompt
    assert "behind" in prompt


def test_build_premarket_prompt_survival_mode_note():
    prompt = build_premarket_prompt(
        us_close={}, gift_nifty="", prev_vix=17.0,
        fii_dii="", events_today=[], events_this_week=[],
        level_memory={}, edge_tracker={},
        risk_state={"mtd_pnl_pct": -5.5, "pacing": "survival", "survival_mode": True},
        fo_ban_list=[], recent_lessons=[],
    )
    assert "survival" in prompt.lower()
    assert "theta only" in prompt.lower() or "no directional" in prompt.lower()


# ── Response Parsing ────────────────────────────────────────────────────


def test_parse_playbook_valid_json():
    raw = json.dumps({
        "date": "2026-03-11",
        "day_classification": "LIKELY_TREND_UP",
        "nifty_plan": {
            "bias": "bullish",
            "key_levels": {"resistance_1": 24350, "support_1": 24150},
            "setups": [
                {
                    "id": "N1", "priority": 1, "type": "breakout_long",
                    "trigger": "15-min close above 24350",
                    "instrument": "NIFTY CE", "strike_logic": "delta 0.45",
                    "target": 24500, "stoploss": 24280,
                    "max_risk_pct": 1.5, "conviction": "high",
                },
            ],
            "no_trade_zone": "24200-24300",
        },
        "stock_plans": [
            {
                "id": "H1", "priority": 3, "symbol": "HDFCBANK",
                "type": "breakout_long",
                "trigger": "15-min close above 1625",
                "instrument": "HDFCBANK CE", "strike_logic": "delta 0.45",
                "target": 1660, "stoploss": 1610,
                "max_risk_pct": 1.0, "conviction": "medium",
            },
        ],
        "risk_budget": {
            "max_capital_at_risk_today_pct": 4.0,
            "max_trades_today": 4,
            "max_per_trade_risk_pct": 1.5,
            "survival_mode": False,
        },
        "no_trade_conditions": ["VIX > 22"],
        "carry_rules": {
            "carry_if": "profit > 1.5%, VIX < 20, DTE > 3",
        },
        "theta_plan": {"action": "hold"},
        "market_context": {
            "us_close": "+0.3%", "gift_nifty": "24250",
            "vix": 17.8, "fii_dii": "FII -1200cr",
        },
    })
    pb = parse_playbook_response(raw, today=date(2026, 3, 11))
    assert pb is not None
    assert pb.day_classification == DayClassification.LIKELY_TREND_UP
    assert len(pb.nifty_setups) == 1
    assert pb.nifty_setups[0].conviction == Conviction.HIGH
    assert len(pb.stock_plans) == 1
    assert pb.stock_plans[0].symbol == "HDFCBANK"


def test_parse_playbook_extracts_json_from_markdown():
    raw = """Here's my analysis:

```json
{
    "date": "2026-03-11",
    "day_classification": "LIKELY_RANGE",
    "nifty_plan": {
        "bias": "neutral",
        "key_levels": {},
        "setups": [],
        "no_trade_zone": ""
    },
    "stock_plans": [],
    "risk_budget": {
        "max_capital_at_risk_today_pct": 4.0,
        "max_trades_today": 4,
        "max_per_trade_risk_pct": 1.5,
        "survival_mode": false
    },
    "no_trade_conditions": [],
    "carry_rules": {},
    "theta_plan": {"action": "hold"},
    "market_context": {}
}
```

This is a rangebound day."""
    pb = parse_playbook_response(raw, today=date(2026, 3, 11))
    assert pb is not None
    assert pb.day_classification == DayClassification.LIKELY_RANGE


def test_parse_playbook_returns_none_on_garbage():
    pb = parse_playbook_response("This is not JSON at all", today=date(2026, 3, 11))
    assert pb is None


def test_parse_playbook_caps_stock_plans_at_3():
    raw = json.dumps({
        "date": "2026-03-11",
        "day_classification": "LIKELY_TREND_UP",
        "nifty_plan": {"bias": "bullish", "key_levels": {}, "setups": [], "no_trade_zone": ""},
        "stock_plans": [
            {"id": f"S{i}", "priority": i, "symbol": f"SYM{i}",
             "type": "breakout_long", "trigger": f"trigger {i}",
             "instrument": f"SYM{i} CE", "strike_logic": "delta 0.45",
             "target": 100 + i, "stoploss": 90 + i,
             "max_risk_pct": 1.0, "conviction": "medium"}
            for i in range(5)  # Claude sends 5, we cap at 3
        ],
        "risk_budget": {"max_capital_at_risk_today_pct": 4.0, "max_trades_today": 4,
                        "max_per_trade_risk_pct": 1.5, "survival_mode": False},
        "no_trade_conditions": [], "carry_rules": {},
        "theta_plan": {"action": "hold"}, "market_context": {},
    })
    pb = parse_playbook_response(raw, today=date(2026, 3, 11))
    assert pb is not None
    assert len(pb.stock_plans) == 3  # capped


# ── Setup Parsing ───────────────────────────────────────────────────────


def test_parse_setup_extracts_trigger_level():
    raw = json.dumps({
        "date": "2026-03-11",
        "day_classification": "LIKELY_TREND_UP",
        "nifty_plan": {
            "bias": "bullish", "key_levels": {},
            "setups": [
                {
                    "id": "N1", "priority": 1, "type": "breakout_long",
                    "trigger": "15-min candle close above 24350 with volume > 1.5x",
                    "instrument": "NIFTY CE", "strike_logic": "delta 0.45",
                    "target": 24500, "stoploss": 24280,
                    "max_risk_pct": 1.5, "conviction": "high",
                },
            ],
            "no_trade_zone": "",
        },
        "stock_plans": [],
        "risk_budget": {"max_capital_at_risk_today_pct": 4.0, "max_trades_today": 4,
                        "max_per_trade_risk_pct": 1.5, "survival_mode": False},
        "no_trade_conditions": [], "carry_rules": {},
        "theta_plan": {"action": "hold"}, "market_context": {},
    })
    pb = parse_playbook_response(raw, today=date(2026, 3, 11))
    setup = pb.nifty_setups[0]
    assert setup.trigger_level == 24350.0  # extracted from trigger text or target/stoploss
    assert setup.stoploss == 24280.0
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_strategist.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'v7.strategist'`

- [ ] **Step 3: Implement prompt building and response parsing**

```python
# v7/strategist.py
"""Claude-powered Strategist for V7.

Generates and adapts the daily trading playbook.
Calls Claude at scheduled times:
  8:45 AM  — Pre-market playbook (Sonnet)
  9:45 AM  — Opening read (Sonnet)
  10:30 AM — Check-in 1 (Sonnet)
  1:00 PM  — Check-in 2 (Sonnet)
  On exception — Emergency (Sonnet)
  3:30 PM  — EOD review (Haiku)

Uses ClaudeCLIClient from config.py which wraps either the Anthropic SDK
or the `claude` CLI depending on available auth.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

from v7.types import (
    Playbook, Setup, SetupType, DayClassification, Conviction,
    RiskBudget, PacingStatus, CarryRules,
)

log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# Max stock plans per playbook (spec: 2-3)
MAX_STOCK_PLANS = 3

# ── System Prompts ──────────────────────────────────────────────────────

STRATEGIST_SYSTEM = """You are the head of a professional Indian F&O trading desk managing ₹3-5L capital.

Your job: generate a structured daily trading playbook that the mechanical executor will follow exactly.

Rules:
- Max 2-3 setups per instrument (Plan A and Plan B, not an encyclopedia)
- Max 2-3 stock plans per day (pick the best)
- Every setup has a SPECIFIC price trigger level (not "buy if bullish")
- No-trade zones explicitly defined
- Priority ranking determines execution order
- Conviction: "high" (2% risk), "medium" (1.5% risk), "low" (0.75% risk)
- All trigger levels must be numbers, not text
- Respond with ONLY a JSON playbook — no commentary before or after

Instrument universe: NIFTY, BANKNIFTY, RELIANCE, HDFCBANK, ICICIBANK, TCS, TATAMOTORS, BAJFINANCE, SBIN, INFY"""

OPENING_READ_SYSTEM = """You are updating the morning playbook after the first 30 minutes of price discovery.
Review the opening range, gap behavior, volume, and OI shifts.
Output an updated playbook JSON with:
- Day type confirmation or override
- Opening range levels added
- Setups adjusted if levels invalidated
- Possible: "no good setups today, theta only"
Respond with ONLY the updated JSON playbook."""

CHECKIN_SYSTEM = """You are doing a mid-session check-in on the trading playbook.
Review current P&L, open positions, fired/unfired setups, level tests, OI changes, VIX.
Output an updated playbook JSON with:
- Confirm or modify remaining setups
- Add new setup ONLY if a clear opportunity emerged
- Declare "no trade rest of day" if choppy
Respond with ONLY the updated JSON playbook."""

EXCEPTION_SYSTEM = """You are handling an EXCEPTION in the trading session.
Something unexpected happened that the playbook doesn't cover.
Respond with ONLY a JSON object:
{
  "action": "flatten_all" | "hold_no_new" | "adjust_sl" | "specific_action",
  "details": "what to do",
  "new_sl_levels": {"SYMBOL": new_sl_price} (if action is adjust_sl),
  "close_symbols": ["SYMBOL"] (if action is flatten_all or specific_action)
}"""

EOD_SYSTEM = """You are grading today's trades for the trading journal.
For each trade, assign:
- entry_grade: A (trigger + confirmation) / B (trigger, weak confirmation) / C (FOMO)
- exit_grade: A (plan followed) / B (minor deviation) / C (panic/held too long)
- lesson: one sentence

Also provide a day summary.
Respond with ONLY a JSON object."""


# ── Prompt Building ─────────────────────────────────────────────────────


def build_premarket_prompt(
    us_close: dict,
    gift_nifty: str,
    prev_vix: float,
    fii_dii: str,
    events_today: list[str],
    events_this_week: list[str],
    level_memory: dict,
    edge_tracker: dict,
    risk_state: dict,
    fo_ban_list: list[str],
    recent_lessons: list[str],
) -> str:
    """Build the pre-market prompt with all inputs."""
    parts = [
        "Generate today's trading playbook.\n",
        f"## Market Data\n",
        f"- US Close: {json.dumps(us_close)}",
        f"- GIFT Nifty: {gift_nifty}",
        f"- India VIX (prev close): {prev_vix}",
        f"- FII/DII: {fii_dii}",
        f"- Events today: {events_today if events_today else 'None'}",
        f"- Events this week: {events_this_week if events_this_week else 'None'}",
        f"- F&O ban list: {fo_ban_list if fo_ban_list else 'None'}",
        "",
        f"## Key Levels (from memory)\n",
        json.dumps(level_memory, indent=2) if level_memory else "No levels stored yet.",
        "",
        f"## Edge Tracker (historical performance)\n",
        json.dumps(edge_tracker, indent=2) if edge_tracker else "No trade history yet.",
        "",
        f"## Risk State\n",
        f"- MTD P&L: {risk_state.get('mtd_pnl_pct', 0):.1f}%",
        f"- Pacing: {risk_state.get('pacing', 'on_track')}",
    ]

    if risk_state.get("survival_mode"):
        parts.append("\n**SURVIVAL MODE ACTIVE**: MTD drawdown > 5%. No directional trades allowed. Theta only. Generate a theta-only playbook with wider wings.")

    if recent_lessons:
        parts.append(f"\n## Recent Lessons\n")
        for lesson in recent_lessons[-5:]:  # last 5 lessons
            parts.append(f"- {lesson}")

    parts.append("\n## Output Format")
    parts.append("Respond with a JSON playbook matching this schema:")
    parts.append(_playbook_schema_hint())

    return "\n".join(parts)


def build_opening_read_prompt(
    current_playbook: dict,
    opening_range_high: float,
    opening_range_low: float,
    gap_direction: str,
    gap_behavior: str,
    first_30min_volume_ratio: float,
    oi_changes: dict,
) -> str:
    """Build the opening read prompt (9:45 AM)."""
    return "\n".join([
        "Update the playbook after first 30 minutes.\n",
        f"## Current Playbook\n{json.dumps(current_playbook, indent=2)}\n",
        f"## Opening Data",
        f"- Opening range: {opening_range_low:.2f} — {opening_range_high:.2f}",
        f"- Gap: {gap_direction} ({gap_behavior})",
        f"- First 30-min volume vs 20-day avg: {first_30min_volume_ratio:.1%}",
        f"- OI changes from previous close: {json.dumps(oi_changes)}\n",
        "Update the playbook JSON. Add opening_range levels. Confirm or override day type.",
    ])


def build_checkin_prompt(
    current_playbook: dict,
    daily_pnl: float,
    open_positions: list[dict],
    setups_fired: list[str],
    levels_tested: list[dict],
    oi_changes: dict,
    current_vix: float,
    checkin_number: int,
) -> str:
    """Build check-in prompt (10:30 AM or 1:00 PM)."""
    return "\n".join([
        f"Check-in #{checkin_number}. Update the playbook.\n",
        f"## Current Playbook\n{json.dumps(current_playbook, indent=2)}\n",
        f"## Session Status",
        f"- Daily P&L: ₹{daily_pnl:.0f}",
        f"- Open positions: {json.dumps(open_positions) if open_positions else 'None'}",
        f"- Setups fired: {setups_fired if setups_fired else 'None yet'}",
        f"- Levels tested since last update: {json.dumps(levels_tested) if levels_tested else 'None'}",
        f"- OI changes: {json.dumps(oi_changes)}",
        f"- Current VIX: {current_vix}",
        "",
        "Update the playbook JSON. Confirm/modify remaining setups.",
    ])


def build_exception_prompt(
    exception_type: str,
    details: dict,
    current_playbook: dict,
    open_positions: list[dict],
) -> str:
    """Build exception prompt for unexpected events."""
    return "\n".join([
        f"EXCEPTION: {exception_type}\n",
        f"## Details\n{json.dumps(details, indent=2)}\n",
        f"## Open Positions\n{json.dumps(open_positions, indent=2)}\n",
        f"## Current Playbook\n{json.dumps(current_playbook, indent=2)}\n",
        "What action should we take?",
    ])


def build_eod_prompt(
    trades_today: list[dict],
    daily_pnl: float,
    day_classification_predicted: str,
    day_classification_actual: str,
) -> str:
    """Build EOD review prompt (3:30 PM, Haiku)."""
    return "\n".join([
        "Grade today's trades and provide a day summary.\n",
        f"## Trades\n{json.dumps(trades_today, indent=2)}\n",
        f"## Summary",
        f"- Daily P&L: ₹{daily_pnl:.0f}",
        f"- Day type predicted: {day_classification_predicted}",
        f"- Day type actual: {day_classification_actual}",
        "",
        "Grade each trade (entry_grade, exit_grade, lesson). Provide day summary.",
    ])


def _playbook_schema_hint() -> str:
    return """```json
{
  "date": "YYYY-MM-DD",
  "day_classification": "LIKELY_TREND_UP|LIKELY_TREND_DOWN|LIKELY_RANGE|UNCERTAIN|EVENT_DAY|NO_TRADE",
  "nifty_plan": {
    "bias": "bullish|bearish|neutral",
    "key_levels": {"resistance_1": float, "support_1": float, ...},
    "setups": [
      {
        "id": "N1",
        "priority": 1,
        "type": "breakout_long|breakout_short|support_bounce|resistance_fade|credit_spread_bull|credit_spread_bear",
        "trigger": "specific condition with PRICE LEVEL number",
        "instrument": "NIFTY CE|PE",
        "strike_logic": "delta description",
        "target": float,
        "stoploss": float,
        "max_risk_pct": float,
        "conviction": "high|medium|low"
      }
    ],
    "no_trade_zone": "low-high"
  },
  "stock_plans": [same structure with added "symbol" field, max 3],
  "risk_budget": {
    "max_capital_at_risk_today_pct": 4.0,
    "max_trades_today": 4,
    "max_per_trade_risk_pct": 1.5,
    "survival_mode": false
  },
  "no_trade_conditions": ["condition1", ...],
  "carry_rules": {"carry_if": "conditions"},
  "theta_plan": {"action": "hold|enter|adjust|exit", "details": "..."},
  "market_context": {"us_close": "...", "gift_nifty": "...", "vix": float, "fii_dii": "..."}
}
```"""


# ── Response Parsing ────────────────────────────────────────────────────


def _extract_json(text: str) -> dict | None:
    """Extract JSON from Claude's response, handling markdown code blocks."""
    # Try direct JSON parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` block
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { ... last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


def _extract_trigger_level(trigger_text: str, target: float, stoploss: float) -> float:
    """Extract a numeric trigger level from the trigger description.

    Looks for numbers in the trigger text. Falls back to midpoint of
    target and stoploss if no number found.
    """
    numbers = re.findall(r"[\d]+(?:\.[\d]+)?", trigger_text)
    # Filter for price-like numbers (> 100 for Nifty-scale, or > 10 for stock premium)
    candidates = [float(n) for n in numbers if float(n) > 10]

    if candidates:
        # Pick the number closest to the midpoint of target/stoploss
        midpoint = (target + stoploss) / 2
        return min(candidates, key=lambda x: abs(x - midpoint))

    # Fallback: midpoint
    return (target + stoploss) / 2


def _parse_setup(d: dict, symbol: str = "NIFTY") -> Setup:
    """Parse a setup dict from Claude's response into a Setup object."""
    target = float(d.get("target", 0))
    stoploss = float(d.get("stoploss", 0))
    trigger_text = d.get("trigger", "")

    # Map type string to SetupType enum
    type_map = {
        "breakout_long": SetupType.BREAKOUT_LONG,
        "breakout_short": SetupType.BREAKOUT_SHORT,
        "support_bounce": SetupType.SUPPORT_BOUNCE,
        "resistance_fade": SetupType.RESISTANCE_FADE,
        "credit_spread_bull": SetupType.CREDIT_SPREAD_BULL,
        "credit_spread_bear": SetupType.CREDIT_SPREAD_BEAR,
        "iron_condor": SetupType.IRON_CONDOR,
    }
    setup_type = type_map.get(d.get("type", ""), SetupType.BREAKOUT_LONG)

    conviction_map = {
        "high": Conviction.HIGH,
        "medium": Conviction.MEDIUM,
        "low": Conviction.LOW,
    }
    conviction = conviction_map.get(d.get("conviction", "medium"), Conviction.MEDIUM)

    trigger_level = _extract_trigger_level(trigger_text, target, stoploss)

    return Setup(
        id=d.get("id", "X1"),
        priority=int(d.get("priority", 99)),
        type=setup_type,
        symbol=d.get("symbol", symbol),
        trigger_level=trigger_level,
        trigger_condition=trigger_text,
        instrument=d.get("instrument", f"{symbol} CE"),
        strike_logic=d.get("strike_logic", "delta 0.45"),
        target=target,
        stoploss=stoploss,
        max_risk_pct=float(d.get("max_risk_pct", 1.5)),
        conviction=conviction,
    )


def parse_playbook_response(raw: str, today: date | None = None) -> Playbook | None:
    """Parse Claude's response into a Playbook object.

    Returns None if parsing fails.
    """
    today = today or date.today()
    data = _extract_json(raw)
    if data is None:
        log.error("Failed to extract JSON from Strategist response")
        return None

    try:
        # Day classification
        dc_map = {
            "LIKELY_TREND_UP": DayClassification.LIKELY_TREND_UP,
            "LIKELY_TREND_DOWN": DayClassification.LIKELY_TREND_DOWN,
            "LIKELY_RANGE": DayClassification.LIKELY_RANGE,
            "UNCERTAIN": DayClassification.UNCERTAIN,
            "EVENT_DAY": DayClassification.EVENT_DAY,
            "NO_TRADE": DayClassification.NO_TRADE,
        }
        day_class = dc_map.get(
            data.get("day_classification", "UNCERTAIN"),
            DayClassification.UNCERTAIN,
        )

        # Nifty setups
        nifty_plan = data.get("nifty_plan", {})
        nifty_setups = [
            _parse_setup(s, symbol="NIFTY")
            for s in nifty_plan.get("setups", [])
        ]

        # Stock plans (cap at MAX_STOCK_PLANS)
        stock_plans_raw = data.get("stock_plans", [])[:MAX_STOCK_PLANS]
        stock_plans = [
            _parse_setup(s, symbol=s.get("symbol", "UNKNOWN"))
            for s in stock_plans_raw
        ]

        # Risk budget
        rb_data = data.get("risk_budget", {})
        risk_budget = RiskBudget(
            max_capital_at_risk_today_pct=rb_data.get("max_capital_at_risk_today_pct", 4.0),
            max_trades_today=rb_data.get("max_trades_today", 4),
            max_per_trade_risk_pct=rb_data.get("max_per_trade_risk_pct", 1.5),
            survival_mode=rb_data.get("survival_mode", False),
        )

        # Carry rules
        carry_data = data.get("carry_rules", {})
        carry_rules = CarryRules(
            min_profit_pct=carry_data.get("min_profit_pct", 1.5),
            max_vix=carry_data.get("max_vix", 20.0),
            min_dte=carry_data.get("min_dte", 3),
            max_hedge_cost=carry_data.get("max_hedge_cost", 500.0),
        )

        # Theta action
        theta_plan = data.get("theta_plan", {})
        theta_action = theta_plan.get("action", "hold")
        theta_details = theta_plan.get("details", "")

        playbook = Playbook(
            date=today,
            day_classification=day_class,
            nifty_bias=nifty_plan.get("bias", "neutral"),
            nifty_setups=nifty_setups,
            stock_plans=stock_plans,
            risk_budget=risk_budget,
            no_trade_conditions=data.get("no_trade_conditions", []),
            carry_rules=carry_rules,
            market_context=data.get("market_context", {}),
            theta_action=theta_action,
            theta_details=theta_details,
        )

        return playbook

    except Exception as e:
        log.error(f"Failed to parse playbook: {e}")
        return None


# ── Fallback Playbook ───────────────────────────────────────────────────


def build_fallback_playbook(
    today: date | None = None,
    prev_playbook: Playbook | None = None,
) -> Playbook:
    """Conservative fallback when Claude is unreachable.

    If previous playbook available: reuse with halved risk budgets, index only.
    Otherwise: no-trade playbook (theta only).
    """
    today = today or date.today()

    if prev_playbook:
        # Reuse previous playbook with conservative adjustments
        return Playbook(
            date=today,
            day_classification=DayClassification.UNCERTAIN,
            nifty_bias=prev_playbook.nifty_bias,
            nifty_setups=[
                Setup(
                    id=s.id, priority=s.priority, type=s.type,
                    symbol=s.symbol, trigger_level=s.trigger_level,
                    trigger_condition=s.trigger_condition,
                    instrument=s.instrument, strike_logic=s.strike_logic,
                    target=s.target, stoploss=s.stoploss,
                    max_risk_pct=s.max_risk_pct * 0.5,  # halved
                    conviction=Conviction.LOW,  # downgraded
                )
                for s in prev_playbook.nifty_setups[:2]  # max 2 setups
            ],
            stock_plans=[],  # no stock setups without fresh analysis
            risk_budget=RiskBudget(
                max_capital_at_risk_today_pct=2.0,  # halved
                max_trades_today=2,  # halved
                max_per_trade_risk_pct=0.75,  # halved
                survival_mode=False,
            ),
            no_trade_conditions=prev_playbook.no_trade_conditions + ["Claude API unavailable — conservative mode"],
            carry_rules=CarryRules(),
            market_context={"note": "Fallback playbook — Claude unavailable"},
            theta_action="hold",
        )

    # No previous playbook — absolute minimum
    return Playbook(
        date=today,
        day_classification=DayClassification.NO_TRADE,
        nifty_bias="neutral",
        nifty_setups=[],
        stock_plans=[],
        risk_budget=RiskBudget(
            max_capital_at_risk_today_pct=0.0,
            max_trades_today=0,
            max_per_trade_risk_pct=0.0,
            survival_mode=False,
        ),
        no_trade_conditions=["Claude API unavailable — no playbook generated"],
        carry_rules=CarryRules(),
        market_context={"note": "No-trade fallback — Claude unavailable, no prior playbook"},
        theta_action="hold",
    )


# ── Exception Response Parsing ──────────────────────────────────────────


def parse_exception_response(raw: str) -> dict | None:
    """Parse Claude's exception response."""
    data = _extract_json(raw)
    if data is None:
        return None
    if "action" not in data:
        return None
    return data


def default_exception_action(exception_type: str) -> dict:
    """Default action when Claude is unreachable during an exception."""
    defaults = {
        "vix_spike": {"action": "hold_no_new", "details": "VIX spike — hold positions, no new trades. Close 50% if VIX > 25."},
        "flash_crash": {"action": "flatten_all", "details": "Flash crash — close all positions immediately."},
        "3_sl_hits": {"action": "hold_no_new", "details": "3 SL hits — stop trading for the day."},
        "margin_warning": {"action": "hold_no_new", "details": "High margin — no new positions until margin frees."},
        "stock_spike": {"action": "hold_no_new", "details": "Large stock move — hold, check for news."},
    }
    return defaults.get(exception_type, {"action": "hold_no_new", "details": f"Unknown exception: {exception_type}. Defaulting to hold."})


# ── Strategist Class ────────────────────────────────────────────────────


class Strategist:
    """Claude-powered playbook generator.

    Handles all scheduled Claude calls and fallback logic.
    """

    def __init__(self, state_dir: str | Path = "data/v7"):
        from config import get_anthropic_client, CLAUDE_MODEL, CLAUDE_MODEL_LIGHT
        self._client = get_anthropic_client()
        self._model = CLAUDE_MODEL
        self._model_light = CLAUDE_MODEL_LIGHT
        self._max_retries = 3
        self._retry_delay = 120  # seconds

        from v7.state import StateManager
        from pathlib import Path
        self._state = StateManager(Path(state_dir))

    def _call_claude(
        self, prompt: str, system: str, model: str | None = None,
        max_tokens: int = 4096,
    ) -> str | None:
        """Call Claude with retry logic. Returns response text or None."""
        import time as time_mod

        model = model or self._model
        for attempt in range(self._max_retries):
            try:
                response = self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except Exception as e:
                log.warning(f"Claude call failed (attempt {attempt + 1}/{self._max_retries}): {e}")
                if attempt < self._max_retries - 1:
                    time_mod.sleep(self._retry_delay)
        return None

    def generate_premarket_playbook(
        self,
        us_close: dict, gift_nifty: str, prev_vix: float,
        fii_dii: str, events_today: list[str], events_this_week: list[str],
        level_memory: dict, edge_tracker: dict, risk_state: dict,
        fo_ban_list: list[str], recent_lessons: list[str],
    ) -> Playbook:
        """Generate the pre-market playbook (8:45 AM call).

        Falls back to conservative playbook if Claude is unreachable.
        """
        prompt = build_premarket_prompt(
            us_close=us_close, gift_nifty=gift_nifty, prev_vix=prev_vix,
            fii_dii=fii_dii, events_today=events_today,
            events_this_week=events_this_week, level_memory=level_memory,
            edge_tracker=edge_tracker, risk_state=risk_state,
            fo_ban_list=fo_ban_list, recent_lessons=recent_lessons,
        )

        raw = self._call_claude(prompt, system=STRATEGIST_SYSTEM)
        if raw:
            playbook = parse_playbook_response(raw)
            if playbook:
                self._state.save_playbook(playbook)
                return playbook
            log.error("Claude returned unparseable playbook — using fallback")

        # Fallback
        prev = self._state.load_playbook()
        fallback = build_fallback_playbook(prev_playbook=prev)
        self._state.save_playbook(fallback)
        return fallback

    def opening_read(
        self,
        opening_range_high: float, opening_range_low: float,
        gap_direction: str, gap_behavior: str,
        first_30min_volume_ratio: float, oi_changes: dict,
    ) -> Playbook | None:
        """Update playbook after opening (9:45 AM call).

        Returns updated playbook or None if Claude unavailable (executor continues with current playbook).
        """
        current = self._state.load_playbook()
        if current is None:
            log.warning("No current playbook for opening read")
            return None

        prompt = build_opening_read_prompt(
            current_playbook=current.to_dict(),
            opening_range_high=opening_range_high,
            opening_range_low=opening_range_low,
            gap_direction=gap_direction,
            gap_behavior=gap_behavior,
            first_30min_volume_ratio=first_30min_volume_ratio,
            oi_changes=oi_changes,
        )

        raw = self._call_claude(prompt, system=OPENING_READ_SYSTEM)
        if raw:
            playbook = parse_playbook_response(raw)
            if playbook:
                playbook.opening_range = {
                    "high": opening_range_high,
                    "low": opening_range_low,
                }
                self._state.save_playbook(playbook)
                return playbook

        # If Claude fails, just add opening range to existing playbook
        current.opening_range = {
            "high": opening_range_high,
            "low": opening_range_low,
        }
        self._state.save_playbook(current)
        return current

    def checkin(
        self,
        daily_pnl: float, open_positions: list[dict],
        setups_fired: list[str], levels_tested: list[dict],
        oi_changes: dict, current_vix: float, checkin_number: int,
    ) -> Playbook | None:
        """Check-in update (10:30 AM or 1:00 PM).

        Returns updated playbook or None (executor continues unchanged).
        """
        current = self._state.load_playbook()
        if current is None:
            return None

        prompt = build_checkin_prompt(
            current_playbook=current.to_dict(),
            daily_pnl=daily_pnl, open_positions=open_positions,
            setups_fired=setups_fired, levels_tested=levels_tested,
            oi_changes=oi_changes, current_vix=current_vix,
            checkin_number=checkin_number,
        )

        raw = self._call_claude(prompt, system=CHECKIN_SYSTEM)
        if raw:
            playbook = parse_playbook_response(raw)
            if playbook:
                # Preserve opening range from earlier
                playbook.opening_range = current.opening_range
                self._state.save_playbook(playbook)
                return playbook

        return None  # continue with current playbook

    def handle_exception(
        self,
        exception_type: str, details: dict,
        open_positions: list[dict],
    ) -> dict:
        """Handle an exception event.

        Returns action dict. Falls back to default if Claude unreachable.
        """
        current = self._state.load_playbook()
        prompt = build_exception_prompt(
            exception_type=exception_type,
            details=details,
            current_playbook=current.to_dict() if current else {},
            open_positions=open_positions,
        )

        raw = self._call_claude(prompt, system=EXCEPTION_SYSTEM)
        if raw:
            action = parse_exception_response(raw)
            if action:
                return action

        return default_exception_action(exception_type)

    def eod_review(self, trades_today: list[dict], daily_pnl: float,
                   predicted_day_type: str, actual_day_type: str) -> dict | None:
        """EOD review (3:30 PM, Haiku). Returns grades dict or None."""
        prompt = build_eod_prompt(
            trades_today=trades_today, daily_pnl=daily_pnl,
            day_classification_predicted=predicted_day_type,
            day_classification_actual=actual_day_type,
        )

        raw = self._call_claude(prompt, system=EOD_SYSTEM, model=self._model_light)
        if raw:
            return _extract_json(raw)
        return None
```

- [ ] **Step 4: Run tests**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_strategist.py -v`
Expected: All PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add v7/strategist.py tests/test_v7_strategist.py
git commit -m "feat(v7): add Strategist — prompt building, playbook parsing, Claude fallback"
```

---

### Task 5: Strategist — Claude call integration and fallback tests

- [ ] **Step 1: Add integration tests with mocked Claude**

Append to `tests/test_v7_strategist.py`:

```python
# ── Fallback Playbook ───────────────────────────────────────────────────


def test_fallback_no_previous_playbook():
    from v7.strategist import build_fallback_playbook
    fb = build_fallback_playbook(today=date(2026, 3, 11), prev_playbook=None)
    assert fb.day_classification == DayClassification.NO_TRADE
    assert fb.risk_budget.max_trades_today == 0
    assert len(fb.nifty_setups) == 0
    assert len(fb.stock_plans) == 0


def test_fallback_with_previous_playbook():
    from v7.strategist import build_fallback_playbook
    prev = Playbook(
        date=date(2026, 3, 10),
        day_classification=DayClassification.LIKELY_TREND_UP,
        nifty_bias="bullish",
        nifty_setups=[
            Setup(id="N1", priority=1, type=SetupType.BREAKOUT_LONG,
                  symbol="NIFTY", trigger_level=24350.0,
                  trigger_condition="15-min close above",
                  instrument="NIFTY CE", strike_logic="delta 0.45",
                  target=24500.0, stoploss=24280.0, max_risk_pct=1.5,
                  conviction=Conviction.HIGH),
        ],
        stock_plans=[
            Setup(id="H1", priority=3, type=SetupType.BREAKOUT_LONG,
                  symbol="HDFCBANK", trigger_level=1625.0,
                  trigger_condition="close above 1625",
                  instrument="HDFCBANK CE", strike_logic="delta 0.45",
                  target=1660.0, stoploss=1610.0, max_risk_pct=1.0),
        ],
        risk_budget=RiskBudget(),
        no_trade_conditions=["VIX > 22"],
        carry_rules=CarryRules(),
    )
    fb = build_fallback_playbook(today=date(2026, 3, 11), prev_playbook=prev)
    assert fb.day_classification == DayClassification.UNCERTAIN
    assert fb.risk_budget.max_trades_today == 2  # halved
    assert fb.risk_budget.max_capital_at_risk_today_pct == 2.0  # halved
    assert len(fb.nifty_setups) >= 1
    assert fb.nifty_setups[0].conviction == Conviction.LOW  # downgraded
    assert len(fb.stock_plans) == 0  # no stock setups in fallback


# ── Exception Handling ──────────────────────────────────────────────────


def test_default_exception_vix_spike():
    from v7.strategist import default_exception_action
    action = default_exception_action("vix_spike")
    assert action["action"] == "hold_no_new"


def test_default_exception_flash_crash():
    from v7.strategist import default_exception_action
    action = default_exception_action("flash_crash")
    assert action["action"] == "flatten_all"


def test_default_exception_unknown():
    from v7.strategist import default_exception_action
    action = default_exception_action("alien_invasion")
    assert action["action"] == "hold_no_new"


def test_parse_exception_response_valid():
    from v7.strategist import parse_exception_response
    raw = json.dumps({"action": "flatten_all", "details": "close everything"})
    result = parse_exception_response(raw)
    assert result["action"] == "flatten_all"


def test_parse_exception_response_no_action():
    from v7.strategist import parse_exception_response
    raw = json.dumps({"details": "missing action field"})
    result = parse_exception_response(raw)
    assert result is None


# ── Strategist Class (mocked Claude) ────────────────────────────────────


def test_strategist_generate_premarket_mocked(tmp_path):
    """Test full Strategist flow with mocked Claude client."""
    playbook_json = json.dumps({
        "date": "2026-03-11",
        "day_classification": "LIKELY_TREND_UP",
        "nifty_plan": {
            "bias": "bullish",
            "key_levels": {"resistance_1": 24350},
            "setups": [
                {"id": "N1", "priority": 1, "type": "breakout_long",
                 "trigger": "close above 24350", "instrument": "NIFTY CE",
                 "strike_logic": "delta 0.45", "target": 24500,
                 "stoploss": 24280, "max_risk_pct": 1.5, "conviction": "high"},
            ],
            "no_trade_zone": "",
        },
        "stock_plans": [],
        "risk_budget": {"max_capital_at_risk_today_pct": 4.0,
                        "max_trades_today": 4, "max_per_trade_risk_pct": 1.5,
                        "survival_mode": False},
        "no_trade_conditions": [], "carry_rules": {},
        "theta_plan": {"action": "hold"}, "market_context": {},
    })

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=playbook_json)]
    mock_client.messages.create.return_value = mock_response

    with patch("v7.strategist.Strategist.__init__", lambda self, **kw: None):
        strat = Strategist.__new__(Strategist)
        strat._client = mock_client
        strat._model = "sonnet"
        strat._model_light = "haiku"
        strat._max_retries = 1
        strat._retry_delay = 0

        from v7.state import StateManager
        strat._state = StateManager(tmp_path / "v7_state")

        pb = strat.generate_premarket_playbook(
            us_close={}, gift_nifty="24250", prev_vix=17.8,
            fii_dii="", events_today=[], events_this_week=[],
            level_memory={}, edge_tracker={}, risk_state={"mtd_pnl_pct": 0, "pacing": "on_track", "survival_mode": False},
            fo_ban_list=[], recent_lessons=[],
        )
        assert pb.day_classification == DayClassification.LIKELY_TREND_UP
        assert len(pb.nifty_setups) == 1


def test_strategist_falls_back_on_claude_failure(tmp_path):
    """If Claude fails, Strategist returns a fallback playbook."""
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("Claude down")

    with patch("v7.strategist.Strategist.__init__", lambda self, **kw: None):
        strat = Strategist.__new__(Strategist)
        strat._client = mock_client
        strat._model = "sonnet"
        strat._model_light = "haiku"
        strat._max_retries = 1
        strat._retry_delay = 0

        from v7.state import StateManager
        strat._state = StateManager(tmp_path / "v7_state")

        pb = strat.generate_premarket_playbook(
            us_close={}, gift_nifty="", prev_vix=17.0,
            fii_dii="", events_today=[], events_this_week=[],
            level_memory={}, edge_tracker={}, risk_state={"mtd_pnl_pct": 0, "pacing": "on_track", "survival_mode": False},
            fo_ban_list=[], recent_lessons=[],
        )
        assert pb.day_classification == DayClassification.NO_TRADE
        assert pb.risk_budget.max_trades_today == 0


# ── Prompt building helpers ─────────────────────────────────────────────


def test_build_opening_read_prompt():
    from v7.strategist import build_opening_read_prompt
    prompt = build_opening_read_prompt(
        current_playbook={"date": "2026-03-11", "nifty_plan": {"bias": "bullish"}},
        opening_range_high=24350.0,
        opening_range_low=24200.0,
        gap_direction="up",
        gap_behavior="extending",
        first_30min_volume_ratio=1.2,
        oi_changes={"24500CE": "+5L OI"},
    )
    assert "24350" in prompt
    assert "24200" in prompt
    assert "extending" in prompt


def test_build_checkin_prompt():
    from v7.strategist import build_checkin_prompt
    prompt = build_checkin_prompt(
        current_playbook={"date": "2026-03-11"},
        daily_pnl=-1500.0,
        open_positions=[{"symbol": "NIFTY", "pnl": 500}],
        setups_fired=["N1"],
        levels_tested=[{"price": 24350, "held": True}],
        oi_changes={},
        current_vix=18.2,
        checkin_number=1,
    )
    assert "-1500" in prompt or "1500" in prompt
    assert "N1" in prompt
    assert "18.2" in prompt


def test_build_exception_prompt():
    from v7.strategist import build_exception_prompt
    prompt = build_exception_prompt(
        exception_type="vix_spike",
        details={"vix_change": 3.5, "current_vix": 22.5},
        current_playbook={"date": "2026-03-11"},
        open_positions=[{"symbol": "NIFTY", "direction": "bullish"}],
    )
    assert "vix_spike" in prompt.lower()
    assert "22.5" in prompt


def test_build_eod_prompt():
    from v7.strategist import build_eod_prompt
    prompt = build_eod_prompt(
        trades_today=[{"symbol": "NIFTY", "pnl": 3000}],
        daily_pnl=3000.0,
        day_classification_predicted="LIKELY_TREND_UP",
        day_classification_actual="TREND_UP",
    )
    assert "3000" in prompt
    assert "LIKELY_TREND_UP" in prompt
```

- [ ] **Step 2: Run all strategist tests**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_strategist.py -v`
Expected: All PASS (22 tests)

- [ ] **Step 3: Commit**

```bash
git add tests/test_v7_strategist.py
git commit -m "test(v7): add Strategist integration tests — fallback, exception, mocked Claude"
```

---

## Chunk 4: Final Integration

### Task 6: Cross-module integration test

- [ ] **Step 1: Write integration test verifying modules work together**

```python
# tests/test_v7_plan2_integration.py
"""Integration tests: Strategist + Risk Engine + Level Memory working together."""
import json
import pytest
from datetime import date
from pathlib import Path
from v7.level_memory import LevelMemory
from v7.risk_engine import RiskEngine
from v7.strategist import parse_playbook_response, build_premarket_prompt
from v7.types import Conviction, Position, PacingStatus


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path / "v7_state"


@pytest.fixture
def level_mem(tmp_dir):
    lm = LevelMemory(state_dir=tmp_dir)
    lm.add_level("NIFTY", 24000.0, "support", "tested 3x")
    lm.add_level("NIFTY", 24500.0, "resistance", "OI wall")
    lm.update_oi_walls("NIFTY", 24500, 24000, 1.1)
    lm.add_level("HDFCBANK", 1600.0, "support", "round number")
    return lm


@pytest.fixture
def risk_eng(tmp_dir):
    return RiskEngine(capital=300_000, state_dir=tmp_dir)


def test_level_memory_feeds_into_strategist_prompt(level_mem):
    """Level Memory context is included in Strategist prompt."""
    ctx = level_mem.to_strategist_context(["NIFTY", "HDFCBANK"])
    prompt = build_premarket_prompt(
        us_close={}, gift_nifty="24250", prev_vix=17.0,
        fii_dii="", events_today=[], events_this_week=[],
        level_memory=ctx,
        edge_tracker={},
        risk_state={"mtd_pnl_pct": 0, "pacing": "on_track", "survival_mode": False},
        fo_ban_list=[], recent_lessons=[],
    )
    assert "24000" in prompt
    assert "24500" in prompt
    assert "1600" in prompt


def test_risk_engine_gates_playbook_setups(risk_eng):
    """Risk Engine pre-trade check validates a setup from a parsed playbook."""
    raw = json.dumps({
        "date": "2026-03-11",
        "day_classification": "LIKELY_TREND_UP",
        "nifty_plan": {
            "bias": "bullish", "key_levels": {},
            "setups": [
                {"id": "N1", "priority": 1, "type": "breakout_long",
                 "trigger": "close above 24350", "instrument": "NIFTY CE",
                 "strike_logic": "delta 0.45", "target": 24500,
                 "stoploss": 24280, "max_risk_pct": 1.5, "conviction": "high"},
            ],
            "no_trade_zone": "",
        },
        "stock_plans": [],
        "risk_budget": {"max_capital_at_risk_today_pct": 4.0,
                        "max_trades_today": 4, "max_per_trade_risk_pct": 1.5,
                        "survival_mode": False},
        "no_trade_conditions": [], "carry_rules": {},
        "theta_plan": {"action": "hold"}, "market_context": {},
    })
    pb = parse_playbook_response(raw, today=date(2026, 3, 11))
    setup = pb.nifty_setups[0]

    allowed, reason, risk_amount = risk_eng.pre_trade_check(
        symbol=setup.symbol,
        conviction=setup.conviction,
        direction="bullish",
        trade_value=5000.0,
        open_positions=[],
        current_risk=0.0,
    )
    assert allowed is True
    assert risk_amount == 6000.0  # HIGH conviction = 2% of 300k


def test_risk_engine_blocks_when_daily_limit_hit(risk_eng):
    """After daily loss cap hit, Risk Engine blocks setup execution."""
    risk_eng.record_daily_pnl(-7000.0)  # > 2% of 300k

    allowed, reason, _ = risk_eng.pre_trade_check(
        symbol="NIFTY", conviction=Conviction.MEDIUM,
        direction="bullish", trade_value=5000.0,
        open_positions=[], current_risk=0.0,
    )
    assert allowed is False


def test_survival_mode_blocks_directional_allows_theta(risk_eng):
    """Survival mode blocks directional trades but allows theta."""
    risk_eng.update_mtd_pnl(-16000.0)  # > 5%

    allowed, reason = risk_eng.can_open_trade()
    assert allowed is False

    theta_ok, _ = risk_eng.can_open_theta()
    assert theta_ok is True


def test_fo_ban_integrated_with_pre_trade(risk_eng):
    """F&O banned symbol is caught by pre_trade_check."""
    risk_eng.update_fo_ban_list(["TATAMOTORS"])

    allowed, reason, _ = risk_eng.pre_trade_check(
        symbol="TATAMOTORS", conviction=Conviction.MEDIUM,
        direction="bullish", trade_value=5000.0,
        open_positions=[], current_risk=0.0,
    )
    assert allowed is False
    assert "ban" in reason.lower()


def test_risk_state_feeds_into_prompt(risk_eng):
    """Risk Engine state summary is included in Strategist prompt."""
    risk_eng.record_daily_pnl(-2000.0)
    risk_eng.record_sl_hit()
    summary = risk_eng.get_state_summary()

    prompt = build_premarket_prompt(
        us_close={}, gift_nifty="", prev_vix=17.0,
        fii_dii="", events_today=[], events_this_week=[],
        level_memory={}, edge_tracker={},
        risk_state=summary,
        fo_ban_list=[], recent_lessons=[],
    )
    # Risk state values should appear in the prompt
    assert str(summary["mtd_pnl_pct"]) in prompt or "0.0" in prompt
```

- [ ] **Step 2: Run integration tests**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_plan2_integration.py -v`
Expected: All PASS (6 tests)

- [ ] **Step 3: Run all Plan 2 tests together**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_level_memory.py tests/test_v7_risk_engine.py tests/test_v7_strategist.py tests/test_v7_plan2_integration.py -v`
Expected: All PASS (~65 tests total)

- [ ] **Step 4: Commit**

```bash
git add tests/test_v7_plan2_integration.py
git commit -m "test(v7): add Plan 2 integration tests — Strategist + Risk Engine + Level Memory"
```

---

## Summary

| Module | File | Tests | Key Capabilities |
|--------|------|-------|------------------|
| Level Memory | `v7/level_memory.py` | `test_v7_level_memory.py` (17 tests) | Add/strengthen/weaken/flip/stale-prune levels, OI walls, persistence |
| Risk Engine | `v7/risk_engine.py` | `test_v7_risk_engine.py` (37 tests) | Conviction sizing, daily limits, concurrent risk, pacing, survival, chop, F&O ban, brokerage |
| Strategist | `v7/strategist.py` | `test_v7_strategist.py` (22 tests) | Prompt building (5 call types), JSON parsing, fallback playbook, exception defaults |
| Integration | — | `test_v7_plan2_integration.py` (6 tests) | Cross-module data flow verification |

**Total: ~82 tests across 4 test files, 3 implementation files.**

### Dependencies on Plan 1

| Plan 1 Module | Used By |
|---------------|---------|
| `v7/types.py` (Playbook, Setup, Position, Conviction, etc.) | All three modules |
| `v7/state.py` (StateManager) | Strategist (save/load playbook) |
| `v7/config_v7.py` (WATCHLIST, RISK_LIMITS) | Risk Engine defaults, Data Feed |
| `config.py` (ClaudeCLIClient, get_anthropic_client) | Strategist |

### What Plan 3 (Executor) will consume

- **Playbook** from Strategist (loaded via StateManager)
- **pre_trade_check()** from Risk Engine (before every entry)
- **Level Memory** levels for trigger checking
- **Chop detection** signals from Risk Engine
- **Exception handling** via Strategist.handle_exception()
