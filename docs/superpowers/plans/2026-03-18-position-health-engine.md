# Position Health Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add continuous position health scoring, graduated exits, partial profit booking, profit protection ratchet, concentration guard, and enriched checkin data to eliminate silent position death and repeat losing trades.

**Architecture:** New `PositionHealthScorer` class computes a 0-100 health score per position every tick. Executor uses the score for graduated actions (tighten trail, partial exit, full health exit). EdgeTracker gets a new `symbol_setup_performance()` method for per-symbol+setup stats. Strategist checkin prompts are enriched with health scores and setup performance. Concentration guard added as a pre-entry check.

**Tech Stack:** Python 3.13, existing v7 framework (Kite data, Executor tick loop, EdgeTracker, Strategist prompts)

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `v7/health_scorer.py` | **Create** | `PositionHealthScorer` — computes health score per position from 5 factors |
| `v7/config_v7.py` | Modify | Add `HEALTH_SCORE`, `PARTIAL_EXIT`, `PROFIT_RATCHET` config sections |
| `v7/types.py` | Modify | Add `entry_time`, `initial_quantity`, `partial_exit_done`, `health_score` to `Position` |
| `v7/executor.py` | Modify | Integrate health scorer in `_manage_positions()`, add partial exit, ratchet trail, concentration guard |
| `v7/edge_tracker.py` | Modify | Add `symbol_setup_performance()`, lower kill threshold, add `by_symbol_setup` |
| `v7/market_intel.py` | Modify | Add `setup_performance_summary()` for checkin enrichment |
| `v7/strategist.py` | Modify | Enrich checkin prompt with health scores + setup performance |
| `tests/test_v7_health_scorer.py` | **Create** | Tests for health score computation |
| `tests/test_v7_executor_health.py` | **Create** | Tests for graduated actions, partial exits, concentration guard |
| `tests/test_v7_edge_tracker.py` | Modify | Tests for `symbol_setup_performance()` |

---

### Task 1: Position Data Model Updates

**Files:**
- Modify: `v7/types.py:123-190` (Position class)
- Test: `tests/test_v7_types.py`

- [ ] **Step 1: Write failing tests for new Position fields**

```python
# tests/test_v7_types.py — add to existing file

def test_position_entry_time():
    """Position tracks entry time for age calculation."""
    pos = Position(
        symbol="RELIANCE", instrument="RELIANCE26MAR1420CE",
        direction="bullish", entry_price=15.0, quantity=500,
        lot_size=250, allocated=7500.0, stoploss=1380.0, target=1460.0,
        entry_date=date(2026, 3, 18), setup_id="S3",
        entry_time=time(10, 15),
    )
    assert pos.entry_time == time(10, 15)
    assert pos.initial_quantity == 500
    assert pos.partial_exit_done is False
    assert pos.health_score == 100.0


def test_position_age_minutes():
    """Position computes age in minutes from entry_time."""
    pos = Position(
        symbol="NIFTY", instrument="NIFTY26MAR23000CE",
        direction="bullish", entry_price=73.5, quantity=65,
        lot_size=65, allocated=4777.5, stoploss=23800.0, target=24200.0,
        entry_date=date(2026, 3, 18), setup_id="N1",
        entry_time=time(10, 0),
    )
    # 12:30 PM = 150 minutes after 10:00 AM
    assert pos.age_minutes(time(12, 30)) == 150


def test_position_premium_health():
    """Position computes premium health ratio."""
    pos = Position(
        symbol="RELIANCE", instrument="RELIANCE26MAR1420CE",
        direction="bullish", entry_price=15.75, quantity=500,
        lot_size=250, allocated=7875.0, stoploss=1380.0, target=1460.0,
        entry_date=date(2026, 3, 18), setup_id="S3",
        entry_time=time(10, 0),
    )
    # Premium dropped from 15.75 to 13.20 = 83.8% health
    assert abs(pos.premium_health(13.20) - 0.838) < 0.01


def test_position_to_dict_new_fields():
    """New fields serialize/deserialize correctly."""
    pos = Position(
        symbol="NIFTY", instrument="NIFTY26MAR23000CE",
        direction="bullish", entry_price=73.5, quantity=65,
        lot_size=65, allocated=4777.5, stoploss=23800.0, target=24200.0,
        entry_date=date(2026, 3, 18), setup_id="N1",
        entry_time=time(10, 15),
        initial_quantity=130,
        partial_exit_done=True,
        health_score=65.0,
    )
    d = pos.to_dict()
    assert d["entry_time"] == "10:15:00"
    assert d["initial_quantity"] == 130
    assert d["partial_exit_done"] is True
    assert d["health_score"] == 65.0

    restored = Position.from_dict(d)
    assert restored.entry_time == time(10, 15)
    assert restored.initial_quantity == 130
    assert restored.partial_exit_done is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_types.py -v -k "entry_time or age_minutes or premium_health or to_dict_new" 2>&1 | tail -20`
Expected: FAIL — `Position.__init__() got an unexpected keyword argument 'entry_time'`

- [ ] **Step 3: Add new fields to Position dataclass**

In `v7/types.py`, add to the `Position` class after `hedge_cost` (line 140):

```python
    entry_time: time | None = None
    initial_quantity: int = 0
    partial_exit_done: bool = False
    health_score: float = 100.0

    def __post_init__(self):
        if self.peak_price == 0.0:
            self.peak_price = self.entry_price
        if self.initial_quantity == 0:
            self.initial_quantity = self.quantity

    def age_minutes(self, current_time: time) -> int:
        """Minutes since entry. Handles same-day only."""
        if self.entry_time is None:
            return 0
        entry_min = self.entry_time.hour * 60 + self.entry_time.minute
        current_min = current_time.hour * 60 + current_time.minute
        return max(0, current_min - entry_min)

    def premium_health(self, current_premium: float) -> float:
        """Ratio of current premium to entry premium. 1.0 = no decay, 0.5 = 50% lost."""
        if self.entry_price <= 0:
            return 1.0
        return current_premium / self.entry_price
```

Also add `time` to the imports at the top of the file (`from datetime import date, time`).

Update `to_dict()` to include new fields:

```python
    "entry_time": str(self.entry_time) if self.entry_time else None,
    "initial_quantity": self.initial_quantity,
    "partial_exit_done": self.partial_exit_done,
    "health_score": self.health_score,
```

Update `from_dict()` to parse new fields:

```python
    entry_time=time.fromisoformat(d["entry_time"]) if d.get("entry_time") else None,
    initial_quantity=d.get("initial_quantity", d.get("quantity", 0)),
    partial_exit_done=d.get("partial_exit_done", False),
    health_score=d.get("health_score", 100.0),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_types.py -v -k "entry_time or age_minutes or premium_health or to_dict_new" 2>&1 | tail -20`
Expected: PASS

- [ ] **Step 5: Run full test suite to check nothing broke**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_types.py -v 2>&1 | tail -10`
Expected: All existing tests still pass

- [ ] **Step 6: Commit**

```bash
cd ~/financial-agent-india
git add v7/types.py tests/test_v7_types.py
git commit -m "feat: add entry_time, initial_quantity, partial_exit_done, health_score to Position"
```

---

### Task 2: Config — Health Score, Partial Exit, Profit Ratchet

**Files:**
- Modify: `v7/config_v7.py:98-112`
- Test: `tests/test_v7_config.py`

- [ ] **Step 1: Write failing test for new config sections**

```python
# tests/test_v7_config.py — add to existing file

def test_health_score_config():
    from v7.config_v7 import HEALTH_SCORE
    assert HEALTH_SCORE["progress_weight"] == 0.30
    assert HEALTH_SCORE["momentum_weight"] == 0.25
    assert HEALTH_SCORE["premium_weight"] == 0.20
    assert HEALTH_SCORE["volume_weight"] == 0.15
    assert HEALTH_SCORE["sl_distance_weight"] == 0.10
    assert HEALTH_SCORE["exit_threshold"] == 30
    assert HEALTH_SCORE["partial_threshold"] == 50
    assert HEALTH_SCORE["tighten_threshold"] == 70


def test_partial_exit_config():
    from v7.config_v7 import PARTIAL_EXIT
    assert PARTIAL_EXIT["first_target_rr"] == 1.0
    assert PARTIAL_EXIT["first_exit_pct"] == 0.50
    assert PARTIAL_EXIT["second_target_rr"] == 2.0
    assert PARTIAL_EXIT["second_exit_pct"] == 0.50


def test_profit_ratchet_config():
    from v7.config_v7 import PROFIT_RATCHET
    assert PROFIT_RATCHET["breakeven_to_1r"] == 1.2
    assert PROFIT_RATCHET["1r_to_2r"] == 0.8
    assert PROFIT_RATCHET["above_2r"] == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_config.py -v -k "health_score_config or partial_exit or profit_ratchet" 2>&1 | tail -10`
Expected: FAIL — `cannot import name 'HEALTH_SCORE'`

- [ ] **Step 3: Add config sections to config_v7.py**

Add after the `TRAILING` section (after line 103):

```python
# ── Position Health Score ─────────────────────────────────────────────
HEALTH_SCORE = {
    "progress_weight": 0.30,    # progress toward target vs time elapsed
    "momentum_weight": 0.25,    # recent candle direction alignment
    "premium_weight": 0.20,     # option premium health (current/entry)
    "volume_weight": 0.15,      # volume confirmation
    "sl_distance_weight": 0.10, # how much SL buffer consumed
    "exit_threshold": 30,       # health < 30 → full exit
    "partial_threshold": 50,    # health < 50 → partial exit (if not done)
    "tighten_threshold": 70,    # health < 70 → tighten trail
    "cooldown_on_exit": True,   # health exit → same-day symbol cooldown
}

# ── Partial Profit Booking ────────────────────────────────────────────
PARTIAL_EXIT = {
    "first_target_rr": 1.0,     # book first tranche at 1:1 R:R
    "first_exit_pct": 0.50,     # exit 50% at first target
    "second_target_rr": 2.0,    # book second tranche at 2:1 R:R
    "second_exit_pct": 0.50,    # exit 50% of remaining at second target
}

# ── Profit Protection Ratchet (ATR multiplier by profit stage) ────────
PROFIT_RATCHET = {
    "breakeven_to_1r": 1.2,     # slightly tighter than default 1.5x
    "1r_to_2r": 0.8,            # much tighter — protect gains
    "above_2r": 0.5,            # very tight — lock it in
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_config.py -v -k "health_score_config or partial_exit or profit_ratchet" 2>&1 | tail -10`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd ~/financial-agent-india
git add v7/config_v7.py tests/test_v7_config.py
git commit -m "feat: add HEALTH_SCORE, PARTIAL_EXIT, PROFIT_RATCHET configs"
```

---

### Task 3: PositionHealthScorer

**Files:**
- Create: `v7/health_scorer.py`
- Create: `tests/test_v7_health_scorer.py`

- [ ] **Step 1: Write failing tests for health score computation**

```python
# tests/test_v7_health_scorer.py

from datetime import date, time
from v7.health_scorer import PositionHealthScorer
from v7.types import Position


def _make_position(**kwargs) -> Position:
    defaults = dict(
        symbol="RELIANCE", instrument="RELIANCE26MAR1420CE",
        direction="bullish", entry_price=15.75, quantity=500,
        lot_size=250, allocated=7875.0,
        stoploss=1380.0, target=1460.0,
        entry_date=date(2026, 3, 18), setup_id="S3",
        entry_time=time(10, 0),
    )
    defaults.update(kwargs)
    return Position(**defaults)


class TestProgressScore:
    """Progress: target progress vs time elapsed."""

    def test_good_progress(self):
        """50% toward target in 25% of time → high score."""
        scorer = PositionHealthScorer()
        pos = _make_position()
        # underlying at 1420 (entry trigger), target 1460, SL 1380
        # Now at 1440 = 50% toward target
        score = scorer.progress_score(
            pos, underlying_ltp=1440.0, current_time=time(10, 45)
        )
        assert score > 80

    def test_no_progress_late(self):
        """0% toward target after 3 hours → low score."""
        scorer = PositionHealthScorer()
        pos = _make_position()
        # Still at entry level after 3 hours
        score = scorer.progress_score(
            pos, underlying_ltp=1400.0, current_time=time(13, 0)
        )
        assert score < 30

    def test_negative_progress(self):
        """Moving away from target → very low score."""
        scorer = PositionHealthScorer()
        pos = _make_position()
        score = scorer.progress_score(
            pos, underlying_ltp=1385.0, current_time=time(12, 0)
        )
        assert score < 15


class TestMomentumScore:
    """Momentum: are recent candles aligned with direction?"""

    def test_bullish_with_rising_candles(self):
        # Last 5 closes: rising
        candles = [
            [0, 100, 102, 99, 101, 1000],
            [0, 101, 103, 100, 102, 1200],
            [0, 102, 105, 101, 104, 1100],
            [0, 104, 106, 103, 105, 1300],
            [0, 105, 107, 104, 106, 1400],
        ]
        scorer = PositionHealthScorer()
        score = scorer.momentum_score("bullish", candles)
        assert score > 80

    def test_bullish_with_falling_candles(self):
        # Last 5 closes: falling
        candles = [
            [0, 106, 107, 105, 106, 1000],
            [0, 105, 106, 104, 104, 1200],
            [0, 104, 105, 102, 103, 1100],
            [0, 103, 104, 101, 101, 1300],
            [0, 101, 102, 99, 100, 1400],
        ]
        scorer = PositionHealthScorer()
        score = scorer.momentum_score("bullish", candles)
        assert score < 30


class TestPremiumScore:
    def test_healthy_premium(self):
        scorer = PositionHealthScorer()
        pos = _make_position(entry_price=15.75)
        score = scorer.premium_score(pos, current_premium=15.0)
        assert score > 80  # ~95% of entry premium

    def test_eroded_premium(self):
        scorer = PositionHealthScorer()
        pos = _make_position(entry_price=15.75)
        score = scorer.premium_score(pos, current_premium=10.0)
        assert score < 40  # 63% of entry — significant erosion


class TestSLDistanceScore:
    def test_near_entry(self):
        scorer = PositionHealthScorer()
        pos = _make_position(stoploss=1380.0)
        # Underlying at 1410, entry was ~1400, SL 1380
        score = scorer.sl_distance_score(pos, underlying_ltp=1410.0)
        assert score > 70

    def test_near_stoploss(self):
        scorer = PositionHealthScorer()
        pos = _make_position(stoploss=1380.0)
        # Underlying at 1382, very close to SL
        score = scorer.sl_distance_score(pos, underlying_ltp=1382.0)
        assert score < 20


class TestCompositeScore:
    def test_healthy_position(self):
        """All factors positive → high composite score."""
        scorer = PositionHealthScorer()
        pos = _make_position()
        rising_candles = [
            [0, 100, 102, 99, 101, 1000],
            [0, 101, 103, 100, 102, 1200],
            [0, 102, 105, 101, 104, 1100],
            [0, 104, 106, 103, 105, 1300],
            [0, 105, 107, 104, 106, 1400],
        ]
        score = scorer.compute(
            pos,
            underlying_ltp=1440.0,
            option_ltp=18.0,
            current_time=time(10, 45),
            candles=rising_candles,
        )
        assert score > 70

    def test_dying_position(self):
        """All factors negative → low composite score."""
        scorer = PositionHealthScorer()
        pos = _make_position()
        falling_candles = [
            [0, 106, 107, 105, 106, 1000],
            [0, 105, 106, 104, 104, 800],
            [0, 104, 105, 102, 103, 600],
            [0, 103, 104, 101, 101, 400],
            [0, 101, 102, 99, 100, 300],
        ]
        score = scorer.compute(
            pos,
            underlying_ltp=1385.0,
            option_ltp=10.0,
            current_time=time(13, 0),
            candles=falling_candles,
        )
        assert score < 30
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_health_scorer.py -v 2>&1 | tail -20`
Expected: FAIL — `ModuleNotFoundError: No module named 'v7.health_scorer'`

- [ ] **Step 3: Implement PositionHealthScorer**

```python
# v7/health_scorer.py
"""Position Health Scorer — continuous position quality assessment.

Computes a 0-100 health score per position every tick based on 5 factors:
1. Progress toward target vs time elapsed (30%)
2. Momentum alignment of recent candles (25%)
3. Option premium health vs entry premium (20%)
4. Volume confirmation (15%)
5. SL distance consumed (10%)
"""
from __future__ import annotations

from datetime import time

from v7.types import Position
from v7.config_v7 import HEALTH_SCORE


# Expected trade duration by setup type (minutes)
_EXPECTED_DURATION = {
    "breakout_long": 120,
    "breakout_short": 120,
    "support_bounce": 180,
    "resistance_fade": 180,
    "credit_spread_bull": 300,
    "credit_spread_bear": 300,
    "iron_condor": 300,
}
DEFAULT_DURATION = 150


class PositionHealthScorer:
    """Computes health score (0-100) for a position."""

    def progress_score(
        self, pos: Position, underlying_ltp: float, current_time: time,
        setup_type: str = "",
    ) -> float:
        """Score based on progress toward target vs time elapsed.

        Good: 50% toward target in 25% of expected time → 100
        Bad: 0% toward target after 100% of expected time → 0
        """
        # Progress ratio: how far toward target from entry trigger level
        total_distance = abs(pos.target - pos.stoploss)
        if total_distance == 0:
            return 50.0

        if pos.direction == "bullish":
            progress = (underlying_ltp - pos.stoploss) / total_distance
        else:
            progress = (pos.stoploss - underlying_ltp) / total_distance

        progress = max(0.0, min(1.0, progress))

        # Time ratio: how much of expected duration has elapsed
        age = pos.age_minutes(current_time)
        expected = _EXPECTED_DURATION.get(setup_type, DEFAULT_DURATION)
        time_ratio = min(age / expected, 2.0)  # cap at 2x expected

        # Score: progress should be ahead of time
        if time_ratio == 0:
            return 100.0  # just entered

        efficiency = progress / time_ratio
        # efficiency > 1 = ahead of schedule → high score
        # efficiency < 0.3 = way behind → low score
        score = min(100.0, efficiency * 100.0)
        return max(0.0, score)

    def momentum_score(self, direction: str, candles: list) -> float:
        """Score based on last 5 candle closes alignment with direction.

        Bullish + rising closes → high score
        Bullish + falling closes → low score
        """
        if not candles or len(candles) < 3:
            return 50.0  # neutral if insufficient data

        closes = [c[4] for c in candles[-5:]]
        if len(closes) < 2:
            return 50.0

        # Count aligned moves (close > prev_close for bullish)
        aligned = 0
        total = len(closes) - 1
        for i in range(1, len(closes)):
            if direction == "bullish" and closes[i] > closes[i - 1]:
                aligned += 1
            elif direction == "bearish" and closes[i] < closes[i - 1]:
                aligned += 1

        # Weight recent candles more (last candle = 2x weight)
        weighted_aligned = 0.0
        weighted_total = 0.0
        for i in range(1, len(closes)):
            weight = 1.0 + (i / total)  # later candles weighted higher
            is_aligned = (
                (direction == "bullish" and closes[i] > closes[i - 1])
                or (direction == "bearish" and closes[i] < closes[i - 1])
            )
            if is_aligned:
                weighted_aligned += weight
            weighted_total += weight

        ratio = weighted_aligned / weighted_total if weighted_total else 0.5
        return ratio * 100.0

    def premium_score(self, pos: Position, current_premium: float) -> float:
        """Score based on option premium health.

        1.0+ (premium gained) → 100
        0.85 (15% decay) → 70
        0.60 (40% decay) → 20
        """
        health = pos.premium_health(current_premium)
        if health >= 1.0:
            return 100.0
        # Linear scale: 1.0 → 100, 0.5 → 0
        return max(0.0, min(100.0, (health - 0.5) * 200.0))

    def volume_score(self, candles: list) -> float:
        """Score based on volume trend. Rising volume on move = confirmation.

        Simple: compare last 3 candles volume to prior 3.
        """
        if not candles or len(candles) < 6:
            return 50.0  # neutral if insufficient data

        recent_vol = sum(c[5] for c in candles[-3:])
        prior_vol = sum(c[5] for c in candles[-6:-3])

        if prior_vol == 0:
            return 50.0

        ratio = recent_vol / prior_vol
        # ratio > 1.2 = increasing volume → good
        # ratio < 0.8 = decreasing volume → bad
        score = 50.0 + (ratio - 1.0) * 100.0
        return max(0.0, min(100.0, score))

    def sl_distance_score(self, pos: Position, underlying_ltp: float) -> float:
        """Score based on how much SL buffer is consumed.

        At entry level → 100 (full buffer remaining)
        Halfway to SL → 50
        At SL → 0
        """
        total_risk = abs(pos.target - pos.stoploss)
        if total_risk == 0:
            return 50.0

        if pos.direction == "bullish":
            remaining = underlying_ltp - pos.stoploss
        else:
            remaining = pos.stoploss - underlying_ltp

        target_distance = abs(pos.target - pos.stoploss)
        ratio = remaining / target_distance if target_distance else 0
        return max(0.0, min(100.0, ratio * 100.0))

    def compute(
        self,
        pos: Position,
        underlying_ltp: float,
        option_ltp: float,
        current_time: time,
        candles: list | None = None,
        setup_type: str = "",
    ) -> float:
        """Composite health score (0-100) from all 5 factors."""
        w = HEALTH_SCORE

        progress = self.progress_score(pos, underlying_ltp, current_time, setup_type)
        momentum = self.momentum_score(pos.direction, candles or [])
        premium = self.premium_score(pos, option_ltp)
        volume = self.volume_score(candles or [])
        sl_dist = self.sl_distance_score(pos, underlying_ltp)

        composite = (
            progress * w["progress_weight"]
            + momentum * w["momentum_weight"]
            + premium * w["premium_weight"]
            + volume * w["volume_weight"]
            + sl_dist * w["sl_distance_weight"]
        )
        return round(max(0.0, min(100.0, composite)), 1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_health_scorer.py -v 2>&1 | tail -30`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
cd ~/financial-agent-india
git add v7/health_scorer.py tests/test_v7_health_scorer.py
git commit -m "feat: add PositionHealthScorer with 5-factor scoring"
```

---

### Task 4: Executor — Partial Profit Booking

**Files:**
- Modify: `v7/executor.py:386-451` (`_manage_positions` and new `_check_partial_exit`)
- Test: `tests/test_v7_executor_health.py`

- [ ] **Step 1: Write failing tests for partial exit**

```python
# tests/test_v7_executor_health.py

from datetime import date, time, datetime
from unittest.mock import MagicMock, patch
from v7.types import Position, Setup, SetupType, Playbook, RiskBudget, CarryRules
from v7.executor import Executor


def _make_executor():
    """Create executor with mocked dependencies."""
    data = MagicMock()
    orders = MagicMock()
    state = MagicMock()
    strategist = MagicMock()
    state.load_positions.return_value = []
    state.load_playbook.return_value = None
    state.load_daily_state.return_value = {}
    executor = Executor(data=data, orders=orders, state=state, strategist=strategist)
    executor._initialized = True
    executor._capital = 300_000
    executor._vix = 15.0
    executor._quotes = {}
    executor._daily = {"closed_trades": []}
    return executor


def _make_position(**kwargs) -> Position:
    defaults = dict(
        symbol="RELIANCE", instrument="RELIANCE26MAR1420CE",
        direction="bullish", entry_price=15.0, quantity=500,
        lot_size=250, allocated=7500.0,
        stoploss=1380.0, target=1460.0,
        entry_date=date(2026, 3, 18), setup_id="S3",
        entry_time=time(10, 0),
        initial_quantity=500,
    )
    defaults.update(kwargs)
    return Position(**defaults)


class TestPartialExit:
    def test_partial_exit_at_1rr(self):
        """At 1:1 R:R, 50% of position is exited."""
        executor = _make_executor()
        pos = _make_position(
            entry_price=15.0, stoploss=1380.0, target=1460.0,
            quantity=500, initial_quantity=500,
        )
        # entry_price on underlying would be around stoploss + risk
        # risk = target - stoploss for simplicity of the check
        # 1:1 R:R means underlying moved 1x risk from entry
        # SL = 1380, TGT = 1460, range = 80
        # Entry was when SL was set, so effective entry ~ 1380 + some buffer
        # Let's say underlying at 1460 (target) - that's more than 1RR
        # Actually the check should be: underlying moved enough that
        # unrealized PnL on option >= 1x risk amount

        executor._orders.place_exit_order.return_value = MagicMock(
            filled=True, fill_price=20.0, order_id="exit1"
        )

        exited_qty = executor._check_partial_exit(pos, underlying_ltp=1440.0, option_ltp=20.0,
                                                    now=datetime(2026, 3, 18, 11, 30))
        assert exited_qty == 250  # 50% of 500
        assert pos.partial_exit_done is True
        assert pos.quantity == 250

    def test_no_partial_if_already_done(self):
        """Don't partial exit twice."""
        executor = _make_executor()
        pos = _make_position(partial_exit_done=True, quantity=250, initial_quantity=500)
        exited_qty = executor._check_partial_exit(pos, underlying_ltp=1450.0, option_ltp=22.0,
                                                    now=datetime(2026, 3, 18, 12, 0))
        assert exited_qty == 0

    def test_no_partial_if_below_1rr(self):
        """Don't partial exit if not at 1:1 R:R yet."""
        executor = _make_executor()
        pos = _make_position()
        exited_qty = executor._check_partial_exit(pos, underlying_ltp=1400.0, option_ltp=14.0,
                                                    now=datetime(2026, 3, 18, 11, 0))
        assert exited_qty == 0
        assert pos.partial_exit_done is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_executor_health.py::TestPartialExit -v 2>&1 | tail -15`
Expected: FAIL — `Executor has no attribute '_check_partial_exit'`

- [ ] **Step 3: Implement `_check_partial_exit` in executor.py**

Add after `_check_breakeven` method (after line 477):

```python
    def _check_partial_exit(self, pos: Position, underlying_ltp: float,
                            option_ltp: float, now: datetime) -> int:
        """Book partial profit at 1:1 R:R. Returns quantity exited."""
        from v7.config_v7 import PARTIAL_EXIT

        if pos.partial_exit_done:
            return 0

        # Check if underlying has reached 1:1 R:R
        risk = abs(pos.target - pos.stoploss)
        rr_threshold = PARTIAL_EXIT["first_target_rr"]

        if pos.direction == "bullish":
            target_level = pos.stoploss + risk * (1 + rr_threshold)
            reached = underlying_ltp >= target_level
        else:
            target_level = pos.stoploss - risk * (1 + rr_threshold)
            reached = underlying_ltp <= target_level

        if not reached:
            return 0

        # Calculate exit quantity (50% of initial, rounded to lot size)
        exit_pct = PARTIAL_EXIT["first_exit_pct"]
        exit_qty = int(pos.initial_quantity * exit_pct)
        # Round down to lot size
        if pos.lot_size > 0:
            exit_qty = (exit_qty // pos.lot_size) * pos.lot_size
        if exit_qty <= 0:
            return 0

        # Place partial exit order
        from v7.order_manager import OrderSide
        side = OrderSide.SELL if pos.direction == "bullish" else OrderSide.BUY
        limit_price = option_ltp - 0.50 if side == OrderSide.SELL else option_ltp + 0.50

        result = self._orders.place_exit_order(
            tradingsymbol=pos.instrument,
            exchange="NFO",
            side=side,
            quantity=exit_qty,
            limit_price=limit_price,
            is_sl_exit=False,
        )

        if result.filled:
            exit_price = result.fill_price
            partial_pnl = exit_qty * (exit_price - pos.entry_price)
            pos.quantity -= exit_qty
            pos.partial_exit_done = True
            pos.allocated = pos.entry_price * pos.quantity  # recalculate allocated

            self._daily["daily_pnl"] = self._daily.get("daily_pnl", 0) + partial_pnl
            logger.info("PARTIAL EXIT: %s qty=%d @ %.2f, partial_pnl=%.2f, remaining=%d",
                        pos.instrument, exit_qty, exit_price, partial_pnl, pos.quantity)

            # Move SL to breakeven on remaining after partial
            if pos.direction == "bullish":
                pos.stoploss = max(pos.stoploss, pos.entry_price)
            else:
                pos.stoploss = min(pos.stoploss, pos.entry_price)

            self._send_partial_exit_alert(pos, exit_qty, exit_price, partial_pnl)
            return exit_qty

        return 0

    def _send_partial_exit_alert(self, pos: Position, qty: int, price: float, pnl: float) -> None:
        """Send Telegram alert for partial exit."""
        try:
            from v7.telegram import send_alert
            msg = (f"PARTIAL EXIT\n{pos.symbol} {pos.direction.upper()}\n"
                   f"Qty: {qty} @ {price:.2f}\n"
                   f"Partial P&L: ₹{pnl:,.0f}\n"
                   f"Remaining: {pos.quantity}")
            send_alert(msg)
        except Exception:
            pass
```

- [ ] **Step 4: Integrate partial exit into `_manage_positions()`**

In `_manage_positions()`, add after the breakeven check (after line 438) and before trailing stop:

```python
                # Partial profit booking at 1:1 R:R
                if option_ltp is not None:
                    self._check_partial_exit(pos, underlying_ltp, option_ltp, now)
                    if pos.quantity <= 0:
                        positions_to_remove.append(pos)
                        continue
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_executor_health.py::TestPartialExit -v 2>&1 | tail -15`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
cd ~/financial-agent-india
git add v7/executor.py tests/test_v7_executor_health.py
git commit -m "feat: add partial profit booking at 1:1 R:R"
```

---

### Task 5: Executor — Profit Protection Ratchet

**Files:**
- Modify: `v7/executor.py:479-503` (`_compute_trailing_sl`)
- Test: `tests/test_v7_executor_health.py`

- [ ] **Step 1: Write failing tests for profit ratchet**

```python
# tests/test_v7_executor_health.py — add to file

class TestProfitRatchet:
    def test_default_trail_when_no_profit(self):
        """Before breakeven, use default 1.5x ATR."""
        executor = _make_executor()
        pos = _make_position(stoploss=1380.0, target=1460.0, peak_price=1400.0)
        # Underlying below entry, not yet profitable
        multiplier = executor._get_trailing_multiplier(pos, underlying_ltp=1395.0)
        assert multiplier == 1.5  # default

    def test_tighter_trail_at_1r(self):
        """After 1R profit, trail at 1.2x ATR."""
        executor = _make_executor()
        pos = _make_position(stoploss=1380.0, target=1460.0, peak_price=1440.0)
        # Risk = 80pts, 1R = entry + 80 = ~1460
        # If SL moved to breakeven (entry), profit stage = breakeven_to_1r
        pos.stoploss = 1400.0  # breakeven
        multiplier = executor._get_trailing_multiplier(pos, underlying_ltp=1440.0)
        assert multiplier == 1.2

    def test_very_tight_trail_above_2r(self):
        """After 2R profit, trail at 0.5x ATR."""
        executor = _make_executor()
        pos = _make_position(stoploss=1380.0, target=1460.0, peak_price=1520.0)
        pos.stoploss = 1460.0  # SL well past breakeven
        multiplier = executor._get_trailing_multiplier(pos, underlying_ltp=1520.0)
        assert multiplier == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_executor_health.py::TestProfitRatchet -v 2>&1 | tail -10`
Expected: FAIL — `Executor has no attribute '_get_trailing_multiplier'`

- [ ] **Step 3: Implement `_get_trailing_multiplier` and update `_compute_trailing_sl`**

Add new method before `_compute_trailing_sl` (before line 479):

```python
    def _get_trailing_multiplier(self, pos: Position, underlying_ltp: float) -> float:
        """Dynamic ATR multiplier based on profit stage.

        Tighter trail as profit grows — don't give back big winners.
        """
        from v7.config_v7 import PROFIT_RATCHET, TRAILING

        risk = abs(pos.target - pos.stoploss)
        if risk == 0:
            return TRAILING["atr_multiplier"]

        # Calculate current profit in R multiples
        if pos.direction == "bullish":
            profit = underlying_ltp - pos.stoploss
        else:
            profit = pos.stoploss - underlying_ltp

        r_multiple = profit / risk if risk > 0 else 0

        # SL at or above breakeven is the precondition
        at_breakeven = (
            (pos.direction == "bullish" and pos.stoploss >= pos.entry_price) or
            (pos.direction == "bearish" and pos.stoploss <= pos.entry_price)
        )

        if not at_breakeven:
            return TRAILING["atr_multiplier"]  # default 1.5

        if r_multiple >= 2.0:
            return PROFIT_RATCHET["above_2r"]
        elif r_multiple >= 1.0:
            return PROFIT_RATCHET["1r_to_2r"]
        else:
            return PROFIT_RATCHET["breakeven_to_1r"]
```

Modify `_compute_trailing_sl` (line 497) to use dynamic multiplier:

Replace:
```python
            trailing_distance = 1.5 * atr
```
With:
```python
            multiplier = self._get_trailing_multiplier(pos, ltp)
            trailing_distance = multiplier * atr
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_executor_health.py::TestProfitRatchet -v 2>&1 | tail -10`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd ~/financial-agent-india
git add v7/executor.py tests/test_v7_executor_health.py
git commit -m "feat: add profit protection ratchet — tighter trail as profit grows"
```

---

### Task 6: Executor — Health-Based Graduated Exits

**Files:**
- Modify: `v7/executor.py:386-451` (`_manage_positions`)
- Test: `tests/test_v7_executor_health.py`

- [ ] **Step 1: Write failing tests for health-based exits**

```python
# tests/test_v7_executor_health.py — add to file

class TestHealthBasedExit:
    def test_health_exit_below_threshold(self):
        """Health < 30 → full exit with reason 'health_exit'."""
        executor = _make_executor()
        pos = _make_position()
        pos.health_score = 25.0
        executor._positions = [pos]

        # Mock the health scorer to return low score
        mock_scorer = MagicMock()
        mock_scorer.compute.return_value = 25.0
        executor._health_scorer = mock_scorer

        executor._orders.place_exit_order.return_value = MagicMock(
            filled=True, fill_price=12.0, order_id="exit_health"
        )

        # Mock data fetching
        executor._get_ltp_for_symbol = MagicMock(return_value=1385.0)
        executor._get_ltp_for_instrument = MagicMock(return_value=12.0)
        executor._data.get_candles = MagicMock(return_value=[])
        executor._state.append_trade = MagicMock()

        executor._apply_health_action(pos, 25.0, 1385.0, 12.0,
                                       datetime(2026, 3, 18, 13, 0))

        # Should have exited
        executor._orders.place_exit_order.assert_called_once()

    def test_health_tighten_at_threshold(self):
        """Health 50-70 → tighten trail but don't exit."""
        executor = _make_executor()
        pos = _make_position(stoploss=1380.0)
        old_sl = pos.stoploss

        executor._apply_health_action(pos, 60.0, 1400.0, 14.5,
                                       datetime(2026, 3, 18, 12, 0))

        # Should NOT have exited
        executor._orders.place_exit_order.assert_not_called()

    def test_health_cooldown_on_exit(self):
        """Health exit adds symbol to same-day cooldown list."""
        executor = _make_executor()
        pos = _make_position()
        executor._positions = [pos]
        executor._orders.place_exit_order.return_value = MagicMock(
            filled=True, fill_price=12.0, order_id="exit_health"
        )
        executor._state.append_trade = MagicMock()

        executor._apply_health_action(pos, 20.0, 1385.0, 12.0,
                                       datetime(2026, 3, 18, 13, 0))

        assert "RELIANCE" in executor._daily.get("cooldown_symbols", [])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_executor_health.py::TestHealthBasedExit -v 2>&1 | tail -10`
Expected: FAIL — `Executor has no attribute '_apply_health_action'`

- [ ] **Step 3: Implement health scoring integration in executor**

Add to `Executor.__init__` (around line 60):

```python
        from v7.health_scorer import PositionHealthScorer
        self._health_scorer = PositionHealthScorer()
```

Add new method after `_check_partial_exit`:

```python
    def _apply_health_action(self, pos: Position, health: float,
                             underlying_ltp: float, option_ltp: float,
                             now: datetime) -> bool:
        """Apply graduated action based on health score. Returns True if position was exited."""
        from v7.config_v7 import HEALTH_SCORE

        pos.health_score = health

        # Health < exit_threshold → full exit
        if health < HEALTH_SCORE["exit_threshold"]:
            logger.info("HEALTH EXIT: %s health=%.1f < %d, exiting",
                        pos.instrument, health, HEALTH_SCORE["exit_threshold"])
            self._exit_position(pos, option_ltp, "health_exit", now)

            # Add to cooldown if configured
            if HEALTH_SCORE.get("cooldown_on_exit"):
                cooldown = self._daily.setdefault("cooldown_symbols", [])
                if pos.symbol not in cooldown:
                    cooldown.append(pos.symbol)

            return True

        # Health < partial_threshold → partial exit if not done
        if health < HEALTH_SCORE["partial_threshold"] and not pos.partial_exit_done:
            self._check_partial_exit(pos, underlying_ltp, option_ltp, now)

        # Health < tighten_threshold → tighten trail to 1.0x ATR
        if health < HEALTH_SCORE["tighten_threshold"]:
            try:
                candles = self._data.get_candles(pos.symbol, interval="FIVE_MINUTE", days=1)
                if candles and len(candles) >= 14:
                    atr = self._compute_atr(candles[-14:])
                    if atr > 0:
                        tight_distance = 1.0 * atr  # tighter than normal
                        if pos.direction == "bullish":
                            new_sl = pos.peak_price - tight_distance
                        else:
                            new_sl = pos.peak_price + tight_distance
                        if self._is_better_sl(pos, new_sl):
                            pos.stoploss = new_sl
                            logger.info("HEALTH TIGHTEN: %s health=%.1f, SL → %.2f",
                                        pos.instrument, health, new_sl)
            except Exception:
                pass

        return False
```

Integrate into `_manage_positions()` — add after trailing SL update (after line 448), before the end of the position loop:

```python
                # Health score check (skip for carried positions)
                if not pos.carried and option_ltp is not None:
                    try:
                        candles = self._data.get_candles(pos.symbol, interval="FIVE_MINUTE", days=1)
                    except Exception:
                        candles = []
                    setup_type = self._get_setup_type(pos.setup_id)
                    health = self._health_scorer.compute(
                        pos, underlying_ltp, option_ltp,
                        now.time(), candles, setup_type,
                    )
                    if self._apply_health_action(pos, health, underlying_ltp, option_ltp, now):
                        positions_to_remove.append(pos)
                        continue
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_executor_health.py::TestHealthBasedExit -v 2>&1 | tail -15`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd ~/financial-agent-india
git add v7/executor.py tests/test_v7_executor_health.py
git commit -m "feat: add health-based graduated exits (tighten/partial/full)"
```

---

### Task 7: Executor — Concentration Guard

**Files:**
- Modify: `v7/executor.py:231-280` (`_evaluate_single_trigger`)
- Test: `tests/test_v7_executor_health.py`

- [ ] **Step 1: Write failing tests for concentration guard**

```python
# tests/test_v7_executor_health.py — add to file

class TestConcentrationGuard:
    def test_block_duplicate_symbol(self):
        """Can't enter RELIANCE if already holding RELIANCE position."""
        executor = _make_executor()
        existing = _make_position(symbol="RELIANCE")
        executor._positions = [existing]

        assert executor._has_symbol_concentration("RELIANCE") is True

    def test_allow_different_symbol(self):
        """Can enter SBIN if holding RELIANCE."""
        executor = _make_executor()
        existing = _make_position(symbol="RELIANCE")
        executor._positions = [existing]

        assert executor._has_symbol_concentration("SBIN") is False

    def test_block_cooldown_symbol(self):
        """Can't enter symbol that was health-exited today."""
        executor = _make_executor()
        executor._daily["cooldown_symbols"] = ["RELIANCE"]

        assert executor._is_symbol_cooled_down("RELIANCE") is True

    def test_allow_non_cooldown_symbol(self):
        executor = _make_executor()
        executor._daily["cooldown_symbols"] = ["RELIANCE"]

        assert executor._is_symbol_cooled_down("SBIN") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_executor_health.py::TestConcentrationGuard -v 2>&1 | tail -10`
Expected: FAIL — `Executor has no attribute '_has_symbol_concentration'`

- [ ] **Step 3: Implement concentration guard**

Add methods to Executor (before `_evaluate_triggers`):

```python
    def _has_symbol_concentration(self, symbol: str) -> bool:
        """Check if already holding a position in this symbol."""
        return any(p.symbol == symbol for p in self._positions)

    def _is_symbol_cooled_down(self, symbol: str) -> bool:
        """Check if symbol was health-exited today (same-day cooldown)."""
        return symbol in self._daily.get("cooldown_symbols", [])
```

In `_evaluate_single_trigger` (after line 248, after `if not triggered: return`), add:

```python
        # Concentration guard: max 1 position per symbol
        if self._has_symbol_concentration(setup.symbol):
            logger.info("Setup %s: already holding %s — concentration guard", setup.id, setup.symbol)
            return

        # Cooldown guard: symbol was health-exited today
        if self._is_symbol_cooled_down(setup.symbol):
            logger.info("Setup %s: %s cooled down today — skipping", setup.id, setup.symbol)
            return
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_executor_health.py::TestConcentrationGuard -v 2>&1 | tail -10`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd ~/financial-agent-india
git add v7/executor.py tests/test_v7_executor_health.py
git commit -m "feat: add concentration guard — max 1 position per symbol + cooldown"
```

---

### Task 8: Executor — Set entry_time on Position Entry

**Files:**
- Modify: `v7/executor.py:282-379` (`_enter_position`)
- Test: `tests/test_v7_executor_health.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_v7_executor_health.py — add to file

class TestEntryTimeTracking:
    def test_entry_sets_time(self):
        """Position created with entry_time set to current time."""
        executor = _make_executor()
        executor._data.get_option_chain = MagicMock(return_value=[{"strike": 1420}])
        executor._data.can_trade = MagicMock(return_value=True)

        with patch("v7.executor.select_directional_strike") as mock_select:
            mock_select.return_value = {
                "tradingsymbol": "RELIANCE26MAR1420CE",
                "strike": 1420, "delta": 0.45, "premium": 15.0,
                "lot_size": 250,
            }
            executor._orders.place_entry_order.return_value = MagicMock(
                filled=True, fill_price=15.0, order_id="entry1"
            )

            setup = Setup(
                id="S3", symbol="RELIANCE", priority=1,
                type=SetupType.BREAKOUT_LONG,
                trigger_level=1420.0, trigger_condition="breakout",
                instrument="CE", strike_logic="delta 0.45",
                target=1460.0, stoploss=1380.0,
                conviction="medium", max_risk_pct=1.5,
            )

            now = datetime(2026, 3, 18, 10, 30, 0)
            executor._enter_position(setup, 1421.0, now)

            assert len(executor._positions) == 1
            assert executor._positions[0].entry_time == time(10, 30)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_executor_health.py::TestEntryTimeTracking -v 2>&1 | tail -10`
Expected: FAIL — `entry_time` is None

- [ ] **Step 3: Set entry_time in `_enter_position`**

In `_enter_position` (line 362), add `entry_time=now.time()` to the Position constructor:

```python
        pos = Position(
            symbol=setup.symbol,
            instrument=tradingsymbol,
            direction=direction,
            entry_price=result.fill_price,
            quantity=quantity,
            lot_size=actual_lot_size,
            allocated=result.fill_price * quantity,
            stoploss=setup.stoploss,
            target=setup.target,
            entry_date=now.date(),
            setup_id=setup.id,
            entry_time=now.time(),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_executor_health.py::TestEntryTimeTracking -v 2>&1 | tail -10`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd ~/financial-agent-india
git add v7/executor.py tests/test_v7_executor_health.py
git commit -m "feat: track entry_time on position creation"
```

---

### Task 9: EdgeTracker — Per-Symbol+Setup Performance

**Files:**
- Modify: `v7/edge_tracker.py:95-178`
- Test: `tests/test_v7_edge_tracker.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_v7_edge_tracker.py — add to existing file

def test_symbol_setup_performance():
    """Get performance for a specific symbol+setup combo."""
    from v7.edge_tracker import EdgeTracker
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = EdgeTracker(data_dir=tmpdir)

        # Record 4 RELIANCE breakout_long losses
        for i in range(4):
            from v7.types import TradeResult, SetupType
            trade = TradeResult(
                symbol="RELIANCE", instrument=f"RELIANCE26MAR1420CE",
                setup_type=SetupType.BREAKOUT_LONG, setup_id="S3",
                direction="bullish", entry_price=15.0 + i * 0.25,
                exit_price=13.0, quantity=500,
                pnl=-1000 - i * 100, pnl_pct=-13.0 - i,
                costs=0, entry_date="2026-03-17", exit_date="2026-03-17",
                exit_reason="wind_down",
            )
            tracker.record(trade, strategy="momentum", time_bucket="9:45-11:00")

        # Record 1 SBIN breakout_short win
        trade = TradeResult(
            symbol="SBIN", instrument="SBIN26MAR1045PE",
            setup_type=SetupType.BREAKOUT_SHORT, setup_id="S2",
            direction="bearish", entry_price=21.05, exit_price=21.95,
            quantity=750, pnl=675, pnl_pct=4.27,
            costs=0, entry_date="2026-03-16", exit_date="2026-03-16",
            exit_reason="target",
        )
        tracker.record(trade, strategy="momentum", time_bucket="9:45-11:00")

        perf = tracker.symbol_setup_performance("RELIANCE", "breakout_long")
        assert perf["trades"] == 4
        assert perf["wins"] == 0
        assert perf["win_rate"] == 0.0
        assert perf["net_pnl"] < 0

        perf2 = tracker.symbol_setup_performance("SBIN", "breakout_short")
        assert perf2["trades"] == 1
        assert perf2["wins"] == 1

        # Non-existent combo
        perf3 = tracker.symbol_setup_performance("TCS", "breakout_long")
        assert perf3["trades"] == 0


def test_kill_candidates_lower_threshold():
    """Kill candidates with lower threshold (10 trades)."""
    from v7.edge_tracker import EdgeTracker
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = EdgeTracker(data_dir=tmpdir)

        # Record 12 losing momentum trades
        for i in range(12):
            from v7.types import TradeResult, SetupType
            trade = TradeResult(
                symbol="NIFTY", instrument=f"NIFTY26MAR{23000+i*50}CE",
                setup_type=SetupType.BREAKOUT_LONG, setup_id="N1",
                direction="bullish", entry_price=50.0,
                exit_price=30.0, quantity=65,
                pnl=-1300, pnl_pct=-40.0,
                costs=0, entry_date=f"2026-03-{10+i}", exit_date=f"2026-03-{10+i}",
                exit_reason="stoploss",
            )
            tracker.record(trade, strategy="momentum", time_bucket="9:45-11:00")

        # With default threshold (30) → no kills
        kills_30 = tracker.kill_candidates(min_trades=30)
        assert kills_30 == []

        # With lower threshold (10) → momentum is a kill candidate
        kills_10 = tracker.kill_candidates(min_trades=10)
        assert "momentum" in kills_10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_edge_tracker.py -v -k "symbol_setup_performance or kill_candidates_lower" 2>&1 | tail -10`
Expected: FAIL — `EdgeTracker has no attribute 'symbol_setup_performance'`

- [ ] **Step 3: Implement `symbol_setup_performance` and update `get_stats`**

Add to `EdgeTracker` class (after `_instrument_stats`, around line 169):

```python
    def symbol_setup_performance(self, symbol: str, setup_type: str) -> dict:
        """Get performance for a specific symbol + setup_type combo."""
        matching = [
            t for t in self._trades
            if t["symbol"] == symbol and t["setup_type"] == setup_type
        ]
        if not matching:
            return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "net_pnl": 0.0}

        wins = [t for t in matching if t["is_win"]]
        weighted = self._weighted_overall(matching)
        return {
            "trades": len(matching),
            "wins": len(wins),
            "losses": len(matching) - len(wins),
            "win_rate": len(wins) / len(matching),
            "net_pnl": sum(t["pnl"] for t in matching),
            "avg_pnl": sum(t["pnl"] for t in matching) / len(matching),
            **weighted,
        }

    def all_symbol_setup_combos(self) -> dict[str, dict]:
        """Get performance for all symbol+setup combos (for checkin prompt)."""
        combos: dict[str, list] = {}
        for t in self._trades:
            key = f"{t['symbol']}:{t['setup_type']}"
            combos.setdefault(key, []).append(t)

        result = {}
        for key, trades in combos.items():
            wins = [t for t in trades if t["is_win"]]
            weighted = self._weighted_overall(trades)
            result[key] = {
                "trades": len(trades),
                "wins": len(wins),
                "win_rate": len(wins) / len(trades),
                "net_pnl": sum(t["pnl"] for t in trades),
                **weighted,
            }
        return result
```

Also add `by_symbol_setup` to the `get_stats` return value:

In `get_stats()` return dict (line 112-125), add:

```python
            "by_symbol_setup": self.all_symbol_setup_combos(),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_edge_tracker.py -v -k "symbol_setup_performance or kill_candidates_lower" 2>&1 | tail -10`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd ~/financial-agent-india
git add v7/edge_tracker.py tests/test_v7_edge_tracker.py
git commit -m "feat: add symbol_setup_performance and all_symbol_setup_combos to EdgeTracker"
```

---

### Task 10: Enriched Checkin Prompts

**Files:**
- Modify: `v7/market_intel.py:512-555`
- Modify: `v7/strategist.py` (checkin prompt builder)
- Test: `tests/test_v7_market_intel.py`

- [ ] **Step 1: Write failing test for setup_performance_summary**

```python
# tests/test_v7_market_intel.py — add to existing file

def test_setup_performance_summary():
    """Generate per-symbol+setup performance text for checkin."""
    from v7.market_intel import setup_performance_summary
    from unittest.mock import MagicMock

    edge_tracker = MagicMock()
    edge_tracker.all_symbol_setup_combos.return_value = {
        "RELIANCE:breakout_long": {
            "trades": 4, "wins": 0, "win_rate": 0.0,
            "net_pnl": -4550, "weighted_win_rate": 0.0,
        },
        "SBIN:breakout_short": {
            "trades": 1, "wins": 1, "win_rate": 1.0,
            "net_pnl": 675, "weighted_win_rate": 1.0,
        },
    }

    text = setup_performance_summary(edge_tracker)
    assert "RELIANCE:breakout_long" in text
    assert "0W/4L" in text
    assert "SBIN:breakout_short" in text
    assert "1W/0L" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_market_intel.py -v -k "setup_performance_summary" 2>&1 | tail -10`
Expected: FAIL — `cannot import name 'setup_performance_summary'`

- [ ] **Step 3: Implement `setup_performance_summary` in market_intel.py**

Add function to `market_intel.py` (after `build_premarket_intel`, near end of file):

```python
def setup_performance_summary(edge_tracker) -> str:
    """Generate per-symbol+setup performance text for checkin prompts."""
    try:
        combos = edge_tracker.all_symbol_setup_combos()
    except Exception:
        return ""

    if not combos:
        return "No setup performance data yet."

    lines = ["## Recent Setup Performance (rolling, 14-day weighted)"]
    for key, data in sorted(combos.items(), key=lambda x: x[1]["net_pnl"]):
        wins = data["wins"]
        losses = data["trades"] - wins
        trend = "↑" if data["weighted_win_rate"] > 0.5 else "↓" if data["weighted_win_rate"] < 0.3 else "→"
        lines.append(
            f"- {key}: {wins}W/{losses}L | "
            f"WR: {data['win_rate']:.0%} | "
            f"Net: ₹{data['net_pnl']:+,.0f} | "
            f"Trend: {trend}"
        )
    return "\n".join(lines)
```

- [ ] **Step 4: Integrate into strategist checkin prompt**

In `v7/strategist.py`, find the checkin prompt builder method. Add after the existing checkin data:

```python
        # Setup performance for informed tactical decisions
        from v7.market_intel import setup_performance_summary
        if hasattr(self, '_edge_tracker') and self._edge_tracker:
            setup_perf = setup_performance_summary(self._edge_tracker)
            if setup_perf:
                prompt_parts.append(setup_perf)
```

Also add health scores for open positions to the checkin data:

```python
        # Open position health
        if positions:
            health_lines = ["## Open Position Health"]
            for pos in positions:
                status = "healthy" if pos.health_score > 70 else "warning" if pos.health_score > 40 else "critical"
                health_lines.append(
                    f"- {pos.symbol} {pos.instrument} | Health: {pos.health_score:.0f}/100 ({status}) | "
                    f"P&L: {pos.unrealized_pnl_pct(pos.entry_price):.1f}%"
                )
            prompt_parts.append("\n".join(health_lines))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_market_intel.py -v -k "setup_performance_summary" 2>&1 | tail -10`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
cd ~/financial-agent-india
git add v7/market_intel.py v7/strategist.py tests/test_v7_market_intel.py
git commit -m "feat: enrich checkin prompts with health scores + setup performance"
```

---

### Task 11: Add health_exit to Trade Exit Reasons + Telegram

**Files:**
- Modify: `v7/executor.py:522-573` (`_exit_position`)
- Modify: `v7/telegram.py` (alert formatting)

- [ ] **Step 1: Verify health_exit flows through existing exit path**

`_exit_position` already accepts any string `reason`, so `"health_exit"` will work. But the Telegram alert should format it distinctly.

- [ ] **Step 2: Update Telegram exit alert to show health exit reason**

In `v7/telegram.py`, find the exit alert function and add health_exit formatting:

```python
    # In the exit alert function, add:
    if reason == "health_exit":
        reason_text = "HEALTH EXIT (position dying — momentum/premium/progress failed)"
    elif reason == "time_stop":
        reason_text = "TIME STOP (no progress toward target)"
    else:
        reason_text = reason.upper()
```

- [ ] **Step 3: Run full test suite**

Run: `cd ~/financial-agent-india && python -m pytest tests/ -v --tb=short -k "v7" 2>&1 | tail -30`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
cd ~/financial-agent-india
git add v7/telegram.py
git commit -m "feat: add health_exit and time_stop alert formatting to Telegram"
```

---

### Task 12: Integration Test — Full Tick with Health Engine

**Files:**
- Create: `tests/test_v7_health_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_v7_health_integration.py
"""Integration test: full tick cycle with health engine active."""

from datetime import date, time, datetime
from unittest.mock import MagicMock, patch
from v7.types import Position, Playbook, RiskBudget, CarryRules
from v7.executor import Executor


def test_tick_with_health_engine():
    """Full tick with health scoring — dying position gets health-exited."""
    data = MagicMock()
    orders = MagicMock()
    state = MagicMock()
    strategist = MagicMock()

    # Position: RELIANCE CE entered at 10 AM, now 1 PM, premium eroded
    pos = Position(
        symbol="RELIANCE", instrument="RELIANCE26MAR1420CE",
        direction="bullish", entry_price=15.75, quantity=500,
        lot_size=250, allocated=7875.0,
        stoploss=1380.0, target=1460.0,
        entry_date=date(2026, 3, 18), setup_id="S3",
        entry_time=time(10, 0),
    )

    state.load_positions.return_value = [pos]
    state.load_playbook.return_value = None
    state.load_daily_state.return_value = {}

    executor = Executor(data=data, orders=orders, state=state, strategist=strategist)
    executor._initialized = True
    executor._capital = 300_000
    executor._vix = 15.0
    executor._positions = [pos]
    executor._daily = {"closed_trades": []}

    # Mock: underlying barely moved (1390, near SL), option premium tanked
    executor._get_ltp_for_symbol = MagicMock(return_value=1390.0)
    executor._get_ltp_for_instrument = MagicMock(return_value=10.0)

    # Falling candles (momentum against)
    falling_candles = [
        [0, 1410, 1412, 1408, 1410, 1000],
        [0, 1408, 1410, 1405, 1406, 900],
        [0, 1405, 1407, 1400, 1402, 800],
        [0, 1400, 1403, 1395, 1396, 700],
        [0, 1395, 1398, 1388, 1390, 600],
        [0, 1395, 1398, 1388, 1390, 600],  # need 6+ for volume score
    ]
    data.get_candles.return_value = falling_candles + falling_candles  # 12 candles

    orders.place_exit_order.return_value = MagicMock(
        filled=True, fill_price=10.0, order_id="health_exit_1"
    )
    orders.cancel_order = MagicMock()
    state.append_trade = MagicMock()

    # Simulate tick at 1 PM (3 hours after entry)
    now = datetime(2026, 3, 18, 13, 0, 0)

    # The health score should be low:
    # - progress: near SL, 0% toward target after 3 hours → very low
    # - momentum: falling candles for bullish → low
    # - premium: 10/15.75 = 63% → low
    # - volume: declining → low
    # - SL distance: very close to SL → low

    executor._manage_positions(now)

    # Position should have been health-exited
    assert len(executor._positions) == 0 or executor._positions[0].health_score < 30


def test_tick_healthy_position_holds():
    """Healthy position is not prematurely exited."""
    data = MagicMock()
    orders = MagicMock()
    state = MagicMock()
    strategist = MagicMock()

    pos = Position(
        symbol="NIFTY", instrument="NIFTY26MAR23000CE",
        direction="bullish", entry_price=73.5, quantity=65,
        lot_size=65, allocated=4777.5,
        stoploss=23800.0, target=24200.0,
        entry_date=date(2026, 3, 18), setup_id="N1",
        entry_time=time(10, 0),
    )

    state.load_positions.return_value = [pos]
    state.load_playbook.return_value = None
    state.load_daily_state.return_value = {}

    executor = Executor(data=data, orders=orders, state=state, strategist=strategist)
    executor._initialized = True
    executor._capital = 300_000
    executor._vix = 15.0
    executor._positions = [pos]
    executor._daily = {"closed_trades": []}

    # Underlying moving toward target, premium healthy
    executor._get_ltp_for_symbol = MagicMock(return_value=24050.0)
    executor._get_ltp_for_instrument = MagicMock(return_value=95.0)

    rising_candles = [
        [0, 23900, 23920, 23890, 23910, 1000],
        [0, 23910, 23940, 23900, 23930, 1100],
        [0, 23930, 23960, 23920, 23950, 1200],
        [0, 23950, 23980, 23940, 23970, 1300],
        [0, 23970, 24010, 23960, 24000, 1400],
        [0, 24000, 24050, 23990, 24040, 1500],
    ]
    data.get_candles.return_value = rising_candles + rising_candles

    now = datetime(2026, 3, 18, 11, 0, 0)
    executor._manage_positions(now)

    # Position should still be held
    assert len(executor._positions) == 1
    assert executor._positions[0].health_score > 60
```

- [ ] **Step 2: Run integration tests**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_health_integration.py -v 2>&1 | tail -20`
Expected: PASS

- [ ] **Step 3: Run full v7 test suite**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_*.py -v --tb=short 2>&1 | tail -30`
Expected: All tests pass (existing + new)

- [ ] **Step 4: Commit**

```bash
cd ~/financial-agent-india
git add tests/test_v7_health_integration.py
git commit -m "test: add integration tests for health engine in full tick cycle"
```

---

### Task 13: Deploy to Pi

**Files:**
- No new files — deploy existing code

- [ ] **Step 1: Sync code to Pi**

```bash
rsync -avz --exclude='venv/' --exclude='__pycache__/' --exclude='data/' \
  ~/financial-agent-india/v7/ pi@100.103.11.7:~/financial-agent-india/v7/
rsync -avz ~/financial-agent-india/tests/test_v7_health*.py \
  pi@100.103.11.7:~/financial-agent-india/tests/
```

- [ ] **Step 2: Run tests on Pi**

```bash
ssh pi@100.103.11.7 "cd ~/financial-agent-india && ./venv/bin/python -m pytest tests/test_v7_health_scorer.py tests/test_v7_executor_health.py tests/test_v7_health_integration.py -v 2>&1 | tail -30"
```
Expected: All PASS

- [ ] **Step 3: Verify cron runs without errors**

```bash
ssh pi@100.103.11.7 "tail -20 ~/financial-agent-india/data/v7/cron.log"
```
Expected: No import errors, health scorer initializes

- [ ] **Step 4: Monitor first trading day**

Watch Telegram for:
- Health score updates in position alerts
- Partial exit alerts
- Health exit alerts (if triggered)
- Tighter trailing alerts

---

## Summary of Changes

| Component | Before | After |
|-----------|--------|-------|
| **Position entry** | Fire and forget | Tracks entry_time, initial_quantity. Concentration guard blocks duplicate symbols. |
| **Position mid-life** | Only SL/target/trailing | Health score every tick. Graduated: tighten → partial exit → full health exit. |
| **Partial profit** | None (all-or-nothing) | 50% booked at 1:1 R:R, SL moved to breakeven on remainder. |
| **Trailing stop** | Fixed 1.5x ATR | Ratchet: 1.5x → 1.2x → 0.8x → 0.5x as profit grows. |
| **Wind_down** | Force-close everything not profitable | Stays the same — but fewer positions reach wind_down because health exits happen earlier. |
| **Edge tracker** | by_strategy, by_instrument | + by_symbol_setup. symbol_setup_performance() for granular lookups. |
| **Claude checkins** | P&L number + open positions | + Health scores per position + per-symbol+setup win rates. Claude can make informed tactical decisions. |
| **Cooldown** | None | Health exit → same-day symbol cooldown. Prevents re-entering dying setups. |
