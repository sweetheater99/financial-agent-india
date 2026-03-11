# V7 Plan 1: Core Infrastructure, Data Feed & State Persistence

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the foundational layer that all V7 components depend on — shared types, state persistence, data feed abstraction, and config updates.

**Architecture:** V7 code lives in a new `v7/` package within the existing repo. It imports existing battle-tested modules (kite_data, connect, risk_manager, regime, indicators, greeks) directly. New modules define the V7-specific types, state management, and data feed abstraction.

**Tech Stack:** Python 3.13, Kite Connect, AngelOne SmartAPI, JSON file persistence, pytest

**Spec:** `docs/superpowers/specs/2026-03-11-v7-professional-trader-bot-design.md`

**Depends on:** Nothing (this is the foundation)
**Blocks:** Plan 2 (Strategist), Plan 3 (Executor), Plan 4 (Journal)

---

## File Structure

```
v7/
├── __init__.py              # Package init
├── types.py                 # Shared types: Playbook, Setup, Position, TradeResult, DayPhase
├── config_v7.py             # V7-specific config (watchlist, risk budgets, phase times)
├── state.py                 # State persistence: load/save all state files
├── data_feed.py             # Unified data feed (Kite primary, AngelOne protect-only)
├── strike_selector.py       # Mechanical strike selection (delta, liquidity filters)
├── margin.py                # Margin estimation and budget tracking
tests/
├── test_v7_types.py
├── test_v7_state.py
├── test_v7_data_feed.py
├── test_v7_strike_selector.py
├── test_v7_margin.py
├── test_v7_config.py
```

---

## Chunk 1: Package Setup, Types & Config

### Task 1: Create v7 package and shared types

**Files:**
- Create: `v7/__init__.py`
- Create: `v7/types.py`
- Test: `tests/test_v7_types.py`

- [ ] **Step 1: Write failing tests for types**

```python
# tests/test_v7_types.py
"""Tests for V7 shared types."""
import pytest
from datetime import date, time
from v7.types import (
    DayPhase, DayClassification, Conviction, SetupType,
    Setup, Playbook, Position, TradeResult, CarryRules,
    RiskBudget, PacingStatus, MarketContext, KeyLevels,
)


def test_day_phase_from_time_premarket():
    assert DayPhase.from_time(time(8, 45)) == DayPhase.PRE_MARKET


def test_day_phase_from_time_opening_read():
    assert DayPhase.from_time(time(9, 20)) == DayPhase.OPENING_READ


def test_day_phase_from_time_active():
    assert DayPhase.from_time(time(11, 0)) == DayPhase.ACTIVE_TRADING


def test_day_phase_from_time_wind_down():
    assert DayPhase.from_time(time(14, 45)) == DayPhase.WIND_DOWN


def test_day_phase_from_time_post_close():
    assert DayPhase.from_time(time(15, 20)) == DayPhase.POST_CLOSE


def test_day_phase_from_time_outside_hours():
    assert DayPhase.from_time(time(7, 0)) == DayPhase.OUTSIDE_HOURS
    assert DayPhase.from_time(time(16, 0)) == DayPhase.OUTSIDE_HOURS


def test_setup_creation():
    s = Setup(
        id="N1",
        priority=1,
        type=SetupType.BREAKOUT_LONG,
        symbol="NIFTY",
        trigger_level=24350.0,
        trigger_condition="15-min close above with volume > 1.5x",
        instrument="NIFTY CE",
        strike_logic="slightly OTM, delta 0.40-0.50",
        target=24500.0,
        stoploss=24280.0,
        max_risk_pct=1.5,
    )
    assert s.id == "N1"
    assert s.type == SetupType.BREAKOUT_LONG
    assert s.fired is False
    assert s.cancelled is False


def test_setup_to_dict_roundtrip():
    s = Setup(
        id="N1", priority=1, type=SetupType.BREAKOUT_LONG,
        symbol="NIFTY", trigger_level=24350.0,
        trigger_condition="15-min close above",
        instrument="NIFTY CE", strike_logic="delta 0.45",
        target=24500.0, stoploss=24280.0, max_risk_pct=1.5,
    )
    d = s.to_dict()
    s2 = Setup.from_dict(d)
    assert s2.id == s.id
    assert s2.trigger_level == s.trigger_level
    assert s2.type == s.type


def test_position_pnl_long():
    p = Position(
        symbol="NIFTY", instrument="NIFTY CE 24400",
        direction="bullish", entry_price=100.0,
        quantity=75, lot_size=75, allocated=7500.0,
        stoploss=80.0, target=150.0,
        entry_date=date(2026, 3, 11), setup_id="N1",
    )
    assert p.unrealized_pnl(120.0) == 75 * 20.0  # 1500
    assert p.unrealized_pnl(80.0) == 75 * -20.0   # -1500


def test_position_to_dict_roundtrip():
    p = Position(
        symbol="NIFTY", instrument="NIFTY CE 24400",
        direction="bullish", entry_price=100.0,
        quantity=75, lot_size=75, allocated=7500.0,
        stoploss=80.0, target=150.0,
        entry_date=date(2026, 3, 11), setup_id="N1",
    )
    d = p.to_dict()
    p2 = Position.from_dict(d)
    assert p2.symbol == p.symbol
    assert p2.entry_price == p.entry_price
    assert p2.stoploss == p.stoploss


def test_trade_result_creation():
    tr = TradeResult(
        symbol="NIFTY", instrument="NIFTY CE 24400",
        direction="bullish", entry_price=100.0, exit_price=150.0,
        quantity=75, entry_date=date(2026, 3, 11),
        exit_date=date(2026, 3, 11), exit_reason="target",
        pnl=3750.0, pnl_pct=50.0, costs=120.0,
        setup_id="N1", setup_type=SetupType.BREAKOUT_LONG,
        entry_grade="A", exit_grade="A",
    )
    assert tr.pnl == 3750.0
    assert tr.exit_reason == "target"


def test_risk_budget_can_allocate():
    rb = RiskBudget(
        max_capital_at_risk_today_pct=4.0,
        max_trades_today=4,
        max_per_trade_risk_pct=1.5,
        survival_mode=False,
    )
    # At 3L capital, 4% = 12000 max concurrent risk
    assert rb.can_allocate(4500, 0, 300_000) is True
    assert rb.can_allocate(4500, 10000, 300_000) is False  # 10000+4500 > 12000


def test_risk_budget_can_enter_trade():
    rb = RiskBudget(
        max_capital_at_risk_today_pct=4.0,
        max_trades_today=4,
        max_per_trade_risk_pct=1.5,
        survival_mode=False,
    )
    # All guards pass
    assert rb.can_enter_trade(
        new_risk=4500, current_risk=0, capital=300_000,
        trades_today=0, consecutive_sl_hits=0, daily_pnl=-1000,
    ) is True
    # Max trades hit
    assert rb.can_enter_trade(
        new_risk=4500, current_risk=0, capital=300_000,
        trades_today=4, consecutive_sl_hits=0, daily_pnl=0,
    ) is False
    # 3 consecutive SL hits
    assert rb.can_enter_trade(
        new_risk=4500, current_risk=0, capital=300_000,
        trades_today=1, consecutive_sl_hits=3, daily_pnl=0,
    ) is False
    # Daily loss > 2%
    assert rb.can_enter_trade(
        new_risk=4500, current_risk=0, capital=300_000,
        trades_today=1, consecutive_sl_hits=0, daily_pnl=-6500,
    ) is False


def test_risk_budget_survival_mode():
    rb = RiskBudget(
        max_capital_at_risk_today_pct=4.0,
        max_trades_today=4,
        max_per_trade_risk_pct=1.5,
        survival_mode=True,
    )
    # In survival mode, no directional trades allowed
    assert rb.allows_directional() is False
    assert rb.allows_theta() is True


def test_risk_budget_full_stop():
    rb = RiskBudget(
        max_capital_at_risk_today_pct=4.0,
        max_trades_today=4,
        max_per_trade_risk_pct=1.5,
        survival_mode=False,
        pacing_status=PacingStatus.FULL_STOP,
    )
    # In full stop, nothing allowed
    assert rb.allows_directional() is False
    assert rb.allows_theta() is False


def test_playbook_serialization():
    pb = Playbook(
        date=date(2026, 3, 11),
        day_classification=DayClassification.LIKELY_TREND_UP,
        nifty_bias="bullish",
        nifty_setups=[],
        stock_plans=[],
        risk_budget=RiskBudget(
            max_capital_at_risk_today_pct=4.0,
            max_trades_today=4,
            max_per_trade_risk_pct=1.5,
            survival_mode=False,
        ),
        no_trade_conditions=["VIX > 22"],
        carry_rules=CarryRules(
            min_profit_pct=1.5, max_vix=20.0, min_dte=3,
            max_hedge_cost=500.0, never_carry=["expiry_day"],
        ),
    )
    d = pb.to_dict()
    pb2 = Playbook.from_dict(d)
    assert pb2.date == pb.date
    assert pb2.day_classification == pb.day_classification
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'v7'`

- [ ] **Step 3: Create package init**

```python
# v7/__init__.py
"""V7 Professional Trader Bot."""
```

- [ ] **Step 4: Implement types**

```python
# v7/types.py
"""Shared types for V7 trading bot.

All data structures that cross component boundaries are defined here.
Dataclasses with to_dict/from_dict for JSON persistence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time
from enum import Enum
from typing import Any


# ── Enums ──────────────────────────────────────────────────────────────


class DayPhase(Enum):
    """Current phase of the trading day. Drives executor behavior."""
    PRE_MARKET = "pre_market"          # before 9:15
    OPENING_READ = "opening_read"      # 9:15-9:45
    ACTIVE_TRADING = "active_trading"  # 9:45-14:30
    WIND_DOWN = "wind_down"            # 14:30-15:15
    POST_CLOSE = "post_close"          # 15:15-15:30
    OUTSIDE_HOURS = "outside_hours"    # before 8:30 or after 15:30

    @classmethod
    def from_time(cls, t: time) -> DayPhase:
        mins = t.hour * 60 + t.minute
        if mins < 510:       # before 8:30
            return cls.OUTSIDE_HOURS
        if mins < 555:       # 8:30-9:14
            return cls.PRE_MARKET
        if mins < 585:       # 9:15-9:44
            return cls.OPENING_READ
        if mins < 870:       # 9:45-14:29
            return cls.ACTIVE_TRADING
        if mins < 915:       # 14:30-15:14
            return cls.WIND_DOWN
        if mins <= 930:       # 15:15-15:30
            return cls.POST_CLOSE
        return cls.OUTSIDE_HOURS


class DayClassification(Enum):
    LIKELY_TREND_UP = "LIKELY_TREND_UP"
    LIKELY_TREND_DOWN = "LIKELY_TREND_DOWN"
    LIKELY_RANGE = "LIKELY_RANGE"
    UNCERTAIN = "UNCERTAIN"
    EVENT_DAY = "EVENT_DAY"
    NO_TRADE = "NO_TRADE"


class Conviction(Enum):
    HIGH = "high"      # 2.0% risk
    MEDIUM = "medium"  # 1.5% risk
    LOW = "low"        # 0.75% risk


class SetupType(Enum):
    BREAKOUT_LONG = "breakout_long"
    BREAKOUT_SHORT = "breakout_short"
    SUPPORT_BOUNCE = "support_bounce"
    RESISTANCE_FADE = "resistance_fade"
    CREDIT_SPREAD_BULL = "credit_spread_bull"
    CREDIT_SPREAD_BEAR = "credit_spread_bear"
    IRON_CONDOR = "iron_condor"


# ── Data Classes ───────────────────────────────────────────────────────


@dataclass
class Setup:
    """A single trade setup from the playbook."""
    id: str
    priority: int
    type: SetupType
    symbol: str
    trigger_level: float
    trigger_condition: str
    instrument: str          # e.g. "NIFTY CE 24400" or "HDFCBANK PE"
    strike_logic: str        # how to pick the strike
    target: float
    stoploss: float
    max_risk_pct: float
    conviction: Conviction = Conviction.MEDIUM
    fired: bool = False
    cancelled: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id, "priority": self.priority,
            "type": self.type.value, "symbol": self.symbol,
            "trigger_level": self.trigger_level,
            "trigger_condition": self.trigger_condition,
            "instrument": self.instrument,
            "strike_logic": self.strike_logic,
            "target": self.target, "stoploss": self.stoploss,
            "max_risk_pct": self.max_risk_pct,
            "conviction": self.conviction.value,
            "fired": self.fired, "cancelled": self.cancelled,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Setup:
        return cls(
            id=d["id"], priority=d["priority"],
            type=SetupType(d["type"]), symbol=d["symbol"],
            trigger_level=d["trigger_level"],
            trigger_condition=d["trigger_condition"],
            instrument=d["instrument"],
            strike_logic=d["strike_logic"],
            target=d["target"], stoploss=d["stoploss"],
            max_risk_pct=d["max_risk_pct"],
            conviction=Conviction(d.get("conviction", "medium")),
            fired=d.get("fired", False),
            cancelled=d.get("cancelled", False),
        )


@dataclass
class Position:
    """An open trading position."""
    symbol: str
    instrument: str          # e.g. "NIFTY CE 24400"
    direction: str           # "bullish" or "bearish"
    entry_price: float
    quantity: int
    lot_size: int
    allocated: float         # capital committed
    stoploss: float
    target: float
    entry_date: date
    setup_id: str
    peak_price: float = 0.0
    sl_order_id: str | None = None  # exchange SL order ID
    carried: bool = False
    hedge_instrument: str | None = None
    hedge_cost: float = 0.0

    def __post_init__(self):
        if self.peak_price == 0.0:
            self.peak_price = self.entry_price

    def unrealized_pnl(self, current_price: float) -> float:
        if self.direction == "bullish":
            return self.quantity * (current_price - self.entry_price)
        return self.quantity * (self.entry_price - current_price)

    def unrealized_pnl_pct(self, current_price: float) -> float:
        cost = self.entry_price * self.quantity
        if cost == 0:
            return 0.0
        return (self.unrealized_pnl(current_price) / cost) * 100

    def risk_amount(self) -> float:
        """Max loss if SL is hit."""
        if self.direction == "bullish":
            return self.quantity * (self.entry_price - self.stoploss)
        return self.quantity * (self.stoploss - self.entry_price)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "instrument": self.instrument,
            "direction": self.direction, "entry_price": self.entry_price,
            "quantity": self.quantity, "lot_size": self.lot_size,
            "allocated": self.allocated, "stoploss": self.stoploss,
            "target": self.target,
            "entry_date": str(self.entry_date), "setup_id": self.setup_id,
            "peak_price": self.peak_price,
            "sl_order_id": self.sl_order_id,
            "carried": self.carried,
            "hedge_instrument": self.hedge_instrument,
            "hedge_cost": self.hedge_cost,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Position:
        return cls(
            symbol=d["symbol"], instrument=d["instrument"],
            direction=d["direction"], entry_price=d["entry_price"],
            quantity=d["quantity"], lot_size=d["lot_size"],
            allocated=d["allocated"], stoploss=d["stoploss"],
            target=d["target"],
            entry_date=date.fromisoformat(d["entry_date"]),
            setup_id=d["setup_id"],
            peak_price=d.get("peak_price", d["entry_price"]),
            sl_order_id=d.get("sl_order_id"),
            carried=d.get("carried", False),
            hedge_instrument=d.get("hedge_instrument"),
            hedge_cost=d.get("hedge_cost", 0.0),
        )


@dataclass
class TradeResult:
    """A completed (closed) trade."""
    symbol: str
    instrument: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: int
    entry_date: date
    exit_date: date
    exit_reason: str         # "target", "stoploss", "trailing", "wind_down", "carry_gap", etc.
    pnl: float
    pnl_pct: float
    costs: float
    setup_id: str
    setup_type: SetupType
    entry_grade: str = "B"   # A/B/C
    exit_grade: str = "B"
    lesson: str = ""

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "instrument": self.instrument,
            "direction": self.direction,
            "entry_price": self.entry_price, "exit_price": self.exit_price,
            "quantity": self.quantity,
            "entry_date": str(self.entry_date),
            "exit_date": str(self.exit_date),
            "exit_reason": self.exit_reason,
            "pnl": self.pnl, "pnl_pct": self.pnl_pct,
            "costs": self.costs, "setup_id": self.setup_id,
            "setup_type": self.setup_type.value,
            "entry_grade": self.entry_grade,
            "exit_grade": self.exit_grade,
            "lesson": self.lesson,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TradeResult:
        return cls(
            symbol=d["symbol"], instrument=d["instrument"],
            direction=d["direction"],
            entry_price=d["entry_price"], exit_price=d["exit_price"],
            quantity=d["quantity"],
            entry_date=date.fromisoformat(d["entry_date"]),
            exit_date=date.fromisoformat(d["exit_date"]),
            exit_reason=d["exit_reason"],
            pnl=d["pnl"], pnl_pct=d["pnl_pct"],
            costs=d["costs"], setup_id=d["setup_id"],
            setup_type=SetupType(d["setup_type"]),
            entry_grade=d.get("entry_grade", "B"),
            exit_grade=d.get("exit_grade", "B"),
            lesson=d.get("lesson", ""),
        )


@dataclass
class CarryRules:
    """Rules for overnight carry decisions."""
    min_profit_pct: float = 1.5
    max_vix: float = 20.0
    min_dte: int = 3
    max_hedge_cost: float = 500.0
    never_carry: list[str] = field(default_factory=lambda: ["expiry_day", "event_tomorrow", "vix_above_22"])

    def to_dict(self) -> dict:
        return {
            "min_profit_pct": self.min_profit_pct,
            "max_vix": self.max_vix, "min_dte": self.min_dte,
            "max_hedge_cost": self.max_hedge_cost,
            "never_carry": self.never_carry,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CarryRules:
        return cls(**d)


class PacingStatus(Enum):
    ON_TRACK = "on_track"
    AHEAD = "ahead"
    BEHIND = "behind"
    SURVIVAL = "survival"
    FULL_STOP = "full_stop"


@dataclass
class RiskBudget:
    """Daily risk budget set by the Strategist."""
    max_capital_at_risk_today_pct: float = 4.0
    max_trades_today: int = 4
    max_per_trade_risk_pct: float = 1.5
    survival_mode: bool = False
    pacing_status: PacingStatus = PacingStatus.ON_TRACK
    mtd_pnl_pct: float = 0.0
    monthly_target_pct: float = 5.0

    def can_allocate(self, new_risk: float, current_risk: float, capital: float) -> bool:
        """Check concurrent risk budget only."""
        max_risk = capital * (self.max_capital_at_risk_today_pct / 100)
        return (current_risk + new_risk) <= max_risk

    def can_enter_trade(
        self, new_risk: float, current_risk: float, capital: float,
        trades_today: int, consecutive_sl_hits: int, daily_pnl: float,
    ) -> bool:
        """Comprehensive entry gate — checks ALL daily guards."""
        # Guard 1: concurrent risk budget
        if not self.can_allocate(new_risk, current_risk, capital):
            return False
        # Guard 2: max trades per day
        if trades_today >= self.max_trades_today:
            return False
        # Guard 3: 3 consecutive SL hits
        if consecutive_sl_hits >= 3:
            return False
        # Guard 4: daily loss > 2% of capital
        if daily_pnl < -(capital * 0.02):
            return False
        # Guard 5: survival or full stop mode
        if not self.allows_directional():
            return False
        return True

    def allows_directional(self) -> bool:
        if self.pacing_status == PacingStatus.FULL_STOP:
            return False
        return not self.survival_mode

    def allows_theta(self) -> bool:
        if self.pacing_status == PacingStatus.FULL_STOP:
            return False
        return True  # theta allowed even in survival mode

    def risk_pct_for_conviction(self, conviction: Conviction) -> float:
        base = {
            Conviction.HIGH: 2.0,
            Conviction.MEDIUM: 1.5,
            Conviction.LOW: 0.75,
        }[conviction]
        if self.pacing_status == PacingStatus.AHEAD:
            return base * 0.75  # protect gains
        return base

    def to_dict(self) -> dict:
        return {
            "max_capital_at_risk_today_pct": self.max_capital_at_risk_today_pct,
            "max_trades_today": self.max_trades_today,
            "max_per_trade_risk_pct": self.max_per_trade_risk_pct,
            "survival_mode": self.survival_mode,
            "pacing_status": self.pacing_status.value,
            "mtd_pnl_pct": self.mtd_pnl_pct,
            "monthly_target_pct": self.monthly_target_pct,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RiskBudget:
        return cls(
            max_capital_at_risk_today_pct=d["max_capital_at_risk_today_pct"],
            max_trades_today=d["max_trades_today"],
            max_per_trade_risk_pct=d["max_per_trade_risk_pct"],
            survival_mode=d.get("survival_mode", False),
            pacing_status=PacingStatus(d.get("pacing_status", "on_track")),
            mtd_pnl_pct=d.get("mtd_pnl_pct", 0.0),
            monthly_target_pct=d.get("monthly_target_pct", 5.0),
        )


@dataclass
class MarketContext:
    """Structured market context for the playbook."""
    us_close: str = ""             # e.g. "+0.3%"
    gift_nifty: str = ""           # e.g. "24250 (+0.2%)"
    vix: float = 0.0
    fii_dii: str = ""              # e.g. "FII -1200cr, DII +800cr"
    events_today: list[str] = field(default_factory=list)
    events_this_week: list[str] = field(default_factory=list)
    fo_ban_list: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "us_close": self.us_close, "gift_nifty": self.gift_nifty,
            "vix": self.vix, "fii_dii": self.fii_dii,
            "events_today": self.events_today,
            "events_this_week": self.events_this_week,
            "fo_ban_list": self.fo_ban_list,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MarketContext:
        return cls(
            us_close=d.get("us_close", ""),
            gift_nifty=d.get("gift_nifty", ""),
            vix=d.get("vix", 0.0),
            fii_dii=d.get("fii_dii", ""),
            events_today=d.get("events_today", []),
            events_this_week=d.get("events_this_week", []),
            fo_ban_list=d.get("fo_ban_list", []),
        )


@dataclass
class KeyLevels:
    """Key price levels for an instrument."""
    resistance_1: float = 0.0
    resistance_2: float = 0.0
    support_1: float = 0.0
    support_2: float = 0.0
    opening_range_high: float | None = None
    opening_range_low: float | None = None
    no_trade_zone: tuple[float, float] | None = None  # (low, high) price band

    def to_dict(self) -> dict:
        return {
            "resistance_1": self.resistance_1, "resistance_2": self.resistance_2,
            "support_1": self.support_1, "support_2": self.support_2,
            "opening_range_high": self.opening_range_high,
            "opening_range_low": self.opening_range_low,
            "no_trade_zone": list(self.no_trade_zone) if self.no_trade_zone else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> KeyLevels:
        ntz = d.get("no_trade_zone")
        return cls(
            resistance_1=d.get("resistance_1", 0.0),
            resistance_2=d.get("resistance_2", 0.0),
            support_1=d.get("support_1", 0.0),
            support_2=d.get("support_2", 0.0),
            opening_range_high=d.get("opening_range_high"),
            opening_range_low=d.get("opening_range_low"),
            no_trade_zone=tuple(ntz) if ntz else None,
        )

    def in_no_trade_zone(self, price: float) -> bool:
        if self.no_trade_zone is None:
            return False
        return self.no_trade_zone[0] <= price <= self.no_trade_zone[1]


@dataclass
class Playbook:
    """The daily trading plan generated by the Strategist."""
    date: date
    day_classification: DayClassification
    nifty_bias: str
    nifty_setups: list[Setup]
    stock_plans: list[Setup]
    risk_budget: RiskBudget
    no_trade_conditions: list[str]
    carry_rules: CarryRules
    market_context: MarketContext = field(default_factory=MarketContext)
    key_levels: dict[str, KeyLevels] = field(default_factory=dict)  # symbol -> KeyLevels
    opening_range: dict[str, float] | None = None  # filled after 9:45
    theta_action: str = "hold"  # "hold", "enter", "adjust", "exit"
    theta_details: str = ""

    def all_setups(self) -> list[Setup]:
        return sorted(self.nifty_setups + self.stock_plans, key=lambda s: s.priority)

    def active_setups(self) -> list[Setup]:
        return [s for s in self.all_setups() if not s.fired and not s.cancelled]

    def to_dict(self) -> dict:
        return {
            "date": str(self.date),
            "day_classification": self.day_classification.value,
            "nifty_bias": self.nifty_bias,
            "nifty_setups": [s.to_dict() for s in self.nifty_setups],
            "stock_plans": [s.to_dict() for s in self.stock_plans],
            "risk_budget": self.risk_budget.to_dict(),
            "no_trade_conditions": self.no_trade_conditions,
            "carry_rules": self.carry_rules.to_dict(),
            "market_context": self.market_context.to_dict(),
            "key_levels": {k: v.to_dict() for k, v in self.key_levels.items()},
            "opening_range": self.opening_range,
            "theta_action": self.theta_action,
            "theta_details": self.theta_details,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Playbook:
        mc = d.get("market_context", {})
        market_context = MarketContext.from_dict(mc) if mc else MarketContext()
        kl = d.get("key_levels", {})
        key_levels = {k: KeyLevels.from_dict(v) for k, v in kl.items()}
        return cls(
            date=date.fromisoformat(d["date"]),
            day_classification=DayClassification(d["day_classification"]),
            nifty_bias=d["nifty_bias"],
            nifty_setups=[Setup.from_dict(s) for s in d["nifty_setups"]],
            stock_plans=[Setup.from_dict(s) for s in d["stock_plans"]],
            risk_budget=RiskBudget.from_dict(d["risk_budget"]),
            no_trade_conditions=d["no_trade_conditions"],
            carry_rules=CarryRules.from_dict(d["carry_rules"]),
            market_context=market_context,
            key_levels=key_levels,
            opening_range=d.get("opening_range"),
            theta_action=d.get("theta_action", "hold"),
            theta_details=d.get("theta_details", ""),
        )
```

- [ ] **Step 5: Run tests**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_types.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add v7/__init__.py v7/types.py tests/test_v7_types.py
git commit -m "feat(v7): add shared types — Playbook, Setup, Position, RiskBudget"
```

---

### Task 2: V7 config

**Files:**
- Create: `v7/config_v7.py`
- Test: `tests/test_v7_config.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_v7_config.py
"""Tests for V7-specific configuration."""
import pytest
from v7.config_v7 import (
    WATCHLIST, CAPITAL, PHASE_TIMES,
    RISK_LIMITS, THETA_LIMITS, BROKERAGE,
    is_15min_boundary, get_conviction_risk_pct,
)
from datetime import time


def test_watchlist_has_10_instruments():
    assert len(WATCHLIST) == 10
    assert "NIFTY" in [w["symbol"] for w in WATCHLIST]
    assert "BANKNIFTY" in [w["symbol"] for w in WATCHLIST]


def test_watchlist_has_lot_sizes():
    nifty = next(w for w in WATCHLIST if w["symbol"] == "NIFTY")
    assert nifty["lot_size"] == 75
    assert nifty["type"] == "index"


def test_capital_defaults():
    assert CAPITAL["initial"] == 300_000
    assert CAPITAL["cash_reserve_pct"] == 0.20


def test_phase_times():
    assert PHASE_TIMES["opening_read_end"] == time(9, 45)
    assert PHASE_TIMES["active_start"] == time(9, 45)
    assert PHASE_TIMES["wind_down_start"] == time(14, 30)


def test_risk_limits():
    assert RISK_LIMITS["max_daily_risk_pct"] == 4.0
    assert RISK_LIMITS["max_per_trade_risk_pct"] == 1.5
    assert RISK_LIMITS["max_trades_per_day"] == 4
    assert RISK_LIMITS["max_concurrent_positions"] == 4
    assert RISK_LIMITS["survival_mode_threshold_pct"] == 5.0
    assert RISK_LIMITS["full_stop_threshold_pct"] == 8.0


def test_theta_limits():
    assert THETA_LIMITS["max_margin_pct"] == 0.40
    assert THETA_LIMITS["min_vix"] == 14.0
    assert THETA_LIMITS["max_vix"] == 20.0
    assert THETA_LIMITS["profit_target_pct"] == 0.50


def test_brokerage():
    assert BROKERAGE["flat_per_order"] == 20.0
    assert BROKERAGE["min_trade_value"] == 2000.0


def test_is_15min_boundary():
    assert is_15min_boundary(time(10, 0)) is True
    assert is_15min_boundary(time(10, 15)) is True
    assert is_15min_boundary(time(10, 30)) is True
    assert is_15min_boundary(time(10, 45)) is True
    assert is_15min_boundary(time(10, 3)) is False
    assert is_15min_boundary(time(10, 14)) is False
    # Allow 1-min tolerance for cron timing
    assert is_15min_boundary(time(10, 1)) is True
    assert is_15min_boundary(time(10, 16)) is True


def test_get_conviction_risk_pct():
    assert get_conviction_risk_pct("high") == 2.0
    assert get_conviction_risk_pct("medium") == 1.5
    assert get_conviction_risk_pct("low") == 0.75
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_config.py -v`
Expected: FAIL

- [ ] **Step 3: Implement config**

```python
# v7/config_v7.py
"""V7-specific configuration.

Imports shared config from parent config.py for credentials, market hours,
holidays, etc. Defines V7-specific constants.
"""
from datetime import time


# ── Watchlist ──────────────────────────────────────────────────────────
WATCHLIST = [
    {"symbol": "NIFTY",      "type": "index", "lot_size": 75,  "token": "99926000", "exchange": "NSE"},
    {"symbol": "BANKNIFTY",  "type": "index", "lot_size": 30,  "token": "99926009", "exchange": "NSE"},
    {"symbol": "RELIANCE",   "type": "stock", "lot_size": 250, "token": "2885",     "exchange": "NSE"},
    {"symbol": "HDFCBANK",   "type": "stock", "lot_size": 550, "token": "1333",     "exchange": "NSE"},
    {"symbol": "ICICIBANK",  "type": "stock", "lot_size": 700, "token": "4963",     "exchange": "NSE"},
    {"symbol": "TCS",        "type": "stock", "lot_size": 175, "token": "11536",    "exchange": "NSE"},
    {"symbol": "TATAMOTORS", "type": "stock", "lot_size": 575, "token": "3456",     "exchange": "NSE"},
    {"symbol": "BAJFINANCE", "type": "stock", "lot_size": 125, "token": "317",      "exchange": "NSE"},
    {"symbol": "SBIN",       "type": "stock", "lot_size": 750, "token": "3045",     "exchange": "NSE"},
    {"symbol": "INFY",       "type": "stock", "lot_size": 300, "token": "1594",     "exchange": "NSE"},
]

# ── Capital ────────────────────────────────────────────────────────────
CAPITAL = {
    "initial": 300_000,
    "cash_reserve_pct": 0.20,       # never deploy more than 80%
    "margin_buffer_pct": 0.30,      # keep 30% margin free for MTM
}

# ── Phase Times (IST) ─────────────────────────────────────────────────
PHASE_TIMES = {
    "premarket_start": time(8, 45),
    "market_open": time(9, 15),
    "opening_read_end": time(9, 45),
    "active_start": time(9, 45),
    "checkin_1": time(10, 30),
    "checkin_2": time(13, 0),
    "wind_down_start": time(14, 30),
    "wind_down_end": time(15, 15),
    "post_close_end": time(15, 30),
    "eod_review": time(15, 33),
}

# ── Risk Limits ────────────────────────────────────────────────────────
RISK_LIMITS = {
    "max_daily_risk_pct": 4.0,          # concurrent capital at risk
    "max_per_trade_risk_pct": 1.5,      # per trade (medium conviction)
    "max_trades_per_day": 4,
    "max_concurrent_positions": 4,       # at 3-5L capital
    "max_consecutive_sl_daily": 3,       # stop after 3 SL hits
    "survival_mode_threshold_pct": 5.0,  # MTD drawdown → theta only
    "full_stop_threshold_pct": 8.0,      # MTD drawdown → no trading
    "drawdown_reduce_pct": 3.0,          # reduce size by 25%
}

# ── Theta Engine Limits ────────────────────────────────────────────────
THETA_LIMITS = {
    "max_margin_pct": 0.40,          # 40% of margin for theta
    "min_vix": 14.0,
    "max_vix": 20.0,
    "short_delta": 0.20,
    "wing_gap_nifty": 200,           # points OTM for hedge
    "profit_target_pct": 0.50,       # close at 50% credit captured
    "close_by_day": "wednesday",     # don't hold to expiry
    "survival_delta": 0.15,          # wider wings in survival mode
    "max_risk_pct": 3.0,             # max 3% of capital at risk
}

# ── State & Recovery ──────────────────────────────────────────────────
STATE_DIR = "data/v7"                       # all state files live here
RESTART_COOLDOWN_SECONDS = 300              # 5 min cooldown after Pi restart

# ── Brokerage (Zerodha) ───────────────────────────────────────────────
BROKERAGE = {
    "flat_per_order": 20.0,
    "min_trade_value": 2000.0,       # brokerage < 1% of trade value
    "opt_stt_sell_pct": 0.000625,
    "opt_exchange_pct": 0.000495,
    "opt_stamp_duty_pct": 0.00003,
    "gst_pct": 0.18,
    "slippage_pct": 0.015,           # 1.5% for options
}

# ── Strike Selection ───────────────────────────────────────────────────
STRIKE_FILTERS = {
    "min_oi": 500,
    "min_volume": 100,
    "max_bid_ask_nifty": 2.0,
    "max_bid_ask_banknifty": 5.0,
    "max_bid_ask_stock": 3.0,
    "min_premium": 10.0,
    "directional_delta_range": (0.40, 0.50),
    "spread_sell_delta": 0.25,
    "hedge_delta": 0.10,
}

# ── Trailing Stop ──────────────────────────────────────────────────────
TRAILING = {
    "atr_period": 14,
    "atr_multiplier": 1.5,          # trail at 1.5x ATR on 5-min
    "breakeven_rr": 1.0,            # move SL to breakeven at 1:1 R:R
}

# ── Carry Rules ────────────────────────────────────────────────────────
CARRY = {
    "min_profit_pct": 1.5,
    "max_vix": 20.0,
    "min_dte": 3,
    "max_hedge_cost": 500.0,
    "never_carry": ["expiry_day", "event_tomorrow", "vix_above_22"],
}

# ── Telegram ───────────────────────────────────────────────────────────
TELEGRAM = {
    "heartbeat_interval_min": 30,
    "alert_on": ["entry", "exit", "exception", "carry", "eod"],
}


def is_15min_boundary(t: time) -> bool:
    """Check if time is within 1 minute of a 15-min candle close."""
    return t.minute % 15 <= 1


def get_conviction_risk_pct(conviction: str) -> float:
    """Get risk % per trade for a given conviction level."""
    return {"high": 2.0, "medium": 1.5, "low": 0.75}.get(conviction, 1.5)
```

- [ ] **Step 4: Run tests**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_config.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add v7/config_v7.py tests/test_v7_config.py
git commit -m "feat(v7): add V7 config — watchlist, risk limits, phase times"
```

---

### Task 3: State persistence

**Files:**
- Create: `v7/state.py`
- Test: `tests/test_v7_state.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_v7_state.py
"""Tests for V7 state persistence."""
import json
import pytest
from datetime import date, time
from pathlib import Path
from v7.state import StateManager
from v7.types import (
    Playbook, Setup, SetupType, Position, TradeResult,
    RiskBudget, CarryRules, DayClassification, Conviction,
)


@pytest.fixture
def tmp_state_dir(tmp_path):
    return tmp_path / "v7_state"


@pytest.fixture
def state(tmp_state_dir):
    return StateManager(tmp_state_dir)


@pytest.fixture
def sample_playbook():
    return Playbook(
        date=date(2026, 3, 11),
        day_classification=DayClassification.LIKELY_TREND_UP,
        nifty_bias="bullish",
        nifty_setups=[
            Setup(id="N1", priority=1, type=SetupType.BREAKOUT_LONG,
                  symbol="NIFTY", trigger_level=24350.0,
                  trigger_condition="15-min close above",
                  instrument="NIFTY CE", strike_logic="delta 0.45",
                  target=24500.0, stoploss=24280.0, max_risk_pct=1.5),
        ],
        stock_plans=[],
        risk_budget=RiskBudget(),
        no_trade_conditions=["VIX > 22"],
        carry_rules=CarryRules(),
    )


@pytest.fixture
def sample_position():
    return Position(
        symbol="NIFTY", instrument="NIFTY CE 24400",
        direction="bullish", entry_price=100.0,
        quantity=75, lot_size=75, allocated=7500.0,
        stoploss=80.0, target=150.0,
        entry_date=date(2026, 3, 11), setup_id="N1",
    )


def test_state_creates_directory(state, tmp_state_dir):
    assert tmp_state_dir.exists()


def test_save_and_load_playbook(state, sample_playbook):
    state.save_playbook(sample_playbook)
    loaded = state.load_playbook()
    assert loaded is not None
    assert loaded.date == sample_playbook.date
    assert loaded.day_classification == sample_playbook.day_classification
    assert len(loaded.nifty_setups) == 1


def test_load_playbook_returns_none_if_missing(state):
    assert state.load_playbook() is None


def test_load_playbook_returns_none_if_stale(state, sample_playbook):
    state.save_playbook(sample_playbook)
    # Simulate loading on a different day
    loaded = state.load_playbook(today=date(2026, 3, 12))
    assert loaded is None


def test_save_and_load_positions(state, sample_position):
    state.save_positions([sample_position])
    loaded = state.load_positions()
    assert len(loaded) == 1
    assert loaded[0].symbol == "NIFTY"
    assert loaded[0].entry_price == 100.0


def test_load_positions_empty_if_missing(state):
    assert state.load_positions() == []


def test_save_and_load_daily_state(state):
    daily = {
        "date": "2026-03-11",
        "trades_today": 2,
        "sl_hits_today": 1,
        "daily_pnl": -1500.0,
        "current_risk": 4500.0,
    }
    state.save_daily_state(daily)
    loaded = state.load_daily_state()
    assert loaded["trades_today"] == 2
    assert loaded["daily_pnl"] == -1500.0


def test_daily_state_resets_on_new_day(state):
    daily = {"date": "2026-03-10", "trades_today": 3, "sl_hits_today": 2,
             "daily_pnl": -2000.0, "current_risk": 0.0}
    state.save_daily_state(daily)
    loaded = state.load_daily_state(today=date(2026, 3, 11))
    assert loaded["trades_today"] == 0  # reset for new day


def test_save_and_load_trade_history(state):
    tr = TradeResult(
        symbol="NIFTY", instrument="NIFTY CE 24400",
        direction="bullish", entry_price=100.0, exit_price=150.0,
        quantity=75, entry_date=date(2026, 3, 11),
        exit_date=date(2026, 3, 11), exit_reason="target",
        pnl=3750.0, pnl_pct=50.0, costs=120.0,
        setup_id="N1", setup_type=SetupType.BREAKOUT_LONG,
    )
    state.append_trade(tr)
    history = state.load_trade_history()
    assert len(history) == 1
    assert history[0].pnl == 3750.0


def test_append_trade_preserves_existing(state):
    for i in range(3):
        tr = TradeResult(
            symbol=f"SYM{i}", instrument=f"OPT{i}",
            direction="bullish", entry_price=100.0, exit_price=110.0,
            quantity=75, entry_date=date(2026, 3, 11),
            exit_date=date(2026, 3, 11), exit_reason="target",
            pnl=750.0, pnl_pct=10.0, costs=40.0,
            setup_id=f"S{i}", setup_type=SetupType.BREAKOUT_LONG,
        )
        state.append_trade(tr)
    assert len(state.load_trade_history()) == 3


def test_save_and_load_level_memory(state):
    levels = {
        "NIFTY": {
            "levels": [
                {"price": 24000, "type": "support", "strength": 3,
                 "source": "tested 3x", "last_tested": "2026-03-10"},
            ],
            "oi_walls": {"call_max_oi_strike": 24500, "put_max_oi_strike": 24000},
        }
    }
    state.save_level_memory(levels)
    loaded = state.load_level_memory()
    assert "NIFTY" in loaded
    assert loaded["NIFTY"]["levels"][0]["strength"] == 3


def test_save_and_load_edge_tracker(state):
    edge = {
        "overall_win_rate": 0.55,
        "by_strategy": {
            "momentum_breakout": {"trades": 45, "win_rate": 0.58, "avg_rr": 1.8},
        },
        "by_instrument": {
            "NIFTY": {"trades": 35, "net_pnl": 12000},
        },
        "by_time": {
            "9:45-11:00": {"trades": 40, "win_rate": 0.60},
        },
    }
    state.save_edge_tracker(edge)
    loaded = state.load_edge_tracker()
    assert loaded["overall_win_rate"] == 0.55
    assert loaded["by_strategy"]["momentum_breakout"]["trades"] == 45


def test_save_and_load_monthly_state(state):
    monthly = {
        "month": "2026-03",
        "mtd_pnl": 5000.0,
        "mtd_pnl_pct": 1.67,
        "trades_this_month": 15,
        "survival_mode": False,
    }
    state.save_monthly_state(monthly)
    loaded = state.load_monthly_state()
    assert loaded["mtd_pnl"] == 5000.0
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_state.py -v`
Expected: FAIL

- [ ] **Step 3: Implement state manager**

```python
# v7/state.py
"""State persistence for V7 trading bot.

All runtime state is persisted to JSON files in data/v7/.
Handles load, save, reset, and stale-data detection.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from v7.types import Playbook, Position, TradeResult


class StateManager:
    """File-backed state persistence for V7."""

    def __init__(self, state_dir: str | Path):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.dir / name

    def _read_json(self, name: str) -> dict | list | None:
        path = self._path(name)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def _write_json(self, name: str, data: dict | list) -> None:
        path = self._path(name)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    # ── Playbook ───────────────────────────────────────────────────

    def save_playbook(self, playbook: Playbook) -> None:
        self._write_json("playbook.json", playbook.to_dict())

    def load_playbook(self, today: date | None = None) -> Playbook | None:
        data = self._read_json("playbook.json")
        if data is None:
            return None
        pb = Playbook.from_dict(data)
        today = today or date.today()
        if pb.date != today:
            return None  # stale playbook from a different day
        return pb

    # ── Positions ──────────────────────────────────────────────────

    def save_positions(self, positions: list[Position]) -> None:
        self._write_json("positions.json", [p.to_dict() for p in positions])

    def load_positions(self) -> list[Position]:
        data = self._read_json("positions.json")
        if data is None:
            return []
        return [Position.from_dict(d) for d in data]

    # ── Daily State ────────────────────────────────────────────────

    def save_daily_state(self, state: dict) -> None:
        self._write_json("daily_state.json", state)

    def load_daily_state(self, today: date | None = None) -> dict:
        today = today or date.today()
        data = self._read_json("daily_state.json")
        if data is None or data.get("date") != str(today):
            return {
                "date": str(today),
                "trades_today": 0,
                "sl_hits_today": 0,
                "daily_pnl": 0.0,
                "current_risk": 0.0,
            }
        return data

    # ── Trade History ──────────────────────────────────────────────

    def append_trade(self, trade: TradeResult) -> None:
        history = self._read_json("trade_history.json") or []
        history.append(trade.to_dict())
        self._write_json("trade_history.json", history)

    def load_trade_history(self) -> list[TradeResult]:
        data = self._read_json("trade_history.json")
        if data is None:
            return []
        return [TradeResult.from_dict(d) for d in data]

    # ── Level Memory ───────────────────────────────────────────────

    def save_level_memory(self, levels: dict) -> None:
        self._write_json("level_memory.json", levels)

    def load_level_memory(self) -> dict:
        return self._read_json("level_memory.json") or {}

    # ── Monthly State ──────────────────────────────────────────────

    def save_monthly_state(self, state: dict) -> None:
        self._write_json("monthly_state.json", state)

    def load_monthly_state(self) -> dict:
        data = self._read_json("monthly_state.json")
        if data is None:
            return {
                "month": str(date.today())[:7],
                "mtd_pnl": 0.0,
                "mtd_pnl_pct": 0.0,
                "trades_this_month": 0,
                "survival_mode": False,
            }
        return data

    # ── Edge Tracker ───────────────────────────────────────────────

    def save_edge_tracker(self, data: dict) -> None:
        self._write_json("edge_tracker.json", data)

    def load_edge_tracker(self) -> dict:
        return self._read_json("edge_tracker.json") or {
            "overall_win_rate": 0.0,
            "total_trades": 0,
            "by_strategy": {},
            "by_instrument": {},
            "by_time": {},
        }
```

- [ ] **Step 4: Run tests**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_state.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add v7/state.py tests/test_v7_state.py
git commit -m "feat(v7): add state persistence — playbook, positions, daily/monthly state"
```

---

## Chunk 2: Data Feed & Strike Selector

### Task 4: Unified data feed

**Files:**
- Create: `v7/data_feed.py`
- Test: `tests/test_v7_data_feed.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_v7_data_feed.py
"""Tests for V7 unified data feed.

These tests use mocks — actual API calls tested via integration tests.
"""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from v7.data_feed import DataFeed, DataFeedError, ProtectOnlyMode


def test_data_feed_init():
    feed = DataFeed(use_kite=False, use_angelone=False)
    assert feed.mode == "offline"


def test_data_feed_kite_mode():
    with patch("kite_data.get_kite") as mock_kite:
        mock_kite.return_value = MagicMock()
        feed = DataFeed(use_kite=True, use_angelone=False)
        assert feed.mode == "kite"


def test_data_feed_protect_only_on_kite_failure():
    with patch("kite_data.get_kite", side_effect=Exception("Token expired")):
        with patch("connect.get_session") as mock_angel:
            mock_angel.return_value = MagicMock()
            feed = DataFeed(use_kite=True, use_angelone=True)
            assert feed.mode == "protect_only"


def test_batch_ltp_returns_dict():
    feed = DataFeed(use_kite=False, use_angelone=False)
    with patch.object(feed, "_fetch_ltp_batch") as mock:
        mock.return_value = {"NIFTY": 24200.0, "BANKNIFTY": 52000.0}
        prices = feed.get_batch_ltp(["NIFTY", "BANKNIFTY"])
        assert prices["NIFTY"] == 24200.0


def test_cannot_trade_in_protect_only():
    feed = DataFeed(use_kite=False, use_angelone=False)
    feed.mode = "protect_only"
    assert feed.can_trade() is False


def test_can_trade_in_kite_mode():
    feed = DataFeed(use_kite=False, use_angelone=False)
    feed.mode = "kite"
    assert feed.can_trade() is True


def test_get_candles_raises_in_protect_only():
    feed = DataFeed(use_kite=False, use_angelone=False)
    feed.mode = "protect_only"
    with pytest.raises(ProtectOnlyMode):
        feed.get_candles("NIFTY", interval="5minute", days=1)


def test_get_option_chain_raises_in_protect_only():
    feed = DataFeed(use_kite=False, use_angelone=False)
    feed.mode = "protect_only"
    with pytest.raises(ProtectOnlyMode):
        feed.get_option_chain("NIFTY", "26MAR2026")
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_data_feed.py -v`
Expected: FAIL

- [ ] **Step 3: Implement data feed**

```python
# v7/data_feed.py
"""Unified data feed for V7.

Kite primary, AngelOne protect-only fallback.
When Kite is unavailable, bot can only monitor LTP for existing positions.
"""
from __future__ import annotations

import logging
import time as time_mod
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))


class DataFeedError(Exception):
    pass


class ProtectOnlyMode(DataFeedError):
    """Raised when trying to do trading operations in protect-only mode."""
    pass


class DataFeed:
    """Abstraction over Kite + AngelOne data sources.

    Modes:
      - "kite": full trading capability
      - "protect_only": AngelOne LTP only, no new trades
      - "offline": no data source (for testing)
    """

    def __init__(self, use_kite: bool = True, use_angelone: bool = True):
        self.kite = None
        self.angelone = None
        self.mode = "offline"
        self._last_ltp: dict[str, float] = {}
        self._last_ltp_time: datetime | None = None

        if use_kite:
            try:
                from kite_data import get_kite
                self.kite = get_kite()
                self.mode = "kite"
                log.info("DataFeed: Kite connected")
            except Exception as e:
                log.warning(f"DataFeed: Kite unavailable — {e}")

        if use_angelone and self.mode != "kite":
            try:
                from connect import get_session
                self.angelone = get_session()
                self.mode = "protect_only"
                log.warning("DataFeed: Protect-only mode (AngelOne LTP only)")
            except Exception as e:
                log.error(f"DataFeed: AngelOne also unavailable — {e}")

    def can_trade(self) -> bool:
        return self.mode == "kite"

    def is_data_stale(self, max_age_seconds: int = 60) -> bool:
        if self._last_ltp_time is None:
            return True
        age = (datetime.now(IST) - self._last_ltp_time).total_seconds()
        return age > max_age_seconds

    # ── LTP ────────────────────────────────────────────────────────

    def get_batch_ltp(self, symbols: list[str]) -> dict[str, float]:
        """Fetch LTP for multiple symbols. Works in all modes."""
        try:
            prices = self._fetch_ltp_batch(symbols)
            self._last_ltp.update(prices)
            self._last_ltp_time = datetime.now(IST)
            return prices
        except Exception as e:
            log.error(f"DataFeed: LTP fetch failed — {e}")
            return self._last_ltp  # return cached if fetch fails

    def _fetch_ltp_batch(self, symbols: list[str]) -> dict[str, float]:
        if self.mode == "kite" and self.kite:
            from v7.config_v7 import WATCHLIST
            kite_symbols = []
            sym_map = {}
            for sym in symbols:
                wl = next((w for w in WATCHLIST if w["symbol"] == sym), None)
                if wl:
                    key = f"NSE:{sym}"
                    if wl["type"] == "index":
                        key = f"NSE:{sym} 50" if sym == "NIFTY" else f"NSE:{sym}"
                    kite_symbols.append(key)
                    sym_map[key] = sym
            if not kite_symbols:
                return {}
            quotes = self.kite.quote(kite_symbols)
            return {sym_map[k]: v["last_price"] for k, v in quotes.items() if k in sym_map}

        elif self.mode == "protect_only" and self.angelone:
            from v7.config_v7 import WATCHLIST
            prices = {}
            for sym in symbols:
                wl = next((w for w in WATCHLIST if w["symbol"] == sym), None)
                if wl:
                    try:
                        data = self.angelone.ltpData("NSE", sym, wl["token"])
                        if data and data.get("data"):
                            prices[sym] = float(data["data"]["ltp"])
                        time_mod.sleep(0.5)  # rate limit
                    except Exception:
                        pass
            return prices

        return {}

    # ── Candles ────────────────────────────────────────────────────

    def get_candles(self, symbol: str, interval: str = "5minute",
                    days: int = 1) -> list:
        """Fetch OHLCV candles. Only available in Kite mode."""
        if not self.can_trade():
            raise ProtectOnlyMode("Cannot fetch candles in protect-only mode")
        from kite_data import fetch_candles_kite, resolve_token
        from v7.config_v7 import WATCHLIST
        wl = next((w for w in WATCHLIST if w["symbol"] == symbol), None)
        if not wl:
            raise DataFeedError(f"Symbol {symbol} not in watchlist")
        token = resolve_token(symbol)
        return fetch_candles_kite(symbol, token, "NSE", interval, days)

    # ── Option Chain ───────────────────────────────────────────────

    def get_option_chain(self, symbol: str, expiry: str) -> list[dict]:
        """Fetch option chain. Only available in Kite mode."""
        if not self.can_trade():
            raise ProtectOnlyMode("Cannot fetch option chain in protect-only mode")
        from kite_data import fetch_option_chain_kite
        return fetch_option_chain_kite(symbol, expiry)

    # ── VIX ────────────────────────────────────────────────────────

    def get_vix(self) -> float:
        """Fetch India VIX."""
        if self.mode == "kite" and self.kite:
            from kite_data import get_vix_kite
            return get_vix_kite()
        # Fallback: try AngelOne
        if self.angelone:
            try:
                data = self.angelone.ltpData("NSE", "India VIX", "26017")
                if data and data.get("data"):
                    return float(data["data"]["ltp"])
            except Exception:
                pass
        return 0.0  # unknown

    # ── Health Check ───────────────────────────────────────────────

    def health_check(self) -> dict:
        """Return current data feed status."""
        return {
            "mode": self.mode,
            "can_trade": self.can_trade(),
            "stale": self.is_data_stale(),
            "last_update": str(self._last_ltp_time) if self._last_ltp_time else None,
            "cached_symbols": len(self._last_ltp),
        }

    def try_reconnect_kite(self) -> bool:
        """Attempt to reconnect to Kite. Returns True if successful."""
        try:
            from kite_data import get_kite
            self.kite = get_kite()
            self.mode = "kite"
            log.info("DataFeed: Kite reconnected")
            return True
        except Exception as e:
            log.warning(f"DataFeed: Kite reconnect failed — {e}")
            return False
```

- [ ] **Step 4: Run tests**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_data_feed.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add v7/data_feed.py tests/test_v7_data_feed.py
git commit -m "feat(v7): add unified data feed — Kite primary, protect-only fallback"
```

---

### Task 5: Strike selector

**Files:**
- Create: `v7/strike_selector.py`
- Test: `tests/test_v7_strike_selector.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_v7_strike_selector.py
"""Tests for mechanical strike selection."""
import pytest
from v7.strike_selector import (
    select_directional_strike, select_spread_strikes,
    select_hedge_strike, passes_liquidity_filter,
)


def make_chain_entry(strike, ce_ltp, pe_ltp, ce_oi=1000, pe_oi=1000,
                     ce_delta=0.5, pe_delta=-0.5,
                     ce_bid_ask=1.0, pe_bid_ask=1.0):
    return {
        "strikePrice": strike,
        "CE": {
            "ltp": ce_ltp, "oi": ce_oi, "volume": 500,
            "delta": ce_delta, "bidPrice": ce_ltp - ce_bid_ask/2,
            "askPrice": ce_ltp + ce_bid_ask/2,
        },
        "PE": {
            "ltp": pe_ltp, "oi": pe_oi, "volume": 500,
            "delta": pe_delta, "bidPrice": pe_ltp - pe_bid_ask/2,
            "askPrice": pe_ltp + pe_bid_ask/2,
        },
    }


@pytest.fixture
def sample_chain():
    """Nifty-like chain around 24200."""
    return [
        make_chain_entry(24000, 250, 50, ce_delta=0.65, pe_delta=-0.35, ce_oi=50000, pe_oi=40000),
        make_chain_entry(24100, 180, 80, ce_delta=0.58, pe_delta=-0.42, ce_oi=45000, pe_oi=35000),
        make_chain_entry(24200, 120, 120, ce_delta=0.50, pe_delta=-0.50, ce_oi=60000, pe_oi=60000),
        make_chain_entry(24300, 75, 175, ce_delta=0.42, pe_delta=-0.58, ce_oi=55000, pe_oi=30000),
        make_chain_entry(24400, 40, 240, ce_delta=0.33, pe_delta=-0.67, ce_oi=70000, pe_oi=20000),
        make_chain_entry(24500, 20, 320, ce_delta=0.22, pe_delta=-0.78, ce_oi=80000, pe_oi=15000),
        make_chain_entry(24600, 10, 410, ce_delta=0.14, pe_delta=-0.86, ce_oi=40000, pe_oi=10000),
    ]


def test_select_directional_call(sample_chain):
    result = select_directional_strike(
        chain=sample_chain, direction="bullish", spot=24200,
        risk_budget=4500, lot_size=75, symbol="NIFTY",
    )
    assert result is not None
    assert result["option_type"] == "CE"
    assert 0.35 <= abs(result["delta"]) <= 0.55
    assert result["premium"] * 75 <= 4500  # within budget


def test_select_directional_put(sample_chain):
    result = select_directional_strike(
        chain=sample_chain, direction="bearish", spot=24200,
        risk_budget=4500, lot_size=75, symbol="NIFTY",
    )
    assert result is not None
    assert result["option_type"] == "PE"


def test_select_directional_respects_budget(sample_chain):
    result = select_directional_strike(
        chain=sample_chain, direction="bullish", spot=24200,
        risk_budget=1000, lot_size=75, symbol="NIFTY",
    )
    # At lot_size=75, can only afford premium < 13.33
    # Only 24600 CE at 10 fits
    if result:
        assert result["premium"] * 75 <= 1000


def test_passes_liquidity_filter():
    assert passes_liquidity_filter(oi=1000, volume=200, bid_ask_spread=1.5, symbol="NIFTY")
    assert not passes_liquidity_filter(oi=100, volume=200, bid_ask_spread=1.5, symbol="NIFTY")  # low OI
    assert not passes_liquidity_filter(oi=1000, volume=200, bid_ask_spread=5.0, symbol="NIFTY")  # wide spread


def test_select_spread_strikes(sample_chain):
    result = select_spread_strikes(
        chain=sample_chain, direction="bearish", spot=24200,
        risk_budget=4500, lot_size=75, symbol="NIFTY",
    )
    if result:
        assert result["sell_strike"] > result["buy_strike"]  # bear put spread
        assert result["max_loss"] <= 4500


def test_select_hedge_strike(sample_chain):
    result = select_hedge_strike(
        chain=sample_chain, direction="bullish", spot=24200,
        max_cost=500, lot_size=75,
    )
    if result:
        # Hedge for bullish position = buy PE (protection against downside)
        assert result["option_type"] == "PE"
        assert result["premium"] * 75 <= 500
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_strike_selector.py -v`
Expected: FAIL

- [ ] **Step 3: Implement strike selector**

```python
# v7/strike_selector.py
"""Mechanical strike selection for V7.

No Claude calls. Pure rules based on delta, liquidity, and budget.
"""
from __future__ import annotations

from v7.config_v7 import STRIKE_FILTERS


def passes_liquidity_filter(oi: int, volume: int, bid_ask_spread: float,
                            symbol: str) -> bool:
    """Check if a strike passes minimum liquidity requirements."""
    if oi < STRIKE_FILTERS["min_oi"]:
        return False
    if volume < STRIKE_FILTERS["min_volume"]:
        return False

    max_spread = STRIKE_FILTERS["max_bid_ask_stock"]
    if symbol == "NIFTY":
        max_spread = STRIKE_FILTERS["max_bid_ask_nifty"]
    elif symbol == "BANKNIFTY":
        max_spread = STRIKE_FILTERS["max_bid_ask_banknifty"]

    if bid_ask_spread > max_spread:
        return False
    return True


def _get_bid_ask_spread(option_data: dict) -> float:
    bid = option_data.get("bidPrice", 0)
    ask = option_data.get("askPrice", 0)
    if bid and ask:
        return ask - bid
    return 999.0  # unknown → treat as illiquid


def select_directional_strike(
    chain: list[dict], direction: str, spot: float,
    risk_budget: float, lot_size: int, symbol: str,
) -> dict | None:
    """Select the best strike for a directional option buy.

    Returns dict with: strike, option_type, premium, delta, oi
    or None if nothing fits.
    """
    option_type = "CE" if direction == "bullish" else "PE"
    delta_min, delta_max = STRIKE_FILTERS["directional_delta_range"]
    min_premium = STRIKE_FILTERS["min_premium"]
    max_premium = risk_budget / lot_size

    candidates = []
    for entry in chain:
        opt = entry.get(option_type)
        if not opt or not opt.get("ltp"):
            continue

        delta = abs(opt.get("delta", 0))
        premium = opt["ltp"]
        oi = opt.get("oi", 0)
        volume = opt.get("volume", 0)
        spread = _get_bid_ask_spread(opt)

        if delta < delta_min or delta > delta_max:
            continue
        if premium < min_premium or premium > max_premium:
            continue
        if not passes_liquidity_filter(oi, volume, spread, symbol):
            continue

        candidates.append({
            "strike": entry["strikePrice"],
            "option_type": option_type,
            "premium": premium,
            "delta": delta,
            "oi": oi,
            "volume": volume,
            "bid_ask_spread": spread,
        })

    if not candidates:
        # Fallback: widen delta range to 0.30-0.60
        for entry in chain:
            opt = entry.get(option_type)
            if not opt or not opt.get("ltp"):
                continue
            delta = abs(opt.get("delta", 0))
            premium = opt["ltp"]
            if 0.30 <= delta <= 0.60 and min_premium <= premium <= max_premium:
                oi = opt.get("oi", 0)
                volume = opt.get("volume", 0)
                spread = _get_bid_ask_spread(opt)
                if passes_liquidity_filter(oi, volume, spread, symbol):
                    candidates.append({
                        "strike": entry["strikePrice"],
                        "option_type": option_type,
                        "premium": premium,
                        "delta": delta,
                        "oi": oi,
                        "volume": volume,
                        "bid_ask_spread": spread,
                    })

    if not candidates:
        return None

    # Pick: closest to target delta (0.45), then highest OI
    target_delta = 0.45
    candidates.sort(key=lambda c: (abs(c["delta"] - target_delta), -c["oi"]))
    return candidates[0]


def select_spread_strikes(
    chain: list[dict], direction: str, spot: float,
    risk_budget: float, lot_size: int, symbol: str,
) -> dict | None:
    """Select strikes for a credit spread.

    Bear: sell PE at delta ~0.25, buy PE further OTM.
    Bull: sell CE at delta ~0.25, buy CE further OTM.

    Returns dict with: sell_strike, buy_strike, sell_premium, buy_premium,
                       net_credit, max_loss, option_type
    """
    sell_delta = STRIKE_FILTERS["spread_sell_delta"]

    if direction == "bearish":
        option_type = "PE"
    else:
        option_type = "CE"

    # Find sell strike (delta ~0.25)
    sell_candidates = []
    for entry in chain:
        opt = entry.get(option_type)
        if not opt or not opt.get("ltp"):
            continue
        delta = abs(opt.get("delta", 0))
        if abs(delta - sell_delta) < 0.10:
            oi = opt.get("oi", 0)
            volume = opt.get("volume", 0)
            spread = _get_bid_ask_spread(opt)
            if passes_liquidity_filter(oi, volume, spread, symbol):
                sell_candidates.append({
                    "strike": entry["strikePrice"],
                    "premium": opt["ltp"],
                    "delta": delta,
                })

    if not sell_candidates:
        return None

    sell_candidates.sort(key=lambda c: abs(c["delta"] - sell_delta))
    sell = sell_candidates[0]

    # Find buy strike: 2-3 strikes further OTM
    strikes = sorted(set(e["strikePrice"] for e in chain))
    sell_idx = strikes.index(sell["strike"]) if sell["strike"] in strikes else -1
    if sell_idx < 0:
        return None

    if direction == "bearish":
        # Bear put: buy further OTM put (lower strike)
        buy_idx = sell_idx - 2 if sell_idx >= 2 else 0
    else:
        # Bull call: buy further OTM call (higher strike)
        buy_idx = sell_idx + 2 if sell_idx + 2 < len(strikes) else len(strikes) - 1

    buy_strike = strikes[buy_idx]
    buy_entry = next((e for e in chain if e["strikePrice"] == buy_strike), None)
    if not buy_entry:
        return None

    buy_opt = buy_entry.get(option_type)
    if not buy_opt or not buy_opt.get("ltp"):
        return None

    net_credit = sell["premium"] - buy_opt["ltp"]
    if net_credit < 15:  # minimum credit per lot
        return None

    strike_width = abs(sell["strike"] - buy_strike)
    max_loss = (strike_width - net_credit) * lot_size

    if max_loss > risk_budget:
        return None

    return {
        "sell_strike": sell["strike"],
        "buy_strike": buy_strike,
        "sell_premium": sell["premium"],
        "buy_premium": buy_opt["ltp"],
        "net_credit": net_credit,
        "max_loss": max_loss,
        "option_type": option_type,
    }


def select_hedge_strike(
    chain: list[dict], direction: str, spot: float,
    max_cost: float, lot_size: int,
) -> dict | None:
    """Select a protective hedge for overnight carry.

    Bullish position → buy OTM PE (downside protection).
    Bearish position → buy OTM CE (upside protection).
    """
    option_type = "PE" if direction == "bullish" else "CE"
    max_premium = max_cost / lot_size

    candidates = []
    for entry in chain:
        opt = entry.get(option_type)
        if not opt or not opt.get("ltp"):
            continue
        premium = opt["ltp"]
        if premium <= 0 or premium > max_premium:
            continue
        # Prefer 3-4 strikes OTM
        distance = abs(entry["strikePrice"] - spot)
        candidates.append({
            "strike": entry["strikePrice"],
            "option_type": option_type,
            "premium": premium,
            "delta": abs(opt.get("delta", 0)),
            "distance": distance,
        })

    if not candidates:
        return None

    # Sort by: reasonable distance from spot (not too close, not too far)
    # Target: 3-4% OTM
    target_distance = spot * 0.03
    candidates.sort(key=lambda c: abs(c["distance"] - target_distance))
    return candidates[0]
```

- [ ] **Step 4: Run tests**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_strike_selector.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add v7/strike_selector.py tests/test_v7_strike_selector.py
git commit -m "feat(v7): add mechanical strike selection — directional, spread, hedge"
```

---

### Task 6: Margin estimator

**Files:**
- Create: `v7/margin.py`
- Test: `tests/test_v7_margin.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_v7_margin.py
"""Tests for margin estimation."""
import pytest
from v7.margin import MarginTracker


@pytest.fixture
def tracker():
    return MarginTracker(capital=300_000)


def test_initial_state(tracker):
    assert tracker.available_margin() == 300_000
    assert tracker.utilization_pct() == 0.0


def test_add_bought_option(tracker):
    # Bought option: margin = premium paid
    tracker.add_position("NIFTY CE 24400", margin=7500)
    assert tracker.used_margin() == 7500
    assert tracker.available_margin() == 292_500


def test_can_add_within_limit(tracker):
    # 70% limit = 210_000
    assert tracker.can_add(200_000) is True
    assert tracker.can_add(220_000) is False


def test_remove_position(tracker):
    tracker.add_position("NIFTY CE", margin=7500)
    tracker.remove_position("NIFTY CE")
    assert tracker.used_margin() == 0


def test_theta_budget(tracker):
    # 40% of 300K = 120K for theta
    assert tracker.theta_budget() == 120_000


def test_directional_budget(tracker):
    # 60% of 300K = 180K for directional (minus buffer)
    # But actual available = capital - buffer(30%) = 210K, minus theta reservation
    assert tracker.directional_budget() > 0


def test_utilization_with_positions(tracker):
    tracker.add_position("pos1", margin=100_000)
    tracker.add_position("pos2", margin=50_000)
    assert tracker.utilization_pct() == pytest.approx(50.0, rel=0.01)


def test_estimate_option_buy_margin():
    tracker = MarginTracker(capital=300_000)
    m = tracker.estimate_option_buy_margin(premium=100, lot_size=75)
    assert m == 7500  # premium * lot_size
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_margin.py -v`
Expected: FAIL

- [ ] **Step 3: Implement margin tracker**

```python
# v7/margin.py
"""Margin estimation and budget tracking for V7.

Tracks margin utilization across theta and directional positions.
Uses conservative estimates (actual SPAN margin via Kite API for live trading).
"""
from __future__ import annotations

from v7.config_v7 import CAPITAL, THETA_LIMITS


class MarginTracker:
    """Track margin utilization and enforce budget limits."""

    def __init__(self, capital: float = CAPITAL["initial"]):
        self.capital = capital
        self._positions: dict[str, float] = {}  # instrument → margin
        self._max_utilization_pct = 70.0  # 30% buffer for MTM

    def add_position(self, instrument: str, margin: float) -> None:
        self._positions[instrument] = margin

    def remove_position(self, instrument: str) -> None:
        self._positions.pop(instrument, None)

    def used_margin(self) -> float:
        return sum(self._positions.values())

    def available_margin(self) -> float:
        return self.capital - self.used_margin()

    def utilization_pct(self) -> float:
        if self.capital == 0:
            return 100.0
        return (self.used_margin() / self.capital) * 100

    def can_add(self, new_margin: float) -> bool:
        """Check if adding this margin stays within 70% utilization."""
        total = self.used_margin() + new_margin
        return (total / self.capital * 100) <= self._max_utilization_pct

    def theta_budget(self) -> float:
        """Max margin available for theta engine."""
        return self.capital * THETA_LIMITS["max_margin_pct"]

    def directional_budget(self) -> float:
        """Max margin available for directional trades."""
        max_deploy = self.capital * (self._max_utilization_pct / 100)
        theta_reserved = self.theta_budget()
        return max_deploy - theta_reserved

    @staticmethod
    def estimate_option_buy_margin(premium: float, lot_size: int) -> float:
        """For bought options, margin = total premium paid."""
        return premium * lot_size

    @staticmethod
    def estimate_spread_margin(strike_width: float, lot_size: int,
                                net_credit: float = 0) -> float:
        """For spreads, margin ≈ max loss = (width - credit) × lots."""
        return (strike_width - net_credit) * lot_size

    def to_dict(self) -> dict:
        return {
            "capital": self.capital,
            "positions": dict(self._positions),
            "used": self.used_margin(),
            "available": self.available_margin(),
            "utilization_pct": round(self.utilization_pct(), 1),
        }
```

- [ ] **Step 4: Run tests**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_margin.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add v7/margin.py tests/test_v7_margin.py
git commit -m "feat(v7): add margin tracker — budget split, utilization limits"
```

---

### Task 7: Integration smoke test

**Files:**
- Create: `tests/test_v7_integration.py`

- [ ] **Step 1: Write integration test that exercises the full core layer**

```python
# tests/test_v7_integration.py
"""Integration test: core V7 infrastructure works together."""
import pytest
from datetime import date, time
from pathlib import Path
from v7.types import (
    Playbook, Setup, SetupType, Position, TradeResult,
    RiskBudget, CarryRules, DayClassification, DayPhase,
)
from v7.config_v7 import WATCHLIST, RISK_LIMITS, is_15min_boundary
from v7.state import StateManager
from v7.margin import MarginTracker


@pytest.fixture
def state(tmp_path):
    return StateManager(tmp_path / "v7")


def test_full_day_lifecycle(state):
    """Simulate a complete trading day through the core layer."""

    # 1. Create morning playbook
    playbook = Playbook(
        date=date(2026, 3, 11),
        day_classification=DayClassification.LIKELY_TREND_UP,
        nifty_bias="bullish",
        nifty_setups=[
            Setup(id="N1", priority=1, type=SetupType.BREAKOUT_LONG,
                  symbol="NIFTY", trigger_level=24350.0,
                  trigger_condition="15-min close above with volume",
                  instrument="NIFTY CE", strike_logic="delta 0.45",
                  target=24500.0, stoploss=24280.0, max_risk_pct=1.5),
        ],
        stock_plans=[
            Setup(id="H1", priority=2, type=SetupType.SUPPORT_BOUNCE,
                  symbol="HDFCBANK", trigger_level=1625.0,
                  trigger_condition="15-min close above 1625",
                  instrument="HDFCBANK CE", strike_logic="ATM",
                  target=1660.0, stoploss=1610.0, max_risk_pct=1.0),
        ],
        risk_budget=RiskBudget(max_capital_at_risk_today_pct=4.0,
                               max_trades_today=4, max_per_trade_risk_pct=1.5),
        no_trade_conditions=["VIX > 22"],
        carry_rules=CarryRules(),
    )

    # 2. Save and reload playbook
    state.save_playbook(playbook)
    loaded = state.load_playbook(today=date(2026, 3, 11))
    assert loaded is not None
    assert len(loaded.active_setups()) == 2

    # 3. Simulate a trade entry
    margin = MarginTracker(capital=300_000)
    position = Position(
        symbol="NIFTY", instrument="NIFTY CE 24400",
        direction="bullish", entry_price=80.0,
        quantity=75, lot_size=75, allocated=6000.0,
        stoploss=60.0, target=130.0,
        entry_date=date(2026, 3, 11), setup_id="N1",
    )
    margin.add_position("NIFTY CE 24400", margin=6000)
    assert margin.can_add(5000)  # can add another trade

    # 4. Save position
    state.save_positions([position])
    loaded_pos = state.load_positions()
    assert len(loaded_pos) == 1
    assert loaded_pos[0].unrealized_pnl(100.0) == 75 * 20  # 1500

    # 5. Simulate trade exit
    result = TradeResult(
        symbol="NIFTY", instrument="NIFTY CE 24400",
        direction="bullish", entry_price=80.0, exit_price=120.0,
        quantity=75, entry_date=date(2026, 3, 11),
        exit_date=date(2026, 3, 11), exit_reason="target",
        pnl=3000.0, pnl_pct=50.0, costs=80.0,
        setup_id="N1", setup_type=SetupType.BREAKOUT_LONG,
    )
    state.append_trade(result)
    state.save_positions([])  # no open positions
    margin.remove_position("NIFTY CE 24400")

    # 6. Verify final state
    assert len(state.load_positions()) == 0
    assert len(state.load_trade_history()) == 1
    assert margin.used_margin() == 0

    # 7. Update daily state
    daily = state.load_daily_state(today=date(2026, 3, 11))
    daily["trades_today"] = 1
    daily["daily_pnl"] = 3000.0
    state.save_daily_state(daily)

    reloaded = state.load_daily_state(today=date(2026, 3, 11))
    assert reloaded["trades_today"] == 1
    assert reloaded["daily_pnl"] == 3000.0


def test_phase_transitions():
    """Verify day phases are correct for key times."""
    assert DayPhase.from_time(time(8, 45)) == DayPhase.PRE_MARKET
    assert DayPhase.from_time(time(9, 15)) == DayPhase.OPENING_READ
    assert DayPhase.from_time(time(9, 45)) == DayPhase.ACTIVE_TRADING
    assert DayPhase.from_time(time(14, 30)) == DayPhase.WIND_DOWN
    assert DayPhase.from_time(time(15, 15)) == DayPhase.POST_CLOSE
    assert DayPhase.from_time(time(15, 30)) == DayPhase.POST_CLOSE


def test_risk_budget_concurrent_limit():
    """4% concurrent risk at 3L = 12K. Can't exceed."""
    rb = RiskBudget(max_capital_at_risk_today_pct=4.0)
    assert rb.can_allocate(4500, 0, 300_000)       # 4500 < 12000
    assert rb.can_allocate(4500, 4500, 300_000)     # 9000 < 12000
    assert rb.can_allocate(4500, 9000, 300_000)     # 13500 > 12000 → False
    assert rb.can_allocate(4500, 9000, 300_000) is False
```

- [ ] **Step 2: Run all V7 tests**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_*.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_v7_integration.py
git commit -m "test(v7): add integration smoke test for core infrastructure"
```

---

## Summary

Plan 1 delivers **6 modules** that form the foundation for V7:

| Module | Purpose | LOC (est) |
|---|---|---|
| `v7/types.py` | Shared types: Playbook, Setup, Position, etc. | ~350 |
| `v7/config_v7.py` | Watchlist, risk limits, phase times | ~120 |
| `v7/state.py` | File-backed persistence for all runtime state | ~120 |
| `v7/data_feed.py` | Kite primary + protect-only fallback | ~150 |
| `v7/strike_selector.py` | Mechanical strike selection | ~180 |
| `v7/margin.py` | Margin tracking and budget enforcement | ~80 |

**Next:** Plan 2 (Strategist + Risk Engine) builds on these types and state management.
