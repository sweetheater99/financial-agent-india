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
    PRE_MARKET = "pre_market"
    OPENING_READ = "opening_read"
    ACTIVE_TRADING = "active_trading"
    WIND_DOWN = "wind_down"
    POST_CLOSE = "post_close"
    OUTSIDE_HOURS = "outside_hours"

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
    instrument: str
    strike_logic: str
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
    instrument: str
    direction: str
    entry_price: float
    quantity: int
    lot_size: int
    allocated: float
    stoploss: float
    target: float
    entry_date: date
    setup_id: str
    peak_price: float = 0.0
    sl_order_id: str | None = None
    carried: bool = False
    hedge_instrument: str | None = None
    hedge_cost: float = 0.0
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
        if self.entry_time is None:
            return 0
        entry_min = self.entry_time.hour * 60 + self.entry_time.minute
        current_min = current_time.hour * 60 + current_time.minute
        return max(0, current_min - entry_min)

    def premium_health(self, current_premium: float) -> float:
        if self.entry_price <= 0:
            return 1.0
        return current_premium / self.entry_price

    def unrealized_pnl(self, current_price: float) -> float:
        """P&L for long option positions (both CE and PE buys)."""
        return self.quantity * (current_price - self.entry_price)

    def unrealized_pnl_pct(self, current_price: float) -> float:
        cost = self.entry_price * self.quantity
        if cost == 0:
            return 0.0
        return (self.unrealized_pnl(current_price) / cost) * 100

    def risk_amount(self) -> float:
        """Max loss = total premium paid (for long options, max loss is 100% of premium)."""
        return self.allocated

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
            "entry_time": str(self.entry_time) if self.entry_time else None,
            "initial_quantity": self.initial_quantity,
            "partial_exit_done": self.partial_exit_done,
            "health_score": self.health_score,
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
            entry_time=time.fromisoformat(d["entry_time"]) if d.get("entry_time") else None,
            initial_quantity=d.get("initial_quantity", d.get("quantity", 0)),
            partial_exit_done=d.get("partial_exit_done", False),
            health_score=d.get("health_score", 100.0),
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
    exit_reason: str
    pnl: float
    pnl_pct: float
    costs: float
    setup_id: str
    setup_type: SetupType
    entry_grade: str = "B"
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
        if not self.can_allocate(new_risk, current_risk, capital):
            return False
        if trades_today >= self.max_trades_today:
            return False
        if consecutive_sl_hits >= 3:
            return False
        if daily_pnl < -(capital * 0.02):
            return False
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
        return True

    def risk_pct_for_conviction(self, conviction: Conviction) -> float:
        base = {
            Conviction.HIGH: 2.0,
            Conviction.MEDIUM: 1.5,
            Conviction.LOW: 0.75,
        }[conviction]
        if self.pacing_status == PacingStatus.AHEAD:
            return base * 0.75
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
    us_close: str = ""
    gift_nifty: str = ""
    vix: float = 0.0
    fii_dii: str = ""
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
    no_trade_zone: tuple[float, float] | None = None

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
    key_levels: dict[str, KeyLevels] = field(default_factory=dict)
    opening_range: dict[str, float] | None = None
    theta_action: str = "hold"
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
