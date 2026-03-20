"""V7 Theta Engine — Weekly Nifty iron condor strategy.

Independent background income. Own risk budget (max 3% capital at risk).
Max 40% margin utilization.

Entry: Friday/Monday when VIX 16-25 (NSE weekly expiry = Tuesday since Sep 2025)
Exit: profit target (70%), delta breach, Monday EOD, never Tuesday (expiry)

Spec ref: Component 5 (lines 597-642), Margin Budget (lines 942-957)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

logger = logging.getLogger("v7.theta_engine")


@dataclass
class CondorLeg:
    """A single leg of an iron condor."""
    tradingsymbol: str
    strike: float
    premium: float      # entry premium
    quantity: int
    option_type: str    # "CE" or "PE"
    order_id: str = ""
    current_price: float = 0.0

    def to_dict(self) -> dict:
        return {
            "tradingsymbol": self.tradingsymbol, "strike": self.strike,
            "premium": self.premium, "quantity": self.quantity,
            "option_type": self.option_type, "order_id": self.order_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CondorLeg:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class CondorPosition:
    """A complete iron condor (4 legs)."""
    entry_date: date
    expiry_date: date
    short_ce: CondorLeg
    long_ce: CondorLeg
    short_pe: CondorLeg
    long_pe: CondorLeg
    net_credit: float           # per-lot net credit received
    short_ce_delta: float = 0.0 # current delta of short CE
    short_pe_delta: float = 0.0 # current delta of short PE (negative)

    @property
    def wing_width_ce(self) -> float:
        return abs(self.long_ce.strike - self.short_ce.strike)

    @property
    def wing_width_pe(self) -> float:
        return abs(self.short_pe.strike - self.long_pe.strike)

    def current_value(self, prices: dict) -> float:
        """Current net cost to close all legs.

        prices: {tradingsymbol_or_key: ltp}
        """
        def _price(leg: CondorLeg) -> float:
            for key in [f"NFO:{leg.tradingsymbol}", leg.tradingsymbol]:
                if key in prices:
                    return prices[key]
            return leg.premium  # fallback to entry if no quote

        # Cost to close = buy back shorts - sell longs
        short_ce_cost = _price(self.short_ce)
        long_ce_value = _price(self.long_ce)
        short_pe_cost = _price(self.short_pe)
        long_pe_value = _price(self.long_pe)

        return (short_ce_cost - long_ce_value) + (short_pe_cost - long_pe_value)

    def profit_pct(self, prices: dict) -> float:
        """Percentage of credit captured."""
        if self.net_credit <= 0:
            return 0.0
        current_val = self.current_value(prices)
        profit = self.net_credit - current_val
        return profit / self.net_credit

    def to_dict(self) -> dict:
        return {
            "entry_date": str(self.entry_date),
            "expiry_date": str(self.expiry_date),
            "short_ce": self.short_ce.to_dict(),
            "long_ce": self.long_ce.to_dict(),
            "short_pe": self.short_pe.to_dict(),
            "long_pe": self.long_pe.to_dict(),
            "net_credit": self.net_credit,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CondorPosition:
        return cls(
            entry_date=date.fromisoformat(d["entry_date"]),
            expiry_date=date.fromisoformat(d["expiry_date"]),
            short_ce=CondorLeg.from_dict(d["short_ce"]),
            long_ce=CondorLeg.from_dict(d["long_ce"]),
            short_pe=CondorLeg.from_dict(d["short_pe"]),
            long_pe=CondorLeg.from_dict(d["long_pe"]),
            net_credit=d["net_credit"],
        )


# Constants
THETA_MAX_RISK_PCT = 3.0        # max 3% of capital at risk
THETA_MAX_MARGIN_PCT = 40.0     # max 40% margin utilization
VIX_ENTRY_LOW = 16.0
VIX_ENTRY_HIGH = 25.0
ENTRY_DELTA = 0.18
SURVIVAL_ENTRY_DELTA = 0.15
WING_WIDTH = 200                # points between short and long strikes
MIN_CREDIT_PER_LOT = 40.0      # minimum net credit ₹30/lot
PROFIT_TARGET_PCT = 0.70  # Research: close at 70-80% of max profit beats holding to expiry
SURVIVAL_PROFIT_TARGET_PCT = 0.50  # Survival: still take 50%


class ThetaEngine:
    """Weekly Nifty iron condor manager.

    Lifecycle:
    1. Entry check: Friday/Monday, VIX 16-25, no existing condor
    2. Strike selection: short at delta 0.20, long at +200pts OTM
    3. Daily management: delta monitoring, profit target
    4. Exit: profit target, delta breach, time cutoff
    """

    def __init__(
        self,
        data_feed,
        order_mgr,
        state_mgr,
        strike_selector,
        margin_tracker,
        capital: float = 300_000,
    ):
        self._data = data_feed
        self._orders = order_mgr
        self._state = state_mgr
        self._strikes = strike_selector
        self._margin = margin_tracker
        self._capital = capital

        self._condor: Optional[CondorPosition] = None
        self._survival_mode = False

        # Load persisted state
        saved = self._state.load_theta_state()
        if saved:
            self._condor = CondorPosition.from_dict(saved)

    # ── Public Interface ──────────────────────────────────────────────

    def tick(self, today: Optional[date] = None) -> None:
        """Called each executor tick. Manages condor lifecycle."""
        today = today or date.today()

        if self._condor is None:
            # No condor — check if we should enter
            if self._should_enter_today(today):
                self._enter_condor(today)
        else:
            # Have a condor — manage it
            self._manage_condor(today)

    # ── Entry Logic ───────────────────────────────────────────────────

    def _should_enter_today(self, today: date) -> bool:
        """Check all entry conditions for a new condor."""
        # Already have a condor
        if self._condor is not None:
            return False

        # Friday(4) or Monday(0) — 2 days before Tuesday weekly expiry
        if today.weekday() not in (4, 0):
            return False

        # VIX range check
        vix = self._data.get_vix()
        if vix is None or vix < VIX_ENTRY_LOW or vix > VIX_ENTRY_HIGH:
            return False

        # Margin check: condor shouldn't push margin above 40%
        current_margin = self._margin.current_utilization_pct()
        if current_margin > THETA_MAX_MARGIN_PCT:
            return False

        return True

    def _enter_condor(self, today: date) -> None:
        """Select strikes and enter a 4-leg iron condor."""
        nifty_ltp = self._data.get_batch_ltp(["NIFTY"]).get("NIFTY")
        if nifty_ltp is None:
            logger.warning("Cannot enter condor: no Nifty LTP")
            return

        chain = self._data.get_option_chain("NIFTY")
        if not chain:
            logger.warning("Cannot enter condor: no option chain")
            return

        delta = self._entry_delta()

        # Find short CE: delta ~0.20 above ATM
        short_ce_strike = self._find_strike_by_delta(chain, "CE", delta)
        # Find short PE: delta ~0.20 below ATM
        short_pe_strike = self._find_strike_by_delta(chain, "PE", delta)

        if short_ce_strike is None or short_pe_strike is None:
            logger.warning("Cannot find suitable strikes for condor")
            return

        # Long strikes: WING_WIDTH further OTM
        long_ce_strike = short_ce_strike + WING_WIDTH
        long_pe_strike = short_pe_strike - WING_WIDTH

        # Get premiums from chain
        short_ce_prem = self._get_premium(chain, short_ce_strike, "CE")
        long_ce_prem = self._get_premium(chain, long_ce_strike, "CE")
        short_pe_prem = self._get_premium(chain, short_pe_strike, "PE")
        long_pe_prem = self._get_premium(chain, long_pe_strike, "PE")

        if any(p is None for p in [short_ce_prem, long_ce_prem, short_pe_prem, long_pe_prem]):
            logger.warning("Missing premiums for condor strikes")
            return

        net_credit = (short_ce_prem + short_pe_prem) - (long_ce_prem + long_pe_prem)
        if net_credit < MIN_CREDIT_PER_LOT:
            logger.info("Net credit %.1f < min %.1f, skipping", net_credit, MIN_CREDIT_PER_LOT)
            return

        # Risk budget check
        lot_size = 75
        if not self._is_within_risk_budget(WING_WIDTH, net_credit, lot_size):
            logger.info("Condor risk exceeds 3%% budget, skipping")
            return

        # Compute next Tuesday expiry (NSE weekly expiry moved to Tuesday Sep 2025)
        days_to_tue = (1 - today.weekday()) % 7
        if days_to_tue == 0:
            days_to_tue = 7
        expiry = today + timedelta(days=days_to_tue)

        logger.info(
            "THETA ENTRY: Nifty iron condor %d/%d/%d/%d credit=%.1f expiry=%s",
            long_pe_strike, short_pe_strike, short_ce_strike, long_ce_strike,
            net_credit, expiry,
        )

        lot_mult = self._lot_multiplier()
        qty = int(lot_size * lot_mult)

        self._condor = CondorPosition(
            entry_date=today,
            expiry_date=expiry,
            short_ce=CondorLeg(f"NIFTY{expiry.strftime('%y%m%d')}{short_ce_strike}CE",
                               short_ce_strike, short_ce_prem, qty, "CE"),
            long_ce=CondorLeg(f"NIFTY{expiry.strftime('%y%m%d')}{long_ce_strike}CE",
                              long_ce_strike, long_ce_prem, qty, "CE"),
            short_pe=CondorLeg(f"NIFTY{expiry.strftime('%y%m%d')}{short_pe_strike}PE",
                               short_pe_strike, short_pe_prem, qty, "PE"),
            long_pe=CondorLeg(f"NIFTY{expiry.strftime('%y%m%d')}{long_pe_strike}PE",
                              long_pe_strike, long_pe_prem, qty, "PE"),
            net_credit=net_credit,
        )

        # Place 4-leg orders via Kite (NRML product for multi-day)
        from v7.order_manager import OrderSide
        legs_to_place = [
            (self._condor.short_ce, OrderSide.SELL, "sell short CE"),
            (self._condor.long_ce, OrderSide.BUY, "buy long CE"),
            (self._condor.short_pe, OrderSide.SELL, "sell short PE"),
            (self._condor.long_pe, OrderSide.BUY, "buy long PE"),
        ]
        all_filled = True
        for leg, side, desc in legs_to_place:
            result = self._orders.place_entry_order(
                tradingsymbol=leg.tradingsymbol,
                exchange="NFO",
                side=side,
                quantity=leg.quantity,
                limit_price=leg.premium + (2.0 if side == OrderSide.BUY else -2.0),
            )
            if result.filled:
                leg.order_id = result.order_id or ""
                logger.info("THETA ORDER: %s %s @ %.2f filled", desc, leg.tradingsymbol, result.fill_price)
            else:
                logger.warning("THETA ORDER FAILED: %s %s", desc, leg.tradingsymbol)
                all_filled = False

        if not all_filled:
            logger.error("THETA: Not all legs filled — manual intervention needed!")
            # TODO: unwind partial fills

        # Send Telegram alert
        self._send_theta_alert("ENTRY", f"Iron Condor {short_pe_strike}/{short_ce_strike} wings {WING_WIDTH}pt credit={net_credit:.1f}")

        self._state.save_theta_state(self._condor.to_dict())

    def _send_theta_alert(self, action: str, details: str) -> None:
        """Send Telegram alert for theta engine events."""
        try:
            import os, requests
            token = os.environ.get("DEAL_BOT_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN")
            chat = os.environ.get("TELEGRAM_FORUM_CHAT_ID") or os.environ.get("DEAL_BOT_CHAT_ID")
            topic = os.environ.get("TELEGRAM_TOPIC_STOCKS")
            if not token or not chat:
                return
            msg = "<b>V7 THETA " + action + "</b>\n\n" + details + "\nVIX: " + str(round(self._data.get_vix() or 0, 1))
            data = {"chat_id": chat, "parse_mode": "HTML", "text": msg}
            if topic:
                data["message_thread_id"] = topic
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=5)
        except Exception as e:
            logger.warning("Theta Telegram alert failed: %s", e)

    # ── Daily Management ──────────────────────────────────────────────

    def _manage_condor(self, today: date) -> None:
        """Daily management: profit, delta, time checks."""
        if self._condor is None:
            return

        # Time management first
        if self._should_close_for_time(today):
            logger.info("THETA: closing condor for time management (day=%s)", today.strftime("%A"))
            self._close_condor("time_management")
            return

        # Profit check
        action = self._evaluate_condor_management()
        if action == "close_profit":
            logger.info("THETA: closing condor — profit target reached")
            self._close_condor("profit_target")
            return

        # Delta risk check
        delta_action = self._evaluate_delta_risk()
        if delta_action == "close_all":
            logger.info("THETA: closing condor — delta > 0.50")
            self._close_condor("delta_breach")
        elif delta_action == "close_ce_side":
            logger.info("THETA: closing CE side — delta > 0.45")
            self._close_side("CE")
        elif delta_action == "close_pe_side":
            logger.info("THETA: closing PE side — delta > 0.45")
            self._close_side("PE")
        elif delta_action == "tighten_ce":
            logger.info("THETA: tightening CE hedge — delta > 0.35")
            # Would buy closer CE protection
        elif delta_action == "tighten_pe":
            logger.info("THETA: tightening PE hedge — delta > 0.35")

    def _evaluate_condor_management(self) -> Optional[str]:
        """Check profit target."""
        if self._condor is None:
            return None

        from v7.config_v7 import WATCHLIST
        prices = self._data.get_batch_ltp([w["symbol"] for w in WATCHLIST])
        profit_pct = self._condor.profit_pct(prices)
        target = self._profit_target_pct()

        if profit_pct >= target:
            return "close_profit"
        return None

    def _evaluate_delta_risk(self) -> Optional[str]:
        """Check short strike deltas for risk thresholds."""
        if self._condor is None:
            return None

        ce_delta = abs(self._condor.short_ce_delta)
        pe_delta = abs(self._condor.short_pe_delta)

        # Delta > 0.50 → close everything
        if ce_delta > 0.50 or pe_delta > 0.50:
            return "close_all"

        # Delta > 0.45 → close threatened side
        if ce_delta > 0.45:
            return "close_ce_side"
        if pe_delta > 0.45:
            return "close_pe_side"

        # Delta > 0.35 → tighten hedge
        if ce_delta > 0.35:
            return "tighten_ce"
        if pe_delta > 0.35:
            return "tighten_pe"

        return None

    # ── Time Management ───────────────────────────────────────────────

    def _should_close_for_time(self, today: date) -> bool:
        """Close by Monday EOD or never hold to Tuesday (expiry day)."""
        if self._condor is None:
            return False
        # Tuesday = expiry day → must close
        if today.weekday() == 1:  # Tuesday
            return True
        # Monday → close if still open (gamma risk before Tuesday expiry)
        if today.weekday() == 0:  # Monday
            return True
        return False

    # ── Close Operations ──────────────────────────────────────────────

    def _close_condor(self, reason: str) -> None:
        """Close all 4 legs of the condor."""
        if self._condor is None:
            return

        logger.info("THETA CLOSE: reason=%s", reason)

        from v7.order_manager import OrderSide
        # Buy back shorts, sell longs
        for leg, side in [
            (self._condor.short_ce, OrderSide.BUY),  # buy back short CE
            (self._condor.short_pe, OrderSide.BUY),  # buy back short PE
            (self._condor.long_ce, OrderSide.SELL),   # sell long CE
            (self._condor.long_pe, OrderSide.SELL),   # sell long PE
        ]:
            self._orders.place_exit_order(
                tradingsymbol=leg.tradingsymbol,
                exchange="NFO",
                side=side,
                quantity=leg.quantity,
                limit_price=leg.current_price or leg.premium,
            )

        self._condor = None
        self._state.save_theta_state(None)

    def _close_side(self, side: str) -> None:
        """Close one side of the condor (CE or PE)."""
        if self._condor is None:
            return

        from v7.order_manager import OrderSide
        if side == "CE":
            legs = [(self._condor.short_ce, OrderSide.BUY),
                    (self._condor.long_ce, OrderSide.SELL)]
        else:
            legs = [(self._condor.short_pe, OrderSide.BUY),
                    (self._condor.long_pe, OrderSide.SELL)]

        for leg, order_side in legs:
            self._orders.place_exit_order(
                tradingsymbol=leg.tradingsymbol,
                exchange="NFO",
                side=order_side,
                quantity=leg.quantity,
                limit_price=leg.current_price or leg.premium,
            )

        logger.info("THETA: closed %s side", side)

    # ── Risk Budget ───────────────────────────────────────────────────

    def _is_within_risk_budget(self, wing_width: float, net_credit: float, lot_size: int) -> bool:
        """Check if condor max loss fits within 3% capital risk budget."""
        max_loss = (wing_width - net_credit) * lot_size
        max_allowed = self._capital * (THETA_MAX_RISK_PCT / 100)
        return max_loss <= max_allowed

    # ── Survival Mode ────────────────────────────────────────────────

    def _entry_delta(self) -> float:
        return SURVIVAL_ENTRY_DELTA if self._survival_mode else ENTRY_DELTA

    def _lot_multiplier(self) -> float:
        return 0.5 if self._survival_mode else 1.0

    def _profit_target_pct(self) -> float:
        return SURVIVAL_PROFIT_TARGET_PCT if self._survival_mode else PROFIT_TARGET_PCT

    # ── Strike Helpers ────────────────────────────────────────────────

    def _find_strike_by_delta(self, chain: list, option_type: str, target_delta: float) -> Optional[float]:
        """Find strike closest to target delta from option chain."""
        best_strike = None
        best_diff = float("inf")

        for entry in chain:
            opt = entry.get(option_type)
            if not opt:
                continue
            delta = abs(opt.get("delta", 0))
            diff = abs(delta - target_delta)
            if diff < best_diff:
                best_diff = diff
                best_strike = entry["strikePrice"]

        return best_strike

    def _get_premium(self, chain: list, strike: float, option_type: str) -> Optional[float]:
        """Get last traded price for a specific strike from chain."""
        for entry in chain:
            if entry["strikePrice"] == strike:
                opt = entry.get(option_type)
                if opt:
                    return opt.get("lastTradedPrice")
        return None
