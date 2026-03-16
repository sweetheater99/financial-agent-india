# v7/executor.py
"""V7 Executor — Mechanical tick cycle engine.

Runs every 3 minutes during market hours. No Claude calls. Pure rule execution.
Phase-based behavior:
  - Opening Read (9:15-9:45): only manage carried positions
  - Active Trading (9:45-14:30): full playbook execution on 15-min candle boundaries
  - Wind Down (14:30-15:15): close intraday, carry decisions
  - Post-close (15:15-15:30): final SL placement on carried positions

Spec ref: Component 2 (lines 319-438)
"""
from __future__ import annotations

import logging
from datetime import datetime, date, time
from typing import Optional

from v7.types import (
    DayPhase, Setup, SetupType, Position, Playbook, TradeResult,
    CarryRules, RiskBudget, Conviction,
)

logger = logging.getLogger("v7.executor")


def is_15min_boundary(t: time) -> bool:
    """Check if time falls on a 15-min candle close boundary (:00, :15, :30, :45)."""
    return t.minute % 15 == 0


# Index symbol aliases — kept here to avoid importing kiteconnect at module level
_INDEX_ALIASES = {
    "NIFTY": "NSE:NIFTY 50",
    "BANKNIFTY": "NSE:NIFTY BANK",
    "FINNIFTY": "NSE:NIFTY FIN SERVICE",
    "MIDCPNIFTY": "NSE:NIFTY MID SELECT",
}


class Executor:
    """Mechanical execution engine. No Claude calls.

    Responsibilities:
    - Fetch LTP, candles, VIX each tick
    - Evaluate playbook triggers on 15-min candle boundaries (Active Trading only)
    - Manage open positions: SL, breakeven, trailing, target
    - Wind down: close intraday, evaluate carry
    - Detect exception conditions → delegate to Strategist.handle_exception()
    """

    def __init__(
        self,
        state_mgr,
        data_feed,
        risk_engine,
        order_mgr,
        margin_tracker,
        capital: float = 300_000,
        strategist=None,
    ):
        self._state = state_mgr
        self._data = data_feed
        self._risk = risk_engine
        self._orders = order_mgr
        self._margin = margin_tracker
        self._capital = capital
        self._strategist = strategist  # for exception callbacks

        # Runtime state — loaded from persistence at start
        self._playbook: Optional[Playbook] = None
        self._positions: list[Position] = []
        self._daily: dict = {}
        self._quotes: dict = {}  # latest LTP batch
        self._vix: float = 0.0
        self._initialized = False

    def initialize(self) -> None:
        """Load persisted state. Called once at bot start or after restart."""
        self._playbook = self._state.load_playbook()
        self._positions = self._state.load_positions()
        self._daily = self._state.load_daily_state()
        self._initialized = True

        # Validate playbook is for today
        if self._playbook and self._playbook.date != date.today():
            logger.warning("Playbook is from %s, not today. Entering protect-only mode.",
                           self._playbook.date)
            self._playbook = None

        # Verify SL orders on exchange for carried positions
        for pos in self._positions:
            if pos.carried and pos.sl_order_id:
                if not self._orders.is_sl_order_live(pos.sl_order_id):
                    logger.warning("Carried position %s SL order %s is NOT live! Replacing.",
                                   pos.instrument, pos.sl_order_id)
                    # Will be handled in carried position management

        logger.info("Executor initialized: %d positions, playbook=%s",
                     len(self._positions), "loaded" if self._playbook else "NONE")

    def tick(self) -> None:
        """Single tick of the executor loop. Called every 3 minutes."""
        now = datetime.now()
        phase = DayPhase.from_time(now.time())

        # Outside market hours — exit immediately
        if phase in (DayPhase.OUTSIDE_HOURS, DayPhase.PRE_MARKET):
            return

        if not self._initialized:
            self.initialize()

        # 1. FETCH DATA
        self._fetch_data(now)

        # 2. Phase-specific behavior
        if phase == DayPhase.OPENING_READ:
            self._handle_opening_read(now)
        elif phase == DayPhase.ACTIVE_TRADING:
            self._handle_active_trading(now)
        elif phase == DayPhase.WIND_DOWN:
            self._handle_wind_down(now)
        elif phase == DayPhase.POST_CLOSE:
            self._handle_post_close(now)

        # 3. MANAGE OPEN POSITIONS (every tick, every phase during market hours)
        if self._positions:
            self._manage_positions(now)

        # 4. THETA ENGINE (every tick during market hours)
        if hasattr(self, '_theta_engine') and self._theta_engine:
            self._theta_engine.tick()

        # 5. EXCEPTION DETECTION (every tick)
        self._check_exceptions(now)

        # 6. PERSIST STATE
        if self._playbook:
            self._state.save_playbook(self._playbook)
        self._state.save_positions(self._positions)
        self._state.save_daily_state(self._daily)

    # ── Data Fetching ─────────────────────────────────────────────────

    def _fetch_data(self, now: datetime) -> None:
        """Fetch LTP batch for watchlist + open option positions + VIX."""
        try:
            from v7.config_v7 import WATCHLIST
            symbols = [w["symbol"] for w in WATCHLIST]
            self._quotes = self._data.get_batch_ltp(symbols)
        except Exception as e:
            logger.error("LTP batch fetch failed: %s", e)
            self._quotes = {}

        # Fetch option LTPs for open positions
        if self._positions and self._data.can_trade():
            try:
                option_keys = [f"NFO:{p.instrument}" for p in self._positions if p.instrument]
                if option_keys:
                    option_quotes = self._data.kite.quote(option_keys)
                    for key, q in option_quotes.items():
                        self._quotes[key] = q.get("last_price", 0)
            except Exception as e:
                logger.warning("Option LTP fetch failed: %s", e)

        try:
            self._vix = self._data.get_vix() or 0.0
        except Exception as e:
            logger.warning("VIX fetch failed: %s", e)

        # Track VIX 30min ago for exception detection
        if self._daily.get("vix_30min_ago") is None:
            self._daily["vix_30min_ago"] = self._vix

        # Track Nifty open
        if self._daily.get("nifty_open") is None:
            nifty_ltp = self._get_ltp_for_symbol("NIFTY")
            if nifty_ltp:
                self._daily["nifty_open"] = nifty_ltp

    # ── Phase Handlers ────────────────────────────────────────────────

    def _handle_opening_read(self, now: datetime) -> None:
        """9:15-9:45: Only manage carried positions (gap check). No triggers."""
        carried = [p for p in self._positions if p.carried]
        if carried:
            self._manage_carried_positions_gap(now, carried)

    def _handle_active_trading(self, now: datetime) -> None:
        """9:45-14:30: Check triggers on candle boundaries only."""
        if not self._playbook:
            return

        # Expiry day cutoff: no new positions after 1:00 PM on Thursday
        if self._is_expiry_cutoff(now):
            return

        if is_15min_boundary(now.time()):
            self._evaluate_triggers(now)

            # Auto-refresh: if all setups are stale, ask strategist for new ones
            self._check_stale_setups(now)

    def _handle_wind_down(self, now: datetime) -> None:
        """14:30-15:15: Close intraday, carry decisions."""
        self._wind_down(now)

    def _handle_post_close(self, now: datetime) -> None:
        """15:15-15:30: Final SL placement on carried positions."""
        for pos in self._positions:
            if pos.carried and not pos.sl_order_id:
                self._place_exchange_sl(pos)

    # ── Trigger Evaluation ────────────────────────────────────────────

    def _evaluate_triggers(self, now: datetime) -> None:
        """Check all playbook setups in priority order. Only on candle boundaries."""
        if not self._playbook:
            return

        # Check no-trade conditions
        if self._check_no_trade_conditions():
            return

        all_setups = self._playbook.all_setups()
        for setup in sorted(all_setups, key=lambda s: s.priority):
            if setup.fired or setup.cancelled:
                continue
            self._evaluate_single_trigger(setup, now)

    def _evaluate_single_trigger(self, setup: Setup, now: datetime) -> None:
        """Evaluate a single setup trigger. Enter if conditions met."""
        # Get current price for the setup's symbol
        ltp = self._get_ltp_for_symbol(setup.symbol)
        if ltp is None:
            return

        # Check trigger condition
        triggered = False
        if setup.type in (SetupType.BREAKOUT_LONG, SetupType.SUPPORT_BOUNCE):
            triggered = ltp > setup.trigger_level
        elif setup.type in (SetupType.BREAKOUT_SHORT, SetupType.RESISTANCE_FADE):
            triggered = ltp < setup.trigger_level
        elif setup.type in (SetupType.CREDIT_SPREAD_BULL, SetupType.CREDIT_SPREAD_BEAR):
            triggered = ltp > setup.trigger_level  # simplified

        if not triggered:
            return

        # Skip if price already past target (would enter and exit same tick)
        if setup.type in (SetupType.BREAKOUT_SHORT, SetupType.RESISTANCE_FADE):
            if ltp < setup.target:
                logger.info("Setup %s: price %.2f already past target %.2f — skipping stale trigger",
                            setup.id, ltp, setup.target)
                return
        elif setup.type in (SetupType.BREAKOUT_LONG, SetupType.SUPPORT_BOUNCE):
            if ltp > setup.target:
                logger.info("Setup %s: price %.2f already past target %.2f — skipping stale trigger",
                            setup.id, ltp, setup.target)
                return

        # Risk budget check
        risk_amount = self._capital * (setup.max_risk_pct / 100)
        current_risk = sum(p.risk_amount() for p in self._positions)
        if not self._playbook.risk_budget.can_enter_trade(
            new_risk=risk_amount,
            current_risk=current_risk,
            capital=self._capital,
            trades_today=self._daily.get("trades_today", 0),
            consecutive_sl_hits=self._daily.get("consecutive_sl_hits", 0),
            daily_pnl=self._daily.get("daily_pnl", 0),
        ):
            logger.info("Setup %s triggered but risk budget exhausted", setup.id)
            return

        # F&O ban check, brokerage check would go here

        # ENTER
        logger.info("Setup %s TRIGGERED at %.2f (level=%.2f)", setup.id, ltp, setup.trigger_level)
        self._enter_position(setup, ltp, now)

    def _enter_position(self, setup: Setup, trigger_ltp: float, now: datetime) -> None:
        """Execute entry for a triggered setup.

        1. Fetch option chain for the symbol
        2. Use StrikeSelector to pick the best strike (~0.45 delta)
        3. Place order with actual tradingsymbol and option premium
        """
        setup.fired = True

        direction = "bullish" if setup.type in (
            SetupType.BREAKOUT_LONG, SetupType.SUPPORT_BOUNCE,
            SetupType.CREDIT_SPREAD_BULL,
        ) else "bearish"

        risk_amount = self._capital * (setup.max_risk_pct / 100)

        # Look up lot size from watchlist config
        from v7.config_v7 import WATCHLIST
        wl = next((w for w in WATCHLIST if w["symbol"] == setup.symbol), None)
        lot_size = wl["lot_size"] if wl else 75

        # Fetch option chain and select strike
        try:
            chain = self._data.get_option_chain(setup.symbol, "current")
            if not chain:
                logger.warning("Setup %s: no option chain for %s — skipping", setup.id, setup.symbol)
                setup.fired = False
                return
        except Exception as e:
            logger.error("Setup %s: option chain fetch failed: %s", setup.id, e)
            setup.fired = False
            return

        from v7.strike_selector import select_directional_strike
        selected = select_directional_strike(
            chain=chain,
            direction=direction,
            spot=trigger_ltp,
            risk_budget=risk_amount,
            lot_size=lot_size,
            symbol=setup.symbol,
        )

        if not selected:
            logger.warning("Setup %s: no suitable strike found for %s — skipping", setup.id, setup.symbol)
            setup.fired = False
            return

        tradingsymbol = selected["tradingsymbol"]
        option_premium = selected["premium"]
        actual_lot_size = selected.get("lot_size", lot_size)
        exchange = "NFO"

        if not tradingsymbol:
            logger.warning("Setup %s: strike selected but no tradingsymbol — skipping", setup.id)
            setup.fired = False
            return

        logger.info("Setup %s: selected %s (strike=%s, delta=%.2f, premium=%.2f)",
                     setup.id, tradingsymbol, selected["strike"], selected["delta"], option_premium)

        # Limit price: option premium + small buffer
        limit_price = option_premium + 2.0
        quantity = actual_lot_size

        from v7.order_manager import OrderSide
        result = self._orders.place_entry_order(
            tradingsymbol=tradingsymbol,
            exchange=exchange,
            side=OrderSide.BUY,
            quantity=quantity,
            limit_price=limit_price,
        )

        if not result.filled:
            logger.warning("Entry order for %s did not fill — will retry next boundary", setup.id)
            setup.fired = False
            return

        # Create position
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
        )
        self._positions.append(pos)
        self._daily["trades_today"] = self._daily.get("trades_today", 0) + 1

        logger.info("ENTERED: %s %s @ %.2f, SL=%.2f, TGT=%.2f",
                     setup.id, tradingsymbol, result.fill_price, setup.stoploss, setup.target)

        # Telegram alert with trade reasoning
        self._send_entry_alert(setup, pos)

    # ── Position Management ───────────────────────────────────────────

    def _manage_positions(self, now: datetime) -> None:
        """Update P&L, check SL/target/trailing for each position.

        SL and target are UNDERLYING price levels from the playbook.
        We check the underlying symbol's price against these levels,
        then exit the option position at its current premium.
        """
        positions_to_remove = []

        for pos in self._positions:
            # Get underlying price for SL/target evaluation
            underlying_ltp = self._get_ltp_for_symbol(pos.symbol)
            # Get option premium for P&L and exit
            option_ltp = self._get_ltp_for_instrument(pos.instrument)

            if underlying_ltp is None and option_ltp is None:
                continue

            # Track option premium for P&L
            if option_ltp is not None:
                pnl = pos.unrealized_pnl(option_ltp)

            # Update peak price (track underlying for trailing SL)
            if underlying_ltp is not None:
                if pos.direction == "bullish":
                    pos.peak_price = max(pos.peak_price, underlying_ltp)
                else:
                    pos.peak_price = min(pos.peak_price, underlying_ltp) if pos.peak_price > 0 else underlying_ltp

            # SL/target checks use UNDERLYING price
            if underlying_ltp is not None:
                # SL hit → EXIT immediately
                if self._is_sl_hit(pos, underlying_ltp):
                    exit_price = option_ltp if option_ltp is not None else 0
                    logger.info("SL HIT: %s underlying=%.2f (SL=%.2f), option=%s@%.2f",
                                pos.instrument, underlying_ltp, pos.stoploss, pos.instrument, exit_price)
                    self._exit_position(pos, exit_price, "stoploss", now)
                    positions_to_remove.append(pos)
                    self._daily["consecutive_sl_hits"] = self._daily.get("consecutive_sl_hits", 0) + 1
                    continue

                # Target hit → full exit
                if self._is_target_hit(pos, underlying_ltp):
                    exit_price = option_ltp if option_ltp is not None else 0
                    logger.info("TARGET HIT: %s underlying=%.2f (TGT=%.2f), option=%s@%.2f",
                                pos.instrument, underlying_ltp, pos.target, pos.instrument, exit_price)
                    self._exit_position(pos, exit_price, "target", now)
                    positions_to_remove.append(pos)
                    self._daily["consecutive_sl_hits"] = 0
                    continue

                # 1:1 R:R → move SL to breakeven (underlying levels)
                self._check_breakeven(pos, underlying_ltp)

                # Trailing stop on underlying
                trailing_sl = self._compute_trailing_sl(pos, underlying_ltp)
                if trailing_sl is not None and self._is_better_sl(pos, trailing_sl):
                    old_sl = pos.stoploss
                    pos.stoploss = trailing_sl
                    logger.info("TRAILING SL: %s underlying moved %.2f → %.2f",
                                pos.instrument, old_sl, trailing_sl)
                    if pos.sl_order_id:
                        self._update_exchange_sl(pos)

        for pos in positions_to_remove:
            self._positions.remove(pos)

    def _is_sl_hit(self, pos: Position, ltp: float) -> bool:
        if pos.direction == "bullish":
            return ltp <= pos.stoploss
        return ltp >= pos.stoploss

    def _is_target_hit(self, pos: Position, ltp: float) -> bool:
        if pos.direction == "bullish":
            return ltp >= pos.target
        return ltp <= pos.target

    def _check_breakeven(self, pos: Position, ltp: float) -> None:
        """Move SL to breakeven when 1:1 R:R is achieved."""
        risk = abs(pos.entry_price - pos.stoploss)
        if pos.direction == "bullish":
            if ltp >= pos.entry_price + risk and pos.stoploss < pos.entry_price:
                logger.info("BREAKEVEN: %s SL moved to entry %.2f", pos.instrument, pos.entry_price)
                pos.stoploss = pos.entry_price
                if pos.sl_order_id:
                    self._update_exchange_sl(pos)
        else:
            if ltp <= pos.entry_price - risk and pos.stoploss > pos.entry_price:
                logger.info("BREAKEVEN: %s SL moved to entry %.2f", pos.instrument, pos.entry_price)
                pos.stoploss = pos.entry_price
                if pos.sl_order_id:
                    self._update_exchange_sl(pos)

    def _compute_trailing_sl(self, pos: Position, ltp: float) -> Optional[float]:
        """Trailing stop = peak_price - 1.5x ATR(5min).

        Returns new SL level or None if ATR not available.
        """
        # Fetch 5-min ATR for the instrument
        try:
            candles = self._data.get_candles(
                pos.symbol, interval="FIVE_MINUTE", days=1
            )
            if not candles or len(candles) < 14:
                return None

            # Compute ATR from last 14 candles
            atr = self._compute_atr(candles[-14:])
            if atr <= 0:
                return None

            trailing_distance = 1.5 * atr
            if pos.direction == "bullish":
                return pos.peak_price - trailing_distance
            else:
                return pos.peak_price + trailing_distance
        except Exception:
            return None

    def _compute_atr(self, candles: list) -> float:
        """Simple ATR from candle data. Candles: [[ts, o, h, l, c, v], ...]"""
        if len(candles) < 2:
            return 0.0
        trs = []
        for i in range(1, len(candles)):
            h, l, prev_c = candles[i][2], candles[i][3], candles[i - 1][4]
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            trs.append(tr)
        return sum(trs) / len(trs) if trs else 0.0

    def _is_better_sl(self, pos: Position, new_sl: float) -> bool:
        """Only move SL in the favorable direction (tighter)."""
        if pos.direction == "bullish":
            return new_sl > pos.stoploss
        return new_sl < pos.stoploss

    def _exit_position(self, pos: Position, ltp: float, reason: str, now: datetime) -> None:
        """Execute exit order for a position."""
        from v7.order_manager import OrderSide

        # Cancel exchange SL if active
        if pos.sl_order_id:
            self._orders.cancel_order(pos.sl_order_id)

        # Place exit order
        is_sl_exit = reason == "stoploss"
        side = OrderSide.SELL if pos.direction == "bullish" else OrderSide.BUY
        limit_price = ltp - 0.50 if side == OrderSide.SELL else ltp + 0.50

        result = self._orders.place_exit_order(
            tradingsymbol=pos.instrument,
            exchange="NFO",
            side=side,
            quantity=pos.quantity,
            limit_price=limit_price,
            is_sl_exit=is_sl_exit,
        )

        exit_price = result.fill_price if result.filled else ltp
        pnl = pos.unrealized_pnl(exit_price)
        self._daily["daily_pnl"] = self._daily.get("daily_pnl", 0) + pnl

        logger.info("EXITED: %s reason=%s exit=%.2f pnl=%.2f", pos.instrument, reason, exit_price, pnl)

        # Record trade result for EOD journal and edge tracking
        trade = TradeResult(
            symbol=pos.symbol,
            instrument=pos.instrument,
            setup_type=self._get_setup_type(pos.setup_id),
            setup_id=pos.setup_id,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            pnl=pnl,
            pnl_pct=(pnl / pos.allocated * 100) if pos.allocated else 0,
            costs=0.0,
            entry_date=str(pos.entry_date),
            exit_date=str(now.date()),
            exit_reason=reason,
        )
        self._state.append_trade(trade)
        closed = self._daily.get("closed_trades", [])
        closed.append(trade.to_dict())
        self._daily["closed_trades"] = closed

        # Telegram alert on exit
        self._send_exit_alert(pos, exit_price, reason, pnl)

    # ── Carried Position Management ───────────────────────────────────

    def _manage_carried_positions_gap(self, now: datetime, carried: list[Position]) -> None:
        """Opening read: check gap on carried positions."""
        for pos in carried:
            ltp = self._get_ltp_for_instrument(pos.instrument)
            if ltp is None:
                continue

            sl_distance = abs(pos.entry_price - pos.stoploss)
            gap = ltp - pos.peak_price if pos.direction == "bullish" else pos.peak_price - ltp

            if gap < -sl_distance:
                # Gap against position > SL distance → exit in first 5 min
                logger.info("CARRIED GAP EXIT: %s gap=%.2f > SL distance=%.2f",
                             pos.instrument, abs(gap), sl_distance)
                self._exit_position(pos, ltp, "carry_gap", now)
                self._positions.remove(pos)
            elif gap > 0:
                # Gap in favor → tighten SL to lock gains
                new_sl = pos.entry_price + (gap * 0.5) if pos.direction == "bullish" \
                    else pos.entry_price - (gap * 0.5)
                if self._is_better_sl(pos, new_sl):
                    pos.stoploss = new_sl
                    logger.info("CARRIED GAP TIGHTEN: %s new SL=%.2f", pos.instrument, new_sl)
                    if pos.sl_order_id:
                        self._update_exchange_sl(pos)

    # ── Wind Down ─────────────────────────────────────────────────────

    def _wind_down(self, now: datetime) -> None:
        """Close intraday positions, evaluate carry criteria."""
        if not self._playbook:
            # No playbook — close everything
            for pos in list(self._positions):
                ltp = self._get_ltp_for_instrument(pos.instrument)
                if ltp:
                    self._exit_position(pos, ltp, "wind_down", now)
            self._positions.clear()
            return

        carry_rules = self._playbook.carry_rules
        positions_to_close = []

        for pos in self._positions:
            if pos.carried:
                continue  # already carried from previous day

            ltp = self._get_ltp_for_instrument(pos.instrument)
            if ltp is None:
                positions_to_close.append(pos)
                continue

            # Check carry criteria
            if self._meets_carry_criteria(pos, ltp, carry_rules):
                self._convert_to_carry(pos, ltp, now)
            else:
                self._exit_position(pos, ltp, "wind_down", now)
                positions_to_close.append(pos)

        for pos in positions_to_close:
            if pos in self._positions:
                self._positions.remove(pos)

    def _meets_carry_criteria(self, pos: Position, ltp: float, rules: CarryRules) -> bool:
        """Check if position meets overnight carry criteria.

        Carry criteria (from spec):
        - Profit > 1.5%
        - Trend intact (simplified: in profit)
        - VIX < 20
        - DTE > 3 days
        - Not expiry day, not event tomorrow, not VIX > 22
        """
        pnl_pct = pos.unrealized_pnl_pct(ltp)
        if pnl_pct < rules.min_profit_pct:
            return False
        if self._vix > rules.max_vix:
            return False
        # Check never-carry conditions
        today = date.today()
        if today.weekday() == 3:  # Thursday = expiry day
            return False
        if "vix_above_22" in rules.never_carry and self._vix > 22:
            return False
        return True

    def _convert_to_carry(self, pos: Position, ltp: float, now: datetime) -> None:
        """Convert intraday position to overnight carry with hedge + exchange SL."""
        pos.carried = True
        logger.info("CARRY: %s converting to overnight", pos.instrument)

        # Place protective hedge (buy OTM option opposite side)
        # This would use StrikeSelector to find a 3-4 strike OTM hedge
        # For now, log intent
        logger.info("CARRY: would place protective hedge for %s (max cost ₹500)", pos.instrument)

        # Place exchange-level SL order
        self._place_exchange_sl(pos, product="NRML")

    def _place_exchange_sl(self, pos: Position, product: str = "NRML") -> None:
        """Place SL order on exchange for a position."""
        from v7.order_manager import OrderSide

        side = OrderSide.SELL if pos.direction == "bullish" else OrderSide.BUY
        trigger = pos.stoploss
        limit = trigger - 1.0 if side == OrderSide.SELL else trigger + 1.0

        result = self._orders.place_sl_order(
            tradingsymbol=pos.instrument,
            exchange="NFO",
            side=side,
            quantity=pos.quantity,
            trigger_price=trigger,
            limit_price=limit,
            product=product,
        )

        if result.order_id:
            pos.sl_order_id = result.order_id
            logger.info("Exchange SL placed: %s for %s trigger=%.2f",
                         result.order_id, pos.instrument, trigger)

    def _update_exchange_sl(self, pos: Position) -> None:
        """Update exchange SL order to new stoploss level."""
        if not pos.sl_order_id:
            return
        from v7.order_manager import OrderSide

        side = OrderSide.SELL if pos.direction == "bullish" else OrderSide.BUY
        trigger = pos.stoploss
        limit = trigger - 1.0 if side == OrderSide.SELL else trigger + 1.0

        result = self._orders.update_sl_order(
            old_order_id=pos.sl_order_id,
            tradingsymbol=pos.instrument,
            exchange="NFO",
            side=side,
            quantity=pos.quantity,
            new_trigger_price=trigger,
            new_limit_price=limit,
        )

        if result.order_id:
            pos.sl_order_id = result.order_id

    # ── Exception Detection ───────────────────────────────────────────

    def _check_exceptions(self, now: datetime) -> None:
        """Detect exception conditions and delegate to Strategist."""
        if not self._strategist:
            return

        exceptions = []

        # VIX jump > 2pts in 30 min
        vix_30m_ago = self._daily.get("vix_30min_ago", self._vix)
        if self._vix - vix_30m_ago > 2.0:
            exceptions.append(f"VIX spike: {vix_30m_ago:.1f} → {self._vix:.1f} in 30min")

        # Nifty > 1.5% from open
        nifty_open = self._daily.get("nifty_open")
        nifty_now = self._get_ltp_for_symbol("NIFTY")
        if nifty_open and nifty_now:
            nifty_move_pct = abs(nifty_now - nifty_open) / nifty_open * 100
            if nifty_move_pct > 1.5:
                exceptions.append(f"Nifty move: {nifty_move_pct:.1f}% from open")

        # Margin > 70%
        margin_pct = self._margin.utilization_pct()
        if margin_pct > 70:
            exceptions.append(f"Margin utilization: {margin_pct:.0f}%")

        # 3 consecutive SL hits
        if self._daily.get("consecutive_sl_hits", 0) >= 3:
            exceptions.append(f"3 consecutive SL hits today")

        if exceptions:
            logger.warning("EXCEPTION DETECTED: %s", "; ".join(exceptions))
            try:
                self._strategist.handle_exception(
                    exception_type="; ".join(exceptions),
                    details={"daily_state": self._daily, "vix": self._vix},
                    open_positions=[
                        {"symbol": p.symbol, "instrument": p.instrument,
                         "direction": p.direction, "entry_price": p.entry_price,
                         "stoploss": p.stoploss, "target": p.target}
                        for p in self._positions
                    ],
                )
            except Exception as e:
                logger.error("Exception handler failed: %s", e)

    # ── No-Trade Conditions ───────────────────────────────────────────

    def _check_no_trade_conditions(self) -> bool:
        """Check if any no-trade condition is active. Returns True to skip."""
        if not self._playbook or not self._playbook.no_trade_conditions:
            return False

        for condition in self._playbook.no_trade_conditions:
            cond_lower = condition.lower()
            if "vix" in cond_lower:
                # Parse "VIX > 22" style conditions
                try:
                    threshold = float(cond_lower.split(">")[-1].strip())
                    if self._vix > threshold:
                        logger.info("No-trade: %s (VIX=%.1f)", condition, self._vix)
                        return True
                except (ValueError, IndexError):
                    pass
        return False

    # ── Theta Engine Integration ──────────────────────────────────────

    def set_theta_engine(self, theta_engine) -> None:
        """Attach theta engine for integrated tick cycle."""
        self._theta_engine = theta_engine

    # ── Expiry Day ────────────────────────────────────────────────────

    def _is_expiry_day(self, today: Optional[date] = None) -> bool:
        """Thursday is weekly expiry for index options."""
        today = today or date.today()
        return today.weekday() == 3

    def _expiry_adjusted_sl(self, pos: Position) -> float:
        """On expiry day, tighten SL to half normal distance."""
        normal_distance = abs(pos.entry_price - pos.stoploss)
        half_distance = normal_distance / 2
        if pos.direction == "bullish":
            return pos.entry_price - half_distance
        return pos.entry_price + half_distance

    def _is_expiry_cutoff(self, now: datetime) -> bool:
        """No new positions after 1:00 PM on expiry day."""
        if not self._is_expiry_day(today=now.date()):
            return False
        return now.hour >= 13

    # ── Position Rolling ──────────────────────────────────────────────

    def _should_roll(self, pos: Position, current_price: float, dte: int) -> bool:
        """Check if position should be rolled to next expiry.

        Conditions (from spec):
        - Loss is 20-50% of premium (not too small, not too large)
        - DTE < 5 (theta accelerating)
        - Original thesis still valid (simplified: not at SL)
        - VIX <= 22 (next expiry premium must be affordable)

        Do NOT roll if:
        - Loss > 50% (thesis is wrong)
        - Loss < 20% (position still viable)
        - VIX has spiked above 22
        """
        if pos.direction == "bullish":
            loss_pct = (pos.entry_price - current_price) / pos.entry_price * 100
        else:
            loss_pct = (current_price - pos.entry_price) / pos.entry_price * 100

        # Loss must be 20-50% range
        if loss_pct < 20 or loss_pct > 50:
            return False

        # DTE must be < 5
        if dte >= 5:
            return False

        # VIX check
        if self._vix > 22:
            return False

        return True

    # ── Telegram Alerts ────────────────────────────────────────────────

    def _send_entry_alert(self, setup: Setup, pos: Position) -> None:
        """Send Telegram alert on trade entry with reasoning."""
        from v7.telegram import AlertLevel

        day_type = self._playbook.day_classification.value if self._playbook else "?"
        direction = "LONG" if pos.direction == "bullish" else "SHORT"
        risk_amt = self._capital * (setup.max_risk_pct / 100)

        lines = [
            f"<b>V7 TRADE ENTRY</b>",
            "",
            f"<b>{pos.symbol}</b> {direction} — {setup.type.value}",
            f"Instrument: {pos.instrument}",
            f"Entry: {pos.entry_price:,.2f}  |  Qty: {pos.quantity}",
            f"SL: {pos.stoploss:,.2f}  |  Target: {pos.target:,.2f}",
            f"Risk: Rs.{risk_amt:,.0f} ({setup.max_risk_pct:.1f}%)",
            f"Conviction: {setup.conviction.value}",
            "",
            f"<b>Why:</b> {setup.trigger_condition}",
            f"Day type: {day_type}  |  VIX: {self._vix:.1f}",
            f"Setup: {setup.id}  |  Open positions: {len(self._positions)}",
        ]

        try:
            from v7.telegram import TelegramAlerter
            tg = TelegramAlerter()
            tg.send("\n".join(lines), AlertLevel.HIGH)
        except Exception as e:
            logger.warning("Entry Telegram alert failed: %s", e)

    def _send_exit_alert(self, pos: Position, exit_price: float, reason: str, pnl: float) -> None:
        """Send Telegram alert on trade exit."""
        from v7.telegram import AlertLevel

        direction = "LONG" if pos.direction == "bullish" else "SHORT"
        pnl_label = "WIN" if pnl > 0 else "LOSS"
        daily_pnl = self._daily.get("daily_pnl", 0)

        lines = [
            f"<b>V7 EXIT [{pnl_label}]</b>",
            "",
            f"<b>{pos.symbol}</b> {direction}",
            f"Entry: {pos.entry_price:,.2f} → Exit: {exit_price:,.2f}",
            f"P&amp;L: Rs.{pnl:+,.0f}",
            f"Reason: {reason}",
            f"Day P&amp;L: Rs.{daily_pnl:+,.0f}  |  Open: {len(self._positions)}",
        ]

        try:
            from v7.telegram import TelegramAlerter
            tg = TelegramAlerter()
            tg.send("\n".join(lines), AlertLevel.HIGH)
        except Exception as e:
            logger.warning("Exit Telegram alert failed: %s", e)

    # ── Stale Setup Detection ─────────────────────────────────────────

    def _check_stale_setups(self, now: datetime) -> None:
        """If all active setups are stale (price >1.5% from trigger), ask strategist to refresh.

        Only refreshes once per hour to avoid spamming Claude calls.
        Skips if we have open positions (focus on managing them).
        """
        if not self._playbook or not self._strategist:
            return

        # Don't refresh if we have open positions — focus on management
        if self._positions:
            return

        # Rate limit: once per hour
        last_refresh = self._daily.get("last_stale_refresh")
        if last_refresh:
            from datetime import datetime as dt
            try:
                last_dt = dt.fromisoformat(last_refresh)
                if (now - last_dt).total_seconds() < 3600:
                    return
            except (ValueError, TypeError):
                pass

        active = self._playbook.active_setups()
        if not active:
            # All setups fired/cancelled, budget may still allow trades
            trades_left = self._playbook.risk_budget.max_trades_today - self._daily.get("trades_today", 0)
            if trades_left <= 0:
                return
            logger.info("STALE: No active setups, %d trade slots left — requesting refresh", trades_left)
        else:
            # Check if all active triggers are far from current price
            all_stale = True
            for s in active:
                ltp = self._get_ltp_for_symbol(s.symbol)
                if ltp is None:
                    all_stale = False
                    break
                distance_pct = abs(ltp - s.trigger_level) / ltp * 100
                if distance_pct < 1.5:
                    all_stale = False
                    break
            if not all_stale:
                return
            logger.info("STALE: All %d active setups >1.5%% from trigger — requesting refresh", len(active))

        # Call strategist checkin for a refresh
        try:
            result = self._strategist.checkin(checkin_number=0)  # 0 = ad-hoc refresh
            if result.get("plan_changed"):
                logger.info("STALE REFRESH: Playbook updated — %s", result.get("summary", ""))
                self._playbook = self._state.load_playbook()  # reload
                from v7.telegram import TelegramAlerter, AlertLevel
                tg = TelegramAlerter()
                tg.send(f"V7 auto-refresh: {result.get('summary', 'playbook updated')}", AlertLevel.LOW)
            self._daily["last_stale_refresh"] = now.isoformat()
        except Exception as e:
            logger.warning("Stale refresh failed: %s", e)

    # ── Helpers ───────────────────────────────────────────────────────

    def _get_setup_type(self, setup_id: str) -> SetupType:
        """Look up SetupType from playbook by setup_id."""
        if self._playbook:
            for s in self._playbook.all_setups():
                if s.id == setup_id:
                    return s.type
        return SetupType.BREAKOUT_LONG

    def _get_ltp_for_symbol(self, symbol: str) -> Optional[float]:
        """Get LTP for a symbol from latest quotes."""
        # Try plain symbol first (get_batch_ltp returns {'NIFTY': 23461.5})
        if symbol in self._quotes:
            return self._quotes[symbol]
        # Try NSE:SYMBOL format
        key = f"NSE:{symbol}"
        if key in self._quotes:
            return self._quotes[key]
        # Try index aliases (defined locally, no kiteconnect import needed)
        alias = _INDEX_ALIASES.get(symbol)
        if alias and alias in self._quotes:
            return self._quotes[alias]
        # Try kite_data.INDEX_ALIASES if available
        try:
            from kite_data import INDEX_ALIASES
            alias = INDEX_ALIASES.get(symbol)
            if alias and alias in self._quotes:
                return self._quotes[alias]
        except ImportError:
            pass
        return None

    def _get_ltp_for_instrument(self, instrument: str) -> Optional[float]:
        """Get LTP for a specific instrument (NFO tradingsymbol)."""
        key = f"NFO:{instrument}"
        if key in self._quotes:
            return self._quotes[key]
        # Try direct match
        if instrument in self._quotes:
            return self._quotes[instrument]
        return None
