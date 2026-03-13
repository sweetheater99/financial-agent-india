# v7/order_manager.py
"""V7 Order Manager — Kite order placement, SL management, fill tracking.

Order execution rules from spec:
- Entry: limit order (bid+₹0.50 buy / ask-₹0.50 sell) → 30s timeout → market
- SL exit: limit → 15s timeout → market (urgent)
- Target exit: limit at target, let it fill passively
- Overnight carry: exchange-level SL/SL-M order via Kite place_order(order_type="SL")
- NOT bracket orders — standalone SL orders that survive bot crash
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger("v7.order_manager")


def get_kite():
    """Lazy import of get_kite to avoid kiteconnect import at module level."""
    from kite_data import get_kite as _get_kite
    return _get_kite()


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    SL = "SL"
    SL_M = "SL-M"


@dataclass
class OrderResult:
    """Result of an order placement attempt."""
    order_id: str
    filled: bool
    fill_price: float = 0.0
    fill_quantity: int = 0
    error: str = ""


# Kite transaction types
_SIDE_MAP = {OrderSide.BUY: "BUY", OrderSide.SELL: "SELL"}

# Default timeouts (seconds)
ENTRY_TIMEOUT = 30
SL_EXIT_TIMEOUT = 15


class OrderManager:
    """Manages order lifecycle via Kite Connect.

    Supports:
    - Entry orders with limit→market fallback
    - Exit orders with configurable timeout
    - Exchange-level SL/SL-M orders for overnight carry
    - SL order lifecycle (place, update, check liveness)
    - Dry-run mode for paper trading
    """

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self._kite = None
        self._dry_counter = 0

    def _get_kite(self):
        if self._kite is None:
            self._kite = get_kite()
        return self._kite

    # ── Entry Orders ──────────────────────────────────────────────────

    def place_entry_order(
        self,
        tradingsymbol: str,
        exchange: str,
        side: OrderSide,
        quantity: int,
        limit_price: float,
        timeout_seconds: int = ENTRY_TIMEOUT,
    ) -> OrderResult:
        """Place limit entry order. If not filled within timeout, modify to market.

        Args:
            limit_price: bid+₹0.50 for buy, ask-₹0.50 for sell (caller computes)
            timeout_seconds: seconds to wait before converting to market (default 30)
        """
        if self.dry_run:
            return self._dry_order(limit_price, quantity, prefix="DRY_")

        kite = self._get_kite()

        # Place limit order
        try:
            order_id = kite.place_order(
                variety="regular",
                exchange=exchange,
                tradingsymbol=tradingsymbol,
                transaction_type=_SIDE_MAP[side],
                quantity=quantity,
                product="MIS",  # intraday
                order_type="LIMIT",
                price=limit_price,
            )
        except Exception as e:
            logger.error("Entry order failed for %s: %s", tradingsymbol, e)
            return OrderResult(order_id="", filled=False, error=str(e))

        order_id = str(order_id)
        logger.info("Entry limit order placed: %s for %s @ %.2f", order_id, tradingsymbol, limit_price)

        # Wait for fill
        if timeout_seconds > 0:
            time.sleep(timeout_seconds)

        # Check if filled
        fill = self.get_fill_info(order_id)
        if fill["filled"]:
            return OrderResult(
                order_id=order_id, filled=True,
                fill_price=fill["price"], fill_quantity=fill["quantity"],
            )

        # Not filled — cancel and re-place as aggressive limit
        # (Zerodha blocks MARKET for stock options, and modify fails
        #  after a rejected MARKET attempt — so cancel + fresh order)
        logger.info("Entry order %s not filled after %ds, cancelling and re-placing", order_id, timeout_seconds)
        try:
            kite.cancel_order(variety="regular", order_id=order_id)
        except Exception as e:
            logger.warning("Cancel failed (may already be terminal): %s", e)

        # Place fresh aggressive limit order (+5% above original price)
        aggressive_price = round(limit_price * 1.05, 1)
        try:
            new_order_id = kite.place_order(
                variety="regular",
                exchange=exchange,
                tradingsymbol=tradingsymbol,
                transaction_type=_SIDE_MAP[side],
                quantity=quantity,
                product="MIS",
                order_type="LIMIT",
                price=aggressive_price,
            )
            new_order_id = str(new_order_id)
            logger.info("Aggressive limit order placed: %s @ %.2f (+5%%)", new_order_id, aggressive_price)
        except Exception as e:
            logger.error("Aggressive limit order failed: %s", e)
            return OrderResult(order_id=order_id, filled=False, error=str(e))

        # Wait briefly for fill
        time.sleep(5)
        fill = self.get_fill_info(new_order_id)
        if not fill["filled"]:
            # Last attempt — cancel if still pending
            try:
                kite.cancel_order(variety="regular", order_id=new_order_id)
            except Exception:
                pass
        return OrderResult(
            order_id=new_order_id, filled=fill["filled"],
            fill_price=fill["price"], fill_quantity=fill["quantity"],
        )

    # ── Exit Orders ───────────────────────────────────────────────────

    def place_exit_order(
        self,
        tradingsymbol: str,
        exchange: str,
        side: OrderSide,
        quantity: int,
        limit_price: float,
        is_sl_exit: bool = False,
        timeout_seconds: Optional[int] = None,
    ) -> OrderResult:
        """Place limit exit order. SL exits use 15s timeout; target exits are passive.

        Args:
            is_sl_exit: if True, uses shorter timeout (15s) and converts to market
            timeout_seconds: override timeout (default: 15 for SL, None for target)
        """
        if timeout_seconds is None:
            timeout_seconds = SL_EXIT_TIMEOUT if is_sl_exit else 0

        if self.dry_run:
            return self._dry_order(limit_price, quantity, prefix="DRY_EXIT_")

        kite = self._get_kite()

        try:
            order_id = kite.place_order(
                variety="regular",
                exchange=exchange,
                tradingsymbol=tradingsymbol,
                transaction_type=_SIDE_MAP[side],
                quantity=quantity,
                product="MIS",
                order_type="LIMIT",
                price=limit_price,
            )
        except Exception as e:
            logger.error("Exit order failed for %s: %s", tradingsymbol, e)
            return OrderResult(order_id="", filled=False, error=str(e))

        order_id = str(order_id)
        logger.info("Exit limit order placed: %s for %s @ %.2f (sl_exit=%s)",
                     order_id, tradingsymbol, limit_price, is_sl_exit)

        if timeout_seconds <= 0 and not is_sl_exit:
            # Target exit — passive, don't wait
            return OrderResult(order_id=order_id, filled=False)

        if timeout_seconds > 0:
            time.sleep(timeout_seconds)

        fill = self.get_fill_info(order_id)
        if fill["filled"]:
            return OrderResult(
                order_id=order_id, filled=True,
                fill_price=fill["price"], fill_quantity=fill["quantity"],
            )

        if is_sl_exit:
            # SL exits are urgent — convert to market
            logger.info("SL exit %s not filled, converting to market", order_id)
            try:
                kite.modify_order(
                    variety="regular",
                    order_id=order_id,
                    order_type="MARKET",
                )
            except Exception as e:
                # Zerodha blocks MARKET for stock options — cancel and re-place
                logger.warning("MARKET exit blocked (%s), cancel + aggressive limit (-10%%)", e)
                try:
                    kite.cancel_order(variety="regular", order_id=order_id)
                except Exception:
                    pass
                aggressive_price = max(round(limit_price * 0.90, 1), 0.50)
                try:
                    new_oid = kite.place_order(
                        variety="regular",
                        exchange=exchange,
                        tradingsymbol=tradingsymbol,
                        transaction_type=_SIDE_MAP[side],
                        quantity=quantity,
                        product="MIS",
                        order_type="LIMIT",
                        price=aggressive_price,
                    )
                    order_id = str(new_oid)
                    logger.info("Aggressive exit order: %s @ %.2f", order_id, aggressive_price)
                except Exception as e2:
                    logger.error("Aggressive exit order failed: %s", e2)
                    return OrderResult(order_id=order_id, filled=False, error=str(e2))

            time.sleep(3)
            fill = self.get_fill_info(order_id)

        return OrderResult(
            order_id=order_id, filled=fill["filled"],
            fill_price=fill["price"], fill_quantity=fill["quantity"],
        )

    # ── Exchange SL Orders ────────────────────────────────────────────

    def place_sl_order(
        self,
        tradingsymbol: str,
        exchange: str,
        side: OrderSide,
        quantity: int,
        trigger_price: float,
        limit_price: Optional[float] = None,
        product: str = "NRML",
    ) -> OrderResult:
        """Place exchange-level SL or SL-M order.

        These orders survive bot crash — the exchange holds them.
        Used for: overnight carry SL, intraday exchange SL.

        Args:
            trigger_price: price at which SL triggers
            limit_price: if None, places SL-M (market after trigger);
                        if set, places SL (limit after trigger)
            product: "NRML" for overnight, "MIS" for intraday
        """
        if self.dry_run:
            self._dry_counter += 1
            return OrderResult(
                order_id=f"DRY_SL_{self._dry_counter}",
                filled=False,  # SL orders start unfilled
            )

        kite = self._get_kite()
        order_type = "SL" if limit_price is not None else "SL-M"

        order_params = dict(
            variety="regular",
            exchange=exchange,
            tradingsymbol=tradingsymbol,
            transaction_type=_SIDE_MAP[side],
            quantity=quantity,
            product=product,
            order_type=order_type,
            trigger_price=trigger_price,
        )
        if limit_price is not None:
            order_params["price"] = limit_price

        try:
            order_id = kite.place_order(**order_params)
        except Exception as e:
            logger.error("SL order failed for %s: %s", tradingsymbol, e)
            return OrderResult(order_id="", filled=False, error=str(e))

        order_id = str(order_id)
        logger.info("Exchange %s order placed: %s for %s trigger=%.2f",
                     order_type, order_id, tradingsymbol, trigger_price)
        return OrderResult(order_id=order_id, filled=False)

    def update_sl_order(
        self,
        old_order_id: str,
        tradingsymbol: str,
        exchange: str,
        side: OrderSide,
        quantity: int,
        new_trigger_price: float,
        new_limit_price: Optional[float] = None,
        product: str = "NRML",
    ) -> OrderResult:
        """Update SL: cancel old order, place new one.

        Kite doesn't support modifying trigger_price reliably,
        so we cancel + re-place.
        """
        if not self.dry_run:
            kite = self._get_kite()
            try:
                kite.cancel_order(variety="regular", order_id=old_order_id)
                logger.info("Cancelled old SL order: %s", old_order_id)
            except Exception as e:
                logger.warning("Failed to cancel old SL %s: %s", old_order_id, e)

        return self.place_sl_order(
            tradingsymbol=tradingsymbol, exchange=exchange, side=side,
            quantity=quantity, trigger_price=new_trigger_price,
            limit_price=new_limit_price, product=product,
        )

    def is_sl_order_live(self, order_id: str) -> bool:
        """Check if an exchange SL order is still active (TRIGGER PENDING)."""
        if self.dry_run:
            return True

        kite = self._get_kite()
        try:
            history = kite.order_history(order_id=order_id)
            if history:
                latest = history[-1] if isinstance(history, list) else history
                status = latest.get("status", "")
                return status in ("TRIGGER PENDING", "OPEN", "OPEN PENDING")
        except Exception as e:
            logger.warning("Failed to check SL order %s: %s", order_id, e)
        return False

    def cancel_order(self, order_id: str) -> bool:
        """Cancel any order by ID."""
        if self.dry_run:
            return True
        kite = self._get_kite()
        try:
            kite.cancel_order(variety="regular", order_id=order_id)
            return True
        except Exception as e:
            logger.warning("Failed to cancel order %s: %s", order_id, e)
            return False

    # ── Fill Info ─────────────────────────────────────────────────────

    def get_fill_info(self, order_id: str) -> dict:
        """Get fill status and price for an order.

        Returns: {"filled": bool, "price": float, "quantity": int}
        """
        if self.dry_run:
            return {"filled": True, "price": 0.0, "quantity": 0}

        kite = self._get_kite()
        try:
            history = kite.order_history(order_id=order_id)
            if history:
                latest = history[-1] if isinstance(history, list) else history
                filled = latest.get("status") == "COMPLETE"
                return {
                    "filled": filled,
                    "price": float(latest.get("average_price", 0)),
                    "quantity": int(latest.get("filled_quantity", 0)),
                }
        except Exception as e:
            logger.warning("Failed to get fill info for %s: %s", order_id, e)
        return {"filled": False, "price": 0.0, "quantity": 0}

    # ── Dry Run Helpers ───────────────────────────────────────────────

    def _dry_order(self, price: float, quantity: int, prefix: str = "DRY_") -> OrderResult:
        self._dry_counter += 1
        return OrderResult(
            order_id=f"{prefix}{self._dry_counter}",
            filled=True,
            fill_price=price,
            fill_quantity=quantity,
        )
