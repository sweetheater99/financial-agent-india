"""
Live Execution Bridge — extends paper trading to place real orders via AngelOne SmartAPI.

When LIVE_MODE is enabled, this module:
1. Places actual orders via AngelOne after a paper position is recorded
2. Tracks order IDs alongside paper positions
3. Monitors fill status and updates paper positions with actual fill prices (slippage tracking)

SAFETY: LIVE_MODE is False by default. Must be explicitly enabled in config.
"""

import logging
import os
import time
import urllib.parse
import urllib.request

import config

logger = logging.getLogger("paper_trade")


# ---------------------------------------------------------------------------
# Safety state
# ---------------------------------------------------------------------------

_consecutive_failures = 0


def _reset_failure_count():
    global _consecutive_failures
    _consecutive_failures = 0


def _record_failure():
    """Increment failure counter; disable LIVE_MODE if kill switch triggers."""
    global _consecutive_failures
    _consecutive_failures += 1
    if _consecutive_failures >= config.LIVE_CONSECUTIVE_FAIL_KILL:
        config.LIVE_MODE = False
        msg = (f"KILL SWITCH: {_consecutive_failures} consecutive order failures. "
               f"LIVE_MODE disabled automatically.")
        logger.critical(msg)
        _telegram_send(f"<b>KILL SWITCH ACTIVATED</b>\n{msg}")
        return True
    return False


# ---------------------------------------------------------------------------
# Telegram helpers
# ---------------------------------------------------------------------------

def _telegram_send(text: str) -> None:
    """Send a message via Telegram bot. Fails silently."""
    token = os.environ.get("DEAL_BOT_TOKEN", "")
    forum_chat_id = os.environ.get("TELEGRAM_FORUM_CHAT_ID", "")
    topic_id = os.environ.get("TELEGRAM_TOPIC_STOCKS", "")
    chat_id = os.environ.get("DEAL_BOT_CHAT_ID", "")
    if not token or not (forum_chat_id or chat_id):
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": forum_chat_id or chat_id,
            "text": text,
            "parse_mode": "HTML",
        }
        if forum_chat_id and topic_id:
            payload["message_thread_id"] = topic_id
        data = urllib.parse.urlencode(payload).encode()
        req = urllib.request.Request(url, data=data)
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


def _telegram_confirm(position: dict, action: str) -> bool:
    """Send Telegram confirmation and return True (always proceeds for now).

    In future, this could be extended to wait for a reply.
    Currently it just sends the alert so the user knows what's happening.
    """
    symbol = position.get("symbol", "?")
    qty = abs(position.get("quantity", 0))
    price = position.get("entry_price", 0)
    instrument = position.get("instrument", "EQ")
    direction = position.get("direction", "?")
    order_value = round(price * qty, 2)

    text = (
        f"<b>LIVE ORDER — {action}</b>\n"
        f"Symbol: <b>{symbol}</b>\n"
        f"Direction: {direction}\n"
        f"Instrument: {instrument}\n"
        f"Qty: {qty}\n"
        f"Price: {price:,.2f}\n"
        f"Order value: {order_value:,.0f}"
    )
    _telegram_send(text)
    return True


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_order_value(position: dict) -> tuple[bool, str]:
    """Check that the order value doesn't exceed the safety cap."""
    qty = abs(position.get("quantity", 0))
    price = position.get("entry_price", 0)
    order_value = price * qty

    if order_value > config.LIVE_MAX_ORDER_VALUE:
        return False, (f"Order value {order_value:,.0f} exceeds cap "
                       f"{config.LIVE_MAX_ORDER_VALUE:,.0f}")
    return True, ""


def _validate_price_deviation(expected_price: float, ltp: float) -> tuple[bool, str]:
    """Check that expected price doesn't deviate too much from LTP."""
    if ltp <= 0 or expected_price <= 0:
        return True, ""  # can't validate, allow

    deviation_pct = abs(expected_price - ltp) / ltp * 100
    if deviation_pct > config.LIVE_MAX_PRICE_DEVIATION_PCT:
        return False, (f"Price deviation {deviation_pct:.1f}% exceeds limit "
                       f"{config.LIVE_MAX_PRICE_DEVIATION_PCT:.1f}%")
    return True, ""


def _determine_exchange(position: dict) -> str:
    """Return the correct exchange for the position."""
    instrument = position.get("instrument", "EQ")
    if instrument in ("OPT", "SPREAD", "MOMENTUM", "CONDOR"):
        return "NFO"
    return "NSE"


def _determine_product_type(position: dict) -> str:
    """Return the correct product type for the position."""
    instrument = position.get("instrument", "EQ")
    if instrument in ("OPT", "SPREAD", "MOMENTUM", "CONDOR"):
        return "CARRYFORWARD"  # F&O carry forward
    return "DELIVERY"  # CNC for equity


def _get_trading_symbol(position: dict) -> str:
    """Get the trading symbol for the order.

    For equity: "SYMBOL-EQ"
    For options: uses the trading_symbol field from the position.
    """
    instrument = position.get("instrument", "EQ")
    if instrument == "EQ":
        symbol = position.get("symbol", "")
        # Already has -EQ suffix?
        if symbol.endswith("-EQ"):
            return symbol
        return f"{symbol}-EQ"
    # For options, the position should have trading_symbol set
    return position.get("trading_symbol", position.get("symbol", ""))


# ---------------------------------------------------------------------------
# LiveExecutor
# ---------------------------------------------------------------------------

class LiveExecutor:
    """Bridge between paper trading and live AngelOne order placement.

    Args:
        smart_api: Authenticated SmartConnect session.
        dry_run: If True (default), logs orders but doesn't place them.
    """

    def __init__(self, smart_api, dry_run: bool = True):
        self.smart_api = smart_api
        self.dry_run = dry_run

    # ── Entry ─────────────────────────────────────────────────────────────

    def execute_entry(self, position: dict) -> dict:
        """Place entry order for a position.

        Args:
            position: paper trade position dict with symbol, direction,
                     quantity, entry_price, instrument, trading_symbol, token

        Returns:
            dict with order_id, fill_price, fill_qty, status, slippage
        """
        result = {
            "order_id": None,
            "fill_price": None,
            "fill_qty": 0,
            "status": "skipped",
            "slippage": None,
        }

        # Safety: paper position must exist
        if not position.get("symbol"):
            result["status"] = "error"
            result["rejection_reason"] = "No symbol in position"
            logger.error("Live entry: no symbol in position")
            return result

        # Safety: order value cap
        ok, reason = _validate_order_value(position)
        if not ok:
            result["status"] = "blocked"
            result["rejection_reason"] = reason
            logger.warning("Live entry BLOCKED for %s: %s", position["symbol"], reason)
            _telegram_send(f"<b>ORDER BLOCKED</b>\n{position['symbol']}: {reason}")
            return result

        # Safety: price deviation check
        ltp = position.get("ltp_at_entry", position.get("entry_price", 0))
        ok, reason = _validate_price_deviation(position["entry_price"], ltp)
        if not ok:
            result["status"] = "blocked"
            result["rejection_reason"] = reason
            logger.warning("Live entry BLOCKED for %s: %s", position["symbol"], reason)
            return result

        # Telegram confirmation
        if config.LIVE_CONFIRM_TELEGRAM:
            _telegram_confirm(position, "ENTRY")

        # Dry run — don't place
        if self.dry_run:
            result["status"] = "dry_run"
            logger.info("Live entry DRY RUN for %s (would place order)", position["symbol"])
            return result

        # Place the order
        exchange = _determine_exchange(position)
        product_type = _determine_product_type(position)
        trading_symbol = _get_trading_symbol(position)
        token = position.get("token", "")

        order_params = {
            "variety": "NORMAL",
            "tradingsymbol": trading_symbol,
            "symboltoken": token,
            "transactiontype": "BUY",
            "exchange": exchange,
            "ordertype": "MARKET",
            "producttype": product_type,
            "duration": "DAY",
            "quantity": abs(position.get("quantity", 0)),
            "price": 0,
            "squareoff": 0,
            "stoploss": 0,
            "triggerprice": 0,
        }

        logger.info("Placing LIVE entry order: %s %s x%d @ MARKET",
                     trading_symbol, exchange, order_params["quantity"])

        try:
            order_id = self.smart_api.placeOrder(order_params)
            logger.info("Entry order placed: %s (order_id=%s)", trading_symbol, order_id)
            time.sleep(config.API_DELAY)

            # Check fill
            fill = self.check_fills(order_id)
            result["order_id"] = order_id
            result["status"] = fill.get("status", "pending")
            result["fill_price"] = fill.get("fill_price")
            result["fill_qty"] = fill.get("fill_qty", 0)

            if result["fill_price"] is not None:
                result["slippage"] = round(
                    result["fill_price"] - position["entry_price"], 2
                )

            _reset_failure_count()
            return result

        except Exception as e:
            logger.error("Live entry order FAILED for %s: %s", trading_symbol, e)
            result["status"] = "error"
            result["rejection_reason"] = str(e)
            killed = _record_failure()
            if killed:
                result["kill_switch"] = True
            return result

    # ── Exit ──────────────────────────────────────────────────────────────

    def execute_exit(self, position: dict, exit_price: float, reason: str) -> dict:
        """Place exit order for a position.

        Returns:
            dict with order_id, fill_price, fill_qty, status, slippage
        """
        result = {
            "order_id": None,
            "fill_price": None,
            "fill_qty": 0,
            "status": "skipped",
            "slippage": None,
        }

        if not position.get("symbol"):
            result["status"] = "error"
            result["rejection_reason"] = "No symbol in position"
            return result

        # Telegram confirmation
        if config.LIVE_CONFIRM_TELEGRAM:
            _telegram_confirm(position, f"EXIT ({reason})")

        if self.dry_run:
            result["status"] = "dry_run"
            logger.info("Live exit DRY RUN for %s (%s)", position["symbol"], reason)
            return result

        exchange = _determine_exchange(position)
        product_type = _determine_product_type(position)
        trading_symbol = _get_trading_symbol(position)
        token = position.get("token", "")

        order_params = {
            "variety": "NORMAL",
            "tradingsymbol": trading_symbol,
            "symboltoken": token,
            "transactiontype": "SELL",
            "exchange": exchange,
            "ordertype": "MARKET",
            "producttype": product_type,
            "duration": "DAY",
            "quantity": abs(position.get("quantity", 0)),
            "price": 0,
            "squareoff": 0,
            "stoploss": 0,
            "triggerprice": 0,
        }

        logger.info("Placing LIVE exit order: %s %s x%d @ MARKET (reason=%s)",
                     trading_symbol, exchange, order_params["quantity"], reason)

        try:
            order_id = self.smart_api.placeOrder(order_params)
            logger.info("Exit order placed: %s (order_id=%s)", trading_symbol, order_id)
            time.sleep(config.API_DELAY)

            fill = self.check_fills(order_id)
            result["order_id"] = order_id
            result["status"] = fill.get("status", "pending")
            result["fill_price"] = fill.get("fill_price")
            result["fill_qty"] = fill.get("fill_qty", 0)

            if result["fill_price"] is not None:
                result["slippage"] = round(result["fill_price"] - exit_price, 2)

            _reset_failure_count()
            return result

        except Exception as e:
            logger.error("Live exit order FAILED for %s: %s", trading_symbol, e)
            result["status"] = "error"
            result["rejection_reason"] = str(e)
            _record_failure()
            return result

    # ── GTT Stoploss ──────────────────────────────────────────────────────

    def place_stoploss_gtt(self, position: dict) -> dict:
        """Place a GTT (Good Till Triggered) stoploss order.

        GTT orders persist across sessions -- survives Pi restarts.
        Falls back to regular SL order if GTT fails.
        """
        result = {
            "order_id": None,
            "type": "gtt",
            "status": "skipped",
        }

        sl_price = position.get("stoploss_price")
        if sl_price is None:
            result["status"] = "error"
            result["rejection_reason"] = "No stoploss_price in position"
            return result

        if self.dry_run:
            result["status"] = "dry_run"
            return result

        exchange = _determine_exchange(position)
        product_type = _determine_product_type(position)
        trading_symbol = _get_trading_symbol(position)
        token = position.get("token", "")
        qty = abs(position.get("quantity", 0))

        # SL limit price slightly below trigger for buys (give room to fill)
        sl_limit = round(sl_price * 0.995, 2)

        # Try GTT first
        try:
            gtt_params = {
                "tradingsymbol": trading_symbol,
                "symboltoken": token,
                "exchange": exchange,
                "producttype": product_type,
                "transactiontype": "SELL",
                "price": sl_limit,
                "qty": qty,
                "triggerprice": sl_price,
                "disclosedqty": 0,
            }
            response = self.smart_api.gttCreateRule(gtt_params)
            time.sleep(config.API_DELAY)

            if response and response.get("data"):
                gtt_id = response["data"].get("id")
                result["order_id"] = gtt_id
                result["status"] = "placed"
                result["type"] = "gtt"
                logger.info("GTT SL placed for %s: trigger=%.2f, limit=%.2f (id=%s)",
                            trading_symbol, sl_price, sl_limit, gtt_id)
                _reset_failure_count()
                return result
            else:
                raise RuntimeError(f"GTT create returned: {response}")

        except Exception as e:
            logger.warning("GTT SL failed for %s: %s — falling back to regular SL",
                           trading_symbol, e)

        # Fallback: regular SL order
        try:
            from place_order import place_stoploss_order
            order_id = place_stoploss_order(
                self.smart_api,
                symbol=trading_symbol,
                token=token,
                qty=qty,
                price=sl_limit,
                trigger_price=sl_price,
                dry_run=False,
            )
            time.sleep(config.API_DELAY)

            if order_id:
                result["order_id"] = order_id
                result["status"] = "placed"
                result["type"] = "regular_sl"
                logger.info("Regular SL placed for %s (fallback): order_id=%s",
                            trading_symbol, order_id)
                _reset_failure_count()
            else:
                result["status"] = "error"
                result["rejection_reason"] = "SL order returned None"
                _record_failure()

        except Exception as e:
            logger.error("Fallback SL also failed for %s: %s", trading_symbol, e)
            result["status"] = "error"
            result["rejection_reason"] = str(e)
            _record_failure()

        return result

    # ── Update Stoploss ───────────────────────────────────────────────────

    def update_stoploss(self, position: dict, new_sl_price: float) -> dict:
        """Cancel old SL order and place new one (for trailing stop updates)."""
        result = {
            "order_id": None,
            "status": "skipped",
        }

        if self.dry_run:
            result["status"] = "dry_run"
            return result

        # Cancel old SL if exists
        old_sl_id = position.get("live_sl_order_id")
        old_sl_type = position.get("live_sl_type", "regular_sl")

        if old_sl_id:
            try:
                if old_sl_type == "gtt":
                    self.smart_api.gttCancelRule({"id": old_sl_id})
                else:
                    from place_order import cancel_order
                    cancel_order(self.smart_api, old_sl_id,
                                 variety="STOPLOSS", dry_run=False)
                time.sleep(config.API_DELAY)
                logger.info("Cancelled old SL for %s (id=%s, type=%s)",
                            position["symbol"], old_sl_id, old_sl_type)
            except Exception as e:
                logger.warning("Failed to cancel old SL %s: %s", old_sl_id, e)

        # Place new SL
        position_copy = dict(position)
        position_copy["stoploss_price"] = new_sl_price
        new_result = self.place_stoploss_gtt(position_copy)

        result["order_id"] = new_result.get("order_id")
        result["status"] = new_result.get("status", "error")
        result["type"] = new_result.get("type", "unknown")

        return result

    # ── Order Status ──────────────────────────────────────────────────────

    def check_fills(self, order_id: str) -> dict:
        """Check if order is filled, pending, or rejected.

        Returns: {status, fill_price, fill_qty, rejection_reason}
        """
        result = {
            "status": "unknown",
            "fill_price": None,
            "fill_qty": 0,
            "rejection_reason": None,
        }

        if not order_id:
            return result

        try:
            order_book = self.smart_api.orderBook()
            if not order_book or not order_book.get("data"):
                result["status"] = "unknown"
                return result

            for order in order_book["data"]:
                if order.get("orderid") == order_id:
                    status = order.get("orderstatus", "").lower()

                    if status == "complete":
                        result["status"] = "filled"
                        result["fill_price"] = float(order.get("averageprice", 0))
                        result["fill_qty"] = int(order.get("filledshares", 0))
                    elif status == "rejected":
                        result["status"] = "rejected"
                        result["rejection_reason"] = order.get("text", "Unknown")
                    elif status in ("cancelled", "canceled"):
                        result["status"] = "cancelled"
                    elif status in ("open", "pending", "trigger pending"):
                        result["status"] = "pending"
                    else:
                        result["status"] = status

                    return result

            result["status"] = "not_found"
        except Exception as e:
            logger.error("check_fills failed for %s: %s", order_id, e)
            result["status"] = "error"

        return result

    # ── Order Book ────────────────────────────────────────────────────────

    def get_order_book(self) -> list:
        """Get all orders for today."""
        try:
            order_book = self.smart_api.orderBook()
            if order_book and order_book.get("data"):
                return order_book["data"]
        except Exception as e:
            logger.error("get_order_book failed: %s", e)
        return []

    # ── Reconciliation ────────────────────────────────────────────────────

    def reconcile_positions(self, portfolio: dict) -> list[str]:
        """Compare paper positions vs actual broker positions.

        Returns list of discrepancies.
        """
        discrepancies = []

        # Get actual positions from broker
        try:
            holding_resp = self.smart_api.holding()
            time.sleep(config.API_DELAY)
        except Exception as e:
            discrepancies.append(f"Failed to fetch broker holdings: {e}")
            return discrepancies

        broker_holdings = {}
        if holding_resp and holding_resp.get("data"):
            for h in holding_resp["data"]:
                sym = h.get("tradingsymbol", "")
                qty = int(h.get("quantity", 0))
                if qty > 0:
                    broker_holdings[sym] = qty

        # Compare with paper positions
        open_pos = [p for p in portfolio.get("positions", [])
                    if p.get("status") == "open" and p.get("live_order_id")]

        for pos in open_pos:
            trading_symbol = _get_trading_symbol(pos)
            expected_qty = abs(pos.get("quantity", 0))
            actual_qty = broker_holdings.pop(trading_symbol, 0)

            if actual_qty == 0:
                discrepancies.append(
                    f"PAPER ONLY: {pos['symbol']} x{expected_qty} — "
                    f"not found in broker holdings"
                )
            elif actual_qty != expected_qty:
                discrepancies.append(
                    f"QTY MISMATCH: {pos['symbol']} — "
                    f"paper={expected_qty}, broker={actual_qty}"
                )

        # Check for broker positions not in paper
        for sym, qty in broker_holdings.items():
            discrepancies.append(
                f"BROKER ONLY: {sym} x{qty} — not tracked in paper portfolio"
            )

        if discrepancies:
            logger.warning("Reconciliation found %d discrepancies:", len(discrepancies))
            for d in discrepancies:
                logger.warning("  %s", d)
            _telegram_send(
                "<b>Position Reconciliation</b>\n" +
                "\n".join(discrepancies)
            )

        return discrepancies


# ---------------------------------------------------------------------------
# Integration helpers for paper_trade.py
# ---------------------------------------------------------------------------

def execute_live_entry(smart_api, position: dict) -> dict | None:
    """Convenience function: place a live entry if LIVE_MODE is on.

    Returns the execution result dict, or None if LIVE_MODE is off.
    Mutates position in-place to add live_order_id, live_fill_price, live_slippage.
    """
    if not config.LIVE_MODE:
        return None

    executor = LiveExecutor(smart_api, dry_run=False)
    result = executor.execute_entry(position)

    if result["status"] == "filled":
        position["live_order_id"] = result["order_id"]
        position["live_fill_price"] = result["fill_price"]
        position["live_slippage"] = result.get("slippage")
        logger.info("Live entry filled: %s @ %.2f (slippage: %s)",
                     position["symbol"], result["fill_price"], result.get("slippage"))

        # Place GTT stoploss
        if position.get("stoploss_price"):
            sl_result = executor.place_stoploss_gtt(position)
            if sl_result["status"] == "placed":
                position["live_sl_order_id"] = sl_result["order_id"]
                position["live_sl_type"] = sl_result.get("type", "unknown")

    elif result["status"] == "error":
        logger.error("Live entry failed for %s: %s",
                      position["symbol"], result.get("rejection_reason"))
        _telegram_send(
            f"<b>LIVE ORDER FAILED</b>\n"
            f"{position['symbol']}: {result.get('rejection_reason', 'unknown')}"
        )

    return result


def execute_live_exit(smart_api, position: dict, exit_price: float, reason: str) -> dict | None:
    """Convenience function: place a live exit if LIVE_MODE is on.

    Returns the execution result dict, or None if LIVE_MODE is off.
    """
    if not config.LIVE_MODE:
        return None

    # Only exit if we actually have a live position
    if not position.get("live_order_id"):
        logger.warning("Live exit skipped for %s: no live_order_id", position["symbol"])
        return None

    executor = LiveExecutor(smart_api, dry_run=False)

    # Cancel any existing SL order first
    sl_id = position.get("live_sl_order_id")
    sl_type = position.get("live_sl_type", "regular_sl")
    if sl_id:
        try:
            if sl_type == "gtt":
                smart_api.gttCancelRule({"id": sl_id})
            else:
                from place_order import cancel_order
                cancel_order(smart_api, sl_id, variety="STOPLOSS", dry_run=False)
            time.sleep(config.API_DELAY)
        except Exception as e:
            logger.warning("Failed to cancel SL before exit for %s: %s",
                           position["symbol"], e)

    result = executor.execute_exit(position, exit_price, reason)

    if result["status"] == "filled":
        position["live_exit_order_id"] = result["order_id"]
        position["live_exit_fill_price"] = result["fill_price"]
        position["live_exit_slippage"] = result.get("slippage")
        logger.info("Live exit filled: %s @ %.2f (slippage: %s)",
                     position["symbol"], result["fill_price"], result.get("slippage"))

    return result
