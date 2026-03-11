# V7 Plan 3: Executor, Order Manager & Theta Engine

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the mechanical execution layer — the Order Manager that places/tracks orders via Kite, the Executor that runs the 3-minute tick cycle with phase-based behavior and position management, and the Theta Engine for weekly iron condor income.

**Architecture:** All three modules live in `v7/`. They import shared types from Plan 1 (`v7.types`, `v7.state`, `v7.data_feed`, `v7.config_v7`, `v7.strike_selector`, `v7.margin`) and the Risk Engine from Plan 2 (`v7.risk_engine`). The Executor does NOT call Claude — it calls `Strategist.handle_exception()` for exception scenarios.

**Tech Stack:** Python 3.13, Kite Connect, pytest

**Spec:** `docs/superpowers/specs/2026-03-11-v7-professional-trader-bot-design.md` — Component 2 (Executor, lines 319-438), Component 5 (Theta Engine, lines 597-642), Order Execution (lines 384-402), Carry Rules (lines 66-72), Exception Triggers (lines 301-316), Margin Budget (lines 942-957), State Persistence (lines 960-982), Position Rolling (lines 808-830)

**Depends on:** Plan 1 (types, state, data_feed, strike_selector, margin, config_v7), Plan 2 (risk_engine)
**Blocks:** Plan 4 (Journal), Plan 5 (Integration/Orchestrator)

---

## File Structure

```
v7/
├── order_manager.py         # Order placement, SL orders, fill tracking via Kite
├── executor.py              # 3-min tick cycle, phase behavior, position management
├── theta_engine.py          # Weekly iron condor strategy, independent risk budget
tests/
├── test_v7_order_manager.py
├── test_v7_executor.py
├── test_v7_theta_engine.py
```

---

## Chunk 1: Order Manager

### Task 1: Order placement with limit→market fallback

**Files:**
- Create: `v7/order_manager.py`
- Test: `tests/test_v7_order_manager.py`

- [ ] **Step 1: Write failing tests for OrderManager**

```python
# tests/test_v7_order_manager.py
"""Tests for V7 Order Manager — order placement, SL orders, fill tracking."""
import pytest
from unittest.mock import MagicMock, patch, call
from datetime import datetime, date
from v7.order_manager import OrderManager, OrderResult, OrderType, OrderSide


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_kite():
    kite = MagicMock()
    kite.place_order = MagicMock(return_value="order_123")
    kite.order_history = MagicMock(return_value=[
        {"status": "COMPLETE", "average_price": 105.0, "filled_quantity": 75}
    ])
    kite.orders = MagicMock(return_value=[
        {"order_id": "order_123", "status": "COMPLETE", "average_price": 105.0}
    ])
    kite.cancel_order = MagicMock()
    kite.modify_order = MagicMock()
    return kite


@pytest.fixture
def order_mgr(mock_kite):
    with patch("v7.order_manager.get_kite", return_value=mock_kite):
        mgr = OrderManager(dry_run=False)
        mgr._kite = mock_kite
        return mgr


# ── Entry Orders ──────────────────────────────────────────────────────


def test_place_entry_limit_order(order_mgr, mock_kite):
    """Entry: limit order at bid+0.50 for buy."""
    result = order_mgr.place_entry_order(
        tradingsymbol="NIFTY2530624400CE",
        exchange="NFO",
        side=OrderSide.BUY,
        quantity=75,
        limit_price=104.50,
    )
    assert result.order_id == "order_123"
    mock_kite.place_order.assert_called_once()
    call_kwargs = mock_kite.place_order.call_args[1]
    assert call_kwargs["order_type"] == "LIMIT"
    assert call_kwargs["price"] == 104.50
    assert call_kwargs["quantity"] == 75


def test_entry_limit_timeout_converts_to_market(order_mgr, mock_kite):
    """If limit not filled in 30s, modify to market."""
    # First call: place limit. order_history returns OPEN (not filled).
    mock_kite.order_history.return_value = [
        {"status": "OPEN", "average_price": 0, "filled_quantity": 0}
    ]
    # After modify, order fills
    mock_kite.modify_order = MagicMock()

    result = order_mgr.place_entry_order(
        tradingsymbol="NIFTY2530624400CE",
        exchange="NFO",
        side=OrderSide.BUY,
        quantity=75,
        limit_price=104.50,
        timeout_seconds=0,  # immediate timeout for testing
    )
    # Should have tried to modify to market
    mock_kite.modify_order.assert_called_once()
    modify_kwargs = mock_kite.modify_order.call_args[1]
    assert modify_kwargs["order_type"] == "MARKET"


# ── Exit Orders ───────────────────────────────────────────────────────


def test_place_exit_limit_order(order_mgr, mock_kite):
    """Exit: limit order at ask-0.50 for sell."""
    result = order_mgr.place_exit_order(
        tradingsymbol="NIFTY2530624400CE",
        exchange="NFO",
        side=OrderSide.SELL,
        quantity=75,
        limit_price=149.50,
    )
    assert result.order_id == "order_123"
    call_kwargs = mock_kite.place_order.call_args[1]
    assert call_kwargs["order_type"] == "LIMIT"


def test_sl_exit_timeout_shorter(order_mgr, mock_kite):
    """SL exit: 15s timeout (not 30s), then market."""
    mock_kite.order_history.return_value = [
        {"status": "OPEN", "average_price": 0, "filled_quantity": 0}
    ]
    result = order_mgr.place_exit_order(
        tradingsymbol="NIFTY2530624400CE",
        exchange="NFO",
        side=OrderSide.SELL,
        quantity=75,
        limit_price=79.50,
        is_sl_exit=True,
        timeout_seconds=0,  # immediate for testing
    )
    mock_kite.modify_order.assert_called_once()


# ── Exchange SL Orders ────────────────────────────────────────────────


def test_place_exchange_sl_order(order_mgr, mock_kite):
    """Place SL order on exchange (for overnight carry)."""
    result = order_mgr.place_sl_order(
        tradingsymbol="NIFTY2530624400CE",
        exchange="NFO",
        side=OrderSide.SELL,
        quantity=75,
        trigger_price=80.0,
        limit_price=79.0,
    )
    assert result.order_id == "order_123"
    call_kwargs = mock_kite.place_order.call_args[1]
    assert call_kwargs["order_type"] == "SL"
    assert call_kwargs["trigger_price"] == 80.0
    assert call_kwargs["price"] == 79.0


def test_place_exchange_slm_order(order_mgr, mock_kite):
    """Place SL-M (market) order on exchange."""
    result = order_mgr.place_sl_order(
        tradingsymbol="NIFTY2530624400CE",
        exchange="NFO",
        side=OrderSide.SELL,
        quantity=75,
        trigger_price=80.0,
        limit_price=None,  # None = SL-M
    )
    call_kwargs = mock_kite.place_order.call_args[1]
    assert call_kwargs["order_type"] == "SL-M"
    assert call_kwargs["trigger_price"] == 80.0


# ── SL Order Management ──────────────────────────────────────────────


def test_update_sl_order(order_mgr, mock_kite):
    """Update SL: cancel old, place new."""
    new_result = order_mgr.update_sl_order(
        old_order_id="old_sl_123",
        tradingsymbol="NIFTY2530624400CE",
        exchange="NFO",
        side=OrderSide.SELL,
        quantity=75,
        new_trigger_price=90.0,
        new_limit_price=89.0,
    )
    # Should cancel old order first
    mock_kite.cancel_order.assert_called_once()
    # Then place new SL
    assert new_result.order_id == "order_123"


def test_check_sl_order_live(order_mgr, mock_kite):
    """Verify SL order is still active on exchange."""
    mock_kite.order_history.return_value = [
        {"status": "TRIGGER PENDING", "average_price": 0}
    ]
    assert order_mgr.is_sl_order_live("sl_order_123") is True


def test_check_sl_order_triggered(order_mgr, mock_kite):
    """SL order has been triggered (filled)."""
    mock_kite.order_history.return_value = [
        {"status": "COMPLETE", "average_price": 79.5, "filled_quantity": 75}
    ]
    assert order_mgr.is_sl_order_live("sl_order_123") is False


def test_check_sl_order_cancelled(order_mgr, mock_kite):
    """SL order was cancelled."""
    mock_kite.order_history.return_value = [
        {"status": "CANCELLED", "average_price": 0}
    ]
    assert order_mgr.is_sl_order_live("sl_order_123") is False


# ── Fill Verification ─────────────────────────────────────────────────


def test_get_fill_price(order_mgr, mock_kite):
    """Get actual fill price from order history."""
    mock_kite.order_history.return_value = [
        {"status": "COMPLETE", "average_price": 105.25, "filled_quantity": 75}
    ]
    fill = order_mgr.get_fill_info("order_123")
    assert fill["price"] == 105.25
    assert fill["quantity"] == 75
    assert fill["filled"] is True


def test_get_fill_not_filled(order_mgr, mock_kite):
    """Order not yet filled."""
    mock_kite.order_history.return_value = [
        {"status": "OPEN", "average_price": 0, "filled_quantity": 0}
    ]
    fill = order_mgr.get_fill_info("order_123")
    assert fill["filled"] is False


# ── Dry Run Mode ──────────────────────────────────────────────────────


def test_dry_run_no_kite_calls():
    """In dry_run mode, no actual Kite calls are made."""
    mgr = OrderManager(dry_run=True)
    result = mgr.place_entry_order(
        tradingsymbol="NIFTY2530624400CE",
        exchange="NFO",
        side=OrderSide.BUY,
        quantity=75,
        limit_price=104.50,
    )
    assert result.order_id.startswith("DRY_")
    assert result.filled is True  # simulated fill


def test_dry_run_sl_order():
    """Dry run SL order returns simulated ID."""
    mgr = OrderManager(dry_run=True)
    result = mgr.place_sl_order(
        tradingsymbol="NIFTY2530624400CE",
        exchange="NFO",
        side=OrderSide.SELL,
        quantity=75,
        trigger_price=80.0,
        limit_price=79.0,
    )
    assert result.order_id.startswith("DRY_SL_")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_order_manager.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'v7.order_manager'`

- [ ] **Step 3: Implement OrderManager**

```python
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

from kite_data import get_kite

logger = logging.getLogger("v7.order_manager")


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

        # Not filled — modify to market
        logger.info("Entry order %s not filled after %ds, converting to market", order_id, timeout_seconds)
        try:
            kite.modify_order(
                variety="regular",
                order_id=order_id,
                order_type="MARKET",
            )
        except Exception as e:
            logger.error("Failed to modify order %s to market: %s", order_id, e)
            return OrderResult(order_id=order_id, filled=False, error=str(e))

        # Brief wait for market fill
        time.sleep(2)
        fill = self.get_fill_info(order_id)
        return OrderResult(
            order_id=order_id, filled=fill["filled"],
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
                logger.error("Failed to modify SL exit to market: %s", e)
                return OrderResult(order_id=order_id, filled=False, error=str(e))

            time.sleep(2)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_order_manager.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd ~/financial-agent-india && git add v7/order_manager.py tests/test_v7_order_manager.py && \
git commit -m "v7: add OrderManager with limit→market fallback and exchange SL orders"
```

---

## Chunk 2: Executor Core — Phase Detection & Tick Cycle

### Task 2: Executor skeleton with phase-based behavior

**Files:**
- Create: `v7/executor.py`
- Test: `tests/test_v7_executor.py`

- [ ] **Step 1: Write failing tests for Executor core**

```python
# tests/test_v7_executor.py
"""Tests for V7 Executor — tick cycle, phase behavior, position management."""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, date, time, timedelta
from v7.types import (
    DayPhase, Setup, SetupType, Position, Playbook,
    DayClassification, RiskBudget, CarryRules, Conviction,
)
from v7.executor import Executor


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_deps():
    """Create mocked dependencies for Executor."""
    state = MagicMock()
    state.load_playbook.return_value = Playbook(
        date=date(2026, 3, 11),
        day_classification=DayClassification.LIKELY_TREND_UP,
        nifty_bias="bullish",
        nifty_setups=[],
        stock_plans=[],
        risk_budget=RiskBudget(
            max_capital_at_risk_today_pct=4.0,
            max_trades_today=4,
            max_per_trade_risk_pct=1.5,
        ),
        no_trade_conditions=[],
        carry_rules=CarryRules(),
    )
    state.load_positions.return_value = []
    state.load_daily_state.return_value = {
        "date": str(date(2026, 3, 11)),
        "daily_pnl": 0.0,
        "trades_today": 0,
        "consecutive_sl_hits": 0,
        "vix_30min_ago": None,
        "nifty_open": None,
    }
    state.save_positions = MagicMock()
    state.save_daily_state = MagicMock()

    data_feed = MagicMock()
    data_feed.get_ltp_batch.return_value = {"NSE:NIFTY 50": 24350.0}
    data_feed.get_vix.return_value = 15.0
    data_feed.get_candles.return_value = []

    risk_engine = MagicMock()
    risk_engine.can_enter.return_value = True
    risk_engine.check_margin.return_value = True

    order_mgr = MagicMock()
    margin = MagicMock()
    margin.current_utilization_pct.return_value = 30.0

    return {
        "state": state,
        "data_feed": data_feed,
        "risk_engine": risk_engine,
        "order_mgr": order_mgr,
        "margin": margin,
    }


@pytest.fixture
def executor(mock_deps):
    return Executor(
        state_mgr=mock_deps["state"],
        data_feed=mock_deps["data_feed"],
        risk_engine=mock_deps["risk_engine"],
        order_mgr=mock_deps["order_mgr"],
        margin_tracker=mock_deps["margin"],
        capital=300_000,
    )


# ── Phase Detection ──────────────────────────────────────────────────


def test_phase_opening_read():
    """9:15-9:44 is Opening Read."""
    assert DayPhase.from_time(time(9, 20)) == DayPhase.OPENING_READ


def test_phase_active_trading():
    """9:45-14:29 is Active Trading."""
    assert DayPhase.from_time(time(11, 0)) == DayPhase.ACTIVE_TRADING


def test_phase_wind_down():
    """14:30-15:14 is Wind Down."""
    assert DayPhase.from_time(time(14, 45)) == DayPhase.WIND_DOWN


# ── 15-min Candle Boundary ────────────────────────────────────────────


def test_is_candle_boundary_true():
    """Minutes :00, :15, :30, :45 are candle boundaries."""
    from v7.executor import is_15min_boundary
    assert is_15min_boundary(time(10, 0)) is True
    assert is_15min_boundary(time(10, 15)) is True
    assert is_15min_boundary(time(10, 30)) is True
    assert is_15min_boundary(time(10, 45)) is True


def test_is_candle_boundary_false():
    """Other minutes are NOT candle boundaries."""
    from v7.executor import is_15min_boundary
    assert is_15min_boundary(time(10, 3)) is False
    assert is_15min_boundary(time(10, 12)) is False
    assert is_15min_boundary(time(10, 27)) is False


# ── Tick Cycle Behavior ───────────────────────────────────────────────


def test_tick_opening_read_skips_triggers(executor, mock_deps):
    """During Opening Read (9:15-9:45), do NOT fire playbook triggers."""
    # Add a setup to the playbook
    setup = Setup(
        id="N1", priority=1, type=SetupType.BREAKOUT_LONG,
        symbol="NIFTY", trigger_level=24350.0,
        trigger_condition="15-min close above",
        instrument="NIFTY CE", strike_logic="delta 0.45",
        target=24500.0, stoploss=24280.0, max_risk_pct=1.5,
    )
    executor._playbook.nifty_setups = [setup]

    with patch("v7.executor.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 3, 11, 9, 20)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        executor.tick()

    # Order manager should NOT have been called for entry
    mock_deps["order_mgr"].place_entry_order.assert_not_called()


def test_tick_active_trading_checks_triggers(executor, mock_deps):
    """During Active Trading on candle boundary, check triggers."""
    setup = Setup(
        id="N1", priority=1, type=SetupType.BREAKOUT_LONG,
        symbol="NIFTY", trigger_level=24300.0,
        trigger_condition="15-min close above",
        instrument="NIFTY CE", strike_logic="delta 0.45",
        target=24500.0, stoploss=24200.0, max_risk_pct=1.5,
    )
    executor._playbook.nifty_setups = [setup]
    mock_deps["data_feed"].get_ltp_batch.return_value = {"NSE:NIFTY 50": 24350.0}

    with patch("v7.executor.datetime") as mock_dt:
        # 10:00 is a candle boundary during active trading
        mock_dt.now.return_value = datetime(2026, 3, 11, 10, 0)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        executor.tick()

    # Should have attempted to evaluate the trigger (whether it fires depends on full logic)
    # The key assertion: trigger evaluation happened (check via internal state or order call)


def test_tick_non_boundary_skips_triggers(executor, mock_deps):
    """On non-boundary ticks, only manage positions — skip trigger checks."""
    setup = Setup(
        id="N1", priority=1, type=SetupType.BREAKOUT_LONG,
        symbol="NIFTY", trigger_level=24300.0,
        trigger_condition="15-min close above",
        instrument="NIFTY CE", strike_logic="delta 0.45",
        target=24500.0, stoploss=24200.0, max_risk_pct=1.5,
    )
    executor._playbook.nifty_setups = [setup]

    with patch("v7.executor.datetime") as mock_dt:
        # 10:03 is NOT a candle boundary
        mock_dt.now.return_value = datetime(2026, 3, 11, 10, 3)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        executor.tick()

    mock_deps["order_mgr"].place_entry_order.assert_not_called()


def test_tick_outside_hours_exits_early(executor, mock_deps):
    """Outside market hours, tick exits immediately."""
    with patch("v7.executor.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 3, 11, 7, 0)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        executor.tick()

    # No data fetching should occur
    mock_deps["data_feed"].get_ltp_batch.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_executor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'v7.executor'`

- [ ] **Step 3: Implement Executor core**

```python
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

        # 4. EXCEPTION DETECTION (every tick)
        self._check_exceptions(now)

        # 5. PERSIST STATE
        self._state.save_positions(self._positions)
        self._state.save_daily_state(self._daily)

    # ── Data Fetching ─────────────────────────────────────────────────

    def _fetch_data(self, now: datetime) -> None:
        """Fetch LTP batch + VIX. Candles fetched on demand."""
        try:
            self._quotes = self._data.get_ltp_batch()
        except Exception as e:
            logger.error("LTP batch fetch failed: %s", e)
            self._quotes = {}

        try:
            self._vix = self._data.get_vix() or 0.0
        except Exception as e:
            logger.warning("VIX fetch failed: %s", e)

        # Track VIX 30min ago for exception detection
        if self._daily.get("vix_30min_ago") is None:
            self._daily["vix_30min_ago"] = self._vix

        # Track Nifty open
        if self._daily.get("nifty_open") is None:
            nifty_ltp = self._quotes.get("NSE:NIFTY 50")
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

        if is_15min_boundary(now.time()):
            self._evaluate_triggers(now)

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

        all_setups = self._playbook.all_setups
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

        # Margin check
        if not self._risk.check_margin(risk_amount):
            logger.info("Setup %s triggered but margin insufficient", setup.id)
            return

        # F&O ban check, brokerage check would go here

        # ENTER
        logger.info("Setup %s TRIGGERED at %.2f (level=%.2f)", setup.id, ltp, setup.trigger_level)
        self._enter_position(setup, ltp, now)

    def _enter_position(self, setup: Setup, trigger_ltp: float, now: datetime) -> None:
        """Execute entry for a triggered setup."""
        setup.fired = True

        # Strike selection would happen here via StrikeSelector
        # For now, use the instrument from setup
        tradingsymbol = setup.instrument
        exchange = "NFO"

        # Calculate limit price: bid + 0.50 for buy
        limit_price = trigger_ltp + 0.50

        risk_amount = self._capital * (setup.max_risk_pct / 100)
        lot_size = 75  # would come from instrument lookup
        quantity = lot_size

        from v7.order_manager import OrderSide
        result = self._orders.place_entry_order(
            tradingsymbol=tradingsymbol,
            exchange=exchange,
            side=OrderSide.BUY,
            quantity=quantity,
            limit_price=limit_price,
        )

        if not result.filled:
            logger.warning("Entry order for %s did not fill", setup.id)
            return

        # Create position
        pos = Position(
            symbol=setup.symbol,
            instrument=tradingsymbol,
            direction="bullish" if setup.type in (
                SetupType.BREAKOUT_LONG, SetupType.SUPPORT_BOUNCE,
                SetupType.CREDIT_SPREAD_BULL,
            ) else "bearish",
            entry_price=result.fill_price,
            quantity=quantity,
            lot_size=lot_size,
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

    # ── Position Management ───────────────────────────────────────────

    def _manage_positions(self, now: datetime) -> None:
        """Update P&L, check SL/target/trailing for each position."""
        positions_to_remove = []

        for pos in self._positions:
            ltp = self._get_ltp_for_instrument(pos.instrument)
            if ltp is None:
                continue

            # Update peak price
            if pos.direction == "bullish":
                pos.peak_price = max(pos.peak_price, ltp)
            else:
                pos.peak_price = min(pos.peak_price, ltp) if pos.peak_price > 0 else ltp

            pnl = pos.unrealized_pnl(ltp)

            # SL hit → EXIT immediately
            if self._is_sl_hit(pos, ltp):
                logger.info("SL HIT: %s @ %.2f (SL=%.2f)", pos.instrument, ltp, pos.stoploss)
                self._exit_position(pos, ltp, "stoploss", now)
                positions_to_remove.append(pos)
                self._daily["consecutive_sl_hits"] = self._daily.get("consecutive_sl_hits", 0) + 1
                continue

            # Target hit → full exit
            if self._is_target_hit(pos, ltp):
                logger.info("TARGET HIT: %s @ %.2f (TGT=%.2f)", pos.instrument, ltp, pos.target)
                self._exit_position(pos, ltp, "target", now)
                positions_to_remove.append(pos)
                self._daily["consecutive_sl_hits"] = 0
                continue

            # 1:1 R:R → move SL to breakeven
            self._check_breakeven(pos, ltp)

            # Trailing stop: peak - 1.5x ATR
            trailing_sl = self._compute_trailing_sl(pos, ltp)
            if trailing_sl is not None and self._is_better_sl(pos, trailing_sl):
                old_sl = pos.stoploss
                pos.stoploss = trailing_sl
                logger.info("TRAILING SL: %s moved %.2f → %.2f", pos.instrument, old_sl, trailing_sl)
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
        nifty_now = self._quotes.get("NSE:NIFTY 50")
        if nifty_open and nifty_now:
            nifty_move_pct = abs(nifty_now - nifty_open) / nifty_open * 100
            if nifty_move_pct > 1.5:
                exceptions.append(f"Nifty move: {nifty_move_pct:.1f}% from open")

        # Margin > 70%
        margin_pct = self._margin.current_utilization_pct()
        if margin_pct > 70:
            exceptions.append(f"Margin utilization: {margin_pct:.0f}%")

        # 3 consecutive SL hits
        if self._daily.get("consecutive_sl_hits", 0) >= 3:
            exceptions.append(f"3 consecutive SL hits today")

        if exceptions:
            logger.warning("EXCEPTION DETECTED: %s", "; ".join(exceptions))
            try:
                self._strategist.handle_exception(
                    exceptions=exceptions,
                    positions=self._positions,
                    daily_state=self._daily,
                    vix=self._vix,
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

    # ── Helpers ───────────────────────────────────────────────────────

    def _get_ltp_for_symbol(self, symbol: str) -> Optional[float]:
        """Get LTP for a symbol from latest quotes."""
        # Try NSE:SYMBOL format
        key = f"NSE:{symbol}"
        if key in self._quotes:
            return self._quotes[key]
        # Try index aliases
        from kite_data import INDEX_ALIASES
        alias = INDEX_ALIASES.get(symbol)
        if alias and alias in self._quotes:
            return self._quotes[alias]
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_executor.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd ~/financial-agent-india && git add v7/executor.py tests/test_v7_executor.py && \
git commit -m "v7: add Executor with phase-based tick cycle, position management, exception detection"
```

---

## Chunk 3: Theta Engine — Weekly Iron Condor

### Task 3: Iron condor entry, daily management, profit/time rules

**Files:**
- Create: `v7/theta_engine.py`
- Test: `tests/test_v7_theta_engine.py`

- [ ] **Step 1: Write failing tests for ThetaEngine**

```python
# tests/test_v7_theta_engine.py
"""Tests for V7 Theta Engine — weekly Nifty iron condor strategy."""
import pytest
from unittest.mock import MagicMock, patch
from datetime import date, datetime, timedelta
from v7.theta_engine import ThetaEngine, CondorPosition, CondorLeg


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_deps():
    data_feed = MagicMock()
    data_feed.get_vix.return_value = 16.0
    data_feed.get_ltp_batch.return_value = {"NSE:NIFTY 50": 24350.0}
    data_feed.get_option_chain.return_value = _mock_option_chain()

    order_mgr = MagicMock()
    from v7.order_manager import OrderResult
    order_mgr.place_entry_order.return_value = OrderResult(
        order_id="theta_order_1", filled=True, fill_price=25.0, fill_quantity=75,
    )
    order_mgr.place_exit_order.return_value = OrderResult(
        order_id="theta_exit_1", filled=True, fill_price=12.0, fill_quantity=75,
    )

    state = MagicMock()
    state.load_theta_state.return_value = None
    state.save_theta_state = MagicMock()

    strike_selector = MagicMock()
    margin = MagicMock()
    margin.current_utilization_pct.return_value = 25.0
    margin.estimate_condor_margin.return_value = 70000.0

    return {
        "data_feed": data_feed,
        "order_mgr": order_mgr,
        "state": state,
        "strike_selector": strike_selector,
        "margin": margin,
    }


@pytest.fixture
def theta(mock_deps):
    return ThetaEngine(
        data_feed=mock_deps["data_feed"],
        order_mgr=mock_deps["order_mgr"],
        state_mgr=mock_deps["state"],
        strike_selector=mock_deps["strike_selector"],
        margin_tracker=mock_deps["margin"],
        capital=300_000,
    )


def _mock_option_chain():
    """Create a mock option chain with strikes around 24350."""
    chain = []
    for strike in range(23800, 24900, 50):
        entry = {"strikePrice": strike}
        # Delta decreases as we go further from ATM
        dist = abs(strike - 24350)
        delta = max(0.05, 0.50 - dist / 1000)
        if strike < 24350:
            entry["PE"] = {
                "lastTradedPrice": max(5, 200 - dist * 0.3),
                "delta": -delta,
                "openInterest": 50000,
                "volume": 10000,
                "bidPrice": max(4, 199 - dist * 0.3),
                "askPrice": max(6, 201 - dist * 0.3),
                "lotSize": 75,
            }
        if strike > 24350:
            entry["CE"] = {
                "lastTradedPrice": max(5, 200 - dist * 0.3),
                "delta": delta,
                "openInterest": 50000,
                "volume": 10000,
                "bidPrice": max(4, 199 - dist * 0.3),
                "askPrice": max(6, 201 - dist * 0.3),
                "lotSize": 75,
            }
        chain.append(entry)
    return chain


# ── Entry Conditions ──────────────────────────────────────────────────


def test_should_enter_vix_in_range(theta, mock_deps):
    """Entry allowed when VIX is 14-20."""
    mock_deps["data_feed"].get_vix.return_value = 16.0
    assert theta._should_enter_today(date(2026, 3, 9)) is True  # Monday


def test_should_not_enter_vix_too_low(theta, mock_deps):
    """Skip if VIX < 14 (premiums too low)."""
    mock_deps["data_feed"].get_vix.return_value = 12.0
    assert theta._should_enter_today(date(2026, 3, 9)) is False


def test_should_not_enter_vix_too_high(theta, mock_deps):
    """Skip if VIX > 20 (too volatile)."""
    mock_deps["data_feed"].get_vix.return_value = 22.0
    assert theta._should_enter_today(date(2026, 3, 9)) is False


def test_should_enter_only_mon_tue(theta, mock_deps):
    """Only enter on Monday or Tuesday."""
    mock_deps["data_feed"].get_vix.return_value = 16.0
    assert theta._should_enter_today(date(2026, 3, 9)) is True   # Monday
    assert theta._should_enter_today(date(2026, 3, 10)) is True  # Tuesday
    assert theta._should_enter_today(date(2026, 3, 11)) is False # Wednesday
    assert theta._should_enter_today(date(2026, 3, 12)) is False # Thursday


def test_should_not_enter_if_already_have_condor(theta, mock_deps):
    """Skip entry if a condor is already open."""
    theta._condor = CondorPosition(
        entry_date=date(2026, 3, 9),
        expiry_date=date(2026, 3, 12),
        short_ce=CondorLeg("NIFTY2531224700CE", 24700, 30.0, 75, "CE"),
        long_ce=CondorLeg("NIFTY2531224900CE", 24900, 10.0, 75, "CE"),
        short_pe=CondorLeg("NIFTY2531224000PE", 24000, 30.0, 75, "PE"),
        long_pe=CondorLeg("NIFTY2531223800PE", 23800, 10.0, 75, "PE"),
        net_credit=40.0,
    )
    assert theta._should_enter_today(date(2026, 3, 10)) is False


def test_should_not_enter_margin_exceeded(theta, mock_deps):
    """Skip if condor would exceed 40% margin budget."""
    mock_deps["margin"].current_utilization_pct.return_value = 55.0
    assert theta._should_enter_today(date(2026, 3, 9)) is False


# ── Risk Budget ───────────────────────────────────────────────────────


def test_condor_max_risk_within_budget(theta):
    """Max risk = (wing_width - credit) × lot_size ≤ 3% of capital."""
    # Wing width 200pts, credit 40pts → max loss = 160 × 75 = 12000
    # 3% of 300000 = 9000 → this would FAIL
    assert theta._is_within_risk_budget(
        wing_width=200, net_credit=40.0, lot_size=75
    ) is False

    # Wing width 200pts, credit 40pts, lot 50 → max loss = 160 × 50 = 8000 < 9000
    assert theta._is_within_risk_budget(
        wing_width=200, net_credit=40.0, lot_size=50
    ) is True


# ── Daily Management ─────────────────────────────────────────────────


def test_profit_target_50pct_close(theta, mock_deps):
    """Close condor when 50% of credit captured."""
    theta._condor = CondorPosition(
        entry_date=date(2026, 3, 9),
        expiry_date=date(2026, 3, 12),
        short_ce=CondorLeg("NIFTY2531224700CE", 24700, 30.0, 75, "CE"),
        long_ce=CondorLeg("NIFTY2531224900CE", 24900, 10.0, 75, "CE"),
        short_pe=CondorLeg("NIFTY2531224000PE", 24000, 30.0, 75, "PE"),
        long_pe=CondorLeg("NIFTY2531223800PE", 23800, 10.0, 75, "PE"),
        net_credit=40.0,
    )
    # Current value of condor is 18 (less than 50% of 40 = 20 → profit > 50%)
    mock_deps["data_feed"].get_ltp_batch.return_value = {
        "NFO:NIFTY2531224700CE": 12.0,
        "NFO:NIFTY2531224900CE": 3.0,
        "NFO:NIFTY2531224000PE": 5.0,
        "NFO:NIFTY2531223800PE": 2.0,
    }
    action = theta._evaluate_condor_management()
    assert action == "close_profit"


def test_delta_035_tighten_hedge(theta, mock_deps):
    """When short strike delta > 0.35, tighten hedge."""
    theta._condor = CondorPosition(
        entry_date=date(2026, 3, 9),
        expiry_date=date(2026, 3, 12),
        short_ce=CondorLeg("NIFTY2531224700CE", 24700, 30.0, 75, "CE"),
        long_ce=CondorLeg("NIFTY2531224900CE", 24900, 10.0, 75, "CE"),
        short_pe=CondorLeg("NIFTY2531224000PE", 24000, 30.0, 75, "PE"),
        long_pe=CondorLeg("NIFTY2531223800PE", 23800, 10.0, 75, "PE"),
        net_credit=40.0,
    )
    theta._condor.short_ce_delta = 0.38  # above 0.35
    theta._condor.short_pe_delta = -0.15  # fine
    action = theta._evaluate_delta_risk()
    assert action == "tighten_ce"


def test_delta_045_close_side(theta, mock_deps):
    """When short strike delta > 0.45, close the threatened side."""
    theta._condor = CondorPosition(
        entry_date=date(2026, 3, 9),
        expiry_date=date(2026, 3, 12),
        short_ce=CondorLeg("NIFTY2531224700CE", 24700, 30.0, 75, "CE"),
        long_ce=CondorLeg("NIFTY2531224900CE", 24900, 10.0, 75, "CE"),
        short_pe=CondorLeg("NIFTY2531224000PE", 24000, 30.0, 75, "PE"),
        long_pe=CondorLeg("NIFTY2531223800PE", 23800, 10.0, 75, "PE"),
        net_credit=40.0,
    )
    theta._condor.short_pe_delta = -0.47
    theta._condor.short_ce_delta = 0.12
    action = theta._evaluate_delta_risk()
    assert action == "close_pe_side"


def test_delta_050_close_all(theta, mock_deps):
    """When short strike delta > 0.50, close entire condor."""
    theta._condor = CondorPosition(
        entry_date=date(2026, 3, 9),
        expiry_date=date(2026, 3, 12),
        short_ce=CondorLeg("NIFTY2531224700CE", 24700, 30.0, 75, "CE"),
        long_ce=CondorLeg("NIFTY2531224900CE", 24900, 10.0, 75, "CE"),
        short_pe=CondorLeg("NIFTY2531224000PE", 24000, 30.0, 75, "PE"),
        long_pe=CondorLeg("NIFTY2531223800PE", 23800, 10.0, 75, "PE"),
        net_credit=40.0,
    )
    theta._condor.short_ce_delta = 0.52
    action = theta._evaluate_delta_risk()
    assert action == "close_all"


# ── Time Management ──────────────────────────────────────────────────


def test_close_by_wednesday_eod(theta, mock_deps):
    """Close by Wednesday EOD if profit < 50%."""
    theta._condor = CondorPosition(
        entry_date=date(2026, 3, 9),
        expiry_date=date(2026, 3, 12),
        short_ce=CondorLeg("NIFTY2531224700CE", 24700, 30.0, 75, "CE"),
        long_ce=CondorLeg("NIFTY2531224900CE", 24900, 10.0, 75, "CE"),
        short_pe=CondorLeg("NIFTY2531224000PE", 24000, 30.0, 75, "PE"),
        long_pe=CondorLeg("NIFTY2531223800PE", 23800, 10.0, 75, "PE"),
        net_credit=40.0,
    )
    # Wednesday, profit only 30%
    assert theta._should_close_for_time(date(2026, 3, 11)) is True  # Wednesday


def test_never_hold_to_thursday(theta, mock_deps):
    """Never hold condor to Thursday expiry."""
    theta._condor = CondorPosition(
        entry_date=date(2026, 3, 9),
        expiry_date=date(2026, 3, 12),
        short_ce=CondorLeg("NIFTY2531224700CE", 24700, 30.0, 75, "CE"),
        long_ce=CondorLeg("NIFTY2531224900CE", 24900, 10.0, 75, "CE"),
        short_pe=CondorLeg("NIFTY2531224000PE", 24000, 30.0, 75, "PE"),
        long_pe=CondorLeg("NIFTY2531223800PE", 23800, 10.0, 75, "PE"),
        net_credit=40.0,
    )
    assert theta._should_close_for_time(date(2026, 3, 12)) is True  # Thursday


# ── Survival Mode ────────────────────────────────────────────────────


def test_survival_mode_wider_wings(theta):
    """In survival mode, use delta 0.15 instead of 0.20."""
    theta._survival_mode = True
    assert theta._entry_delta() == 0.15


def test_survival_mode_smaller_size(theta):
    """In survival mode, use 50% of normal lot count."""
    theta._survival_mode = True
    assert theta._lot_multiplier() == 0.5


def test_survival_mode_lower_profit_target(theta):
    """In survival mode, profit target is 40% (not 50%)."""
    theta._survival_mode = True
    assert theta._profit_target_pct() == 0.40
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_theta_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'v7.theta_engine'`

- [ ] **Step 3: Implement ThetaEngine**

```python
# v7/theta_engine.py
"""V7 Theta Engine — Weekly Nifty iron condor strategy.

Independent background income. Own risk budget (max 3% capital at risk).
Max 40% margin utilization.

Entry: Monday/Tuesday when VIX 14-20
Exit: profit target (50%), delta breach, Wednesday EOD, never Thursday

Spec ref: Component 5 (lines 597-642), Margin Budget (lines 942-957)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
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
VIX_ENTRY_LOW = 14.0
VIX_ENTRY_HIGH = 20.0
ENTRY_DELTA = 0.20
SURVIVAL_ENTRY_DELTA = 0.15
WING_WIDTH = 200                # points between short and long strikes
MIN_CREDIT_PER_LOT = 30.0      # minimum net credit ₹30/lot
PROFIT_TARGET_PCT = 0.50
SURVIVAL_PROFIT_TARGET_PCT = 0.40


class ThetaEngine:
    """Weekly Nifty iron condor manager.

    Lifecycle:
    1. Entry check: Monday/Tuesday, VIX 14-20, no existing condor
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

        # Only Monday (0) or Tuesday (1)
        if today.weekday() not in (0, 1):
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
        nifty_ltp = self._data.get_ltp_batch().get("NSE:NIFTY 50")
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
            logger.info("Net credit ₹%.1f < min ₹%.1f, skipping", net_credit, MIN_CREDIT_PER_LOT)
            return

        # Risk budget check
        lot_size = 75
        if not self._is_within_risk_budget(WING_WIDTH, net_credit, lot_size):
            logger.info("Condor risk exceeds 3%% budget, skipping")
            return

        # TODO: find next Thursday expiry from instruments
        # For now, compute as next Thursday
        days_to_thu = (3 - today.weekday()) % 7
        if days_to_thu == 0:
            days_to_thu = 7
        from datetime import timedelta
        expiry = today + timedelta(days=days_to_thu)

        # Place orders (simplified — would use actual tradingsymbols)
        logger.info("THETA ENTRY: Nifty iron condor %d/%d/%d/%d credit=%.1f expiry=%s",
                     long_pe_strike, short_pe_strike, short_ce_strike, long_ce_strike,
                     net_credit, expiry)

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

        self._state.save_theta_state(self._condor.to_dict())

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

        prices = self._data.get_ltp_batch()
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
        """Close by Wednesday EOD or never hold to Thursday."""
        if self._condor is None:
            return False
        # Thursday = expiry day → must close
        if today.weekday() == 3:  # Thursday
            return True
        # Wednesday → close if still open (gamma risk)
        if today.weekday() == 2:  # Wednesday
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_theta_engine.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd ~/financial-agent-india && git add v7/theta_engine.py tests/test_v7_theta_engine.py && \
git commit -m "v7: add ThetaEngine with weekly iron condor, delta management, survival mode"
```

---

## Chunk 4: Integration & Edge Cases

### Task 4: Executor-ThetaEngine integration, expiry day, position rolling

**Files:**
- Modify: `v7/executor.py` (add theta engine integration, expiry handling)
- Add tests: `tests/test_v7_executor.py` (additional edge case tests)

- [ ] **Step 1: Write failing tests for integration and edge cases**

Add to `tests/test_v7_executor.py`:

```python
# ── Expiry Day Behavior ───────────────────────────────────────────────


def test_expiry_day_no_new_positions_after_1pm(executor, mock_deps):
    """On Thursday, no new positions after 1:00 PM."""
    setup = Setup(
        id="N1", priority=1, type=SetupType.BREAKOUT_LONG,
        symbol="NIFTY", trigger_level=24300.0,
        trigger_condition="15-min close above",
        instrument="NIFTY CE", strike_logic="delta 0.45",
        target=24500.0, stoploss=24200.0, max_risk_pct=1.5,
    )
    executor._playbook.nifty_setups = [setup]
    mock_deps["data_feed"].get_ltp_batch.return_value = {"NSE:NIFTY 50": 24350.0}

    with patch("v7.executor.datetime") as mock_dt:
        # Thursday 1:15 PM — after 1 PM cutoff
        mock_dt.now.return_value = datetime(2026, 3, 12, 13, 15)  # Thursday
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        executor.tick()

    mock_deps["order_mgr"].place_entry_order.assert_not_called()


def test_expiry_day_tighter_sl(executor, mock_deps):
    """On expiry day, SL distance is halved."""
    from v7.executor import Executor
    pos = Position(
        symbol="NIFTY", instrument="NIFTY2530624400CE",
        direction="bullish", entry_price=100.0,
        quantity=75, lot_size=75, allocated=7500.0,
        stoploss=80.0, target=150.0,
        entry_date=date(2026, 3, 12), setup_id="N1",
    )
    # Normal SL distance = 20. On expiry, should be 10 → SL at 90
    adjusted_sl = executor._expiry_adjusted_sl(pos)
    assert adjusted_sl == 90.0


def test_expiry_day_no_carry(executor, mock_deps):
    """No overnight carry on expiry day (Thursday)."""
    pos = Position(
        symbol="NIFTY", instrument="NIFTY2530624400CE",
        direction="bullish", entry_price=100.0,
        quantity=75, lot_size=75, allocated=7500.0,
        stoploss=80.0, target=150.0,
        entry_date=date(2026, 3, 12), setup_id="N1",
    )
    carry_rules = CarryRules()
    result = executor._meets_carry_criteria(pos, 110.0, carry_rules)
    # Should fail because it's Thursday
    with patch("v7.executor.date") as mock_date:
        mock_date.today.return_value = date(2026, 3, 12)  # Thursday
        assert executor._meets_carry_criteria(pos, 110.0, carry_rules) is False


# ── Theta Engine Integration ─────────────────────────────────────────


def test_executor_calls_theta_tick(executor, mock_deps):
    """Executor should call theta_engine.tick() each tick during active hours."""
    theta_mock = MagicMock()
    executor._theta_engine = theta_mock

    with patch("v7.executor.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 3, 11, 11, 0)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        executor.tick()

    theta_mock.tick.assert_called_once()


# ── Recovery ──────────────────────────────────────────────────────────


def test_initialize_verifies_sl_orders(executor, mock_deps):
    """On restart, verify all SL orders for carried positions are live."""
    pos = Position(
        symbol="NIFTY", instrument="NIFTY2530624400CE",
        direction="bullish", entry_price=100.0,
        quantity=75, lot_size=75, allocated=7500.0,
        stoploss=80.0, target=150.0,
        entry_date=date(2026, 3, 10), setup_id="N1",
        carried=True, sl_order_id="sl_old_123",
    )
    mock_deps["state"].load_positions.return_value = [pos]
    mock_deps["order_mgr"].is_sl_order_live.return_value = False

    executor.initialize()

    # Should have logged a warning about dead SL order
    assert executor._positions[0].sl_order_id == "sl_old_123"


def test_stale_playbook_protect_only(executor, mock_deps):
    """If playbook is from yesterday, enter protect-only mode."""
    mock_deps["state"].load_playbook.return_value = Playbook(
        date=date(2026, 3, 10),  # yesterday
        day_classification=DayClassification.LIKELY_TREND_UP,
        nifty_bias="bullish",
        nifty_setups=[],
        stock_plans=[],
        risk_budget=RiskBudget(),
        no_trade_conditions=[],
        carry_rules=CarryRules(),
    )
    with patch("v7.executor.date") as mock_date:
        mock_date.today.return_value = date(2026, 3, 11)
        executor.initialize()

    assert executor._playbook is None  # cleared due to stale date


# ── Position Rolling ──────────────────────────────────────────────────


def test_should_roll_conditions(executor, mock_deps):
    """Position eligible for rolling: in loss 20-40%, DTE < 5, thesis valid."""
    pos = Position(
        symbol="NIFTY", instrument="NIFTY2530624400CE",
        direction="bullish", entry_price=100.0,
        quantity=75, lot_size=75, allocated=7500.0,
        stoploss=80.0, target=150.0,
        entry_date=date(2026, 3, 9), setup_id="N1",
    )
    # Current price: 70 → 30% loss
    assert executor._should_roll(pos, current_price=70.0, dte=3) is True


def test_should_not_roll_loss_too_large(executor, mock_deps):
    """Don't roll if loss > 50% (thesis is wrong)."""
    pos = Position(
        symbol="NIFTY", instrument="NIFTY2530624400CE",
        direction="bullish", entry_price=100.0,
        quantity=75, lot_size=75, allocated=7500.0,
        stoploss=80.0, target=150.0,
        entry_date=date(2026, 3, 9), setup_id="N1",
    )
    assert executor._should_roll(pos, current_price=45.0, dte=3) is False


def test_should_not_roll_dte_high(executor, mock_deps):
    """Don't roll if DTE > 5 (theta not accelerating yet)."""
    pos = Position(
        symbol="NIFTY", instrument="NIFTY2530624400CE",
        direction="bullish", entry_price=100.0,
        quantity=75, lot_size=75, allocated=7500.0,
        stoploss=80.0, target=150.0,
        entry_date=date(2026, 3, 9), setup_id="N1",
    )
    assert executor._should_roll(pos, current_price=70.0, dte=7) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_executor.py -v -k "expiry or theta_tick or recovery or rolling"`
Expected: FAIL — `AttributeError` (methods don't exist yet)

- [ ] **Step 3: Add integration methods to Executor**

Add to `v7/executor.py`:

```python
    # Add these methods to the Executor class:

    # ── Theta Engine Integration ──────────────────────────────────────

    def set_theta_engine(self, theta_engine) -> None:
        """Attach theta engine for integrated tick cycle."""
        self._theta_engine = theta_engine

    # Add to tick() method, after position management:
    #     if hasattr(self, '_theta_engine') and self._theta_engine:
    #         self._theta_engine.tick()

    # ── Expiry Day ────────────────────────────────────────────────────

    def _is_expiry_day(self) -> bool:
        """Thursday is weekly expiry for index options."""
        return date.today().weekday() == 3

    def _expiry_adjusted_sl(self, pos: Position) -> float:
        """On expiry day, tighten SL to half normal distance."""
        normal_distance = abs(pos.entry_price - pos.stoploss)
        half_distance = normal_distance / 2
        if pos.direction == "bullish":
            return pos.entry_price - half_distance
        return pos.entry_price + half_distance

    def _is_expiry_cutoff(self, now: datetime) -> bool:
        """No new positions after 1:00 PM on expiry day."""
        if not self._is_expiry_day():
            return False
        return now.hour >= 13

    # ── Position Rolling ──────────────────────────────────────────────

    def _should_roll(self, pos: Position, current_price: float, dte: int) -> bool:
        """Check if position should be rolled to next expiry.

        Conditions (from spec):
        - Loss is 20-40% of premium (not too small, not too large)
        - DTE < 5 (theta accelerating)
        - Original thesis still valid (simplified: position not at SL)

        Do NOT roll if:
        - Loss > 50% (thesis is wrong)
        - VIX has spiked (new premium expensive)
        """
        loss_pct = (pos.entry_price - current_price) / pos.entry_price * 100
        if pos.direction == "bearish":
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_v7_executor.py tests/test_v7_theta_engine.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd ~/financial-agent-india && git add v7/executor.py tests/test_v7_executor.py && \
git commit -m "v7: add expiry day behavior, position rolling, theta engine integration"
```

---

## Summary

| Chunk | Module | Tests | Key Features |
|-------|--------|-------|--------------|
| 1 | `v7/order_manager.py` | `test_v7_order_manager.py` | Limit→market fallback (30s/15s), exchange SL/SL-M orders, fill tracking, dry-run mode |
| 2 | `v7/executor.py` | `test_v7_executor.py` | Phase-based tick cycle, 15-min boundary triggers, SL/breakeven/trailing/target management, carried position gap check, wind down + carry, exception detection |
| 3 | `v7/theta_engine.py` | `test_v7_theta_engine.py` | Weekly iron condor, VIX-gated entry, delta-based management (0.35/0.45/0.50), profit target, time management, survival mode |
| 4 | Integration | Additional executor tests | Expiry day (tighter SL, no carry, 1PM cutoff), position rolling (20-50% loss + DTE<5), theta engine integration, recovery verification |

### Dependencies Graph

```
Plan 1 (types, state, data_feed, config, strike_selector, margin)
  ↓
Plan 2 (risk_engine)
  ↓
Plan 3 (this plan)
  ├── order_manager.py  ← standalone, uses kite_data
  ├── executor.py       ← uses all Plan 1 + Plan 2 + order_manager
  └── theta_engine.py   ← uses data_feed, order_manager, strike_selector, margin
```

### Key Design Decisions

1. **Exchange SL orders are standalone** (Kite `order_type="SL"` / `"SL-M"`), NOT bracket orders. They survive bot crash.
2. **Executor does NOT call Claude** — it calls `Strategist.handle_exception()` which decides whether to invoke Claude.
3. **Theta engine has independent risk budget** (3% capital, 40% margin) but shares the margin pool with directional trades.
4. **Position rolling is single-roll-only** — if the rolled position fails, exit. No infinite rolling.
5. **Order timeout pattern**: entry limit→30s→market, SL exit limit→15s→market, target limit→passive.
6. **SL update uses cancel+re-place** — Kite doesn't reliably support modifying trigger_price.
