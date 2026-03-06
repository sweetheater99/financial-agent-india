"""Fast options monitor -- checks option positions every 5 minutes.

Only monitors positions with instrument="OPT". Handles:
- Target hit (PUT_TARGET_PCT)
- Stoploss / trailing stop (OPT_TRAILING_STOP_PCT from peak)
- Premium crash (emergency exit at -60%)
- Theta decay warning (approaching DTE threshold)
- Partial profit booking (OPT_PARTIAL_PROFIT_PCT)
- DTE-based forced exit (MIN_DTE_TO_HOLD)
- Max hold date expiry

Much lighter than full monitor_positions() -- no equity, no regime detection,
no spreads, no condors, no momentum options.
"""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config
from connect import get_session, refresh_session
from paper_trade import (
    load_portfolio,
    save_portfolio,
    close_position,
    get_ltp_nfo,
    apply_slippage,
    days_to_expiry,
    _today_ist,
    _telegram_send,
    _telegram_notify_exit,
    SessionExpiredError,
    # Option constants
    PUT_TARGET_PCT,
    PUT_STOPLOSS_PCT,
    OPT_TRAILING_STOP_PCT,
    OPT_THETA_DTE_THRESHOLD,
    OPT_THETA_MIN_GAIN_PCT,
    MIN_DTE_TO_HOLD,
)

IST = timezone(timedelta(hours=5, minutes=30))

logger = logging.getLogger("options_monitor")


def monitor_options(smart_api) -> int:
    """Monitor only OPT positions for fast exits. Returns number of exits."""
    portfolio = load_portfolio()

    option_positions = [
        p for p in portfolio["positions"]
        if p.get("status") == "open" and p.get("instrument") == "OPT"
    ]

    if not option_positions:
        logger.info("No open option positions.")
        return 0

    today = _today_ist()
    exits = 0
    session_refreshed = False

    for pos in option_positions:
        symbol = pos["symbol"]
        entry_price = pos["entry_price"]

        # Fetch current premium with session-refresh retry
        ltp = None
        for _attempt in range(2):
            try:
                ltp = get_ltp_nfo(
                    smart_api,
                    pos.get("contract_symbol", symbol),
                    pos["token"],
                )
                break
            except SessionExpiredError:
                if _attempt == 0 and not session_refreshed:
                    logger.info("Session expired, re-authenticating...")
                    try:
                        refresh_session(smart_api)
                        session_refreshed = True
                        time.sleep(config.API_DELAY)
                        continue
                    except Exception as e:
                        logger.warning("Session refresh failed: %s", e)
                        break
                else:
                    break
        time.sleep(config.API_DELAY)

        if ltp is None:
            logger.warning("%s: could not fetch LTP, skipping", symbol)
            continue

        # P&L (options are long positions -- bought put/call)
        pnl_pct = (ltp - entry_price) / entry_price * 100

        # Update peak premium
        peak = pos.get("peak_premium", entry_price)
        if ltp > peak:
            pos["peak_premium"] = ltp
            peak = ltp

        # DTE check
        dte = days_to_expiry(pos.get("expiry", "01JAN2099"))
        dte_expired = dte < MIN_DTE_TO_HOLD

        # Max hold date check
        max_hold_expired = today > pos.get("max_hold_date", "2099-12-31")

        # --- Determine exit reason ---
        reason = None

        # 1. Theta decay: low DTE + low gain -> early exit
        if dte < OPT_THETA_DTE_THRESHOLD and pnl_pct < OPT_THETA_MIN_GAIN_PCT:
            if not dte_expired:
                reason = "theta_decay"

        # 2. Target hit
        if reason is None and pnl_pct >= PUT_TARGET_PCT:
            reason = "target"

        # 3. Trailing stop / stoploss
        if reason is None:
            trailing_sl = peak * (1 - OPT_TRAILING_STOP_PCT / 100)
            fixed_sl = entry_price * (1 + PUT_STOPLOSS_PCT / 100)  # PUT_STOPLOSS_PCT is negative
            effective_sl = max(trailing_sl, fixed_sl)

            if ltp <= effective_sl:
                reason = "trailing_stop" if trailing_sl > fixed_sl else "stoploss"

        # 4. Premium crash (emergency, >60% loss)
        if reason is None and pnl_pct <= -60:
            reason = "premium_crash"

        # 5. DTE forced exit
        if reason is None and dte_expired:
            reason = "expiry"

        # 6. Max hold date
        if reason is None and max_hold_expired:
            reason = "expiry"

        if reason:
            # --- Partial profit booking (V4) ---
            if (reason == "target"
                    and not pos.get("opt_partial_done")
                    and pnl_pct >= config.OPT_PARTIAL_PROFIT_PCT):
                total_lots = pos.get("num_lots", 1)
                partial_lots = max(1, int(total_lots * config.OPT_PARTIAL_RATIO))
                if partial_lots < total_lots:
                    partial_qty = partial_lots * pos.get("lot_size", abs(pos["quantity"]))
                    if partial_qty < abs(pos["quantity"]):
                        exit_price = apply_slippage(ltp, "OPT", "sell")
                        close_position(portfolio, pos, exit_price, "opt_partial_profit",
                                       close_qty=partial_qty)
                        pos["opt_partial_done"] = True
                        pos["num_lots"] = total_lots - partial_lots
                        logger.info(
                            "%s PE%d: OPT PARTIAL PROFIT (%d/%d lots) @ %.2f (+%.1f%%)",
                            symbol, int(pos.get("strike", 0)),
                            partial_lots, total_lots, exit_price, pnl_pct,
                        )
                        exits += 1
                        continue  # don't full-close yet

            # --- Full exit ---
            exit_price = apply_slippage(ltp, "OPT", "sell")
            closed = close_position(portfolio, pos, exit_price, reason)
            exits += 1

            pnl = closed["pnl"]
            closed_pnl_pct = closed["pnl_pct"]
            tag = reason.upper().replace("_", " ")
            logger.info(
                "%s PE%d: EXIT (%s) prem %.2f -> %.2f  P&L: %+.1f%% (%.0f)",
                symbol, int(pos.get("strike", 0)), tag,
                entry_price, exit_price, closed_pnl_pct, pnl,
            )
            _telegram_notify_exit(pos, reason, pnl, closed_pnl_pct, exit_price, portfolio)
        else:
            # HOLD -- log status
            qty = abs(pos["quantity"])
            pos_pnl = (ltp - entry_price) * qty
            trail_display = peak * (1 - OPT_TRAILING_STOP_PCT / 100)
            logger.info(
                "%s PE%d: Prem %.2f  P&L: %+.1f%% (%+,.0f)  DTE:%d  Trail:%.2f  [HOLD]",
                symbol, int(pos.get("strike", 0)),
                ltp, pnl_pct, pos_pnl, dte, trail_display,
            )

    if exits:
        # Clean up closed positions
        portfolio["positions"] = [
            p for p in portfolio["positions"] if p.get("status") == "open"
        ]
        save_portfolio(portfolio)
    else:
        # Still save peak_premium updates
        save_portfolio(portfolio)

    return exits


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("Fast options monitor starting...")
    smart_api = get_session()
    exits = monitor_options(smart_api)
    if exits:
        logger.info("Closed %d option position(s).", exits)
    else:
        logger.info("No exits triggered.")


if __name__ == "__main__":
    main()
