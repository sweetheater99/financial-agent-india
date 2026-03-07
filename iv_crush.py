"""IV Crush Event Strategy for Indian Stock Market.

Sells Nifty iron condors 1-2 days before known high-IV events
(RBI policy, Budget, major expiry) and closes next morning to capture
IV crush profit. IV inflates before events, then collapses after
regardless of direction.

Usage:
    Called from paper_trade.py during open_positions() and monitor_positions().
"""

import calendar
import logging
import time
from datetime import date, datetime, timedelta, timezone

import config
from config import (
    BUDGET_BLOCK_DATES,
    RBI_POLICY_DATES,
    NSE_HOLIDAYS_2026,
    is_trading_day,
    trading_days_between,
    next_trading_day,
)

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# IV Crush strategy constants
IV_CRUSH_MIN_VIX = 15.0          # IV must be elevated enough to crush
IV_CRUSH_MAX_RISK_PCT = 0.01     # 1% of capital (half of normal condor)
IV_CRUSH_PROFIT_TARGET_PCT = 0.40  # close at 40% profit (before event)
IV_CRUSH_LOSS_EXIT_PCT = 0.50     # emergency exit at 50% loss
IV_CRUSH_MIN_CREDIT_PER_LOT = 40  # lower bar than normal condors
IV_CRUSH_OTM_RANGE_MIN = 300      # wider strikes: 300-500 pts OTM (vs 200-400 normal)
IV_CRUSH_OTM_RANGE_MAX = 500
IV_CRUSH_SIZE_MULT = 0.5          # half of normal condor sizing

# Blocked regimes -- too risky for event strategies
IV_CRUSH_BLOCKED_REGIMES = {"VOLATILE", "CASH"}


def _last_thursday_of_month(year: int, month: int) -> date:
    """Return the last Thursday of the given month."""
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day)
    while d.weekday() != 3:  # 3 = Thursday
        d -= timedelta(days=1)
    return d


def _get_monthly_expiry_dates(year: int) -> list[date]:
    """Return last Thursday of each month for a given year (Nifty monthly expiry)."""
    return [_last_thursday_of_month(year, m) for m in range(1, 13)]


def get_upcoming_events(days_ahead: int = 3, _today: date | None = None) -> list[dict]:
    """Check config for RBI policy dates, budget dates, monthly expiry dates.

    Returns events within next `days_ahead` trading days.
    Each event: {"date": "2026-04-09", "type": "RBI_POLICY", "days_until": 2}
    """
    today = _today or datetime.now(IST).date()
    events = []

    # Collect all event dates with types
    event_sources = []

    for d_str in RBI_POLICY_DATES:
        event_sources.append((d_str, "RBI_POLICY"))

    for d_str in BUDGET_BLOCK_DATES:
        event_sources.append((d_str, "BUDGET"))

    # Monthly expiry dates (last Thursday of each month)
    for year in (today.year, today.year + 1):
        for exp_date in _get_monthly_expiry_dates(year):
            event_sources.append((exp_date.strftime("%Y-%m-%d"), "MONTHLY_EXPIRY"))

    for d_str, event_type in event_sources:
        try:
            event_date = datetime.strptime(d_str, "%Y-%m-%d").date()
        except ValueError:
            continue

        # Count trading days from today to event_date
        td = trading_days_between(today, event_date)

        if 0 < td <= days_ahead:
            events.append({
                "date": d_str,
                "type": event_type,
                "days_until": td,
            })

    # Sort by days_until (nearest first)
    events.sort(key=lambda e: e["days_until"])
    return events


def should_enter_iv_crush(smart_api, _today: date | None = None,
                          _vix: float | None = None,
                          _regime: str | None = None,
                          _portfolio: dict | None = None) -> dict | None:
    """Check if IV crush entry conditions are met.

    Called during open_positions() flow.
    Returns a trade setup dict if conditions are met, None otherwise.

    Conditions:
    - An event is 1-2 trading days away
    - Current VIX > 15 (IV elevated enough to crush)
    - No existing IV crush position open
    - Market is not in VOLATILE/CASH regime
    """
    today = _today or datetime.now(IST).date()

    # 1. Check for upcoming events (1-2 trading days ahead)
    events = get_upcoming_events(days_ahead=2, _today=today)
    if not events:
        logger.debug("IV crush: no events within 2 trading days")
        return None

    # Pick the nearest event
    event = events[0]
    event_date_str = event["date"]
    event_type = event["type"]

    # 2. Check VIX
    if _vix is not None:
        vix = _vix
    else:
        from paper_trade import get_ltp, VIX_TOKEN
        vix = get_ltp(smart_api, "India VIX", VIX_TOKEN)
        time.sleep(config.API_DELAY)

    if vix is None or vix < IV_CRUSH_MIN_VIX:
        logger.debug("IV crush: VIX %.1f < %.1f minimum", vix or 0, IV_CRUSH_MIN_VIX)
        return None

    # 3. Check regime
    if _regime is not None:
        regime = _regime
    else:
        # Fetch regime from paper_trade context -- caller should pass this
        regime = "UNKNOWN"

    if regime in IV_CRUSH_BLOCKED_REGIMES:
        logger.debug("IV crush: blocked in %s regime", regime)
        return None

    # 4. Check no existing IV crush position
    if _portfolio is not None:
        has_iv_crush = any(
            p.get("instrument") == "IV_CRUSH" and p.get("status") == "open"
            for p in _portfolio.get("positions", [])
        )
        if has_iv_crush:
            logger.debug("IV crush: already have an open IV_CRUSH position")
            return None

    # 5. Determine expiry -- if event is on Thursday (expiry day), use next week's expiry
    event_date = datetime.strptime(event_date_str, "%Y-%m-%d").date()
    if event_date.weekday() == 3:  # Thursday = expiry day
        # Use next Thursday as expiry
        target_expiry = event_date + timedelta(days=7)
        # Adjust if that's a holiday
        while not is_trading_day(target_expiry):
            target_expiry -= timedelta(days=1)
    else:
        # Use nearest Thursday after event
        target_expiry = event_date
        while target_expiry.weekday() != 3:
            target_expiry += timedelta(days=1)
        # If that Thursday is before event, go to next week
        if target_expiry <= event_date:
            target_expiry += timedelta(days=7)
        # Adjust for holidays
        while not is_trading_day(target_expiry):
            target_expiry -= timedelta(days=1)

    # Max hold: event_date + 1 trading day
    max_hold_date = next_trading_day(event_date)

    return {
        "event": event_type,
        "event_date": event_date_str,
        "strategy": "IV_CRUSH_CONDOR",
        "days_until_event": event["days_until"],
        "vix_at_signal": vix,
        "target_expiry": target_expiry.strftime("%Y-%m-%d"),
        "max_hold_date": max_hold_date.strftime("%Y-%m-%d"),
        "otm_range_min": IV_CRUSH_OTM_RANGE_MIN,
        "otm_range_max": IV_CRUSH_OTM_RANGE_MAX,
        "size_multiplier": IV_CRUSH_SIZE_MULT,
    }


def check_iv_crush_exit(position: dict, smart_api=None,
                        _today: str | None = None,
                        _underlying_ltp: float | None = None) -> tuple[bool, str]:
    """Check exit conditions for an IV crush position.

    Called during monitor_positions() for IV_CRUSH positions.

    Exit conditions:
    1. Event has passed (next morning after event) -> close for IV crush profit
    2. If already at 40%+ profit before event -> close early
    3. If at 50%+ loss -> emergency exit
    4. Max hold: event_date + 1 trading day

    Returns: (should_exit: bool, reason: str)
    """
    today = _today or datetime.now(IST).strftime("%Y-%m-%d")
    today_date = datetime.strptime(today, "%Y-%m-%d").date()

    event_date_str = position.get("event_date", "")
    max_hold_str = position.get("max_hold_date", "")

    try:
        event_date = datetime.strptime(event_date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return (True, "iv_crush_invalid_event_date")

    # Calculate current P&L
    net_credit = position.get("net_credit", 0)
    if net_credit <= 0:
        return (True, "iv_crush_no_credit")

    # Estimate current cost to close from premium data
    # Use the same premium estimation as condor monitoring
    current_pnl_pct = _estimate_pnl_pct(position, smart_api, today,
                                         _underlying_ltp=_underlying_ltp)

    # 1. Emergency exit: 50%+ loss
    if current_pnl_pct is not None and current_pnl_pct <= -IV_CRUSH_LOSS_EXIT_PCT * 100:
        return (True, "iv_crush_emergency_loss")

    # 2. Early profit: 40%+ profit before event
    if current_pnl_pct is not None and current_pnl_pct >= IV_CRUSH_PROFIT_TARGET_PCT * 100:
        return (True, "iv_crush_early_profit")

    # 3. Event has passed -> close for IV crush
    if today_date > event_date:
        return (True, "iv_crush_event_passed")

    # 4. Max hold date exceeded
    if max_hold_str and today >= max_hold_str:
        return (True, "iv_crush_max_hold")

    return (False, "hold")


def _estimate_pnl_pct(position: dict, smart_api=None, today: str = None,
                      _underlying_ltp: float | None = None) -> float | None:
    """Estimate current P&L percentage for an IV crush condor.

    Returns P&L as percentage of net credit (e.g., 40.0 for 40% profit).
    Positive = profit, negative = loss.
    """
    net_credit = position.get("net_credit", 0)
    if net_credit <= 0:
        return None

    # If we have premium estimates passed in (from monitor), use those
    current_value = position.get("_current_value")
    if current_value is not None:
        pnl = net_credit - current_value
        return (pnl / net_credit) * 100

    # Estimate from underlying move if LTP available
    if _underlying_ltp is not None:
        underlying_at_entry = position.get("underlying_at_entry", 0)
        if underlying_at_entry <= 0:
            return None

        # Simple estimation: if underlying hasn't moved much, IV crush = profit
        # More sophisticated: use the premium estimation from paper_trade
        try:
            from paper_trade import estimate_premium_from_underlying

            cs_prem = estimate_premium_from_underlying(
                {"long_leg": position["call_short"],
                 "underlying_at_entry": underlying_at_entry,
                 "expiry": position["expiry"]},
                "long", _underlying_ltp, today)
            cl_prem = estimate_premium_from_underlying(
                {"long_leg": position["call_long"],
                 "underlying_at_entry": underlying_at_entry,
                 "expiry": position["expiry"]},
                "long", _underlying_ltp, today)
            ps_prem = estimate_premium_from_underlying(
                {"long_leg": position["put_short"],
                 "underlying_at_entry": underlying_at_entry,
                 "expiry": position["expiry"]},
                "long", _underlying_ltp, today)
            pl_prem = estimate_premium_from_underlying(
                {"long_leg": position["put_long"],
                 "underlying_at_entry": underlying_at_entry,
                 "expiry": position["expiry"]},
                "long", _underlying_ltp, today)

            quantity = position.get("quantity", 75)
            cost_to_close = (cs_prem - cl_prem + ps_prem - pl_prem) * quantity
            pnl = net_credit - cost_to_close
            return (pnl / net_credit) * 100
        except Exception:
            return None

    return None


def open_iv_crush_condor(portfolio: dict, crush_setup: dict,
                         condor_data: dict, smart_api=None) -> dict | None:
    """Open an IV crush iron condor position.

    Uses the existing condor structure but with IV_CRUSH instrument tag
    and event metadata.

    Args:
        portfolio: portfolio dict
        crush_setup: dict from should_enter_iv_crush()
        condor_data: dict from select_iron_condor_strikes()
        smart_api: SmartAPI session (for notifications)

    Returns:
        position dict if opened, None otherwise
    """
    today = datetime.now(IST).strftime("%Y-%m-%d")
    slippage = config.PAPER_TRADE_SLIPPAGE_PCT
    lot_size = condor_data.get("lot_size", 75)

    # Apply slippage: sell legs get worse (lower), buy legs get worse (higher)
    cs_prem = round(condor_data["call_short_premium"] * (1 - slippage), 2)
    cl_prem = round(condor_data["call_long_premium"] * (1 + slippage), 2)
    ps_prem = round(condor_data["put_short_premium"] * (1 - slippage), 2)
    pl_prem = round(condor_data["put_long_premium"] * (1 + slippage), 2)

    net_credit = round((cs_prem + ps_prem - cl_prem - pl_prem) * lot_size, 2)
    width = condor_data["width"]
    max_loss = round(width * lot_size - net_credit, 2)

    if net_credit <= 0:
        logger.debug("IV crush: negative net credit, skipping")
        return None

    # Allocation: half of normal condor sizing
    allocation = round(max_loss * IV_CRUSH_SIZE_MULT, 2)

    if portfolio["available_capital"] < allocation:
        logger.debug("IV crush: insufficient capital (need ₹%.0f, have ₹%.0f)",
                     allocation, portfolio["available_capital"])
        return None

    position = {
        "id": f"iv_crush_{today}_{crush_setup['event']}",
        "strategy": "iv_crush",
        "symbol": "NIFTY",
        "instrument": "IV_CRUSH",
        "entry_date": today,
        "expiry": condor_data.get("expiry", crush_setup.get("target_expiry", "")),
        "quantity": lot_size,
        "direction": "neutral",
        "entry_price": 0,  # condor uses net_credit
        "call_short": {
            "strike": condor_data["call_short_strike"],
            "entry_premium": cs_prem,
            "option_type": "CE",
        },
        "call_long": {
            "strike": condor_data["call_long_strike"],
            "entry_premium": cl_prem,
            "option_type": "CE",
        },
        "put_short": {
            "strike": condor_data["put_short_strike"],
            "entry_premium": ps_prem,
            "option_type": "PE",
        },
        "put_long": {
            "strike": condor_data["put_long_strike"],
            "entry_premium": pl_prem,
            "option_type": "PE",
        },
        "net_credit": net_credit,
        "max_profit": net_credit,
        "max_loss": max_loss,
        "allocated": allocation,
        "max_hold_date": crush_setup["max_hold_date"],
        "underlying_at_entry": condor_data.get("spot", 0),
        "vix_at_entry": crush_setup.get("vix_at_signal", 0),
        "status": "open",
        "regime_at_entry": "EVENT_PLAY",
        "iv_percentile_at_entry": 0,
        "event_type": crush_setup["event"],
        "event_date": crush_setup["event_date"],
        "strategy_tag": "iv_crush",
        "index_name": "NIFTY",
    }

    portfolio["positions"].append(position)
    portfolio["available_capital"] = round(
        portfolio["available_capital"] - allocation, 2
    )

    logger.info(
        "OPENED IV crush condor for %s (%s): credit=₹%.0f, max_loss=₹%.0f, "
        "event in %d days",
        crush_setup["event"], crush_setup["event_date"],
        net_credit, max_loss, crush_setup["days_until_event"],
    )

    return position
