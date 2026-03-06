"""
Risk guardrails module for the paper trading engine.

Provides additional safety checks beyond the base RiskManager:
- Drawdown circuit breaker (monthly/weekly/consecutive losses)
- Daily loss limit
- Position correlation guard (sector + direction concentration)
- Portfolio heat monitor (total capital at risk)
- Event calendar filter (budget, RBI, F&O expiry, earnings)

All functions return (blocked: bool, reason: str).  When blocked is True the
caller should skip the trade.  Some functions also return an allocation
multiplier via a third element: (blocked, reason, alloc_mult).
"""

from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta, timezone

import config as cfg

IST = timezone(timedelta(hours=5, minutes=30))


def _today_ist() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d")


def _date(s: str) -> date:
    """Parse YYYY-MM-DD string to date."""
    return datetime.strptime(s, "%Y-%m-%d").date()


# ---------------------------------------------------------------------------
# A. Drawdown Circuit Breaker
# ---------------------------------------------------------------------------

def get_consecutive_losses(closed_trades: list[dict]) -> int:
    """Count consecutive losses from the most recent trade backwards."""
    count = 0
    for t in reversed(closed_trades):
        if t.get("pnl", 0) < 0:
            count += 1
        else:
            break
    return count


def _get_month_start_capital(portfolio: dict) -> float:
    """Estimate capital at start of current month.

    Uses total capital minus P&L earned this month from closed trades.
    """
    total_capital = portfolio.get("capital", 100_000)
    total_pnl = portfolio.get("stats", {}).get("total_pnl", 0)
    current_capital = total_capital + total_pnl

    today = _today_ist()
    month_start = today[:8] + "01"  # YYYY-MM-01

    month_pnl = sum(
        t.get("pnl", 0) for t in portfolio.get("closed_trades", [])
        if t.get("exit_date", "") >= month_start
    )
    return current_capital - month_pnl


def _get_week_start_capital(portfolio: dict) -> float:
    """Estimate capital at start of current week (Monday)."""
    total_capital = portfolio.get("capital", 100_000)
    total_pnl = portfolio.get("stats", {}).get("total_pnl", 0)
    current_capital = total_capital + total_pnl

    today_dt = _date(_today_ist())
    monday = today_dt - timedelta(days=today_dt.weekday())
    monday_str = monday.strftime("%Y-%m-%d")

    week_pnl = sum(
        t.get("pnl", 0) for t in portfolio.get("closed_trades", [])
        if t.get("exit_date", "") >= monday_str
    )
    return current_capital - week_pnl


def check_drawdown_circuit_breaker(
    portfolio: dict, _today: str | None = None
) -> tuple[bool, str, float]:
    """Check drawdown circuit breakers.

    Returns (blocked, reason, allocation_multiplier).
    - Monthly DD > 8% → blocked=True
    - Weekly DD > 4% → allocation_multiplier=0.5
    - 5 consecutive losses → blocked=True (2-day pause)
    - 3 consecutive losses → blocked=True (1-day pause)
    """
    today = _today or _today_ist()
    today_dt = _date(today)
    starting_capital = portfolio.get("capital", 100_000)
    closed = portfolio.get("closed_trades", [])
    alloc_mult = 1.0

    # --- Monthly drawdown ---
    month_start = today[:8] + "01"
    month_pnl = sum(
        t.get("pnl", 0) for t in closed
        if t.get("exit_date", "") >= month_start
    )
    if month_pnl < 0 and abs(month_pnl) > starting_capital * cfg.MONTHLY_DRAWDOWN_BLOCK_PCT:
        return (
            True,
            f"Monthly drawdown {abs(month_pnl):.0f} exceeds "
            f"{cfg.MONTHLY_DRAWDOWN_BLOCK_PCT*100:.0f}% of capital "
            f"({starting_capital * cfg.MONTHLY_DRAWDOWN_BLOCK_PCT:.0f}). "
            f"Blocked for rest of month.",
            0.0,
        )

    # --- Weekly drawdown ---
    monday = today_dt - timedelta(days=today_dt.weekday())
    monday_str = monday.strftime("%Y-%m-%d")
    week_pnl = sum(
        t.get("pnl", 0) for t in closed
        if t.get("exit_date", "") >= monday_str
    )
    if week_pnl < 0 and abs(week_pnl) > starting_capital * cfg.WEEKLY_DRAWDOWN_REDUCE_PCT:
        alloc_mult = 0.5

    # --- Consecutive losses cooling off ---
    consec = get_consecutive_losses(closed)

    if consec >= cfg.MAX_CONSECUTIVE_LOSSES_EXTENDED:
        # Find exit_date of the most recent trade
        last_exit = closed[-1].get("exit_date", "") if closed else ""
        if last_exit:
            last_exit_dt = _date(last_exit)
            resume_dt = last_exit_dt + timedelta(days=2)
            # Skip weekends for resume
            while resume_dt.weekday() >= 5:
                resume_dt += timedelta(days=1)
            if today_dt < resume_dt:
                return (
                    True,
                    f"{consec} consecutive losses. Paused until {resume_dt.strftime('%Y-%m-%d')} (2-day cooling off).",
                    0.0,
                )

    elif consec >= cfg.MAX_CONSECUTIVE_LOSSES_PAUSE:
        last_exit = closed[-1].get("exit_date", "") if closed else ""
        if last_exit:
            last_exit_dt = _date(last_exit)
            resume_dt = last_exit_dt + timedelta(days=1)
            while resume_dt.weekday() >= 5:
                resume_dt += timedelta(days=1)
            if today_dt < resume_dt:
                return (
                    True,
                    f"{consec} consecutive losses. Paused until {resume_dt.strftime('%Y-%m-%d')} (1-day cooling off).",
                    0.0,
                )

    reason = ""
    if alloc_mult < 1.0:
        reason = (
            f"Weekly drawdown {abs(week_pnl):.0f} exceeds "
            f"{cfg.WEEKLY_DRAWDOWN_REDUCE_PCT*100:.0f}% of capital. "
            f"Reducing allocation to {alloc_mult*100:.0f}%."
        )
    return (False, reason, alloc_mult)


# ---------------------------------------------------------------------------
# B. Daily Loss Limit
# ---------------------------------------------------------------------------

def check_daily_loss_limit(
    portfolio: dict, _today: str | None = None
) -> tuple[bool, str]:
    """Block new trades if daily realized losses exceed 2% of capital."""
    today = _today or _today_ist()
    starting_capital = portfolio.get("capital", 100_000)
    closed = portfolio.get("closed_trades", [])

    day_pnl = sum(
        t.get("pnl", 0) for t in closed
        if t.get("exit_date", "") == today
    )

    limit = starting_capital * cfg.DAILY_LOSS_LIMIT_PCT
    if day_pnl < 0 and abs(day_pnl) > limit:
        return (
            True,
            f"Daily loss {abs(day_pnl):.0f} exceeds "
            f"{cfg.DAILY_LOSS_LIMIT_PCT*100:.0f}% of capital ({limit:.0f}). "
            f"Blocked for today.",
        )
    return (False, "")


# ---------------------------------------------------------------------------
# C. Position Correlation Guard
# ---------------------------------------------------------------------------

def check_correlation_guard(
    portfolio: dict, new_symbol: str, new_direction: str, symbol_sector: dict,
    _config: object | None = None,
) -> tuple[bool, str, float]:
    """Check sector and direction concentration.

    Returns (blocked, reason, allocation_multiplier).
    - Max 2 positions in same sector (configurable)
    - Max 3 positions in same direction
    - Total open positions cap: 8
    - 2nd position in same sector → reduce allocation by 25%
    """
    c = _config or cfg
    open_pos = [p for p in portfolio.get("positions", []) if p.get("status") == "open"]
    alloc_mult = 1.0

    # Total position cap
    max_positions = getattr(c, "MAX_CONCURRENT_POSITIONS", 8)
    if len(open_pos) >= max_positions:
        return (True, f"Max open positions ({max_positions}) reached.", 0.0)

    # Sector concentration
    new_sector = symbol_sector.get(new_symbol, "Other")
    max_sector = getattr(c, "MAX_SECTOR_POSITIONS", 2)
    sector_count = sum(
        1 for p in open_pos
        if symbol_sector.get(p["symbol"], "Other") == new_sector
    )
    if sector_count >= max_sector:
        return (
            True,
            f"Max sector positions ({max_sector}) reached for {new_sector}.",
            0.0,
        )
    if sector_count == max_sector - 1 and max_sector > 1:
        # Adding 2nd position in same sector → reduce 25%
        alloc_mult *= 0.75

    # Direction concentration
    max_dir = getattr(c, "MAX_DIRECTION_POSITIONS", 3)
    dir_count = sum(1 for p in open_pos if p.get("direction") == new_direction)
    if dir_count >= max_dir:
        return (
            True,
            f"Max same-direction positions ({max_dir}) reached for {new_direction}.",
            0.0,
        )

    reason = ""
    if alloc_mult < 1.0:
        reason = f"2nd position in sector {new_sector} — allocation reduced 25%."
    return (False, reason, alloc_mult)


# ---------------------------------------------------------------------------
# D. Portfolio Heat Monitor
# ---------------------------------------------------------------------------

def check_portfolio_heat(
    portfolio: dict, new_allocation: float, new_risk_pct: float | None = None,
) -> tuple[bool, str]:
    """Check portfolio heat (total capital at risk).

    Portfolio heat = sum of (allocated * risk_per_trade) for all open positions.
    risk_per_trade = |entry - stoploss| / entry  (or a default 3% if unavailable).

    Returns (blocked, reason).
    """
    starting_capital = portfolio.get("capital", 100_000)
    max_heat = starting_capital * cfg.MAX_PORTFOLIO_HEAT_PCT
    open_pos = [p for p in portfolio.get("positions", []) if p.get("status") == "open"]

    current_heat = 0.0
    for p in open_pos:
        entry = p.get("entry_price", 0)
        sl = p.get("stoploss_price", 0)
        alloc = p.get("allocated", 0)
        if entry > 0 and sl > 0:
            risk_pct = abs(entry - sl) / entry
        else:
            risk_pct = 0.03  # default 3%
        current_heat += alloc * risk_pct

    # New position risk
    if new_risk_pct is not None:
        new_heat = new_allocation * new_risk_pct
    else:
        new_heat = new_allocation * 0.03  # default

    total_heat = current_heat + new_heat

    if total_heat > max_heat:
        return (
            True,
            f"Portfolio heat {total_heat:.0f} would exceed "
            f"{cfg.MAX_PORTFOLIO_HEAT_PCT*100:.0f}% limit ({max_heat:.0f}). "
            f"Current heat: {current_heat:.0f}, new: {new_heat:.0f}.",
        )
    return (False, "")


# ---------------------------------------------------------------------------
# E. Event Calendar Filter
# ---------------------------------------------------------------------------

def _last_thursday_of_month(year: int, month: int) -> date:
    """Return the last Thursday of the given month."""
    # Find the last day of the month
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day)
    # Walk backwards to Thursday (weekday 3)
    while d.weekday() != 3:
        d -= timedelta(days=1)
    return d


def _is_last_thursday(d: date) -> bool:
    """Check if date is the last Thursday of its month."""
    return d == _last_thursday_of_month(d.year, d.month)


def _is_thursday(d: date) -> bool:
    return d.weekday() == 3


def check_event_calendar(
    symbol: str, _today: str | None = None
) -> tuple[bool, str, float]:
    """Check event calendar for trade blocking/reduction.

    Returns (blocked, reason, allocation_multiplier).
    - Budget day / RBI policy day → block all new trades
    - Monthly F&O expiry (last Thursday) → reduce allocation 50%
    - Weekly expiry (Thursday) → reduce allocation for option positions
    - Earnings blackout is handled separately by is_near_earnings()
    """
    today = _today or _today_ist()
    today_dt = _date(today)
    alloc_mult = 1.0

    # Budget day block
    budget_dates = getattr(cfg, "BUDGET_BLOCK_DATES", [])
    if today in budget_dates:
        return (True, f"Budget day ({today}). All new trades blocked.", 0.0)

    # RBI policy day block
    rbi_dates = getattr(cfg, "RBI_POLICY_DATES", [])
    if today in rbi_dates:
        return (True, f"RBI policy day ({today}). All new trades blocked.", 0.0)

    # Monthly F&O expiry — reduce allocation
    if _is_last_thursday(today_dt):
        alloc_mult *= cfg.EXPIRY_DAY_ALLOCATION_MULT
        reason = f"Monthly F&O expiry day. Allocation reduced to {alloc_mult*100:.0f}%."
        return (False, reason, alloc_mult)

    # Weekly expiry day (every Thursday) — minor reduction for options
    if _is_thursday(today_dt):
        alloc_mult *= 0.75  # 25% reduction on weekly expiry days
        reason = "Weekly expiry day (Thursday). Allocation reduced 25%."
        return (False, reason, alloc_mult)

    return (False, "", alloc_mult)


# ---------------------------------------------------------------------------
# Master guard-check: run all guards in sequence
# ---------------------------------------------------------------------------

def run_all_guards(
    portfolio: dict,
    new_symbol: str,
    new_direction: str,
    new_allocation: float,
    symbol_sector: dict,
    new_risk_pct: float | None = None,
    _today: str | None = None,
) -> tuple[bool, str, float]:
    """Run all guardrail checks for a candidate trade.

    Returns (blocked, reason, final_allocation_multiplier).
    The caller should multiply its allocation by the returned multiplier.
    """
    alloc_mult = 1.0
    reasons: list[str] = []

    # 1. Drawdown circuit breaker
    blocked, reason, dd_mult = check_drawdown_circuit_breaker(portfolio, _today=_today)
    if blocked:
        return (True, f"[Drawdown] {reason}", 0.0)
    if dd_mult < 1.0:
        alloc_mult *= dd_mult
        reasons.append(reason)

    # 2. Daily loss limit
    blocked, reason = check_daily_loss_limit(portfolio, _today=_today)
    if blocked:
        return (True, f"[DailyLoss] {reason}", 0.0)

    # 3. Correlation guard
    blocked, reason, corr_mult = check_correlation_guard(
        portfolio, new_symbol, new_direction, symbol_sector,
    )
    if blocked:
        return (True, f"[Correlation] {reason}", 0.0)
    if corr_mult < 1.0:
        alloc_mult *= corr_mult
        reasons.append(reason)

    # 4. Portfolio heat
    effective_alloc = new_allocation * alloc_mult
    blocked, reason = check_portfolio_heat(portfolio, effective_alloc, new_risk_pct)
    if blocked:
        return (True, f"[Heat] {reason}", 0.0)

    # 5. Event calendar
    blocked, reason, evt_mult = check_event_calendar(new_symbol, _today=_today)
    if blocked:
        return (True, f"[Event] {reason}", 0.0)
    if evt_mult < 1.0:
        alloc_mult *= evt_mult
        reasons.append(reason)

    combined_reason = " | ".join(reasons) if reasons else ""
    return (False, combined_reason, alloc_mult)
