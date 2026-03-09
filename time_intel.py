"""Time & event intelligence for Indian markets.

Expiry detection, no-trade zones, gap analysis, and IV crush awareness.
Self-contained — no imports from other project files.
"""

from datetime import date, time, timedelta
import calendar


def is_expiry_day(d: date, holidays: list[date] | None = None) -> str | None:
    """Check if a date is an expiry day (weekly or monthly).

    Returns:
        "monthly" if last Thursday of month (or shifted Wednesday if holiday),
        "weekly" if any other Thursday (or shifted Wednesday if holiday),
        None otherwise.
    """
    holidays = holidays or []

    # Find the last Thursday of the month
    cal = calendar.monthcalendar(d.year, d.month)
    # monthcalendar weeks: Mon=0 .. Sun=6; Thursday=3
    last_thursday = None
    for week in reversed(cal):
        if week[3] != 0:
            last_thursday = date(d.year, d.month, week[3])
            break

    # Determine if last Thursday shifts to Wednesday due to holiday
    last_expiry_date = last_thursday
    if last_thursday in holidays:
        last_expiry_date = last_thursday - timedelta(days=1)

    # Check monthly expiry first
    if d == last_expiry_date:
        return "monthly"

    # Check weekly expiry: Thursday (or Wednesday if Thursday is holiday)
    if d.weekday() == 3 and d not in holidays:
        return "weekly"

    # Check if it's a Wednesday that replaces a holiday Thursday
    if d.weekday() == 2:
        thursday = d + timedelta(days=1)
        if thursday in holidays:
            # This Wednesday is the shifted expiry
            # But not if it's already counted as monthly
            return "weekly"

    return None


def get_no_trade_zone(t: time) -> str | None:
    """Identify if current time falls in a no-trade zone.

    Returns zone name or None if safe to trade.
    """
    if time(9, 15) <= t < time(9, 30):
        return "opening_auction"
    if time(13, 0) <= t < time(14, 0):
        return "post_lunch"
    if time(15, 15) <= t < time(15, 30):
        return "closing_auction"
    return None


def compute_gap(prev_close: float, today_open: float) -> dict:
    """Compute gap percentage between previous close and today's open.

    Returns dict with gap_pct (absolute), direction, and signed_pct.
    """
    if prev_close == 0:
        return {"gap_pct": 0.0, "direction": "up", "signed_pct": 0.0}

    signed_pct = ((today_open - prev_close) / prev_close) * 100
    return {
        "gap_pct": round(abs(signed_pct), 4),
        "direction": "up" if signed_pct >= 0 else "down",
        "signed_pct": round(signed_pct, 4),
    }


def is_iv_elevated(current_vix: float, avg_20d_vix: float) -> bool:
    """True if current VIX is more than 20% above the 20-day average."""
    if avg_20d_vix == 0:
        return False
    return current_vix > avg_20d_vix * 1.20


def format_time_context_for_claude(
    current_time: time,
    d: date,
    gap: dict | None,
    vix: float | None,
    avg_vix: float | None,
    holidays: list[date] | None = None,
) -> str:
    """Format all time-based context into a multiline string for Claude prompts.

    Returns empty string if nothing noteworthy.
    """
    lines: list[str] = []

    # Expiry info
    expiry = is_expiry_day(d, holidays)
    if expiry:
        lines.append(f"⚠ Today is {expiry} expiry day — expect elevated volatility and pin risk.")

    # No-trade zone
    zone = get_no_trade_zone(current_time)
    if zone:
        zone_labels = {
            "opening_auction": "Opening auction (9:15-9:30) — avoid new entries, spreads widen.",
            "post_lunch": "Post-lunch lull (13:00-14:00) — low volume, avoid scalping.",
            "closing_auction": "Closing auction (15:15-15:30) — avoid new entries, use MOC orders.",
        }
        lines.append(f"🚫 No-trade zone: {zone_labels.get(zone, zone)}")

    # Gap info
    if gap and gap.get("gap_pct", 0) > 0.1:
        direction = gap["direction"]
        pct = gap["signed_pct"]
        lines.append(f"📊 Gap {direction}: {pct:+.2f}% — {'watch for gap fill' if abs(pct) < 1 else 'significant gap, trend day likely'}.")

    # IV status
    if vix is not None and avg_vix is not None and avg_vix > 0:
        if is_iv_elevated(vix, avg_vix):
            premium = ((vix - avg_vix) / avg_vix) * 100
            lines.append(f"🔥 IV elevated: VIX {vix:.1f} is {premium:.0f}% above 20-day avg ({avg_vix:.1f}) — favor selling premium.")
        elif vix < avg_vix * 0.85:
            lines.append(f"💤 IV depressed: VIX {vix:.1f} vs 20-day avg {avg_vix:.1f} — favor buying options.")

    return "\n".join(lines)
