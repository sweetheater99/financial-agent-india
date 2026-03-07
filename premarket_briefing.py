# Pre-market briefing for Trading System V3.
#
# Cron: 0 3 * * 1-5 (8:30 AM IST = 3:00 AM UTC)
# Add to Pi crontab: cd ~/financial-agent-india && venv/bin/python paper_trade.py briefing

"""
Generates a consolidated pre-market briefing message for Telegram.
Pulls from global intel, regime detection, OI analysis, and event calendar.
"""

import logging
import time
from datetime import datetime, timedelta, timezone, date

import pandas as pd

import config
from config import RBI_POLICY_DATES, BUDGET_BLOCK_DATES
from global_intel import fetch_macro_context
from regime import classify_regime
from oi_analysis import compute_pcr, compute_max_pain, get_top_oi_strikes
from agent_with_options import fetch_option_chain
from paper_trade import fetch_daily_candles, get_ltp

logger = logging.getLogger("paper_trade")

IST = timezone(timedelta(hours=5, minutes=30))

# Nifty / VIX tokens (same as paper_trade.py)
NIFTY_TOKEN = "99926000"
VIX_TOKEN = "26017"


def _fmt_crores(val: float) -> str:
    """Format crore value with sign and label."""
    sign = "+" if val >= 0 else ""
    action = "buying" if val >= 0 else "selling"
    return f"{sign}\u20b9{abs(val):,.0f} Cr ({action})"


def _pcr_label(pcr: float) -> str:
    if pcr < 0.7:
        return "extreme call buying"
    elif pcr < 0.9:
        return "mildly bullish"
    elif pcr <= 1.1:
        return "neutral"
    elif pcr <= 1.3:
        return "mildly bearish"
    else:
        return "extreme put buying"


def _section_global(macro: dict | None) -> str:
    """Build the Global Overnight section."""
    if not macro:
        return "<b>\U0001f30d Global Overnight</b>\nData unavailable"
    sp = macro.get("sp500_pct_change", 0)
    nas = macro.get("nasdaq_pct_change", 0)
    gift_gap = macro.get("gift_nifty_gap_pct", 0)
    gift_ltp = macro.get("gift_nifty_ltp", 0)
    hang = macro.get("hang_seng_pct", 0)
    nikkei = macro.get("nikkei_pct", 0)

    lines = [
        "<b>\U0001f30d Global Overnight</b>",
        f"S&P 500: {sp:+.1f}% | Nasdaq: {nas:+.1f}%",
    ]
    if gift_ltp:
        lines.append(f"GIFT Nifty: {gift_gap:+.1f}% (LTP {gift_ltp:,.0f})")
    else:
        lines.append(f"GIFT Nifty: {gift_gap:+.1f}%")
    lines.append(f"Asia: Hang Seng {hang:+.1f}%, Nikkei {nikkei:+.1f}%")
    return "\n".join(lines)


def _section_fii_dii(macro: dict | None) -> str:
    """Build the FII/DII Flow section."""
    if not macro:
        return "<b>\U0001f4b0 FII/DII Flow</b>\nData unavailable"
    fii = macro.get("fii_net_crores", 0)
    dii = macro.get("dii_net_crores", 0)
    return "\n".join([
        "<b>\U0001f4b0 FII/DII Flow</b>",
        f"FII: {_fmt_crores(fii)}",
        f"DII: {_fmt_crores(dii)}",
    ])


def _section_regime(regime_result: dict | None, vix: float | None) -> str:
    """Build the Market Regime section."""
    if not regime_result:
        return "<b>\U0001f4ca Market Regime: UNKNOWN</b>\nData unavailable"
    regime = regime_result.get("regime", "UNKNOWN")
    conf = regime_result.get("confidence", 0)
    vix_str = f"{vix:.1f}" if vix else "N/A"
    return "\n".join([
        f"<b>\U0001f4ca Market Regime: {regime}</b>",
        f"Confidence: {conf:.0%} | VIX: {vix_str}",
    ])


def _section_oi(pcr: float | None, max_pain: float | None, oi_strikes: dict | None) -> str:
    """Build the OI Signals section."""
    lines = ["<b>\U0001f4c8 OI Signals</b>"]
    if pcr is not None:
        lines.append(f"PCR: {pcr:.2f} ({_pcr_label(pcr)})")
    else:
        lines.append("PCR: N/A")
    if max_pain:
        lines.append(f"Max Pain: {max_pain:,.0f}")
    if oi_strikes:
        supports = oi_strikes.get("put_support", [])
        resistances = oi_strikes.get("call_resistance", [])
        if supports:
            lines.append(f"Support: {supports[0]['strike']:,.0f}")
        if resistances:
            lines.append(f"Resistance: {resistances[0]['strike']:,.0f}")
    return "\n".join(lines)


def _section_gates(macro: dict | None) -> str:
    """Build the Gates Active section."""
    if not macro:
        return "<b>\u26a0\ufe0f Gates Active: N/A</b>"
    gate = macro.get("hard_gate", "NONE")
    reason = macro.get("hard_gate_reason", "")
    if gate == "NONE":
        return "<b>\u26a0\ufe0f Gates Active: None</b>"
    return f"<b>\u26a0\ufe0f Gates Active: {gate}</b>\n{reason}"


def _section_events(today: date | None = None) -> str:
    """Build the Events Today section, checking RBI policy, Budget, and expiry."""
    if today is None:
        today = datetime.now(IST).date()

    today_str = today.isoformat()
    events_today = []
    upcoming = []

    # Check RBI policy dates
    for d in RBI_POLICY_DATES:
        if d == today_str:
            events_today.append("RBI Policy")
        elif d > today_str and len(upcoming) < 1:
            upcoming.append(f"RBI Policy {d}")

    # Check Budget dates
    for d in BUDGET_BLOCK_DATES:
        if d == today_str:
            events_today.append("Union Budget")
        elif d > today_str and len(upcoming) < 1:
            upcoming.append(f"Budget {d}")

    # Check if today is a Thursday (weekly expiry)
    if today.weekday() == 3:
        events_today.append("Weekly Expiry")

    # Check if it's last Thursday of month (monthly expiry)
    if today.weekday() == 3:
        next_thu = today + timedelta(days=7)
        if next_thu.month != today.month:
            events_today.append("Monthly Expiry")

    lines = ["<b>\U0001f4c5 Events Today</b>"]
    if events_today:
        lines[0] = f"<b>\U0001f4c5 Events Today: {', '.join(events_today)}</b>"
    else:
        lines[0] = "<b>\U0001f4c5 Events Today: None</b>"

    if upcoming:
        lines.append(f"Next: {upcoming[0]}")

    return "\n".join(lines)


def _section_playbook(regime: str | None, vix: float | None, gate: str | None) -> str:
    """Build the Today's Playbook section based on regime and conditions."""
    regime = regime or "UNKNOWN"
    vix = vix or 15.0
    gate = gate or "NONE"

    today = datetime.now(IST)
    day_of_week = today.weekday()  # 0=Mon

    # Equity longs
    equity_ok = regime in ("TRENDING_UP", "SIDEWAYS", "UNCERTAIN") and gate in ("NONE", "REDUCE_25", "REDUCE_50")
    equity_icon = "\u2705" if equity_ok else "\u274c"
    equity_note = ""
    if gate == "BLOCK_ALL":
        equity_note = " (blocked by macro gate)"
    elif gate == "BLOCK_BULLISH":
        equity_note = " (bullish blocked)"
    elif regime == "TRENDING_DOWN":
        equity_note = " (bearish regime)"
    elif regime == "CASH":
        equity_note = " (CASH regime)"
    elif regime == "VOLATILE":
        equity_note = " (volatile — reduced size)"

    # Iron condors
    condor_ok = regime == "SIDEWAYS" and config.CONDOR_MIN_VIX <= vix <= config.CONDOR_MAX_VIX
    condor_icon = "\u2705" if condor_ok else "\u274c"
    condor_note = ""
    if regime != "SIDEWAYS":
        condor_note = " (not sideways)"
    elif vix < config.CONDOR_MIN_VIX or vix > config.CONDOR_MAX_VIX:
        condor_note = f" (VIX {vix:.0f} outside {config.CONDOR_MIN_VIX}-{config.CONDOR_MAX_VIX})"

    # Credit spreads
    spread_ok = config.CREDIT_SPREAD_MIN_VIX <= vix <= config.CREDIT_SPREAD_MAX_VIX and gate not in ("BLOCK_ALL", "BLOCK_BULLISH")
    spread_icon = "\u2705" if spread_ok else "\u274c"
    spread_note = ""
    if vix < config.CREDIT_SPREAD_MIN_VIX:
        spread_note = f" (VIX {vix:.0f} too low)"
    elif vix > config.CREDIT_SPREAD_MAX_VIX:
        spread_note = f" (VIX {vix:.0f} too high)"

    # Weekly theta
    theta_ok = day_of_week in (0, 1) and vix >= config.WEEKLY_THETA_MIN_VIX
    theta_icon = "\u2705" if theta_ok else "\u274c"
    theta_note = ""
    if day_of_week not in (0, 1):
        theta_note = " (not Mon/Tue)"
    elif vix < config.WEEKLY_THETA_MIN_VIX:
        theta_note = f" (VIX {vix:.0f} < {config.WEEKLY_THETA_MIN_VIX})"

    lines = [
        "<b>\U0001f3af Today's Playbook</b>",
        f"\u2022 Equity longs: {equity_icon}{equity_note}",
        f"\u2022 Iron condors: {condor_icon}{condor_note}",
        f"\u2022 Credit spreads: {spread_icon}{spread_note}",
        f"\u2022 Weekly theta: {theta_icon}{theta_note}",
    ]
    return "\n".join(lines)


def generate_premarket_briefing(smart_api) -> str:
    """Generate the full pre-market briefing message.

    Each section is wrapped in try/except so one failure doesn't kill the whole briefing.

    Args:
        smart_api: authenticated SmartAPI session

    Returns:
        HTML-formatted string for Telegram
    """
    today = datetime.now(IST)
    header = f"<b>\U0001f305 Pre-Market Briefing \u2014 {today.strftime('%-d %b %Y')}</b>"

    # --- Fetch Nifty candles + VIX ---
    nifty_candles = None
    vix = None
    nifty_df = None
    prev_close = 0

    try:
        nifty_candles = fetch_daily_candles(smart_api, NIFTY_TOKEN, days=50)
        time.sleep(config.API_DELAY)
        if nifty_candles:
            nifty_df = pd.DataFrame(
                nifty_candles,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            prev_close = nifty_candles[-1][4]
    except Exception as e:
        logger.debug("Nifty candles fetch failed: %s", e)

    try:
        vix = get_ltp(smart_api, "India VIX", VIX_TOKEN)
        time.sleep(config.API_DELAY)
    except Exception as e:
        logger.debug("VIX fetch failed: %s", e)

    # --- Section 1: Global intel ---
    macro = None
    try:
        pcr_for_gate = 1.0  # will be updated below if OI available
        macro = fetch_macro_context(prev_nifty_close=prev_close, pcr=pcr_for_gate)
    except Exception as e:
        logger.debug("Global intel failed: %s", e)

    section_global = _section_global(macro)

    # --- Section 2: FII/DII ---
    section_fii = _section_fii_dii(macro)

    # --- Section 3: Regime ---
    regime_result = None
    try:
        if nifty_df is not None and vix is not None:
            regime_result = classify_regime(nifty_df, vix)
    except Exception as e:
        logger.debug("Regime detection failed: %s", e)

    section_regime = _section_regime(regime_result, vix)

    # --- Section 4: OI signals ---
    pcr = None
    max_pain = None
    oi_strikes = None
    try:
        chain = fetch_option_chain(smart_api, "NIFTY", NIFTY_TOKEN)
        time.sleep(config.API_DELAY)
        if chain:
            pcr = compute_pcr(chain)
            max_pain = compute_max_pain(chain)
            oi_strikes = get_top_oi_strikes(chain, top_n=1)
    except Exception as e:
        logger.debug("OI analysis failed: %s", e)

    section_oi = _section_oi(pcr, max_pain, oi_strikes)

    # --- Section 5: Gates ---
    section_gates = _section_gates(macro)

    # --- Section 6: Events ---
    section_events = _section_events(today.date())

    # --- Section 7: Playbook ---
    regime_name = regime_result["regime"] if regime_result else None
    gate_action = macro.get("hard_gate") if macro else None
    section_playbook = _section_playbook(regime_name, vix, gate_action)

    # Assemble
    sections = [
        header,
        "",
        section_global,
        "",
        section_fii,
        "",
        section_regime,
        "",
        section_oi,
        "",
        section_gates,
        "",
        section_events,
        "",
        section_playbook,
    ]

    return "\n".join(sections)
