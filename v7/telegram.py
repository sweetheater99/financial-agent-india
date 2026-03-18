# v7/telegram.py
"""V7 Telegram Integration — all alerts to Ops Hub Stocks topic.

HARD RULES:
- Always HTML parse_mode. NEVER Markdown.
- Use temp file + curl for shell scripts (matching existing pattern).
- Rate limit: don't spam on rapid SL hits.
"""
from __future__ import annotations

import os
import time as time_mod
import urllib.parse
import urllib.request
from collections import deque
from datetime import date
from enum import IntEnum
from typing import Any

from v7.types import (
    Playbook, Position, TradeResult, Setup,
)


class AlertLevel(IntEnum):
    LOW = 1       # informational (playbook, check-in)
    MEDIUM = 2    # EOD summary
    HIGH = 3      # trade entry/exit, exception
    CRITICAL = 4  # error/disconnect


class TelegramAlerter:
    """Send alerts to Telegram with rate limiting."""

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
        topic_id: str | None = None,
    ):
        self._token = token or os.environ.get("DEAL_BOT_TOKEN", "") or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id or os.environ.get("TELEGRAM_FORUM_CHAT_ID", "") or os.environ.get("DEAL_BOT_CHAT_ID", "")
        self._topic_id = topic_id or os.environ.get("TELEGRAM_TOPIC_STOCKS", "")
        self._recent_sends: deque[float] = deque(maxlen=50)
        self._rate_limit_window = 60  # seconds
        self._rate_limit_max = 20     # max messages per window

    def send(self, text: str, level: AlertLevel = AlertLevel.HIGH) -> bool:
        """Send a message. Returns True if sent, False if rate-limited or failed."""
        if not self._token or not self._chat_id:
            return False

        # Rate limiting: skip LOW messages if we're sending too fast
        now = time_mod.time()
        recent_count = sum(1 for t in self._recent_sends if now - t < self._rate_limit_window)
        if recent_count >= self._rate_limit_max and level <= AlertLevel.LOW:
            return False

        self._recent_sends.append(now)
        return self._do_send(text)

    def _do_send(self, text: str) -> bool:
        """Actually send via Telegram API. Split if > 4000 chars."""
        if len(text) <= 4000:
            return self._send_single(text)

        # Split at newline boundaries
        chunks: list[str] = []
        remaining = text
        while remaining:
            if len(remaining) <= 4000:
                chunks.append(remaining)
                break
            split_at = remaining.rfind("\n", 0, 4000)
            if split_at <= 0:
                split_at = 4000
            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:].lstrip("\n")

        total = len(chunks)
        ok = True
        for i, chunk in enumerate(chunks, 1):
            header = f"({i}/{total}) " if total > 1 else ""
            if not self._send_single(header + chunk):
                ok = False
        return ok

    def _send_single(self, text: str) -> bool:
        """Send a single message via Telegram HTTP API. HTML parse_mode always."""
        try:
            url = f"https://api.telegram.org/bot{self._token}/sendMessage"
            payload: dict[str, Any] = {
                "chat_id": self._chat_id,
                "text": text,
                "parse_mode": "HTML",
            }
            if self._chat_id and self._topic_id:
                payload["message_thread_id"] = self._topic_id
            data = urllib.parse.urlencode(payload).encode()
            req = urllib.request.Request(url, data=data)
            urllib.request.urlopen(req, timeout=10)
            return True
        except Exception:
            return False


# ── Format Functions ──────────────────────────────────────────────────
# All return HTML strings. No Markdown. No backticks. No __.


def format_playbook_summary(pb: Playbook) -> str:
    """Format pre-market playbook for Telegram."""
    lines = [
        f"<b>V7 Pre-Market Playbook — {pb.date}</b>",
        "",
        f"Day type: <b>{pb.day_classification.value}</b>",
        f"Nifty bias: {pb.nifty_bias}",
    ]

    ctx = pb.market_context
    if ctx is not None:
        # Support both MarketContext dataclass and dict (for flexibility)
        if hasattr(ctx, "gift_nifty"):
            if ctx.gift_nifty:
                lines.append(f"GIFT Nifty: {ctx.gift_nifty}")
            if ctx.vix:
                lines.append(f"VIX: {ctx.vix}")
            if ctx.fii_dii:
                lines.append(f"FII/DII: {ctx.fii_dii}")
        elif isinstance(ctx, dict):
            if ctx.get("gift_nifty"):
                lines.append(f"GIFT Nifty: {ctx['gift_nifty']}")
            if ctx.get("vix"):
                lines.append(f"VIX: {ctx['vix']}")
            if ctx.get("fii_dii"):
                lines.append(f"FII/DII: {ctx['fii_dii']}")

    if pb.nifty_setups:
        lines.append("")
        lines.append("<b>Nifty setups:</b>")
        for s in pb.nifty_setups:
            lines.append(f"  [{s.id}] {s.type.value}: trigger {s.trigger_level:,.0f}, "
                         f"SL {s.stoploss:,.0f}, target {s.target:,.0f}")

    if pb.stock_plans:
        lines.append("")
        lines.append("<b>Stock setups:</b>")
        for s in pb.stock_plans:
            lines.append(f"  [{s.id}] {s.symbol} {s.type.value}: trigger {s.trigger_level:,.0f}")

    rb = pb.risk_budget
    lines.append("")
    lines.append(f"Risk: {rb.max_capital_at_risk_today_pct}% max, "
                 f"{rb.max_trades_today} trades max")
    if rb.survival_mode:
        lines.append("<b>SURVIVAL MODE — theta only</b>")

    return "\n".join(lines)


def format_trade_entry(pos: Position) -> str:
    """Format trade entry alert."""
    direction = "LONG" if pos.direction == "bullish" else "SHORT"
    return "\n".join([
        f"V7 Entry: <b>{pos.symbol}</b> {direction}",
        f"Instrument: {pos.instrument}",
        f"Price: {pos.entry_price:,.2f}  |  Qty: {pos.quantity}",
        f"SL: {pos.stoploss:,.2f}  |  Target: {pos.target:,.2f}",
        f"Risk: {pos.allocated:,.0f}  |  Setup: {pos.setup_id}",
    ])


def format_trade_exit(
    result: TradeResult,
    daily_pnl: float = 0.0,
    open_count: int = 0,
) -> str:
    """Format trade exit alert."""
    emoji = "WIN" if result.pnl > 0 else "LOSS"

    # Format exit reason with descriptive text for key reasons
    reason = result.exit_reason
    if reason == "health_exit":
        reason_text = "HEALTH EXIT (position dying — momentum/premium/progress failed)"
    elif reason == "time_stop":
        reason_text = "TIME STOP (no progress toward target)"
    else:
        reason_text = reason.upper()

    return "\n".join([
        f"<b>V7 Exit: {result.symbol}</b> [{emoji}]",
        f"Instrument: {result.instrument}",
        f"Entry: {result.entry_price:,.2f} → Exit: {result.exit_price:,.2f}",
        f"P&amp;L: {result.pnl:+,.0f} ({result.pnl_pct:+.1f}%)",
        f"Reason: {reason_text}  |  Costs: {result.costs:,.0f}",
        f"Grade: entry={result.entry_grade} exit={result.exit_grade}",
        "",
        f"Day P&amp;L: {daily_pnl:+,.0f}  |  Open: {open_count}",
    ])


def format_checkin(
    checkin_num: int,
    plan_changed: bool,
    summary: str,
    daily_pnl: float = 0.0,
    open_count: int = 0,
) -> str:
    """Format check-in update."""
    status = "MODIFIED" if plan_changed else "unchanged"
    return "\n".join([
        f"<b>V7 Check-in #{checkin_num}</b> — plan {status}",
        summary,
        "",
        f"Day P&amp;L: {daily_pnl:+,.0f}  |  Open: {open_count}",
    ])


def format_exception(trigger: str, action: str) -> str:
    """Format exception alert."""
    return "\n".join([
        f"<b>V7 EXCEPTION</b>",
        f"Trigger: {trigger}",
        f"Action: {action}",
    ])


def format_eod_summary(
    trades_today: int,
    wins: int,
    losses: int,
    directional_pnl: float,
    theta_pnl: float,
    total_pnl: float,
    capital: float,
    carried_positions: list[dict],
    day_type_predicted: str,
    day_type_actual: str,
    stale_refreshes: int = 0,
) -> str:
    """Format end-of-day summary."""
    pnl_pct = (total_pnl / capital * 100) if capital > 0 else 0.0
    lines = [
        f"<b>V7 EOD Summary</b>",
        "",
        f"Trades: {trades_today} ({wins}W / {losses}L)",
        f"Directional: {directional_pnl:+,.0f}",
        f"Theta: {theta_pnl:+,.0f}",
        f"<b>Total: {total_pnl:+,.0f} ({pnl_pct:+.1f}%)</b>",
        "",
        f"Day type: predicted {day_type_predicted} / actual {day_type_actual}",
        f"Playbook refreshes: {stale_refreshes}",
    ]

    if carried_positions:
        lines.append("")
        lines.append(f"Carried overnight: {len(carried_positions)}")
        for p in carried_positions:
            lines.append(f"  {p.get('symbol', '?')} — P&amp;L {p.get('pnl', 0):+,.0f}")

    return "\n".join(lines)


def format_weekly_report(report_text: str) -> str:
    """Format weekly review report. report_text is pre-formatted by journal."""
    return f"<b>V7 Weekly Review</b>\n\n{report_text}"


def format_error(error: str, detail: str) -> str:
    """Format error/disconnect alert."""
    return "\n".join([
        f"<b>V7 ERROR</b>",
        f"{error}",
        f"{detail}",
    ])


def format_portfolio_status(
    positions: list[dict],
    unrealised_pnl: float,
    position_details: list[dict],
    realised_pnl: float,
    realised_trades: int,
    capital: float,
) -> str:
    """Format portfolio P&L status for Telegram."""
    lines = [
        "<b>V7 Portfolio Status</b>",
        "",
    ]

    # Open positions with unrealised P&L
    if position_details:
        lines.append(f"<b>Open ({len(position_details)}):</b>")
        for pd in position_details:
            lines.append(
                f"  {pd['symbol']} {pd['instrument']}: "
                f"Rs.{pd['pnl']:+,.0f} ({pd['pnl_pct']:+.1f}%)"
            )
        unr_pct = (unrealised_pnl / capital * 100) if capital else 0
        lines.append(f"<b>Unrealised: Rs.{unrealised_pnl:+,.0f} ({unr_pct:+.1f}%)</b>")
    else:
        lines.append("No open positions")

    # Cumulative realised P&L
    lines.append("")
    real_pct = (realised_pnl / capital * 100) if capital else 0
    lines.append(
        f"<b>Realised: Rs.{realised_pnl:+,.0f} ({real_pct:+.1f}%) "
        f"[{realised_trades} trades]</b>"
    )

    # Net (realised + unrealised)
    net = realised_pnl + unrealised_pnl
    net_pct = (net / capital * 100) if capital else 0
    lines.append(f"<b>Net: Rs.{net:+,.0f} ({net_pct:+.1f}%)</b>")

    return "\n".join(lines)
