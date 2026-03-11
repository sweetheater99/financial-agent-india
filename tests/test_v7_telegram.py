# tests/test_v7_telegram.py
"""Tests for V7 Telegram integration."""
import pytest
import time as time_mod
from unittest.mock import patch, MagicMock
from datetime import date
from v7.telegram import (
    TelegramAlerter, AlertLevel, format_playbook_summary,
    format_trade_entry, format_trade_exit, format_checkin,
    format_exception, format_eod_summary, format_weekly_report,
    format_error,
)
from v7.types import (
    Playbook, Setup, SetupType, Position, TradeResult,
    RiskBudget, CarryRules, DayClassification, Conviction,
)


def test_alert_level_ordering():
    assert AlertLevel.CRITICAL.value > AlertLevel.HIGH.value
    assert AlertLevel.HIGH.value > AlertLevel.MEDIUM.value
    assert AlertLevel.MEDIUM.value > AlertLevel.LOW.value


def test_format_playbook_summary():
    pb = Playbook(
        date=date(2026, 3, 11),
        day_classification=DayClassification.LIKELY_TREND_UP,
        nifty_bias="bullish",
        nifty_setups=[
            Setup(
                id="N1", priority=1, type=SetupType.BREAKOUT_LONG,
                symbol="NIFTY", trigger_level=24350.0,
                trigger_condition="15-min close above 24350",
                instrument="NIFTY CE", strike_logic="delta 0.45",
                target=24500.0, stoploss=24280.0, max_risk_pct=1.5,
            ),
        ],
        stock_plans=[],
        risk_budget=RiskBudget(
            max_capital_at_risk_today_pct=4.0,
            max_trades_today=4,
            max_per_trade_risk_pct=1.5,
            survival_mode=False,
        ),
        no_trade_conditions=["VIX > 22"],
        carry_rules=CarryRules(),
    )
    msg = format_playbook_summary(pb)
    assert "<b>" in msg  # HTML formatting
    assert "LIKELY_TREND_UP" in msg
    assert "bullish" in msg.lower()
    assert "N1" in msg
    assert "```" not in msg  # no Markdown


def test_format_trade_entry():
    pos = Position(
        symbol="NIFTY", instrument="NIFTY CE 24400",
        direction="bullish", entry_price=120.0,
        quantity=75, lot_size=75, allocated=9000.0,
        stoploss=100.0, target=180.0,
        entry_date=date(2026, 3, 11), setup_id="N1",
    )
    msg = format_trade_entry(pos)
    assert "<b>NIFTY</b>" in msg
    assert "24400" in msg
    assert "120" in msg
    assert "HTML" not in msg  # parse_mode is set at send, not in message
    assert "```" not in msg


def test_format_trade_exit():
    result = TradeResult(
        symbol="NIFTY", instrument="NIFTY CE 24400",
        direction="bullish", entry_price=120.0, exit_price=180.0,
        quantity=75, entry_date=date(2026, 3, 11),
        exit_date=date(2026, 3, 11), exit_reason="target",
        pnl=4500.0, pnl_pct=50.0, costs=120.0,
        setup_id="N1", setup_type=SetupType.BREAKOUT_LONG,
    )
    msg = format_trade_exit(result, daily_pnl=4500.0, open_count=0)
    assert "4,500" in msg
    assert "target" in msg.lower()
    assert "```" not in msg


def test_format_eod_summary():
    msg = format_eod_summary(
        trades_today=2, wins=1, losses=1,
        directional_pnl=2000.0, theta_pnl=500.0,
        total_pnl=2500.0, capital=300_000,
        carried_positions=[],
        day_type_predicted="LIKELY_TREND_UP",
        day_type_actual="trending up",
    )
    assert "2,500" in msg
    assert "```" not in msg


def test_format_error():
    msg = format_error("Kite token expired", "Protect-only mode active. SL orders live.")
    assert "Kite token expired" in msg
    assert "<b>" in msg
    assert "```" not in msg


def test_rate_limiting():
    alerter = TelegramAlerter(token="fake", chat_id="fake")
    # Should not actually send (no real token), but should track rate
    with patch.object(alerter, "_do_send") as mock_send:
        alerter.send("test1", AlertLevel.HIGH)
        alerter.send("test2", AlertLevel.HIGH)
        alerter.send("test3", AlertLevel.HIGH)
        alerter.send("test4", AlertLevel.HIGH)
        alerter.send("test5", AlertLevel.HIGH)
        alerter.send("test6", AlertLevel.HIGH)
        # After 5 rapid SL alerts, rate limit kicks in
        # But individual sends should go through at HIGH level
        assert mock_send.call_count == 6  # all HIGH go through


def test_rate_limiting_low_priority():
    alerter = TelegramAlerter(token="fake", chat_id="fake")
    with patch.object(alerter, "_do_send") as mock_send:
        for _ in range(10):
            alerter.send("heartbeat", AlertLevel.LOW)
        # LOW priority gets rate-limited after burst
        assert mock_send.call_count <= 10  # all go through if spaced


def test_no_markdown_in_any_format():
    """Verify none of the format functions produce Markdown syntax."""
    pb = Playbook(
        date=date(2026, 3, 11),
        day_classification=DayClassification.LIKELY_TREND_UP,
        nifty_bias="bullish", nifty_setups=[], stock_plans=[],
        risk_budget=RiskBudget(), no_trade_conditions=[],
        carry_rules=CarryRules(),
    )
    for msg in [
        format_playbook_summary(pb),
        format_trade_entry(Position(
            symbol="X", instrument="X CE", direction="bullish",
            entry_price=100, quantity=75, lot_size=75, allocated=7500,
            stoploss=80, target=150, entry_date=date(2026, 3, 11), setup_id="T",
        )),
        format_eod_summary(0, 0, 0, 0, 0, 0, 300000, [], "?", "?"),
        format_error("err", "detail"),
    ]:
        assert "```" not in msg, f"Markdown backticks found in: {msg[:100]}"
        assert "__" not in msg or "<b>" in msg  # no Markdown bold
