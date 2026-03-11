"""Tests for V7 trade journal."""
import json
import pytest
from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock
from v7.journal import (
    Journal, grade_trades_prompt, parse_journal_response,
    format_obsidian_journal, generate_weekly_review_prompt,
    generate_monthly_report_prompt,
)
from v7.types import TradeResult, SetupType, DayClassification


@pytest.fixture
def tmp_journal(tmp_path):
    vault_dir = tmp_path / "trading-journal"
    return Journal(vault_dir=vault_dir, data_dir=tmp_path)


@pytest.fixture
def sample_trades():
    return [
        TradeResult(
            symbol="NIFTY", instrument="NIFTY CE 24400",
            direction="bullish", entry_price=120.0, exit_price=160.0,
            quantity=75, entry_date=date(2026, 3, 11),
            exit_date=date(2026, 3, 11), exit_reason="target",
            pnl=3000.0, pnl_pct=33.3, costs=120.0,
            setup_id="N1", setup_type=SetupType.BREAKOUT_LONG,
        ),
        TradeResult(
            symbol="HDFCBANK", instrument="HDFCBANK CE 1600",
            direction="bullish", entry_price=50.0, exit_price=35.0,
            quantity=550, entry_date=date(2026, 3, 11),
            exit_date=date(2026, 3, 11), exit_reason="stoploss",
            pnl=-8250.0, pnl_pct=-30.0, costs=80.0,
            setup_id="H1", setup_type=SetupType.SUPPORT_BOUNCE,
        ),
    ]


def test_grade_trades_prompt(sample_trades):
    prompt = grade_trades_prompt(
        trades=sample_trades,
        day_classification=DayClassification.LIKELY_TREND_UP,
        playbook_setups=2,
        day_pnl=-5250.0,
    )
    assert "NIFTY" in prompt
    assert "HDFCBANK" in prompt
    assert "entry_grade" in prompt.lower() or "entry quality" in prompt.lower()
    assert "exit_grade" in prompt.lower() or "exit quality" in prompt.lower()


def test_parse_journal_response():
    raw = json.dumps({
        "trades": [
            {
                "setup_id": "N1",
                "entry_grade": "A",
                "exit_grade": "A",
                "lesson": "Clean breakout with volume confirmation.",
            },
            {
                "setup_id": "H1",
                "entry_grade": "C",
                "exit_grade": "B",
                "lesson": "Sector was weak — check sector strength before banking plays.",
            },
        ],
        "day_summary": {
            "day_type_accuracy": "correct",
            "best_trade": "N1",
            "worst_trade": "H1",
            "overall_lesson": "Sector context matters more than individual chart.",
        },
    })
    result = parse_journal_response(raw)
    assert len(result["trades"]) == 2
    assert result["trades"][0]["entry_grade"] == "A"
    assert result["day_summary"]["best_trade"] == "N1"


def test_format_obsidian_journal(sample_trades):
    grading = {
        "trades": [
            {"setup_id": "N1", "entry_grade": "A", "exit_grade": "A",
             "lesson": "Clean breakout."},
            {"setup_id": "H1", "entry_grade": "C", "exit_grade": "B",
             "lesson": "Check sector first."},
        ],
        "day_summary": {
            "day_type_accuracy": "correct",
            "best_trade": "N1",
            "worst_trade": "H1",
            "overall_lesson": "Sector context matters.",
        },
    }
    md = format_obsidian_journal(
        date_str="2026-03-11",
        trades=sample_trades,
        grading=grading,
        day_classification="LIKELY_TREND_UP",
        directional_pnl=-5250.0,
        theta_pnl=500.0,
        total_pnl=-4750.0,
    )
    assert "# Trading Journal — 2026-03-11" in md
    assert "NIFTY" in md
    assert "HDFCBANK" in md
    assert "Clean breakout" in md
    assert "Sector context matters" in md


def test_save_journal_to_obsidian(tmp_journal, sample_trades):
    grading = {
        "trades": [
            {"setup_id": "N1", "entry_grade": "A", "exit_grade": "A", "lesson": "Good."},
            {"setup_id": "H1", "entry_grade": "C", "exit_grade": "B", "lesson": "Bad."},
        ],
        "day_summary": {
            "day_type_accuracy": "correct", "best_trade": "N1",
            "worst_trade": "H1", "overall_lesson": "Learn.",
        },
    }
    path = tmp_journal.save_daily(
        date_str="2026-03-11",
        trades=sample_trades,
        grading=grading,
        day_classification="LIKELY_TREND_UP",
        directional_pnl=-5250.0,
        theta_pnl=500.0,
        total_pnl=-4750.0,
    )
    assert path.exists()
    assert path.name == "2026-03-11.md"
    content = path.read_text()
    assert "Trading Journal" in content


def test_weekly_review_prompt(sample_trades):
    prompt = generate_weekly_review_prompt(
        trades_this_week=sample_trades,
        edge_summary="Overall: 2 trades, 50% WR",
        level_memory={"NIFTY": {"levels": []}},
        watchlist_performance={"NIFTY": 3000.0, "HDFCBANK": -8250.0},
    )
    assert "performance attribution" in prompt.lower() or "weekly" in prompt.lower()
    assert "NIFTY" in prompt
    assert "HDFCBANK" in prompt


def test_monthly_report_prompt():
    prompt = generate_monthly_report_prompt(
        month="2026-03",
        total_pnl=15000.0,
        total_costs=2500.0,
        capital=300_000,
        trades_count=45,
        win_rate=0.55,
        max_drawdown_pct=3.2,
        edge_summary="Overall: 45 trades, 55% WR",
        theta_pnl=5000.0,
        directional_pnl=10000.0,
    )
    assert "monthly" in prompt.lower() or "report" in prompt.lower()
    assert "15,000" in prompt or "15000" in prompt
    assert "tax" in prompt.lower() or "turnover" in prompt.lower()
