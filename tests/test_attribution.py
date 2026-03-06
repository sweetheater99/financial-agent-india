"""Tests for attribution.py — uses synthetic data, no API calls."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from attribution import compute_attribution, generate_attribution_report


def _make_journal_entry(symbol, direction, entry_date, regime="TRENDING_UP",
                        signals_categories=None, instrument="EQ"):
    return {
        "id": f"{symbol}_{entry_date}_{direction}",
        "symbol": symbol,
        "direction": direction,
        "instrument": instrument,
        "entry_price": 100,
        "entry_date": entry_date,
        "thesis": {
            "score": 8.0,
            "regime": regime,
            "signals": {"categories": signals_categories or []},
        },
        "exit": None,
    }


def _make_closed_trade(symbol, direction, instrument, pnl, pnl_pct,
                        entry_date, exit_date, exit_reason="target"):
    return {
        "symbol": symbol,
        "direction": direction,
        "instrument": instrument,
        "entry_price": 100,
        "exit_price": 100 + pnl,
        "quantity": 1,
        "pnl_gross": pnl,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "transaction_costs": 5,
        "entry_date": entry_date,
        "exit_date": exit_date,
        "exit_reason": exit_reason,
    }


class TestEmptyData:
    """Test 1: Empty data returns empty attribution."""

    def test_empty(self):
        attr = compute_attribution([], [])
        assert attr["per_strategy"] == {}
        assert attr["per_signal"] == {}
        assert attr["per_regime"] == {}
        assert attr["per_direction"] == {}
        assert attr["per_sector"] == {}
        assert attr["recommendations"] == []


class TestPerStrategyBreakdown:
    """Test 2: Per-strategy breakdown is correct."""

    def test_strategy_breakdown(self):
        journal = [
            _make_journal_entry("RELIANCE", "bullish", "2026-01-01", instrument="EQ"),
            _make_journal_entry("RELIANCE", "bearish", "2026-01-10", instrument="OPT"),
            _make_journal_entry("SBIN", "bullish", "2026-01-15", instrument="EQ"),
        ]
        trades = [
            _make_closed_trade("RELIANCE", "bullish", "EQ", 500, 5.0, "2026-01-01", "2026-01-05"),
            _make_closed_trade("RELIANCE", "bearish", "OPT", -200, -10.0, "2026-01-10", "2026-01-12"),
            _make_closed_trade("SBIN", "bullish", "EQ", 300, 3.0, "2026-01-15", "2026-01-20"),
        ]
        attr = compute_attribution(journal, trades)

        assert "EQ" in attr["per_strategy"]
        assert "OPT" in attr["per_strategy"]
        assert attr["per_strategy"]["EQ"]["trades"] == 2
        assert attr["per_strategy"]["EQ"]["wins"] == 2
        assert attr["per_strategy"]["EQ"]["pnl"] == 800
        assert attr["per_strategy"]["OPT"]["trades"] == 1
        assert attr["per_strategy"]["OPT"]["wins"] == 0
        assert attr["per_strategy"]["OPT"]["pnl"] == -200


class TestPerSignalWinRate:
    """Test 3: Per-signal win rate calculation."""

    def test_signal_win_rate(self):
        journal = [
            _make_journal_entry("RELIANCE", "bullish", "2026-01-01",
                                signals_categories=["LongBuildUp", "ShortCovering"]),
            _make_journal_entry("SBIN", "bullish", "2026-01-05",
                                signals_categories=["LongBuildUp"]),
            _make_journal_entry("TCS", "bearish", "2026-01-10",
                                signals_categories=["ShortCovering"]),
        ]
        trades = [
            _make_closed_trade("RELIANCE", "bullish", "EQ", 500, 5.0, "2026-01-01", "2026-01-05"),
            _make_closed_trade("SBIN", "bullish", "EQ", -100, -1.0, "2026-01-05", "2026-01-08"),
            _make_closed_trade("TCS", "bearish", "EQ", 200, 2.0, "2026-01-10", "2026-01-15"),
        ]
        attr = compute_attribution(journal, trades)

        lb = attr["per_signal"]["LongBuildUp"]
        assert lb["appeared_in"] == 2
        assert lb["wins"] == 1
        assert lb["losses"] == 1
        assert lb["win_rate"] == 50.0

        sc = attr["per_signal"]["ShortCovering"]
        assert sc["appeared_in"] == 2
        assert sc["wins"] == 2  # both trades with ShortCovering won


class TestRecommendationsGenerated:
    """Test 4: Recommendations generated for losing strategies."""

    def test_losing_strategy_recommendation(self):
        journal = []
        trades = [
            _make_closed_trade("R1", "bullish", "OPT", -100, -5.0, "2026-01-01", "2026-01-02"),
            _make_closed_trade("R2", "bullish", "OPT", -200, -10.0, "2026-01-03", "2026-01-04"),
            _make_closed_trade("R3", "bullish", "OPT", -150, -7.0, "2026-01-05", "2026-01-06"),
        ]
        attr = compute_attribution(journal, trades)
        recs = attr["recommendations"]
        assert any("Kill OPT" in r for r in recs)


class TestTimeAnalysis:
    """Test 5: Time analysis (avg hold days)."""

    def test_hold_days(self):
        journal = []
        trades = [
            # Winner held 5 days
            _make_closed_trade("R1", "bullish", "EQ", 500, 5.0, "2026-01-01", "2026-01-06"),
            # Winner held 3 days
            _make_closed_trade("R2", "bullish", "EQ", 300, 3.0, "2026-01-10", "2026-01-13"),
            # Loser held 10 days
            _make_closed_trade("R3", "bearish", "EQ", -400, -4.0, "2026-01-15", "2026-01-25"),
        ]
        attr = compute_attribution(journal, trades)
        ta = attr["time_analysis"]
        assert ta["avg_winner_hold_days"] == 4.0  # (5+3)/2
        assert ta["avg_loser_hold_days"] == 10.0


class TestReportFormat:
    """Test 6: Report format contains expected sections."""

    def test_report_sections(self):
        journal = [
            _make_journal_entry("RELIANCE", "bullish", "2026-01-01",
                                signals_categories=["LongBuildUp"]),
        ]
        trades = [
            _make_closed_trade("RELIANCE", "bullish", "EQ", 500, 5.0, "2026-01-01", "2026-01-05"),
        ]
        attr = compute_attribution(journal, trades)
        report = generate_attribution_report(attr)

        assert "Performance Attribution" in report
        assert "By Strategy" in report
        assert "By Direction" in report
        assert "By Regime" in report
        assert "Time Analysis" in report
