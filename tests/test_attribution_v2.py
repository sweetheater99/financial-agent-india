"""Tests for attribution.py v2 features -- profit factor, Sharpe ratio, risk metrics."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from attribution import compute_attribution, generate_attribution_report


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


class TestProfitFactor:
    """Test profit factor calculation."""

    def test_known_trades(self):
        trades = [
            _make_closed_trade("R1", "bullish", "EQ", 500, 5.0, "2026-01-01", "2026-01-05"),
            _make_closed_trade("R2", "bullish", "EQ", 300, 3.0, "2026-01-06", "2026-01-10"),
            _make_closed_trade("R3", "bearish", "EQ", -200, -2.0, "2026-01-11", "2026-01-15"),
            _make_closed_trade("R4", "bearish", "EQ", -100, -1.0, "2026-01-16", "2026-01-20"),
        ]
        attr = compute_attribution([], trades)
        # Total wins = 800, total losses = 300
        # Profit factor = 800 / 300 = 2.67
        assert attr["profit_factor"] == 2.67

    def test_no_losses(self):
        trades = [
            _make_closed_trade("R1", "bullish", "EQ", 500, 5.0, "2026-01-01", "2026-01-05"),
            _make_closed_trade("R2", "bullish", "EQ", 300, 3.0, "2026-01-06", "2026-01-10"),
        ]
        attr = compute_attribution([], trades)
        assert attr["profit_factor"] == float('inf')

    def test_no_wins(self):
        trades = [
            _make_closed_trade("R1", "bearish", "EQ", -200, -2.0, "2026-01-01", "2026-01-05"),
            _make_closed_trade("R2", "bearish", "EQ", -100, -1.0, "2026-01-06", "2026-01-10"),
        ]
        attr = compute_attribution([], trades)
        assert attr["profit_factor"] == 0.0


class TestSharpeRatio:
    """Test Sharpe ratio calculation."""

    def test_known_daily_pnls(self):
        # Create trades with known exit dates to get predictable daily P&Ls
        trades = [
            _make_closed_trade("R1", "bullish", "EQ", 500, 5.0, "2026-01-01", "2026-01-05"),
            _make_closed_trade("R2", "bullish", "EQ", 300, 3.0, "2026-01-01", "2026-01-06"),
            _make_closed_trade("R3", "bearish", "EQ", -200, -2.0, "2026-01-01", "2026-01-07"),
        ]
        attr = compute_attribution([], trades)
        # Daily PnLs: Jan 5 = 500, Jan 6 = 300, Jan 7 = -200
        # Daily returns (% of 100k): 0.005, 0.003, -0.002
        # Mean = 0.002, stdev = 0.003606..., sharpe = (0.002/0.003606)*sqrt(252)
        assert attr["sharpe_ratio"] != 0
        assert isinstance(attr["sharpe_ratio"], float)

    def test_single_day(self):
        # Only one exit date -> Sharpe = 0
        trades = [
            _make_closed_trade("R1", "bullish", "EQ", 500, 5.0, "2026-01-01", "2026-01-05"),
            _make_closed_trade("R2", "bullish", "EQ", 300, 3.0, "2026-01-01", "2026-01-05"),
        ]
        attr = compute_attribution([], trades)
        assert attr["sharpe_ratio"] == 0

    def test_empty_trades(self):
        attr = compute_attribution([], [])
        assert attr["sharpe_ratio"] == 0
        assert attr["profit_factor"] == float('inf')


class TestWinLossRatio:
    """Test avg winner, avg loser, win/loss ratio."""

    def test_known_values(self):
        trades = [
            _make_closed_trade("R1", "bullish", "EQ", 600, 6.0, "2026-01-01", "2026-01-05"),
            _make_closed_trade("R2", "bullish", "EQ", 400, 4.0, "2026-01-06", "2026-01-10"),
            _make_closed_trade("R3", "bearish", "EQ", -200, -2.0, "2026-01-11", "2026-01-15"),
            _make_closed_trade("R4", "bearish", "EQ", -300, -3.0, "2026-01-16", "2026-01-20"),
        ]
        attr = compute_attribution([], trades)
        assert attr["avg_winner"] == 500.0  # (600+400)/2
        assert attr["avg_loser"] == 250.0   # (200+300)/2
        assert attr["win_loss_ratio"] == 2.0  # 500/250


class TestReportIncludesRiskMetrics:
    """Test that report includes the new Risk Metrics section."""

    def test_report_has_risk_metrics(self):
        trades = [
            _make_closed_trade("R1", "bullish", "EQ", 500, 5.0, "2026-01-01", "2026-01-05"),
            _make_closed_trade("R2", "bullish", "EQ", 300, 3.0, "2026-01-06", "2026-01-10"),
            _make_closed_trade("R3", "bearish", "EQ", -200, -2.0, "2026-01-11", "2026-01-15"),
        ]
        attr = compute_attribution([], trades)
        report = generate_attribution_report(attr)

        assert "Risk Metrics" in report
        assert "Profit Factor" in report
        assert "Sharpe Ratio" in report
        assert "Avg Winner" in report
        assert "Avg Loser" in report
        assert "Win/Loss Ratio" in report

    def test_report_still_has_existing_sections(self):
        """Verify existing sections are not broken."""
        trades = [
            _make_closed_trade("R1", "bullish", "EQ", 500, 5.0, "2026-01-01", "2026-01-05"),
        ]
        attr = compute_attribution([], trades)
        report = generate_attribution_report(attr)

        assert "Performance Attribution" in report
        assert "By Strategy" in report
        assert "By Direction" in report
        assert "Time Analysis" in report

    def test_report_profit_factor_inf(self):
        """All winners -> profit factor shows inf."""
        trades = [
            _make_closed_trade("R1", "bullish", "EQ", 500, 5.0, "2026-01-01", "2026-01-05"),
        ]
        attr = compute_attribution([], trades)
        report = generate_attribution_report(attr)
        assert "inf" in report
