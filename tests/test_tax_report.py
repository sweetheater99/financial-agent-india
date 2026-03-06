"""Tests for tax_report.py — uses synthetic data, no API calls."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tax_report import (
    generate_tax_report,
    generate_tax_pnl_statement,
    generate_telegram_tax_summary,
)


def _make_trade(symbol="RELIANCE", instrument="EQ", direction="bullish",
                entry_price=100, exit_price=110, quantity=10,
                pnl=100, pnl_pct=10.0, entry_date="2025-06-01",
                exit_date="2025-08-01", exit_reason="target",
                transaction_costs=0):
    return {
        "symbol": symbol,
        "instrument": instrument,
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "quantity": quantity,
        "pnl_gross": pnl,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "entry_date": entry_date,
        "exit_date": exit_date,
        "exit_reason": exit_reason,
        "transaction_costs": transaction_costs,
    }


class TestEquityGainTaxed:
    """Test 1: Equity gain taxed at 15%."""

    def test_stcg_15pct(self):
        trades = [_make_trade(pnl=10000, exit_date="2025-06-15")]
        report = generate_tax_report(trades, "2025-26")
        assert report["equity_stcg"]["tax_at_15pct"] == 1500.0


class TestFOGainTaxed:
    """Test 2: F&O gain taxed at 30%."""

    def test_fo_30pct(self):
        trades = [_make_trade(instrument="OPT", pnl=10000, exit_date="2025-08-01")]
        report = generate_tax_report(trades, "2025-26")
        assert report["fo_business"]["estimated_tax_at_30pct"] == 3000.0


class TestSTTDeducted:
    """Test 3: STT is computed and present in report."""

    def test_stt_present(self):
        trades = [_make_trade(entry_price=500, exit_price=550, quantity=100,
                              pnl=5000, exit_date="2025-07-01")]
        report = generate_tax_report(trades, "2025-26")
        assert report["total_stt_paid"] > 0


class TestLossesNetted:
    """Test 4: Losses correctly netted."""

    def test_losses_netted(self):
        trades = [
            _make_trade(pnl=5000, entry_date="2025-05-01", exit_date="2025-06-01"),
            _make_trade(pnl=-3000, pnl_pct=-3.0, entry_date="2025-06-01", exit_date="2025-07-01"),
        ]
        report = generate_tax_report(trades, "2025-26")
        assert report["equity_stcg"]["net"] == 2000.0
        # Tax only on net gain
        assert report["equity_stcg"]["tax_at_15pct"] == 300.0


class TestTurnoverCalculation:
    """Test 5: Turnover calculation for audit threshold."""

    def test_turnover(self):
        # F&O turnover = absolute profit values (for audit)
        trades = [
            _make_trade(instrument="OPT", pnl=50000, exit_date="2025-06-01",
                        entry_price=200, exit_price=300, quantity=500),
            _make_trade(instrument="OPT", pnl=-30000, exit_date="2025-07-01",
                        entry_price=200, exit_price=140, quantity=500),
        ]
        report = generate_tax_report(trades, "2025-26")
        # F&O turnover = |50000| + |-30000| = 80000
        assert report["fo_turnover"] == 80000.0
        assert report["audit_required_fo"] is False  # well below 2Cr


class TestEmptyTrades:
    """Test 6: Empty trades returns zero."""

    def test_empty(self):
        report = generate_tax_report([], "2025-26")
        assert report["equity_stcg"]["net"] == 0
        assert report["fo_business"]["net"] == 0
        assert report["total_stt_paid"] == 0
        assert report["trade_wise"] == []


class TestMixedTradesSeparated:
    """Test 7: Mixed equity + F&O trades separated correctly."""

    def test_mixed(self):
        trades = [
            _make_trade(symbol="RELIANCE", instrument="EQ", pnl=2000,
                        entry_date="2025-05-01", exit_date="2025-06-01"),
            _make_trade(symbol="NIFTY", instrument="OPT", pnl=3000,
                        entry_date="2025-06-01", exit_date="2025-07-01"),
            _make_trade(symbol="SBIN", instrument="EQ", pnl=-500,
                        entry_date="2025-07-01", exit_date="2025-08-01"),
            _make_trade(symbol="BANKNIFTY", instrument="SPREAD", pnl=1500,
                        entry_date="2025-08-01", exit_date="2025-09-01"),
        ]
        report = generate_tax_report(trades, "2025-26")

        assert report["equity_stcg"]["trades"] == 2
        assert report["equity_stcg"]["gains"] == 2000
        assert report["equity_stcg"]["losses"] == -500

        assert report["fo_business"]["trades"] == 2
        assert report["fo_business"]["gains"] == 4500  # 3000 + 1500


class TestStatementFormat:
    """Test 8: Statement format has required columns."""

    def test_columns_present(self):
        trades = [_make_trade(pnl=1000, exit_date="2025-06-15")]
        statement = generate_tax_pnl_statement(trades, "2025-26")

        assert "P&L STATEMENT" in statement
        assert "Symbol" in statement
        assert "Dir" in statement
        assert "Entry" in statement
        assert "Exit" in statement
        assert "Gross P&L" in statement
        assert "STT" in statement
        assert "Net P&L" in statement
        assert "Category" in statement
        assert "Tax Summary" in statement
