"""
Tests for the fast options monitor.

Mocks SmartAPI / paper_trade functions to test exit logic in isolation.
"""

import json
from unittest.mock import MagicMock, patch, call

import pytest

from paper_trade import (
    PUT_TARGET_PCT,
    PUT_STOPLOSS_PCT,
    OPT_TRAILING_STOP_PCT,
    OPT_THETA_DTE_THRESHOLD,
    OPT_THETA_MIN_GAIN_PCT,
    MIN_DTE_TO_HOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_option_pos(
    symbol="RELIANCE",
    entry_price=100.0,
    strike=2400,
    expiry="27MAR2026",
    peak_premium=None,
    status="open",
    token="12345",
    quantity=75,
    allocated=7500.0,
    max_hold_date="2026-04-10",
    opt_partial_done=False,
    num_lots=2,
    lot_size=75,
    **kwargs,
):
    pos = {
        "symbol": symbol,
        "instrument": "OPT",
        "direction": "bearish",
        "entry_price": entry_price,
        "strike": strike,
        "expiry": expiry,
        "token": token,
        "contract_symbol": f"{symbol}27MAR26{strike}PE",
        "quantity": quantity,
        "allocated": allocated,
        "status": status,
        "entry_date": "2026-03-01",
        "max_hold_date": max_hold_date,
        "peak_premium": peak_premium if peak_premium is not None else entry_price,
        "opt_partial_done": opt_partial_done,
        "num_lots": num_lots,
        "lot_size": lot_size,
    }
    pos.update(kwargs)
    return pos


def _make_portfolio(positions=None):
    return {
        "capital": 100000,
        "available_capital": 50000.0,
        "positions": positions or [],
        "closed_trades": [],
        "stats": {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "total_costs": 0.0,
            "best_trade": None,
            "worst_trade": None,
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOptionsMonitor:
    """Test the monitor_options function."""

    @patch("options_monitor.save_portfolio")
    @patch("options_monitor.load_portfolio")
    def test_no_option_positions_returns_zero(self, mock_load, mock_save):
        """No OPT positions -> 0 exits, no save."""
        mock_load.return_value = _make_portfolio([
            {"status": "open", "instrument": "EQ", "symbol": "TCS"},
        ])
        from options_monitor import monitor_options
        result = monitor_options(MagicMock())
        assert result == 0

    @patch("options_monitor._telegram_notify_exit")
    @patch("options_monitor.save_portfolio")
    @patch("options_monitor.get_ltp_nfo")
    @patch("options_monitor.days_to_expiry", return_value=20)
    @patch("options_monitor._today_ist", return_value="2026-03-06")
    @patch("options_monitor.load_portfolio")
    def test_stoploss_hit(self, mock_load, mock_today, mock_dte,
                          mock_ltp, mock_save, mock_tg):
        """Premium drops below fixed stoploss -> exit."""
        pos = _make_option_pos(entry_price=100.0)
        # PUT_STOPLOSS_PCT = -40, so fixed_sl = 100 * (1 + (-40)/100) = 60
        # Set LTP below 60
        mock_ltp.return_value = 55.0
        mock_load.return_value = _make_portfolio([pos])

        from options_monitor import monitor_options
        result = monitor_options(MagicMock())
        assert result == 1
        mock_save.assert_called_once()
        mock_tg.assert_called_once()

    @patch("options_monitor._telegram_notify_exit")
    @patch("options_monitor.save_portfolio")
    @patch("options_monitor.get_ltp_nfo")
    @patch("options_monitor.days_to_expiry", return_value=20)
    @patch("options_monitor._today_ist", return_value="2026-03-06")
    @patch("options_monitor.load_portfolio")
    def test_target_hit(self, mock_load, mock_today, mock_dte,
                        mock_ltp, mock_save, mock_tg):
        """Premium rises above target -> exit."""
        pos = _make_option_pos(entry_price=100.0, num_lots=1)
        # PUT_TARGET_PCT = 50, so target at 150
        mock_ltp.return_value = 155.0
        mock_load.return_value = _make_portfolio([pos])

        from options_monitor import monitor_options
        result = monitor_options(MagicMock())
        assert result == 1

    @patch("options_monitor._telegram_notify_exit")
    @patch("options_monitor.save_portfolio")
    @patch("options_monitor.get_ltp_nfo")
    @patch("options_monitor.days_to_expiry", return_value=20)
    @patch("options_monitor._today_ist", return_value="2026-03-06")
    @patch("options_monitor.load_portfolio")
    def test_premium_crash_emergency(self, mock_load, mock_today, mock_dte,
                                     mock_ltp, mock_save, mock_tg):
        """Premium crashes >60% -> emergency exit."""
        pos = _make_option_pos(entry_price=100.0)
        # -60% means LTP <= 40
        mock_ltp.return_value = 35.0
        mock_load.return_value = _make_portfolio([pos])

        from options_monitor import monitor_options
        result = monitor_options(MagicMock())
        assert result == 1
        # Check the reason in the close call
        mock_save.assert_called_once()

    @patch("options_monitor._telegram_notify_exit")
    @patch("options_monitor.save_portfolio")
    @patch("options_monitor.get_ltp_nfo")
    @patch("options_monitor.days_to_expiry", return_value=20)
    @patch("options_monitor._today_ist", return_value="2026-03-06")
    @patch("options_monitor.load_portfolio")
    def test_trailing_stop_trigger(self, mock_load, mock_today, mock_dte,
                                   mock_ltp, mock_save, mock_tg):
        """Peak premium was high, current drops below trailing SL."""
        pos = _make_option_pos(entry_price=100.0, peak_premium=200.0)
        # trailing_sl = 200 * (1 - 25/100) = 150
        # fixed_sl = 100 * (1 - 0.4) = 60
        # effective_sl = max(150, 60) = 150
        # LTP below 150 -> trailing_stop
        mock_ltp.return_value = 145.0
        mock_load.return_value = _make_portfolio([pos])

        from options_monitor import monitor_options
        result = monitor_options(MagicMock())
        assert result == 1

    @patch("options_monitor.save_portfolio")
    @patch("options_monitor.get_ltp_nfo")
    @patch("options_monitor.days_to_expiry", return_value=20)
    @patch("options_monitor._today_ist", return_value="2026-03-06")
    @patch("options_monitor.load_portfolio")
    def test_peak_premium_updated(self, mock_load, mock_today, mock_dte,
                                  mock_ltp, mock_save):
        """If LTP > peak, peak_premium should be updated."""
        pos = _make_option_pos(entry_price=100.0, peak_premium=120.0)
        mock_ltp.return_value = 130.0  # new high, no exit yet
        mock_load.return_value = _make_portfolio([pos])

        from options_monitor import monitor_options
        result = monitor_options(MagicMock())
        assert result == 0
        # peak should be updated
        assert pos["peak_premium"] == 130.0

    @patch("options_monitor._telegram_notify_exit")
    @patch("options_monitor.save_portfolio")
    @patch("options_monitor.get_ltp_nfo")
    @patch("options_monitor.days_to_expiry")
    @patch("options_monitor._today_ist", return_value="2026-03-06")
    @patch("options_monitor.load_portfolio")
    def test_multiple_positions_independent(self, mock_load, mock_today, mock_dte,
                                            mock_ltp, mock_save, mock_tg):
        """Two option positions: one exits, one holds."""
        pos1 = _make_option_pos(symbol="RELIANCE", entry_price=100.0, token="111")
        pos2 = _make_option_pos(symbol="HDFCBANK", entry_price=80.0, token="222")

        # pos1: crashes, pos2: healthy
        mock_ltp.side_effect = [35.0, 90.0]  # pos1 crash, pos2 OK
        mock_dte.return_value = 20
        mock_load.return_value = _make_portfolio([pos1, pos2])

        from options_monitor import monitor_options
        result = monitor_options(MagicMock())
        assert result == 1  # only pos1 exits

    @patch("options_monitor.save_portfolio")
    @patch("options_monitor.get_ltp_nfo")
    @patch("options_monitor.days_to_expiry", return_value=20)
    @patch("options_monitor._today_ist", return_value="2026-03-06")
    @patch("options_monitor.load_portfolio")
    def test_no_exit_healthy_premium(self, mock_load, mock_today, mock_dte,
                                     mock_ltp, mock_save):
        """Premium in healthy range -> no exit."""
        pos = _make_option_pos(entry_price=100.0)
        mock_ltp.return_value = 110.0  # +10%, within bounds
        mock_load.return_value = _make_portfolio([pos])

        from options_monitor import monitor_options
        result = monitor_options(MagicMock())
        assert result == 0

    @patch("options_monitor.save_portfolio")
    @patch("options_monitor.get_ltp_nfo")
    @patch("options_monitor.days_to_expiry", return_value=20)
    @patch("options_monitor._today_ist", return_value="2026-03-06")
    @patch("options_monitor.load_portfolio")
    def test_ltp_fetch_failure_skips(self, mock_load, mock_today, mock_dte,
                                     mock_ltp, mock_save):
        """If LTP returns None, position is skipped."""
        pos = _make_option_pos(entry_price=100.0)
        mock_ltp.return_value = None
        mock_load.return_value = _make_portfolio([pos])

        from options_monitor import monitor_options
        result = monitor_options(MagicMock())
        assert result == 0

    @patch("options_monitor._telegram_notify_exit")
    @patch("options_monitor.save_portfolio")
    @patch("options_monitor.get_ltp_nfo")
    @patch("options_monitor.days_to_expiry", return_value=1)
    @patch("options_monitor._today_ist", return_value="2026-03-06")
    @patch("options_monitor.load_portfolio")
    def test_dte_forced_exit(self, mock_load, mock_today, mock_dte,
                             mock_ltp, mock_save, mock_tg):
        """DTE < MIN_DTE_TO_HOLD -> forced exit."""
        pos = _make_option_pos(entry_price=100.0)
        mock_ltp.return_value = 105.0  # small gain, but DTE too low
        mock_load.return_value = _make_portfolio([pos])

        from options_monitor import monitor_options
        result = monitor_options(MagicMock())
        assert result == 1

    @patch("options_monitor._telegram_notify_exit")
    @patch("options_monitor.save_portfolio")
    @patch("options_monitor.get_ltp_nfo")
    @patch("options_monitor.days_to_expiry", return_value=3)
    @patch("options_monitor._today_ist", return_value="2026-03-06")
    @patch("options_monitor.load_portfolio")
    def test_theta_decay_low_dte_low_gain(self, mock_load, mock_today, mock_dte,
                                          mock_ltp, mock_save, mock_tg):
        """DTE < threshold and gain < minimum -> theta_decay exit."""
        pos = _make_option_pos(entry_price=100.0)
        # DTE=3 < OPT_THETA_DTE_THRESHOLD=5, pnl_pct=5% < OPT_THETA_MIN_GAIN_PCT=10%
        mock_ltp.return_value = 105.0
        mock_load.return_value = _make_portfolio([pos])

        from options_monitor import monitor_options
        result = monitor_options(MagicMock())
        assert result == 1

    @patch("options_monitor.save_portfolio")
    @patch("options_monitor.get_ltp_nfo")
    @patch("options_monitor.days_to_expiry", return_value=3)
    @patch("options_monitor._today_ist", return_value="2026-03-06")
    @patch("options_monitor.load_portfolio")
    def test_theta_decay_skipped_with_high_gain(self, mock_load, mock_today, mock_dte,
                                                mock_ltp, mock_save):
        """DTE < threshold but gain >= minimum -> hold (no theta exit)."""
        pos = _make_option_pos(entry_price=100.0)
        # DTE=3, pnl_pct=15% > OPT_THETA_MIN_GAIN_PCT=10% -> no theta exit
        mock_ltp.return_value = 115.0
        mock_load.return_value = _make_portfolio([pos])

        from options_monitor import monitor_options
        result = monitor_options(MagicMock())
        assert result == 0

    @patch("options_monitor._telegram_notify_exit")
    @patch("options_monitor.save_portfolio")
    @patch("options_monitor.get_ltp_nfo")
    @patch("options_monitor.days_to_expiry", return_value=20)
    @patch("options_monitor._today_ist", return_value="2026-03-06")
    @patch("options_monitor.load_portfolio")
    def test_session_expired_retry(self, mock_load, mock_today, mock_dte,
                                   mock_ltp, mock_save, mock_tg):
        """SessionExpiredError on first try -> refresh -> retry succeeds."""
        from paper_trade import SessionExpiredError
        pos = _make_option_pos(entry_price=100.0)
        # First call raises, second succeeds with crash price
        mock_ltp.side_effect = [SessionExpiredError("test"), 35.0]
        mock_load.return_value = _make_portfolio([pos])

        smart_api = MagicMock()
        from options_monitor import monitor_options
        result = monitor_options(smart_api)
        assert result == 1

    @patch("options_monitor.save_portfolio")
    @patch("options_monitor.get_ltp_nfo")
    @patch("options_monitor.days_to_expiry", return_value=20)
    @patch("options_monitor._today_ist", return_value="2026-03-06")
    @patch("options_monitor.load_portfolio")
    def test_equity_positions_skipped(self, mock_load, mock_today, mock_dte,
                                      mock_ltp, mock_save):
        """Equity and SPREAD positions are completely ignored."""
        portfolio = _make_portfolio([
            {"status": "open", "instrument": "EQ", "symbol": "TCS", "entry_price": 100},
            {"status": "open", "instrument": "SPREAD", "symbol": "NIFTY", "entry_price": 50},
            {"status": "open", "instrument": "CONDOR", "symbol": "NIFTY", "entry_price": 30},
            {"status": "open", "instrument": "MOMENTUM", "symbol": "NIFTY", "entry_price": 20},
        ])
        mock_load.return_value = portfolio

        from options_monitor import monitor_options
        result = monitor_options(MagicMock())
        assert result == 0
        # get_ltp_nfo should never be called
        mock_ltp.assert_not_called()
