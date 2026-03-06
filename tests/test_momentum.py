"""Tests for momentum options and Nifty breakout detection."""


def test_nifty_breakout_bullish():
    from paper_trade import check_nifty_breakout

    # 21 candles: first 20 have high of 22300, then today closes at 22350
    candles = [
        [f"2026-01-{i:02d}", 22000 + i * 10, 22100 + i * 10, 21900 + i * 10, 22050 + i * 10, 1000]
        for i in range(1, 21)
    ]
    # Today: breaks above the 20-day high
    candles.append(["2026-01-21", 22300, 22400, 22250, 22350, 2000])
    # 20-day high from lookback = max of highs from candles[0:20]
    result = check_nifty_breakout(candles, period=20)
    assert result == "bullish"


def test_nifty_breakout_bearish():
    from paper_trade import check_nifty_breakout

    candles = [
        [f"2026-01-{i:02d}", 22000 - i * 5, 22050 - i * 5, 21950 - i * 5, 22000 - i * 5, 1000]
        for i in range(1, 21)
    ]
    # Today: breaks below 20-day low
    lowest_low = min(c[3] for c in candles)
    candles.append(["2026-01-21", lowest_low - 50, lowest_low - 30, lowest_low - 100, lowest_low - 80, 2000])
    result = check_nifty_breakout(candles, period=20)
    assert result == "bearish"


def test_nifty_breakout_none():
    from paper_trade import check_nifty_breakout

    candles = [[f"2026-01-{i:02d}", 22000, 22100, 21900, 22050, 1000] for i in range(1, 22)]
    result = check_nifty_breakout(candles, period=20)
    assert result is None


def test_nifty_breakout_insufficient_data():
    from paper_trade import check_nifty_breakout

    candles = [[f"2026-01-{i:02d}", 22000, 22100, 21900, 22050, 1000] for i in range(1, 5)]
    result = check_nifty_breakout(candles, period=20)
    assert result is None


def test_nifty_breakout_empty():
    from paper_trade import check_nifty_breakout

    assert check_nifty_breakout(None, period=20) is None
    assert check_nifty_breakout([], period=20) is None


def test_momentum_exit_target():
    from paper_trade import check_momentum_exit

    pos = {
        "entry_price": 100,
        "target_price": 190,  # 90% target
        "stoploss_price": 65,  # 35% SL
        "max_hold_date": "2026-03-10",
        "peak_premium": 195,
    }
    reason = check_momentum_exit(pos, current_premium=195, today="2026-03-05")
    assert reason == "target"


def test_momentum_exit_stoploss():
    from paper_trade import check_momentum_exit

    pos = {
        "entry_price": 100,
        "target_price": 190,
        "stoploss_price": 65,
        "max_hold_date": "2026-03-10",
        "peak_premium": 100,
    }
    reason = check_momentum_exit(pos, current_premium=60, today="2026-03-05")
    assert reason == "stoploss"


def test_momentum_exit_time():
    from paper_trade import check_momentum_exit

    pos = {
        "entry_price": 100,
        "target_price": 190,
        "stoploss_price": 65,
        "max_hold_date": "2026-03-06",
        "peak_premium": 110,
    }
    # Past max hold date, profit < 15%
    reason = check_momentum_exit(pos, current_premium=110, today="2026-03-07")
    assert reason == "time_exit"


def test_momentum_exit_trailing_stop():
    from paper_trade import check_momentum_exit

    pos = {
        "entry_price": 100,
        "target_price": 190,
        "stoploss_price": 65,
        "max_hold_date": "2026-03-10",
        "peak_premium": 150,  # 50% gain from entry, peak > 1.3 * entry
    }
    # Current premium dropped to 75% of 150 = 112.5; test just below trail
    reason = check_momentum_exit(pos, current_premium=110, today="2026-03-05")
    assert reason == "trailing_stop"


def test_momentum_no_exit():
    from paper_trade import check_momentum_exit

    pos = {
        "entry_price": 100,
        "target_price": 190,
        "stoploss_price": 65,
        "max_hold_date": "2026-03-10",
        "peak_premium": 120,
    }
    reason = check_momentum_exit(pos, current_premium=120, today="2026-03-05")
    assert reason is None


def test_momentum_exit_zero_entry():
    from paper_trade import check_momentum_exit

    pos = {
        "entry_price": 0,
        "target_price": 190,
        "stoploss_price": 65,
        "max_hold_date": "2026-03-10",
        "peak_premium": 0,
    }
    reason = check_momentum_exit(pos, current_premium=100, today="2026-03-05")
    assert reason is None


def test_open_momentum_position():
    from paper_trade import _open_momentum_position

    portfolio = {
        "capital": 100000,
        "available_capital": 50000,
        "positions": [],
        "closed_trades": [],
        "stats": {"total_trades": 0, "winning_trades": 0, "losing_trades": 0,
                  "total_pnl": 0, "total_pnl_pct": 0, "total_costs": 0,
                  "best_trade": None, "worst_trade": None},
    }

    option_data = {"premium": 100, "strike": 22500, "lot_size": 75}

    pos = _open_momentum_position(portfolio, "bullish", option_data, 8000,
                                   "TRENDING_UP", 45, "26MAR2026")

    assert pos is not None
    assert pos["strategy"] == "momentum"
    assert pos["instrument"] == "MOMENTUM"
    assert pos["direction"] == "bullish"
    assert pos["option_type"] == "CE"
    assert pos["strike"] == 22500
    assert pos["entry_price"] > 100  # slippage applied
    assert pos["target_price"] > pos["entry_price"]
    assert pos["stoploss_price"] < pos["entry_price"]
    assert pos["status"] == "open"
    assert len(portfolio["positions"]) == 1
    assert portfolio["available_capital"] == 42000


def test_open_momentum_position_bearish():
    from paper_trade import _open_momentum_position

    portfolio = {
        "capital": 100000,
        "available_capital": 50000,
        "positions": [],
        "closed_trades": [],
        "stats": {"total_trades": 0, "winning_trades": 0, "losing_trades": 0,
                  "total_pnl": 0, "total_pnl_pct": 0, "total_costs": 0,
                  "best_trade": None, "worst_trade": None},
    }

    option_data = {"premium": 80, "strike": 22500, "lot_size": 75}

    pos = _open_momentum_position(portfolio, "bearish", option_data, 6000,
                                   "TRENDING_DOWN", 40, "26MAR2026")

    assert pos["option_type"] == "PE"
    assert pos["direction"] == "bearish"
