"""Tests for iron condor strike selection and exit logic."""


def test_select_iron_condor_strikes_valid():
    from agent_with_options import select_iron_condor_strikes

    chain = [
        # Put side (below spot)
        {"strikePrice": 22000, "PE": {"openInterest": 80000, "lastPrice": 30, "delta": -0.08},
         "CE": {"openInterest": 100, "lastPrice": 550, "delta": 0.95}},
        {"strikePrice": 22100, "PE": {"openInterest": 100000, "lastPrice": 45, "delta": -0.12},
         "CE": {"openInterest": 200, "lastPrice": 480, "delta": 0.92}},
        {"strikePrice": 22200, "PE": {"openInterest": 120000, "lastPrice": 65, "delta": -0.18},
         "CE": {"openInterest": 300, "lastPrice": 420, "delta": 0.88}},
        # Near spot
        {"strikePrice": 22500, "PE": {"openInterest": 150000, "lastPrice": 180, "delta": -0.45},
         "CE": {"openInterest": 150000, "lastPrice": 180, "delta": 0.55}},
        # Call side (above spot)
        {"strikePrice": 22800, "CE": {"openInterest": 120000, "lastPrice": 60, "delta": 0.18},
         "PE": {"openInterest": 300, "lastPrice": 420, "delta": -0.88}},
        {"strikePrice": 22900, "CE": {"openInterest": 100000, "lastPrice": 40, "delta": 0.12},
         "PE": {"openInterest": 200, "lastPrice": 480, "delta": -0.92}},
        {"strikePrice": 23000, "CE": {"openInterest": 80000, "lastPrice": 25, "delta": 0.08},
         "PE": {"openInterest": 100, "lastPrice": 550, "delta": -0.95}},
    ]
    result = select_iron_condor_strikes(chain, spot=22500, lot_size=75, max_loss_budget=30000,
                                        min_credit_per_lot=30)
    assert result is not None
    assert result["call_short_strike"] > 22500  # above spot
    assert result["put_short_strike"] < 22500  # below spot
    assert result["call_long_strike"] > result["call_short_strike"]  # protection above
    assert result["put_long_strike"] < result["put_short_strike"]  # protection below
    assert result["net_credit"] > 0
    assert result["max_loss"] <= 30000


def test_select_iron_condor_none_insufficient_strikes():
    from agent_with_options import select_iron_condor_strikes

    chain = [
        {"strikePrice": 22500, "PE": {"openInterest": 150000, "lastPrice": 180, "delta": -0.45},
         "CE": {"openInterest": 150000, "lastPrice": 180, "delta": 0.55}},
    ]
    result = select_iron_condor_strikes(chain, spot=22500, lot_size=75, max_loss_budget=3000)
    assert result is None


def test_select_iron_condor_none_empty_chain():
    from agent_with_options import select_iron_condor_strikes

    result = select_iron_condor_strikes([], spot=22500, lot_size=75, max_loss_budget=3000)
    assert result is None


def test_select_iron_condor_none_low_oi():
    from agent_with_options import select_iron_condor_strikes

    chain = [
        {"strikePrice": 22200, "PE": {"openInterest": 10, "lastPrice": 65},
         "CE": {"openInterest": 10, "lastPrice": 420}},
        {"strikePrice": 22500, "PE": {"openInterest": 10, "lastPrice": 180},
         "CE": {"openInterest": 10, "lastPrice": 180}},
        {"strikePrice": 22800, "CE": {"openInterest": 10, "lastPrice": 60},
         "PE": {"openInterest": 10, "lastPrice": 420}},
    ]
    result = select_iron_condor_strikes(chain, spot=22500, lot_size=75, max_loss_budget=3000, min_oi=500)
    assert result is None


def test_condor_exit_target():
    from paper_trade import check_condor_exit

    pos = {
        "instrument": "CONDOR",
        "net_credit": 3000,
        "max_profit": 3000,
        "max_loss": 4500,
        "quantity": 75,
        "entry_date": "2026-03-01",
        "max_hold_date": "2026-03-20",
        "expiry": "26MAR2026",
    }
    # cost_to_close = (15-5 + 15-5) * 75 = 1500
    # pnl = 3000 - 1500 = 1500 = 50% of credit -> target
    reason = check_condor_exit(pos, call_short_prem=15, call_long_prem=5,
                                put_short_prem=15, put_long_prem=5)
    assert reason == "condor_target"


def test_condor_exit_stoploss():
    from paper_trade import check_condor_exit

    pos = {
        "instrument": "CONDOR",
        "net_credit": 3000,
        "max_profit": 3000,
        "max_loss": 4500,
        "quantity": 75,
        "entry_date": "2026-03-01",
        "max_hold_date": "2026-03-20",
        "expiry": "26MAR2026",
    }
    # One side went deep ITM
    # cost_to_close = (200-50 + 15-5) * 75 = 12000
    # pnl = 3000 - 12000 = -9000 > 2x credit -> stoploss
    reason = check_condor_exit(pos, call_short_prem=200, call_long_prem=50,
                                put_short_prem=15, put_long_prem=5)
    assert reason == "condor_stoploss"


def test_condor_exit_vix_spike():
    from paper_trade import check_condor_exit

    pos = {
        "instrument": "CONDOR",
        "net_credit": 3000,
        "max_profit": 3000,
        "max_loss": 4500,
        "quantity": 75,
        "entry_date": "2026-03-01",
        "max_hold_date": "2026-03-20",
        "expiry": "26MAR2026",
    }
    reason = check_condor_exit(pos, call_short_prem=30, call_long_prem=10,
                                put_short_prem=30, put_long_prem=10,
                                vix=20, prev_vix=15)  # 33% spike
    assert reason == "condor_vix_spike"


def test_condor_no_exit():
    from paper_trade import check_condor_exit

    pos = {
        "instrument": "CONDOR",
        "net_credit": 3000,
        "max_profit": 3000,
        "max_loss": 4500,
        "quantity": 75,
        "entry_date": "2026-03-01",
        "max_hold_date": "2026-03-20",
        "expiry": "26MAR2026",
    }
    reason = check_condor_exit(pos, call_short_prem=50, call_long_prem=20,
                                put_short_prem=50, put_long_prem=20)
    assert reason is None


def test_open_iron_condor():
    from paper_trade import _open_iron_condor

    portfolio = {
        "capital": 100000,
        "available_capital": 50000,
        "positions": [],
        "closed_trades": [],
        "stats": {"total_trades": 0, "winning_trades": 0, "losing_trades": 0,
                  "total_pnl": 0, "total_pnl_pct": 0, "total_costs": 0,
                  "best_trade": None, "worst_trade": None},
    }

    condor_data = {
        "call_short_strike": 22800,
        "call_long_strike": 22900,
        "put_short_strike": 22200,
        "put_long_strike": 22100,
        "call_short_premium": 60,
        "call_long_premium": 40,
        "put_short_premium": 65,
        "put_long_premium": 45,
        "net_credit": 3000,
        "max_loss": 4500,
        "width": 100,
        "lot_size": 75,
    }

    pos = _open_iron_condor(portfolio, condor_data, 5000, "SIDEWAYS", 55, "26MAR2026")

    assert pos is not None
    assert pos["strategy"] == "iron_condor"
    assert pos["instrument"] == "CONDOR"
    assert pos["status"] == "open"
    assert pos["call_short"]["strike"] == 22800
    assert pos["put_long"]["strike"] == 22100
    assert pos["net_credit"] > 0
    assert len(portfolio["positions"]) == 1
    assert portfolio["available_capital"] == 45000
