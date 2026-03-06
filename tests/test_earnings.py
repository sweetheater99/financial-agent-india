"""Tests for earnings calendar and macro event checking."""


def test_is_near_earnings_true():
    from paper_trade import is_near_earnings

    # RELIANCE earnings on 2026-04-18 (from earnings_calendar.json)
    assert is_near_earnings("RELIANCE", today="2026-04-17", blackout_days=2) is True
    assert is_near_earnings("RELIANCE", today="2026-04-18", blackout_days=2) is True


def test_is_near_earnings_false():
    from paper_trade import is_near_earnings

    assert is_near_earnings("RELIANCE", today="2026-03-01", blackout_days=2) is False


def test_is_near_earnings_unknown_symbol():
    from paper_trade import is_near_earnings

    assert is_near_earnings("UNKNOWN_XYZ", today="2026-04-17") is False


def test_is_near_earnings_day_before():
    from paper_trade import is_near_earnings

    # TCS earnings on 2026-04-10
    assert is_near_earnings("TCS", today="2026-04-09", blackout_days=2) is True


def test_is_near_earnings_too_far():
    from paper_trade import is_near_earnings

    # RELIANCE earnings on 2026-04-18, check 5 days before
    assert is_near_earnings("RELIANCE", today="2026-04-13", blackout_days=2) is False


def test_is_near_macro_event_true():
    from paper_trade import is_near_macro_event

    # Union Budget 2026-02-01
    assert is_near_macro_event(today="2026-01-31", blackout_days=2) is True
    assert is_near_macro_event(today="2026-02-01", blackout_days=2) is True


def test_is_near_macro_event_false():
    from paper_trade import is_near_macro_event

    assert is_near_macro_event(today="2026-03-10", blackout_days=2) is False


def test_is_near_macro_event_rbi():
    from paper_trade import is_near_macro_event

    # RBI MPC 2026-04-09
    assert is_near_macro_event(today="2026-04-08", blackout_days=2) is True


def test_is_near_macro_event_fomc():
    from paper_trade import is_near_macro_event

    # FOMC 2026-03-19
    assert is_near_macro_event(today="2026-03-18", blackout_days=2) is True
    assert is_near_macro_event(today="2026-03-19", blackout_days=2) is True


def test_is_near_macro_event_day_of():
    from paper_trade import is_near_macro_event

    # Expiry 2026-03-26
    assert is_near_macro_event(today="2026-03-26", blackout_days=2) is True


def test_is_near_macro_event_outside_window():
    from paper_trade import is_near_macro_event

    # Well after FOMC 2026-03-19
    assert is_near_macro_event(today="2026-03-22", blackout_days=2) is False
