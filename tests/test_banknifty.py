"""Tests for BankNifty support."""


def test_banknifty_config():
    from config import BANKNIFTY_TOKEN, BANKNIFTY_LOT_SIZE, BANKNIFTY_VIX_MULTIPLIER
    assert BANKNIFTY_TOKEN == "99926009"
    assert BANKNIFTY_LOT_SIZE == 30
    assert BANKNIFTY_VIX_MULTIPLIER == 1.3


def test_banknifty_vix_scaling():
    """BankNifty VIX thresholds should be 1.3x Nifty thresholds."""
    from config import CONDOR_MIN_VIX, CONDOR_MAX_VIX, BANKNIFTY_VIX_MULTIPLIER
    bn_min = CONDOR_MIN_VIX * BANKNIFTY_VIX_MULTIPLIER
    bn_max = CONDOR_MAX_VIX * BANKNIFTY_VIX_MULTIPLIER
    assert bn_min > CONDOR_MIN_VIX
    assert bn_max > CONDOR_MAX_VIX


def test_banknifty_wider_otm():
    """BankNifty strangles should use wider OTM points than Nifty."""
    from config import WEEKLY_THETA_OTM_POINTS_NIFTY, WEEKLY_THETA_OTM_POINTS_BANKNIFTY
    assert WEEKLY_THETA_OTM_POINTS_BANKNIFTY > WEEKLY_THETA_OTM_POINTS_NIFTY
