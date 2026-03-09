"""Tests for V6 config: kill switch flag and safety-critical constants."""

import config


def test_v6_flag_exists():
    """V6_CLAUDE_FIRST flag must exist and be a boolean."""
    assert hasattr(config, "V6_CLAUDE_FIRST")
    assert isinstance(config.V6_CLAUDE_FIRST, bool)
    assert config.V6_CLAUDE_FIRST is True  # default on


def test_kept_constants_exist():
    """Safety-critical constants must survive the V6 cleanup."""
    # Circuit breaker
    assert hasattr(config, "DAILY_LOSS_CIRCUIT_BREAKER_PCT")
    assert config.DAILY_LOSS_CIRCUIT_BREAKER_PCT > 0

    # Portfolio guardrails
    assert hasattr(config, "CASH_RESERVE_PCT")
    assert hasattr(config, "MAX_STRATEGY_ALLOC_PCT")
    assert hasattr(config, "MAX_CONCURRENT_POSITIONS")
    assert hasattr(config, "MAX_SAME_SECTOR")
    assert hasattr(config, "MAX_PORTFOLIO_HEAT_PCT")

    # SL / target / trailing ATR constants
    assert hasattr(config, "TRAILING_SL_ACTIVATION_PCT")
    assert hasattr(config, "TRAILING_SL_ATR_MULT")
    assert hasattr(config, "SPREAD_SL_ATR_MULT")
    assert hasattr(config, "SPREAD_TARGET_ATR_MULT")

    # API delay
    assert hasattr(config, "API_DELAY")
    assert config.API_DELAY > 0

    # Smart monitoring intervals
    assert hasattr(config, "HOT_CHECK_INTERVAL_MIN")
    assert hasattr(config, "WARM_CHECK_INTERVAL_MIN")
    assert hasattr(config, "COLD_CHECK_INTERVAL_MIN")
