# tests/test_v7_plan2_integration.py
"""Integration tests: Strategist + Risk Engine + Level Memory working together."""
import json
import pytest
from datetime import date
from pathlib import Path
from v7.level_memory import LevelMemory
from v7.risk_engine import RiskEngine
from v7.strategist import parse_playbook_response, build_premarket_prompt
from v7.types import Conviction, Position, PacingStatus


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path / "v7_state"


@pytest.fixture
def level_mem(tmp_dir):
    lm = LevelMemory(state_dir=tmp_dir)
    lm.add_level("NIFTY", 24000.0, "support", "tested 3x")
    lm.add_level("NIFTY", 24500.0, "resistance", "OI wall")
    lm.update_oi_walls("NIFTY", 24500, 24000, 1.1)
    lm.add_level("HDFCBANK", 1600.0, "support", "round number")
    return lm


@pytest.fixture
def risk_eng(tmp_dir):
    return RiskEngine(capital=300_000, state_dir=tmp_dir)


def test_level_memory_feeds_into_strategist_prompt(level_mem):
    ctx = level_mem.to_strategist_context(["NIFTY", "HDFCBANK"])
    prompt = build_premarket_prompt(
        us_close={}, gift_nifty="24250", prev_vix=17.0,
        fii_dii="", events_today=[], events_this_week=[],
        level_memory=ctx,
        edge_tracker={},
        risk_state={"mtd_pnl_pct": 0, "pacing": "on_track", "survival_mode": False},
        fo_ban_list=[], recent_lessons=[],
    )
    assert "24000" in prompt
    assert "24500" in prompt
    assert "1600" in prompt


def test_risk_engine_gates_playbook_setups(risk_eng):
    raw = json.dumps({
        "date": "2026-03-11",
        "day_classification": "LIKELY_TREND_UP",
        "nifty_plan": {
            "bias": "bullish", "key_levels": {},
            "setups": [
                {"id": "N1", "priority": 1, "type": "breakout_long",
                 "trigger": "close above 24350", "instrument": "NIFTY CE",
                 "strike_logic": "delta 0.45", "target": 24500,
                 "stoploss": 24280, "max_risk_pct": 1.5, "conviction": "high"},
            ],
            "no_trade_zone": "",
        },
        "stock_plans": [],
        "risk_budget": {"max_capital_at_risk_today_pct": 4.0,
                        "max_trades_today": 4, "max_per_trade_risk_pct": 1.5,
                        "survival_mode": False},
        "no_trade_conditions": [], "carry_rules": {},
        "theta_plan": {"action": "hold"}, "market_context": {},
    })
    pb = parse_playbook_response(raw, today=date(2026, 3, 11))
    setup = pb.nifty_setups[0]

    allowed, reason, risk_amount = risk_eng.pre_trade_check(
        symbol=setup.symbol,
        conviction=setup.conviction,
        direction="bullish",
        trade_value=5000.0,
        open_positions=[],
        current_risk=0.0,
    )
    assert allowed is True
    assert risk_amount == 6000.0


def test_risk_engine_blocks_when_daily_limit_hit(risk_eng):
    risk_eng.record_daily_pnl(-7000.0)
    allowed, reason, _ = risk_eng.pre_trade_check(
        symbol="NIFTY", conviction=Conviction.MEDIUM,
        direction="bullish", trade_value=5000.0,
        open_positions=[], current_risk=0.0,
    )
    assert allowed is False


def test_survival_mode_blocks_directional_allows_theta(risk_eng):
    risk_eng.update_mtd_pnl(-16000.0)
    allowed, reason = risk_eng.can_open_trade()
    assert allowed is False
    theta_ok, _ = risk_eng.can_open_theta()
    assert theta_ok is True


def test_fo_ban_integrated_with_pre_trade(risk_eng):
    risk_eng.update_fo_ban_list(["TATAMOTORS"])
    allowed, reason, _ = risk_eng.pre_trade_check(
        symbol="TATAMOTORS", conviction=Conviction.MEDIUM,
        direction="bullish", trade_value=5000.0,
        open_positions=[], current_risk=0.0,
    )
    assert allowed is False
    assert "ban" in reason.lower()


def test_risk_state_feeds_into_prompt(risk_eng):
    risk_eng.record_daily_pnl(-2000.0)
    risk_eng.record_sl_hit()
    summary = risk_eng.get_state_summary()
    prompt = build_premarket_prompt(
        us_close={}, gift_nifty="", prev_vix=17.0,
        fii_dii="", events_today=[], events_this_week=[],
        level_memory={}, edge_tracker={},
        risk_state=summary,
        fo_ban_list=[], recent_lessons=[],
    )
    assert str(summary["mtd_pnl_pct"]) in prompt or "0.0" in prompt
