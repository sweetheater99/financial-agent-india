# tests/test_v7_strategist.py
"""Tests for V7 Strategist — Claude-powered playbook generation."""
import json
import pytest
from datetime import date, time
from pathlib import Path
from unittest.mock import MagicMock, patch
from v7.strategist import Strategist, build_premarket_prompt, parse_playbook_response
from v7.types import (
    Playbook, Setup, SetupType, DayClassification, Conviction,
    RiskBudget, PacingStatus, CarryRules,
)


# ── Prompt Building ─────────────────────────────────────────────────────


def test_build_premarket_prompt_contains_market_data():
    prompt = build_premarket_prompt(
        us_close={"sp500": "+0.3%", "nasdaq": "+0.5%", "dow": "+0.2%"},
        gift_nifty="24250 (+0.2%)",
        prev_vix=17.8,
        fii_dii="FII -1200cr, DII +800cr",
        events_today=[],
        events_this_week=["RBI policy Thu"],
        level_memory={"NIFTY": {"levels": [{"price": 24000, "type": "support", "strength": 3}], "oi_walls": {}}},
        edge_tracker={"overall_win_rate": 0.55, "by_strategy": {}},
        risk_state={"mtd_pnl_pct": 2.1, "pacing": "on_track", "survival_mode": False},
        fo_ban_list=["DELTACORP"],
        recent_lessons=["HDFCBANK breakout failed — check sector first"],
    )
    assert "24250" in prompt
    assert "RBI" in prompt
    assert "DELTACORP" in prompt
    assert "24000" in prompt
    assert "support" in prompt.lower()
    assert "sector first" in prompt


def test_build_premarket_prompt_includes_risk_state():
    prompt = build_premarket_prompt(
        us_close={}, gift_nifty="", prev_vix=17.0,
        fii_dii="", events_today=[], events_this_week=[],
        level_memory={}, edge_tracker={},
        risk_state={"mtd_pnl_pct": -3.5, "pacing": "behind", "survival_mode": False},
        fo_ban_list=[], recent_lessons=[],
    )
    assert "-3.5" in prompt
    assert "behind" in prompt


def test_build_premarket_prompt_survival_mode_note():
    prompt = build_premarket_prompt(
        us_close={}, gift_nifty="", prev_vix=17.0,
        fii_dii="", events_today=[], events_this_week=[],
        level_memory={}, edge_tracker={},
        risk_state={"mtd_pnl_pct": -5.5, "pacing": "survival", "survival_mode": True},
        fo_ban_list=[], recent_lessons=[],
    )
    assert "survival" in prompt.lower()
    assert "theta only" in prompt.lower() or "no directional" in prompt.lower()


# ── Response Parsing ────────────────────────────────────────────────────


def test_parse_playbook_valid_json():
    raw = json.dumps({
        "date": "2026-03-11",
        "day_classification": "LIKELY_TREND_UP",
        "nifty_plan": {
            "bias": "bullish",
            "key_levels": {"resistance_1": 24350, "support_1": 24150},
            "setups": [
                {
                    "id": "N1", "priority": 1, "type": "breakout_long",
                    "trigger": "15-min close above 24350",
                    "instrument": "NIFTY CE", "strike_logic": "delta 0.45",
                    "target": 24500, "stoploss": 24280,
                    "max_risk_pct": 1.5, "conviction": "high",
                },
            ],
            "no_trade_zone": "24200-24300",
        },
        "stock_plans": [
            {
                "id": "H1", "priority": 3, "symbol": "HDFCBANK",
                "type": "breakout_long",
                "trigger": "15-min close above 1625",
                "instrument": "HDFCBANK CE", "strike_logic": "delta 0.45",
                "target": 1660, "stoploss": 1610,
                "max_risk_pct": 1.0, "conviction": "medium",
            },
        ],
        "risk_budget": {
            "max_capital_at_risk_today_pct": 4.0,
            "max_trades_today": 4,
            "max_per_trade_risk_pct": 1.5,
            "survival_mode": False,
        },
        "no_trade_conditions": ["VIX > 22"],
        "carry_rules": {
            "carry_if": "profit > 1.5%, VIX < 20, DTE > 3",
        },
        "theta_plan": {"action": "hold"},
        "market_context": {
            "us_close": "+0.3%", "gift_nifty": "24250",
            "vix": 17.8, "fii_dii": "FII -1200cr",
        },
    })
    pb = parse_playbook_response(raw, today=date(2026, 3, 11))
    assert pb is not None
    assert pb.day_classification == DayClassification.LIKELY_TREND_UP
    assert len(pb.nifty_setups) == 1
    assert pb.nifty_setups[0].conviction == Conviction.HIGH
    assert len(pb.stock_plans) == 1
    assert pb.stock_plans[0].symbol == "HDFCBANK"


def test_parse_playbook_extracts_json_from_markdown():
    raw = """Here's my analysis:

```json
{
    "date": "2026-03-11",
    "day_classification": "LIKELY_RANGE",
    "nifty_plan": {
        "bias": "neutral",
        "key_levels": {},
        "setups": [],
        "no_trade_zone": ""
    },
    "stock_plans": [],
    "risk_budget": {
        "max_capital_at_risk_today_pct": 4.0,
        "max_trades_today": 4,
        "max_per_trade_risk_pct": 1.5,
        "survival_mode": false
    },
    "no_trade_conditions": [],
    "carry_rules": {},
    "theta_plan": {"action": "hold"},
    "market_context": {}
}
```

This is a rangebound day."""
    pb = parse_playbook_response(raw, today=date(2026, 3, 11))
    assert pb is not None
    assert pb.day_classification == DayClassification.LIKELY_RANGE


def test_parse_playbook_returns_none_on_garbage():
    pb = parse_playbook_response("This is not JSON at all", today=date(2026, 3, 11))
    assert pb is None


def test_parse_playbook_caps_stock_plans_at_3():
    raw = json.dumps({
        "date": "2026-03-11",
        "day_classification": "LIKELY_TREND_UP",
        "nifty_plan": {"bias": "bullish", "key_levels": {}, "setups": [], "no_trade_zone": ""},
        "stock_plans": [
            {"id": f"S{i}", "priority": i, "symbol": f"SYM{i}",
             "type": "breakout_long", "trigger": f"trigger {i}",
             "instrument": f"SYM{i} CE", "strike_logic": "delta 0.45",
             "target": 100 + i, "stoploss": 90 + i,
             "max_risk_pct": 1.0, "conviction": "medium"}
            for i in range(5)  # Claude sends 5, we cap at 3
        ],
        "risk_budget": {"max_capital_at_risk_today_pct": 4.0, "max_trades_today": 4,
                        "max_per_trade_risk_pct": 1.5, "survival_mode": False},
        "no_trade_conditions": [], "carry_rules": {},
        "theta_plan": {"action": "hold"}, "market_context": {},
    })
    pb = parse_playbook_response(raw, today=date(2026, 3, 11))
    assert pb is not None
    assert len(pb.stock_plans) == 3  # capped


# ── Setup Parsing ───────────────────────────────────────────────────────


def test_parse_setup_extracts_trigger_level():
    raw = json.dumps({
        "date": "2026-03-11",
        "day_classification": "LIKELY_TREND_UP",
        "nifty_plan": {
            "bias": "bullish", "key_levels": {},
            "setups": [
                {
                    "id": "N1", "priority": 1, "type": "breakout_long",
                    "trigger": "15-min candle close above 24350 with volume > 1.5x",
                    "instrument": "NIFTY CE", "strike_logic": "delta 0.45",
                    "target": 24500, "stoploss": 24280,
                    "max_risk_pct": 1.5, "conviction": "high",
                },
            ],
            "no_trade_zone": "",
        },
        "stock_plans": [],
        "risk_budget": {"max_capital_at_risk_today_pct": 4.0, "max_trades_today": 4,
                        "max_per_trade_risk_pct": 1.5, "survival_mode": False},
        "no_trade_conditions": [], "carry_rules": {},
        "theta_plan": {"action": "hold"}, "market_context": {},
    })
    pb = parse_playbook_response(raw, today=date(2026, 3, 11))
    setup = pb.nifty_setups[0]
    assert setup.trigger_level == 24350.0  # extracted from trigger text or target/stoploss
    assert setup.stoploss == 24280.0


# ── Fallback Playbook ───────────────────────────────────────────────────


def test_fallback_no_previous_playbook():
    from v7.strategist import build_fallback_playbook
    fb = build_fallback_playbook(today=date(2026, 3, 11), prev_playbook=None)
    assert fb.day_classification == DayClassification.NO_TRADE
    assert fb.risk_budget.max_trades_today == 0
    assert len(fb.nifty_setups) == 0
    assert len(fb.stock_plans) == 0


def test_fallback_with_previous_playbook():
    from v7.strategist import build_fallback_playbook
    prev = Playbook(
        date=date(2026, 3, 10),
        day_classification=DayClassification.LIKELY_TREND_UP,
        nifty_bias="bullish",
        nifty_setups=[
            Setup(id="N1", priority=1, type=SetupType.BREAKOUT_LONG,
                  symbol="NIFTY", trigger_level=24350.0,
                  trigger_condition="15-min close above",
                  instrument="NIFTY CE", strike_logic="delta 0.45",
                  target=24500.0, stoploss=24280.0, max_risk_pct=1.5,
                  conviction=Conviction.HIGH),
        ],
        stock_plans=[
            Setup(id="H1", priority=3, type=SetupType.BREAKOUT_LONG,
                  symbol="HDFCBANK", trigger_level=1625.0,
                  trigger_condition="close above 1625",
                  instrument="HDFCBANK CE", strike_logic="delta 0.45",
                  target=1660.0, stoploss=1610.0, max_risk_pct=1.0),
        ],
        risk_budget=RiskBudget(),
        no_trade_conditions=["VIX > 22"],
        carry_rules=CarryRules(),
    )
    fb = build_fallback_playbook(today=date(2026, 3, 11), prev_playbook=prev)
    assert fb.day_classification == DayClassification.UNCERTAIN
    assert fb.risk_budget.max_trades_today == 2  # halved
    assert fb.risk_budget.max_capital_at_risk_today_pct == 2.0  # halved
    assert len(fb.nifty_setups) >= 1
    assert fb.nifty_setups[0].conviction == Conviction.LOW  # downgraded
    assert len(fb.stock_plans) == 0  # no stock setups in fallback


# ── Exception Handling ──────────────────────────────────────────────────


def test_default_exception_vix_spike():
    from v7.strategist import default_exception_action
    action = default_exception_action("vix_spike")
    assert action["action"] == "hold_no_new"


def test_default_exception_flash_crash():
    from v7.strategist import default_exception_action
    action = default_exception_action("flash_crash")
    assert action["action"] == "flatten_all"


def test_default_exception_unknown():
    from v7.strategist import default_exception_action
    action = default_exception_action("alien_invasion")
    assert action["action"] == "hold_no_new"


def test_parse_exception_response_valid():
    from v7.strategist import parse_exception_response
    raw = json.dumps({"action": "flatten_all", "details": "close everything"})
    result = parse_exception_response(raw)
    assert result["action"] == "flatten_all"


def test_parse_exception_response_no_action():
    from v7.strategist import parse_exception_response
    raw = json.dumps({"details": "missing action field"})
    result = parse_exception_response(raw)
    assert result is None


# ── Strategist Class (mocked Claude) ────────────────────────────────────


def test_strategist_generate_premarket_mocked(tmp_path):
    """Test full Strategist flow with mocked Claude client."""
    playbook_json = json.dumps({
        "date": "2026-03-11",
        "day_classification": "LIKELY_TREND_UP",
        "nifty_plan": {
            "bias": "bullish",
            "key_levels": {"resistance_1": 24350},
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

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=playbook_json)]
    mock_client.messages.create.return_value = mock_response

    with patch("v7.strategist.Strategist.__init__", lambda self, **kw: None):
        strat = Strategist.__new__(Strategist)
        strat._client = mock_client
        strat._model = "sonnet"
        strat._model_light = "haiku"
        strat._max_retries = 1
        strat._retry_delay = 0

        from v7.state import StateManager
        strat._state = StateManager(tmp_path / "v7_state")

        pb = strat.generate_premarket_playbook(
            us_close={}, gift_nifty="24250", prev_vix=17.8,
            fii_dii="", events_today=[], events_this_week=[],
            level_memory={}, edge_tracker={}, risk_state={"mtd_pnl_pct": 0, "pacing": "on_track", "survival_mode": False},
            fo_ban_list=[], recent_lessons=[],
        )
        assert pb.day_classification == DayClassification.LIKELY_TREND_UP
        assert len(pb.nifty_setups) == 1


def test_strategist_falls_back_on_claude_failure(tmp_path):
    """If Claude fails, Strategist returns a fallback playbook."""
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("Claude down")

    with patch("v7.strategist.Strategist.__init__", lambda self, **kw: None):
        strat = Strategist.__new__(Strategist)
        strat._client = mock_client
        strat._model = "sonnet"
        strat._model_light = "haiku"
        strat._max_retries = 1
        strat._retry_delay = 0

        from v7.state import StateManager
        strat._state = StateManager(tmp_path / "v7_state")

        pb = strat.generate_premarket_playbook(
            us_close={}, gift_nifty="", prev_vix=17.0,
            fii_dii="", events_today=[], events_this_week=[],
            level_memory={}, edge_tracker={}, risk_state={"mtd_pnl_pct": 0, "pacing": "on_track", "survival_mode": False},
            fo_ban_list=[], recent_lessons=[],
        )
        assert pb.day_classification == DayClassification.NO_TRADE
        assert pb.risk_budget.max_trades_today == 0


# ── Prompt building helpers ─────────────────────────────────────────────


def test_build_opening_read_prompt():
    from v7.strategist import build_opening_read_prompt
    prompt = build_opening_read_prompt(
        current_playbook={"date": "2026-03-11", "nifty_plan": {"bias": "bullish"}},
        opening_range_high=24350.0,
        opening_range_low=24200.0,
        gap_direction="up",
        gap_behavior="extending",
        first_30min_volume_ratio=1.2,
        oi_changes={"24500CE": "+5L OI"},
    )
    assert "24350" in prompt
    assert "24200" in prompt
    assert "extending" in prompt


def test_build_checkin_prompt():
    from v7.strategist import build_checkin_prompt
    prompt = build_checkin_prompt(
        current_playbook={"date": "2026-03-11"},
        daily_pnl=-1500.0,
        open_positions=[{"symbol": "NIFTY", "pnl": 500}],
        setups_fired=["N1"],
        levels_tested=[{"price": 24350, "held": True}],
        oi_changes={},
        current_vix=18.2,
        checkin_number=1,
    )
    assert "-1500" in prompt or "1500" in prompt
    assert "N1" in prompt
    assert "18.2" in prompt


def test_build_exception_prompt():
    from v7.strategist import build_exception_prompt
    prompt = build_exception_prompt(
        exception_type="vix_spike",
        details={"vix_change": 3.5, "current_vix": 22.5},
        current_playbook={"date": "2026-03-11"},
        open_positions=[{"symbol": "NIFTY", "direction": "bullish"}],
    )
    assert "vix_spike" in prompt.lower()
    assert "22.5" in prompt


def test_build_eod_prompt():
    from v7.strategist import build_eod_prompt
    prompt = build_eod_prompt(
        trades_today=[{"symbol": "NIFTY", "pnl": 3000}],
        daily_pnl=3000.0,
        day_classification_predicted="LIKELY_TREND_UP",
        day_classification_actual="TREND_UP",
    )
    assert "3000" in prompt
    assert "LIKELY_TREND_UP" in prompt
