# tests/test_v7_level_memory.py
"""Tests for V7 Level Memory — persistent key levels and OI walls."""
import json
import pytest
from datetime import date
from pathlib import Path
from v7.level_memory import LevelMemory, Level


@pytest.fixture
def tmp_state_dir(tmp_path):
    return tmp_path / "v7_state"


@pytest.fixture
def lm(tmp_state_dir):
    return LevelMemory(state_dir=tmp_state_dir)


# ── Level CRUD ──────────────────────────────────────────────────────────


def test_add_level(lm):
    lm.add_level(
        symbol="NIFTY", price=24000.0, level_type="support",
        source="tested 3x in last 5 sessions",
    )
    levels = lm.get_levels("NIFTY")
    assert len(levels) == 1
    assert levels[0].price == 24000.0
    assert levels[0].level_type == "support"
    assert levels[0].strength == 1


def test_add_level_deduplicates_nearby(lm):
    """Adding a level within 0.1% of existing level strengthens it instead."""
    lm.add_level("NIFTY", 24000.0, "support", "source1")
    lm.add_level("NIFTY", 24010.0, "support", "source2")  # within 0.1% of 24000
    levels = lm.get_levels("NIFTY")
    assert len(levels) == 1
    assert levels[0].strength == 2  # strengthened


def test_add_level_different_price_creates_new(lm):
    lm.add_level("NIFTY", 24000.0, "support", "source1")
    lm.add_level("NIFTY", 24500.0, "resistance", "source2")
    levels = lm.get_levels("NIFTY")
    assert len(levels) == 2


def test_get_levels_empty_symbol(lm):
    assert lm.get_levels("UNKNOWN") == []


# ── Level Maintenance ───────────────────────────────────────────────────


def test_strengthen_on_retest(lm):
    lm.add_level("NIFTY", 24000.0, "support", "initial")
    lm.retest_level("NIFTY", 24000.0, held=True)
    levels = lm.get_levels("NIFTY")
    assert levels[0].strength == 2
    assert levels[0].last_tested == str(date.today())


def test_weaken_on_break(lm):
    lm.add_level("NIFTY", 24000.0, "support", "initial")
    lm.retest_level("NIFTY", 24000.0, held=True)  # strength=2
    lm.retest_level("NIFTY", 24000.0, held=False)  # strength=1
    levels = lm.get_levels("NIFTY")
    assert levels[0].strength == 1


def test_remove_on_strength_zero(lm):
    lm.add_level("NIFTY", 24000.0, "support", "initial")  # strength=1
    lm.retest_level("NIFTY", 24000.0, held=False)  # strength=0 → removed
    levels = lm.get_levels("NIFTY")
    assert len(levels) == 0


def test_flip_level(lm):
    lm.add_level("NIFTY", 24000.0, "resistance", "OI wall")
    lm.flip_level("NIFTY", 24000.0)
    levels = lm.get_levels("NIFTY")
    assert levels[0].level_type == "support"
    assert levels[0].strength == 1  # reset to 1 on flip


def test_flip_support_to_resistance(lm):
    lm.add_level("NIFTY", 24000.0, "support", "tested")
    lm.flip_level("NIFTY", 24000.0)
    levels = lm.get_levels("NIFTY")
    assert levels[0].level_type == "resistance"


# ── Staleness ───────────────────────────────────────────────────────────


def test_remove_stale_levels(lm):
    lm.add_level("NIFTY", 24000.0, "support", "old level")
    # Manually set last_tested to 15 days ago
    lm._data["NIFTY"]["levels"][0]["last_tested"] = "2026-02-20"
    lm.remove_stale(max_age_days=10, today=date(2026, 3, 11))
    assert len(lm.get_levels("NIFTY")) == 0


def test_keep_fresh_levels(lm):
    lm.add_level("NIFTY", 24000.0, "support", "recent level")
    lm.remove_stale(max_age_days=10, today=date.today())
    assert len(lm.get_levels("NIFTY")) == 1


# ── OI Walls ────────────────────────────────────────────────────────────


def test_update_oi_walls(lm):
    lm.update_oi_walls("NIFTY", call_max_oi_strike=24500, put_max_oi_strike=24000, pcr=1.1)
    walls = lm.get_oi_walls("NIFTY")
    assert walls["call_max_oi_strike"] == 24500
    assert walls["put_max_oi_strike"] == 24000
    assert walls["pcr"] == 1.1


def test_get_oi_walls_empty(lm):
    walls = lm.get_oi_walls("UNKNOWN")
    assert walls == {}


# ── Persistence ─────────────────────────────────────────────────────────


def test_save_and_reload(tmp_state_dir):
    lm1 = LevelMemory(state_dir=tmp_state_dir)
    lm1.add_level("NIFTY", 24000.0, "support", "tested")
    lm1.update_oi_walls("NIFTY", 24500, 24000, 1.1)
    lm1.save()

    lm2 = LevelMemory(state_dir=tmp_state_dir)
    levels = lm2.get_levels("NIFTY")
    assert len(levels) == 1
    assert levels[0].price == 24000.0
    walls = lm2.get_oi_walls("NIFTY")
    assert walls["call_max_oi_strike"] == 24500


def test_to_strategist_context(lm):
    lm.add_level("NIFTY", 24000.0, "support", "tested 3x")
    lm.add_level("NIFTY", 24500.0, "resistance", "OI wall")
    lm.update_oi_walls("NIFTY", 24500, 24000, 1.1)
    ctx = lm.to_strategist_context(["NIFTY"])
    assert "NIFTY" in ctx
    assert len(ctx["NIFTY"]["levels"]) == 2
    assert "oi_walls" in ctx["NIFTY"]


# ── Level dataclass ─────────────────────────────────────────────────────


def test_level_to_dict_roundtrip():
    lv = Level(
        price=24000.0, level_type="support", strength=3,
        source="tested 3x", last_tested="2026-03-10", created="2026-03-05",
    )
    d = lv.to_dict()
    lv2 = Level.from_dict(d)
    assert lv2.price == lv.price
    assert lv2.strength == lv.strength
    assert lv2.level_type == lv.level_type


def test_level_near():
    lv = Level(price=24000.0, level_type="support", strength=1,
               source="x", last_tested="2026-03-10", created="2026-03-10")
    assert lv.is_near(24020.0, threshold_pct=0.1) is True   # 0.08% away
    assert lv.is_near(24100.0, threshold_pct=0.1) is False  # 0.42% away
