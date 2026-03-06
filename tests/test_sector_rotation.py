"""Tests for sector_rotation module."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sector_rotation import classify_sector_momentum, SECTOR_REPRESENTATIVES


def test_classify_empty():
    """Empty sector returns should return safe defaults."""
    result = classify_sector_momentum({})
    assert result["strengthening"] == []
    assert result["weakening"] == []
    assert result["rotation_detected"] is False
    assert result["top_sector"] is None


def test_classify_strengthening():
    """Sectors with momentum > 0.5 are strengthening."""
    data = {
        "IT": {"1w_return": 3.0, "4w_return": 2.0, "momentum_score": 2.5},
        "Banking": {"1w_return": 0.5, "4w_return": 0.4, "momentum_score": 0.4},
        "Metals": {"1w_return": -2.0, "4w_return": 1.0, "momentum_score": -2.25},
    }
    result = classify_sector_momentum(data)
    assert "IT" in result["strengthening"]
    assert "Banking" in result["neutral"]
    assert "Metals" in result["weakening"]
    assert result["top_sector"] == "IT"
    assert result["worst_sector"] == "Metals"


def test_rotation_detected():
    """Rotation detected when spread > 2.0."""
    data = {
        "Pharma": {"1w_return": 5.0, "4w_return": 1.0, "momentum_score": 4.75},
        "IT": {"1w_return": -3.0, "4w_return": 2.0, "momentum_score": -3.5},
    }
    result = classify_sector_momentum(data)
    assert result["rotation_detected"] is True


def test_no_rotation_similar_momentum():
    """No rotation when all sectors have similar momentum."""
    data = {
        "IT": {"1w_return": 1.0, "4w_return": 1.0, "momentum_score": 0.75},
        "Banking": {"1w_return": 0.8, "4w_return": 1.0, "momentum_score": 0.55},
        "Pharma": {"1w_return": 0.5, "4w_return": 0.5, "momentum_score": 0.375},
    }
    result = classify_sector_momentum(data)
    assert result["rotation_detected"] is False


def test_all_sectors_have_representatives():
    """All sectors in SECTOR_REPRESENTATIVES have at least 2 tickers."""
    for sector, tickers in SECTOR_REPRESENTATIVES.items():
        assert len(tickers) >= 2, f"{sector} has < 2 representatives"
