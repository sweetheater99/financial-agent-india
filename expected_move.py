"""Expected Move Accuracy Tracker.

Tracks whether Nifty's IV-implied expected move is accurate vs actual moves.
If IV consistently overprices moves (actual < expected), there is a seller edge.
"""

import json
import logging
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

DATA_DIR = Path(__file__).parent / "data"
MOVES_FILE = DATA_DIR / "expected_moves.json"


def compute_expected_move(vix: float, spot: float, days: int = 1) -> float:
    """Compute IV-implied expected move in points.

    Expected 1-day move = spot * (vix/100) / sqrt(252) * sqrt(days)

    Args:
        vix: India VIX value (e.g., 14.2)
        spot: Nifty spot price (e.g., 22500)
        days: number of days (default 1)

    Returns:
        Expected move in points (always positive).
    """
    if vix <= 0 or spot <= 0 or days <= 0:
        return 0.0
    return spot * (vix / 100.0) / math.sqrt(252) * math.sqrt(days)


def _load_moves() -> list[dict]:
    """Load recorded moves from JSON file."""
    try:
        if MOVES_FILE.exists():
            with open(MOVES_FILE) as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load expected moves data: %s", e)
    return []


def _save_moves(moves: list[dict]) -> None:
    """Save moves to JSON file."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(MOVES_FILE, "w") as f:
            json.dump(moves, f, indent=2)
    except OSError as e:
        logger.error("Failed to save expected moves data: %s", e)


def record_daily_move(spot_open: float, spot_close: float, vix_open: float,
                      record_date: str | None = None) -> dict | None:
    """Record actual vs expected move for a trading day.

    Args:
        spot_open: Nifty opening price
        spot_close: Nifty closing price
        vix_open: India VIX at open
        record_date: date string (YYYY-MM-DD), defaults to today IST

    Returns:
        The recorded entry dict, or None on error.
    """
    if spot_open <= 0 or spot_close <= 0 or vix_open <= 0:
        logger.warning("Invalid inputs for record_daily_move: open=%.2f close=%.2f vix=%.2f",
                       spot_open, spot_close, vix_open)
        return None

    if record_date is None:
        record_date = datetime.now(IST).strftime("%Y-%m-%d")

    actual_move = abs(spot_close - spot_open)
    expected_move = compute_expected_move(vix_open, spot_open, days=1)
    ratio = actual_move / expected_move if expected_move > 0 else 0.0

    entry = {
        "date": record_date,
        "spot_open": round(spot_open, 2),
        "spot_close": round(spot_close, 2),
        "actual_move": round(actual_move, 2),
        "expected_move": round(expected_move, 2),
        "vix": round(vix_open, 2),
        "ratio": round(ratio, 4),
    }

    moves = _load_moves()

    # Avoid duplicate entries for same date
    moves = [m for m in moves if m.get("date") != record_date]
    moves.append(entry)

    # Keep only last 365 days of data
    moves = sorted(moves, key=lambda m: m.get("date", ""))[-365:]

    _save_moves(moves)
    logger.info("Recorded expected move for %s: actual=%.0f expected=%.0f ratio=%.2f",
                record_date, actual_move, expected_move, ratio)
    return entry


def get_move_accuracy(days: int = 30) -> dict:
    """Analyze expected move accuracy over last N days.

    Args:
        days: number of recent days to analyze (default 30)

    Returns:
        dict with accuracy metrics:
            days_analyzed: number of data points
            avg_ratio: average actual/expected ratio (< 1.0 = IV overprices)
            pct_within_expected: % of days actual move < expected move
            seller_edge: True if avg_ratio < 0.85
            avg_expected: average expected move in points
            avg_actual: average actual move in points
    """
    moves = _load_moves()

    if not moves:
        return {
            "days_analyzed": 0,
            "avg_ratio": 0.0,
            "pct_within_expected": 0,
            "seller_edge": False,
            "avg_expected": 0.0,
            "avg_actual": 0.0,
        }

    # Take last N entries
    recent = sorted(moves, key=lambda m: m.get("date", ""))[-days:]

    ratios = [m["ratio"] for m in recent]
    expected_moves = [m["expected_move"] for m in recent]
    actual_moves = [m["actual_move"] for m in recent]

    avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    within_count = sum(1 for m in recent if m["actual_move"] <= m["expected_move"])
    pct_within = round(within_count / len(recent) * 100) if recent else 0

    return {
        "days_analyzed": len(recent),
        "avg_ratio": round(avg_ratio, 2),
        "pct_within_expected": pct_within,
        "seller_edge": avg_ratio < 0.85,
        "avg_expected": round(sum(expected_moves) / len(expected_moves), 2) if expected_moves else 0.0,
        "avg_actual": round(sum(actual_moves) / len(actual_moves), 2) if actual_moves else 0.0,
    }


def generate_move_report() -> str:
    """Generate Telegram HTML formatted expected move report.

    Returns:
        HTML string for Telegram (parse_mode=HTML).
    """
    accuracy = get_move_accuracy(30)

    if accuracy["days_analyzed"] == 0:
        return "<b>Expected Move Accuracy</b>\n  No data recorded yet."

    lines = ["<b>Expected Move Accuracy (30d)</b>"]
    lines.append(
        f"  Avg Expected: +/-{accuracy['avg_expected']:.0f} pts | "
        f"Avg Actual: +/-{accuracy['avg_actual']:.0f} pts"
    )

    overpricing_pct = round((1 - accuracy["avg_ratio"]) * 100)
    lines.append(
        f"  IV Overpricing: {overpricing_pct}% (actual/expected = {accuracy['avg_ratio']:.2f})"
    )
    lines.append(f"  Days Within Expected: {accuracy['pct_within_expected']}%")

    if accuracy["seller_edge"]:
        lines.append("  <b>Verdict: Strong seller edge</b>")
    elif accuracy["avg_ratio"] < 1.0:
        lines.append("  <b>Verdict: Mild seller edge</b>")
    else:
        lines.append("  <b>Verdict: No seller edge (IV underpricing)</b>")

    lines.append(f"  ({accuracy['days_analyzed']} trading days analyzed)")

    return "\n".join(lines)
