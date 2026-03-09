#!/usr/bin/env python3
"""Migrate portfolio and state files from V5 schema to V6."""

import sys
from pathlib import Path

# Allow importing project modules from parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def migrate_portfolio(portfolio: dict) -> dict:
    """Add V6 fields to portfolio positions.

    - Adds entry_thesis to each position if missing
    - Adds _add_count to each position if missing
    - Sets schema_version = 6
    - Never overwrites existing data
    """
    for pos in portfolio.get("positions", []):
        if "entry_thesis" not in pos:
            pos["entry_thesis"] = {
                "reasoning": f"Pre-V6 position opened on {pos.get('entry_date', 'unknown')}",
                "conviction": "medium",
                "key_conditions": {},
                "invalidation": "Manual review required",
                "expected_hold": "unknown",
                "target_scenario": "unknown",
            }
        pos.setdefault("_add_count", 0)

    portfolio["schema_version"] = 6
    return portfolio


def migrate_state(state: dict) -> dict:
    """Add V6 fields to monitor state.

    - Adds position_assessments if missing
    - Adds claude_consecutive_failures if missing
    - Adds claude_lockdown_active if missing
    - Sets schema_version = 6
    """
    state.setdefault("position_assessments", {})
    state.setdefault("claude_consecutive_failures", 0)
    state.setdefault("claude_lockdown_active", False)
    state["schema_version"] = 6
    return state


def main():
    """CLI entry point: migrate portfolio and state files in place."""
    from paper_trade import load_portfolio, save_portfolio
    from smart_monitor import _load_state, _save_state

    # --- Portfolio ---
    portfolio = load_portfolio()
    n_positions = len(portfolio.get("positions", []))
    portfolio = migrate_portfolio(portfolio)
    save_portfolio(portfolio)
    print(f"Portfolio migrated: {n_positions} positions updated to schema v6")

    # --- State ---
    state = _load_state()
    state = migrate_state(state)
    _save_state(state)
    print("Monitor state migrated to schema v6")

    print("Migration complete.")


if __name__ == "__main__":
    main()
