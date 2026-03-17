# v7/oi_pipeline.py
"""OI (Open Interest) data pipeline for V7.

Processes option chain data to compute:
- Put/Call Ratio (PCR)
- Max Pain strike
- OI buildup (top strikes by OI change)
- Snapshot persistence for change tracking
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class OISnapshot:
    """Point-in-time snapshot of option chain OI data."""
    symbol: str
    expiry: str
    timestamp: str
    total_call_oi: int
    total_put_oi: int
    pcr: float
    max_pain_strike: float
    strikes: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> OISnapshot:
        return cls(
            symbol=d["symbol"],
            expiry=d["expiry"],
            timestamp=d["timestamp"],
            total_call_oi=d["total_call_oi"],
            total_put_oi=d["total_put_oi"],
            pcr=d["pcr"],
            max_pain_strike=d["max_pain_strike"],
            strikes=d.get("strikes", []),
        )


class OIPipeline:
    """Processes option chain data for OI analysis."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.snapshot_dir = data_dir / "oi_snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def process_chain(self, symbol: str, chain: list[dict], expiry: str = "current") -> OISnapshot:
        """Process raw option chain into an OISnapshot.

        Args:
            symbol: e.g. "NIFTY"
            chain: list of strike dicts from Kite, each with
                   {"strikePrice": ..., "CE": {"openInterest": ...}, "PE": {"openInterest": ...}}
            expiry: expiry label, defaults to "current"

        Returns:
            OISnapshot with computed PCR, max pain, and per-strike OI.
        """
        total_call_oi = 0
        total_put_oi = 0
        strikes_data = []

        for entry in chain:
            strike = entry.get("strikePrice", 0)
            ce_oi = entry.get("CE", {}).get("openInterest", 0) or 0
            pe_oi = entry.get("PE", {}).get("openInterest", 0) or 0
            total_call_oi += ce_oi
            total_put_oi += pe_oi
            strikes_data.append({
                "strike": strike,
                "call_oi": ce_oi,
                "put_oi": pe_oi,
            })

        pcr = self.compute_pcr(chain)
        max_pain = self.compute_max_pain(chain)

        return OISnapshot(
            symbol=symbol,
            expiry=expiry,
            timestamp=datetime.now(IST).isoformat(),
            total_call_oi=total_call_oi,
            total_put_oi=total_put_oi,
            pcr=pcr,
            max_pain_strike=max_pain,
            strikes=strikes_data,
        )

    def compute_pcr(self, chain: list[dict]) -> float:
        """Compute Put/Call Ratio from chain data.

        Returns sum(put_oi) / sum(call_oi). Returns 0.0 if no call OI.
        """
        total_call = 0
        total_put = 0
        for entry in chain:
            total_call += entry.get("CE", {}).get("openInterest", 0) or 0
            total_put += entry.get("PE", {}).get("openInterest", 0) or 0
        if total_call == 0:
            return 0.0
        return total_put / total_call

    def compute_max_pain(self, chain: list[dict], spot: float = 0.0) -> float:
        """Compute max pain strike — where total loss to option writers is minimized.

        For each candidate settlement strike, compute:
          CE writer loss = sum(max(0, settlement - strike) * call_oi)
          PE writer loss = sum(max(0, strike - settlement) * put_oi)
        The strike with minimum total loss is max pain.

        Args:
            chain: option chain data
            spot: not used in computation, kept for API compatibility

        Returns:
            Max pain strike price. Returns 0.0 if chain is empty.
        """
        if not chain:
            return 0.0

        # Candidate settlement strikes = all strikes in the chain
        candidates = []
        for entry in chain:
            strike = entry.get("strikePrice", 0)
            if strike > 0:
                candidates.append(strike)

        if not candidates:
            return 0.0

        min_loss = float("inf")
        max_pain_strike = candidates[0]

        for settlement in candidates:
            total_loss = 0.0
            for entry in chain:
                strike = entry.get("strikePrice", 0)
                ce_oi = entry.get("CE", {}).get("openInterest", 0) or 0
                pe_oi = entry.get("PE", {}).get("openInterest", 0) or 0

                # CE writer loss: if settlement > strike, calls are ITM
                ce_loss = max(0, settlement - strike) * ce_oi
                # PE writer loss: if strike > settlement, puts are ITM
                pe_loss = max(0, strike - settlement) * pe_oi

                total_loss += ce_loss + pe_loss

            if total_loss < min_loss:
                min_loss = total_loss
                max_pain_strike = settlement

        return max_pain_strike

    def compute_oi_changes(self, symbol: str) -> dict:
        """Compare current vs previous snapshot to compute OI changes.

        Returns dict with pcr_change, max_pain_change, total_call_oi_change,
        total_put_oi_change, and top strike-level changes.
        Returns empty dict if no previous snapshot exists.
        """
        snapshots = self._load_snapshots(symbol)
        if len(snapshots) < 2:
            return {}

        current = snapshots[-1]
        previous = snapshots[-2]

        # Build strike map for previous
        prev_strikes = {s["strike"]: s for s in previous.get("strikes", [])}

        strike_changes = []
        for s in current.get("strikes", []):
            strike = s["strike"]
            prev = prev_strikes.get(strike, {"call_oi": 0, "put_oi": 0})
            call_change = s["call_oi"] - prev.get("call_oi", 0)
            put_change = s["put_oi"] - prev.get("put_oi", 0)
            abs_change = abs(call_change) + abs(put_change)
            if abs_change > 0:
                strike_changes.append({
                    "strike": strike,
                    "call_oi_change": call_change,
                    "put_oi_change": put_change,
                    "abs_change": abs_change,
                })

        # Sort by absolute change descending
        strike_changes.sort(key=lambda x: x["abs_change"], reverse=True)

        return {
            "pcr_change": current["pcr"] - previous["pcr"],
            "max_pain_change": current["max_pain_strike"] - previous["max_pain_strike"],
            "total_call_oi_change": current["total_call_oi"] - previous["total_call_oi"],
            "total_put_oi_change": current["total_put_oi"] - previous["total_put_oi"],
            "top_strike_changes": strike_changes[:10],
        }

    def get_oi_buildup(self, symbol: str) -> list[dict]:
        """Get top 5 strikes by OI buildup (absolute change).

        Returns list of dicts with strike, call_oi_change, put_oi_change, abs_change.
        """
        changes = self.compute_oi_changes(symbol)
        if not changes:
            return []
        return changes.get("top_strike_changes", [])[:5]

    def save_snapshot(self, snapshot: OISnapshot) -> None:
        """Persist snapshot to data/v7/oi_snapshots/{symbol}.json.

        Appends to a list of snapshots, keeping last 20.
        """
        path = self.snapshot_dir / f"{snapshot.symbol}.json"
        snapshots = []
        if path.exists():
            try:
                snapshots = json.loads(path.read_text())
            except (json.JSONDecodeError, ValueError):
                snapshots = []

        snapshots.append(snapshot.to_dict())
        # Keep last 20 snapshots
        snapshots = snapshots[-20:]

        path.write_text(json.dumps(snapshots, indent=2, default=str))
        log.info("OI snapshot saved for %s (%d total)", snapshot.symbol, len(snapshots))

    def load_previous(self, symbol: str) -> OISnapshot | None:
        """Load the most recent previous snapshot for a symbol."""
        snapshots = self._load_snapshots(symbol)
        if len(snapshots) < 2:
            return None
        return OISnapshot.from_dict(snapshots[-2])

    def _load_snapshots(self, symbol: str) -> list[dict]:
        """Load all snapshots for a symbol as raw dicts."""
        path = self.snapshot_dir / f"{symbol}.json"
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, ValueError):
            return []

    def format_for_prompt(self, symbols: list[str]) -> dict:
        """Format OI data for Claude prompt context.

        Returns dict keyed by symbol with pcr, max_pain, and oi_changes.
        """
        result = {}
        for symbol in symbols:
            snapshots = self._load_snapshots(symbol)
            if not snapshots:
                continue

            current = snapshots[-1]
            oi_changes = self.compute_oi_changes(symbol)
            buildup = self.get_oi_buildup(symbol)

            result[symbol] = {
                "pcr": current["pcr"],
                "max_pain": current["max_pain_strike"],
                "total_call_oi": current["total_call_oi"],
                "total_put_oi": current["total_put_oi"],
                "oi_changes": oi_changes,
                "top_buildup": buildup,
            }

        return result
