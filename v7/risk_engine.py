# v7/risk_engine.py
"""Risk Engine for V7 — pure rules, no Claude.

Gates every trade with:
- Per-trade sizing (conviction-based)
- Daily limits (loss cap, SL hits, trade count, margin)
- Concurrent risk budget (not cumulative)
- Monthly pacing and survival mode
- Correlation check
- Chop detection
- F&O ban list
- Brokerage optimization
"""
from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from v7.types import Conviction, PacingStatus, Position

log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

DAILY_LOSS_CAP_PCT = 2.0
MAX_CONSECUTIVE_SL = 3
MAX_TRADES_PER_DAY = 4
MARGIN_UTILIZATION_BLOCK = 0.70
MAX_CONCURRENT_RISK_PCT = 4.0
MAX_SAME_DIRECTION = 2
SURVIVAL_THRESHOLD_PCT = 5.0
FULL_STOP_THRESHOLD_PCT = 8.0
DRAWDOWN_REDUCE_PCT = 3.0
MIN_TRADE_VALUE = 2000.0
BROKERAGE_PER_ORDER = 20.0

CONVICTION_RISK = {
    Conviction.HIGH: 2.0,
    Conviction.MEDIUM: 1.5,
    Conviction.LOW: 0.75,
}


class RiskEngine:
    def __init__(self, capital: float, state_dir: str | Path):
        self._capital = capital
        self._dir = Path(state_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self._dir / "risk_state.json"

        self._daily_pnl = 0.0
        self._sl_hits_today = 0
        self._trades_today = 0
        self._margin_used_pct = 0.0

        self._pacing = PacingStatus.ON_TRACK
        self._mtd_pnl = 0.0
        self._survival_mode = False
        self._full_stop = False

        self._whipsaw_count = 0
        self._opening_range_pct = 0.0
        self._first_hour_volume_ratio = 1.0

        self._fo_ban_list: list[str] = []

        self._load_state()

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            with open(self._state_path) as f:
                data = json.load(f)
            today_str = str(date.today())
            if data.get("date") == today_str:
                self._daily_pnl = data.get("daily_pnl", 0.0)
                self._sl_hits_today = data.get("sl_hits_today", 0)
                self._trades_today = data.get("trades_today", 0)
                self._margin_used_pct = data.get("margin_used_pct", 0.0)
            self._mtd_pnl = data.get("mtd_pnl", 0.0)
            self._pacing = PacingStatus(data.get("pacing", "on_track"))
            self._survival_mode = data.get("survival_mode", False)
            self._full_stop = data.get("full_stop", False)
            self._fo_ban_list = data.get("fo_ban_list", [])
        except (json.JSONDecodeError, KeyError):
            log.warning("Risk state file corrupt — starting fresh")

    def _save_state(self) -> None:
        data = {
            "date": str(date.today()),
            "daily_pnl": self._daily_pnl,
            "sl_hits_today": self._sl_hits_today,
            "trades_today": self._trades_today,
            "margin_used_pct": self._margin_used_pct,
            "mtd_pnl": self._mtd_pnl,
            "pacing": self._pacing.value,
            "survival_mode": self._survival_mode,
            "full_stop": self._full_stop,
            "fo_ban_list": self._fo_ban_list,
        }
        with open(self._state_path, "w") as f:
            json.dump(data, f, indent=2)

    def risk_amount_for_conviction(self, conviction: Conviction) -> float:
        base_pct = CONVICTION_RISK[conviction]
        if self._pacing == PacingStatus.AHEAD:
            base_pct *= 0.75
        if self._mtd_pnl < 0 and abs(self._mtd_pnl) / self._capital * 100 >= DRAWDOWN_REDUCE_PCT:
            base_pct *= 0.75
        return self._capital * (base_pct / 100)

    def calculate_lots(self, risk_amount: float, premium: float, lot_size: int) -> int:
        if premium <= 0 or lot_size <= 0:
            return 0
        cost_per_lot = premium * lot_size
        return math.floor(risk_amount / cost_per_lot)

    def can_open_trade(self) -> tuple[bool, str]:
        if self._full_stop:
            return False, "Full stop: MTD drawdown > 8%. No trading rest of month."
        if self._survival_mode:
            return False, "Survival mode: MTD drawdown > 5%. Theta only — no directional trades."
        daily_loss_limit = self._capital * (DAILY_LOSS_CAP_PCT / 100)
        if self._daily_pnl < 0 and abs(self._daily_pnl) >= daily_loss_limit:
            return False, f"Daily loss cap: ₹{abs(self._daily_pnl):.0f} exceeds {DAILY_LOSS_CAP_PCT}% (₹{daily_loss_limit:.0f})"
        if self._sl_hits_today >= MAX_CONSECUTIVE_SL:
            return False, f"3 SL hits today. No more trades."
        if self._trades_today >= MAX_TRADES_PER_DAY:
            return False, f"Max trades ({MAX_TRADES_PER_DAY}) reached for today."
        if self._margin_used_pct >= MARGIN_UTILIZATION_BLOCK:
            return False, f"Margin utilization {self._margin_used_pct*100:.0f}% exceeds {MARGIN_UTILIZATION_BLOCK*100:.0f}% threshold."
        return True, "OK"

    def can_open_theta(self) -> tuple[bool, str]:
        if self._full_stop:
            return False, "Full stop: no trading rest of month."
        if self._margin_used_pct >= MARGIN_UTILIZATION_BLOCK:
            return False, f"Margin too high for theta: {self._margin_used_pct*100:.0f}%"
        return True, "OK"

    def can_allocate_risk(self, new_risk: float, current_risk: float) -> bool:
        max_risk = self._capital * (MAX_CONCURRENT_RISK_PCT / 100)
        return (current_risk + new_risk) <= max_risk

    def check_correlation(self, open_positions: list[Position], new_direction: str) -> tuple[bool, str]:
        same_dir = sum(1 for p in open_positions if p.direction == new_direction)
        if same_dir >= MAX_SAME_DIRECTION:
            return True, f"Correlation block: {same_dir} positions already {new_direction}. Max {MAX_SAME_DIRECTION}."
        return False, ""

    def set_pacing(self, status: PacingStatus) -> None:
        self._pacing = status
        self._save_state()

    def update_mtd_pnl(self, mtd_pnl: float) -> None:
        self._mtd_pnl = mtd_pnl
        mtd_dd_pct = abs(mtd_pnl) / self._capital * 100 if mtd_pnl < 0 else 0
        if mtd_dd_pct >= FULL_STOP_THRESHOLD_PCT:
            self._full_stop = True
            self._survival_mode = True
        elif mtd_dd_pct >= SURVIVAL_THRESHOLD_PCT:
            self._survival_mode = True
            self._full_stop = False
        elif mtd_dd_pct < DRAWDOWN_REDUCE_PCT:
            self._survival_mode = False
            self._full_stop = False
        self._save_state()

    @property
    def survival_mode(self) -> bool:
        return self._survival_mode

    @property
    def full_stop(self) -> bool:
        return self._full_stop

    def update_fo_ban_list(self, symbols: list[str]) -> None:
        self._fo_ban_list = symbols
        self._save_state()

    def is_fo_banned(self, symbol: str) -> bool:
        return symbol.upper() in [s.upper() for s in self._fo_ban_list]

    def update_chop_signals(self, whipsaw_count: int, opening_range_pct: float, first_hour_volume_ratio: float) -> None:
        self._whipsaw_count = whipsaw_count
        self._opening_range_pct = opening_range_pct
        self._first_hour_volume_ratio = first_hour_volume_ratio

    def is_choppy(self) -> tuple[bool, str]:
        reasons = []
        if self._whipsaw_count >= 3:
            reasons.append(f"{self._whipsaw_count} whipsaws in first hour")
        if 0 < self._opening_range_pct < 0.3:
            reasons.append(f"Narrow opening range: {self._opening_range_pct:.2f}%")
        if 0 < self._first_hour_volume_ratio < 0.5:
            reasons.append(f"Low volume: {self._first_hour_volume_ratio:.0%} of 20-day avg")
        if reasons:
            return True, "; ".join(reasons)
        return False, ""

    def check_min_trade_value(self, trade_value: float) -> tuple[bool, str]:
        if trade_value < MIN_TRADE_VALUE:
            brokerage_pct = (BROKERAGE_PER_ORDER / trade_value * 100) if trade_value > 0 else 100
            return False, f"Trade value ₹{trade_value:.0f} too small. Brokerage would be {brokerage_pct:.1f}%. Min: ₹{MIN_TRADE_VALUE:.0f}"
        return True, ""

    def record_daily_pnl(self, pnl: float) -> None:
        self._daily_pnl = pnl
        self._save_state()

    def record_sl_hit(self) -> None:
        self._sl_hits_today += 1
        self._save_state()

    def record_trade_opened(self) -> None:
        self._trades_today += 1
        self._save_state()

    def set_margin_used(self, pct: float) -> None:
        self._margin_used_pct = pct
        self._save_state()

    def reset_daily(self) -> None:
        self._daily_pnl = 0.0
        self._sl_hits_today = 0
        self._trades_today = 0
        self._margin_used_pct = 0.0
        self._whipsaw_count = 0
        self._opening_range_pct = 0.0
        self._first_hour_volume_ratio = 1.0
        self._save_state()

    def reset_monthly(self) -> None:
        self._mtd_pnl = 0.0
        self._pacing = PacingStatus.ON_TRACK
        self._survival_mode = False
        self._full_stop = False
        self._save_state()

    def pre_trade_check(self, symbol: str, conviction: Conviction, direction: str,
                        trade_value: float, open_positions: list[Position],
                        current_risk: float) -> tuple[bool, str, float]:
        allowed, reason = self.can_open_trade()
        if not allowed:
            return False, reason, 0.0
        if self.is_fo_banned(symbol):
            return False, f"{symbol} is in F&O ban list.", 0.0
        ok, reason = self.check_min_trade_value(trade_value)
        if not ok:
            return False, reason, 0.0
        blocked, reason = self.check_correlation(open_positions, direction)
        if blocked:
            return False, reason, 0.0
        risk_amount = self.risk_amount_for_conviction(conviction)
        if not self.can_allocate_risk(risk_amount, current_risk):
            max_risk = self._capital * (MAX_CONCURRENT_RISK_PCT / 100)
            return False, f"Risk budget full: current ₹{current_risk:.0f} + new ₹{risk_amount:.0f} > max ₹{max_risk:.0f}", 0.0
        choppy, chop_reason = self.is_choppy()
        if choppy:
            log.warning(f"Chop detected: {chop_reason}")
        return True, "OK", risk_amount

    def get_state_summary(self) -> dict:
        return {
            "daily_pnl": self._daily_pnl,
            "sl_hits_today": self._sl_hits_today,
            "trades_today": self._trades_today,
            "margin_used_pct": self._margin_used_pct,
            "mtd_pnl": self._mtd_pnl,
            "mtd_pnl_pct": (self._mtd_pnl / self._capital * 100) if self._capital > 0 else 0,
            "pacing": self._pacing.value,
            "survival_mode": self._survival_mode,
            "full_stop": self._full_stop,
            "fo_ban_list": self._fo_ban_list,
        }
