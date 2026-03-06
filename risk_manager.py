"""
Risk manager for V2 trading system.

Tracks drawdowns at daily/weekly/monthly/total levels and enforces
circuit breakers, strategy pauses, cash mode, and position limits.
All state is in-memory — no persistence or file I/O.
"""

from datetime import date, timedelta

from config import (
    DRAWDOWN_DAILY_HALT,
    DRAWDOWN_WEEKLY_REDUCE,
    DRAWDOWN_MONTHLY_HALT,
    DRAWDOWN_TOTAL_STOP,
    CONSECUTIVE_LOSS_PAUSE,
    MAX_CONCURRENT_POSITIONS,
    MAX_SAME_SECTOR,
    MAX_SIMULTANEOUS_LOSS_PCT,
)


class RiskManager:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self._daily_pnl = 0.0
        self._weekly_pnl = 0.0
        self._monthly_pnl = 0.0
        self._total_pnl = 0.0
        self._consecutive_losses = 0
        self._cash_until: date | None = None  # date when cash mode ends (exclusive)
        self._strategy_losses: dict[str, int] = {}  # {strategy_name: consecutive_loss_count}

    # ── Trade recording ──────────────────────────────────────────────────

    def record_trade_result(self, pnl: float, strategy: str = "equity") -> None:
        """Update all P&L counters and track consecutive losses."""
        self._daily_pnl += pnl
        self._weekly_pnl += pnl
        self._monthly_pnl += pnl
        self._total_pnl += pnl
        self.current_capital += pnl

        # Global consecutive losses
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # Per-strategy consecutive losses
        if pnl < 0:
            self._strategy_losses[strategy] = self._strategy_losses.get(strategy, 0) + 1
        else:
            self._strategy_losses[strategy] = 0

    # ── Reset methods (call at period boundaries) ────────────────────────

    def reset_daily(self) -> None:
        """Call at start of each trading day."""
        self._daily_pnl = 0.0

    def reset_weekly(self) -> None:
        """Call at start of each week."""
        self._weekly_pnl = 0.0

    def reset_monthly(self) -> None:
        """Call at start of each month."""
        self._monthly_pnl = 0.0

    # ── Drawdown circuit breakers ────────────────────────────────────────

    def check_daily_halt(self) -> bool:
        """True if daily loss exceeds DRAWDOWN_DAILY_HALT (3%) of current capital."""
        return self._daily_pnl < -(DRAWDOWN_DAILY_HALT * self.current_capital)

    def check_weekly_reduce(self) -> bool:
        """True if weekly loss exceeds DRAWDOWN_WEEKLY_REDUCE (5%) of current capital."""
        return self._weekly_pnl < -(DRAWDOWN_WEEKLY_REDUCE * self.current_capital)

    def check_monthly_halt(self) -> bool:
        """True if monthly loss exceeds DRAWDOWN_MONTHLY_HALT (8%) of current capital."""
        return self._monthly_pnl < -(DRAWDOWN_MONTHLY_HALT * self.current_capital)

    def check_total_stop(self) -> bool:
        """True if total drawdown exceeds DRAWDOWN_TOTAL_STOP (15%) of initial capital."""
        return self._total_pnl < -(DRAWDOWN_TOTAL_STOP * self.initial_capital)

    # ── Strategy pause ───────────────────────────────────────────────────

    def is_strategy_paused(self, strategy: str) -> bool:
        """True if strategy has CONSECUTIVE_LOSS_PAUSE (3) consecutive losses."""
        return self._strategy_losses.get(strategy, 0) >= CONSECUTIVE_LOSS_PAUSE

    # ── Cash mode ────────────────────────────────────────────────────────

    def enter_cash_mode(self, duration_days: int, _today: date | None = None) -> None:
        """Set cash mode for N calendar days starting from today."""
        today = _today or date.today()
        self._cash_until = today + timedelta(days=duration_days)

    def is_cash_mode(self, _today: date | None = None) -> bool:
        """True if currently in cash mode."""
        if self._cash_until is None:
            return False
        today = _today or date.today()
        return today < self._cash_until

    def cash_days_remaining(self, _today: date | None = None) -> int:
        """Days remaining in cash mode (0 if not in cash mode)."""
        if self._cash_until is None:
            return 0
        today = _today or date.today()
        remaining = (self._cash_until - today).days
        return max(0, remaining)

    # ── Risk state (for regime.py) ───────────────────────────────────────

    def get_risk_state(self) -> dict:
        """Return dict compatible with regime.py classify_regime()."""
        monthly_dd_pct = (self._monthly_pnl / self.initial_capital) * 100
        return {
            "monthly_drawdown_pct": monthly_dd_pct,
            "consecutive_losses": self._consecutive_losses,
        }

    # ── Position limits ──────────────────────────────────────────────────

    def can_open_position(
        self,
        open_positions: list[dict],
        new_sector: str | None = None,
        new_max_loss: float = 0.0,
    ) -> tuple[bool, str]:
        """Check position limits, sector limits, and correlation guard.

        Each position dict has: {"strategy": str, "sector": str, "max_loss": float}
        Returns (allowed, reason).
        """
        # Max concurrent positions
        if len(open_positions) >= MAX_CONCURRENT_POSITIONS:
            return False, f"Max concurrent positions ({MAX_CONCURRENT_POSITIONS}) reached"

        # Sector limit
        if new_sector is not None:
            sector_count = sum(
                1 for p in open_positions if p.get("sector") == new_sector
            )
            if sector_count >= MAX_SAME_SECTOR:
                return False, f"Max same sector ({MAX_SAME_SECTOR}) reached for {new_sector}"

        # Correlation guard: total max loss across all positions
        total_max_loss = sum(p.get("max_loss", 0.0) for p in open_positions) + new_max_loss
        max_allowed = (MAX_SIMULTANEOUS_LOSS_PCT / 100) * self.current_capital
        if total_max_loss > max_allowed:
            return False, (
                f"Correlation guard: total max loss {total_max_loss:.0f} "
                f"exceeds {MAX_SIMULTANEOUS_LOSS_PCT}% of capital ({max_allowed:.0f})"
            )

        return True, "OK"

    # ── Size multiplier ──────────────────────────────────────────────────

    def get_size_multiplier(self) -> float:
        """Return 0.5 if weekly drawdown breached, 1.0 otherwise."""
        if self.check_weekly_reduce():
            return 0.5
        return 1.0
