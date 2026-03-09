# Overnight Hedge Protection — Design Doc

**Date:** 2026-03-09
**Status:** Approved
**Approach:** Standalone module (Approach A)

## Problem

Naked F&O positions (momentum options, single-leg OPT, futures) can carry overnight with zero hedging. No EOD scan, no protective legs, no forced closure. Overnight gaps can cause outsized losses on leveraged positions.

## Scope

**Covered instruments:** MOMENTUM, OPT, FUT (naked F&O)
**Not covered:** SPREAD, CONDOR, IV_CRUSH (already self-hedged)

## Decision Framework

Claude decides per-position with hard guardrails that cannot be overridden:

| Condition | Action | Override? |
|-----------|--------|-----------|
| Position in loss (P&L < 0%) | CLOSE | Never |
| VIX > 20 | HEDGE or CLOSE only | Never |
| Gain < 30% | HEDGE or CLOSE only | Never |
| Expiry <= 2 trading days | CLOSE | Never |
| Gain > 50% + strong signals + VIX < 20 | Claude can approve CARRY_NAKED | Only this case |

If Claude says CARRY_NAKED but gain < 50%, override to HEDGE.

## Architecture

### New Module: `overnight_hedge.py`

Single entry point: `run_overnight_hedge_scan()` called at 3:15 PM IST.

```
3:15 PM cron tick
  |
  +-- Load portfolio, filter naked F&O (instrument in MOMENTUM, OPT, FUT)
  |
  +-- Hard Guardrails (deterministic, no Claude):
  |     +-- In loss -> CLOSE
  |     +-- VIX > 20 -> HEDGE or CLOSE
  |     +-- Gain < 30% -> HEDGE or CLOSE
  |     +-- Expiry <= 2 days -> CLOSE
  |     +-- Passed all? -> Send to Claude
  |
  +-- Claude Decision:
  |     Input: position data, LTP, P&L%, regime, VIX, global intel, OI context
  |     Output: {action: "close"|"hedge"|"carry_naked", reasoning: "..."}
  |     Hard override: carry_naked requires gain > 50%
  |
  +-- Execute:
  |     CLOSE -> existing close logic, exit_reason="overnight_risk"
  |     HEDGE -> add protective leg (details below)
  |     CARRY_NAKED -> tighten stop-loss by 20%, log approval
  |
  +-- Telegram summary
  +-- Save portfolio
```

### Hedge Leg Mechanics

When HEDGE is chosen:

1. **Strike selection:**
   - Bullish position -> buy OTM PE (250 pts OTM for Nifty, 500 for BankNifty)
   - Bearish position -> buy OTM CE (same distances)
   - Futures -> buy OTM option opposite to direction

2. **Cost cap:** Max 1% of position value. If hedge costs more -> fallback to CLOSE.

3. **Position tracking:** New `hedge_leg` dict on the position:
   ```json
   {
     "hedge_leg": {
       "option_type": "PE",
       "strike": 24800,
       "premium": 45.0,
       "quantity": 75,
       "cost": 3375.0,
       "added_date": "2026-03-09"
     }
   }
   ```

4. **Live execution:** Hedge leg order flows through existing `live_execution.py` — same Telegram confirmation, price deviation check, order ID tracking, kill switch integration.

### Morning Unwind

Next trading day at 9:30 AM (first monitor tick):

1. Detect positions with `hedge_leg` field
2. Get current premium of the hedge option
3. Close the hedge leg
4. Calculate hedge P&L:
   - Gap went against position -> hedge made money -> log as "hedge_saved"
   - Gap was favorable -> hedge lost premium -> log as "insurance_cost"
5. Remove `hedge_leg` from position, continue normal monitoring

### Carry Naked Logic

When CARRY_NAKED is approved (gain > 50%, VIX < 20, strong signals):

1. Tighten stop-loss by 20% (closer to current price)
2. Log Claude's reasoning
3. Morning after: no special handling, normal monitoring continues

## Config Additions (`config.py`)

```python
# Overnight hedge protection
OVERNIGHT_HEDGE_ENABLED = True
OVERNIGHT_HEDGE_TIME_START = "15:15"
OVERNIGHT_HEDGE_TIME_END = "15:25"
OVERNIGHT_MIN_GAIN_FOR_NAKED_CARRY = 0.50
OVERNIGHT_MIN_GAIN_FOR_HEDGE = 0.30
OVERNIGHT_VIX_NAKED_BLOCK = 20
OVERNIGHT_EXPIRY_CLOSE_DAYS = 2
OVERNIGHT_HEDGE_MAX_COST_PCT = 0.01
OVERNIGHT_HEDGE_OTM_POINTS_NIFTY = 250
OVERNIGHT_HEDGE_OTM_POINTS_BANKNIFTY = 500
OVERNIGHT_STOP_TIGHTEN_PCT = 0.20
```

## Cron Integration

Add to `scripts/paper_trade_cron.sh` between monitor and EOD wrap:

```bash
# --- 3:15-3:25 PM: Overnight hedge scan ---
if [ "$HOUR" -eq 15 ] && [ "$MIN" -ge 13 ] && [ "$MIN" -le 27 ]; then
    echo "[HEDGE] Overnight hedge scan" >> "$LOG"
    python overnight_hedge.py >> "$LOG" 2>&1
fi
```

## Telegram Message Format

```
OVERNIGHT HEDGE SCAN

CLOSED (2):
- momentum_2026-03-09_NIFTY_bullish | Loss -12% | closed
- opt_RELIANCE_CE | Expiry tomorrow | closed

HEDGED (1):
- fut_NIFTY | +35% gain | bought 24800 PE @ 45
  Cost: 3,375 (0.8% of position)

CARRY NAKED (0):
(none approved today)

VIX: 16.2 | Regime: TRENDING_UP
```

## Files Touched

| File | Change |
|------|--------|
| `overnight_hedge.py` | NEW — main module (~300 lines) |
| `config.py` | Add overnight hedge config block |
| `paper_trade.py` | Add hedge_leg handling in monitor + morning unwind |
| `scripts/paper_trade_cron.sh` | Add 3:15 PM cron entry |
| `claude_intel.py` | Add overnight hedge prompt template |
| `tests/test_overnight_hedge.py` | NEW — test guardrails + decision logic |

## Edge Cases

1. **No naked positions at 3:15 PM** — skip silently, no Telegram spam
2. **SmartAPI down at 3:15 PM** — can't fetch LTP/option chain -> close all naked (fail-safe)
3. **Claude timeout** — default to HEDGE for profitable, CLOSE for unprofitable
4. **Hedge option illiquid** — if bid-ask spread > 5%, skip hedge -> CLOSE instead
5. **Multiple naked positions** — process sequentially, each gets independent decision
6. **Position already has hedge_leg** — skip (already hedged from previous day)
7. **Morning unwind fails** — retry on next monitor tick, alert via Telegram
