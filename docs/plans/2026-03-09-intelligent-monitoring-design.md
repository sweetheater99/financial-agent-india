# Intelligent Monitoring & Adaptive Learning Design

> **Goal:** Transform the paper trading bot from a 30-min cron into a full-time F&O desk trader — smart tiered monitoring, structured Telegram comms, daily circuit breakers, correlated risk management, and a learning loop that gets smarter over time.

## 1. Smart Tiered Monitoring

Replace flat 30-min cron with urgency-based checking.

### Classification (every run)
For each open position, compute `urgency`:
- **HOT** (check every 5 min): within 2x ATR of target/SL, options DTE < 5, VIX spike day (>15% intraday change), position P&L > +5% or < -3%
- **WARM** (check every 10 min): all other open positions
- **COLD** (check every 30 min): no open positions — heartbeat + regime pulse only

### Cron Schedule
```
# New schedule: every 5 min during market hours
*/5 9-15 * * 1-5  ~/financial-agent-india/scripts/paper_trade_cron.sh
```

Single script determines what to check based on last-check timestamps stored in `data/paper_trades/monitor_state.json`:
```json
{
  "last_check": {"MCX": "2026-03-09T11:40:00", "BPCL": "2026-03-09T11:35:00"},
  "last_regime_check": "2026-03-09T11:30:00",
  "circuit_breaker_active": false,
  "daily_realized_loss": 0
}
```

### Auction Window Protection
- 9:15-9:30 and 3:15-3:30: widen exit thresholds by 0.5x ATR (avoid false exits on auction volatility)
- No new entries during first 5 min (9:15-9:20) — let prices settle

## 2. Structured Telegram Notifications

### Morning Briefing (9:10 AM)
- Overnight global cues (US markets, GIFT Nifty gap, FII/DII)
- Regime expectation + VIX
- Open positions and what to watch
- Claude's market view for the day
- Triggered by: `paper_trade_cron.sh briefing` (move from 3 AM to 9:10 AM)

### Immediate Alerts
- Entry executed + Claude reasoning
- Exit executed + Claude reasoning
- Claude HOLD override (rules said exit, Claude said wait)
- Regime change mid-day (e.g., SIDEWAYS → VOLATILE)
- Position hits breakeven (trailing SL moved to entry)
- Circuit breaker activated
- Sector correlation alert (tightened stops)

### Hourly Digest (10:00, 11:00, 12:00, 1:00, 2:00)
- All positions: symbol, LTP, P&L %, distance to target/SL
- Unrealized + realized P&L
- Current regime + VIX
- Positions flagged HOT (close to exit)

### EOD Wrap (3:20 PM)
- Day's entries and exits with P&L
- Portfolio P&L (daily + cumulative)
- Positions carrying overnight with risk assessment
- Claude's daily review (what went well, what didn't)

## 3. Intraday Re-entry (12:45 PM Pass)

Second lighter screening at ~12:45 PM for afternoon setups:
- Only candidates with score > 6
- Must pass Claude approval (high conviction only)
- Max 2 new positions in afternoon pass
- No afternoon entries if circuit breaker active
- No afternoon entries on Friday (weekend risk)

## 4. Daily Drawdown Circuit Breaker

```python
DAILY_LOSS_CIRCUIT_BREAKER_PCT = 3.0  # % of capital
```

- Track daily realized losses in monitor_state.json
- If cumulative daily realized loss exceeds 3% of capital:
  - Block ALL new entries for rest of day
  - Send Telegram alert: "Circuit breaker: daily loss ₹X (Y%) — no new trades today"
  - Continue monitoring exits (don't block exits)
  - Reset at 9:10 AM next trading day

## 5. Sector Correlation Protection

During every hot/warm check:
- Fetch Nifty LTP (already done for regime)
- If Nifty moves > 1.5% against position direction since entry:
  - Count how many open positions are in the same direction
  - If >= 3 correlated positions exposed: tighten all their SLs by 0.5x ATR
  - Send Telegram: "Correlation alert: Nifty down 2%, tightened SL on 3 bullish positions"

For sector-specific:
- If 2+ positions share a sector and sector index drops > 2%: same treatment
- Track `nifty_at_entry` on each position for comparison

## 6. Adaptive Learning

### Post-Trade Review (on every exit)
After `close_position()`, Claude reviews the trade:
```
Entry thesis: [categories, score, regime at entry]
What happened: [P&L, hold time, exit reason]
Was the entry good? Was the exit timing right?
One lesson learned.
```
Stored in `data/trade_journal/lessons.json` (append-only, last 100 lessons).

### Pattern Detection (weekly, Sunday 10 AM)
Claude reviews all lessons from the past 2 weeks:
- Group by instrument, direction, regime, sector
- Identify patterns: "bearish puts in VIX>20 have 65% win rate" or "IT sector shorts underperform when DII buying"
- Store in `data/trade_journal/patterns.json`

### Feedback Into Decisions
- `evaluate_entry()` prompt includes last 5 relevant lessons + active patterns
- `evaluate_exit()` prompt includes lessons about exit timing
- Patterns influence Claude's conviction and allocation adjustment

### Confidence Tracking
- Store `claude_conviction` on every position
- Weekly report includes: high/medium/low conviction win rates
- If high-conviction calls underperform medium, flag for prompt review

### Monthly Health Check
- If profit factor drops below 1.0 over trailing 30 trades: alert
- If win rate drops below 40%: alert with Claude's diagnosis
- Claude suggests specific parameter tweaks (not auto-applied — user decides)

## 7. Implementation Priority

1. **Smart tiered monitoring** — new cron schedule + monitor_state.json + urgency classification
2. **Circuit breaker** — simple, critical safety net
3. **Structured Telegram** — morning briefing, hourly digest, EOD wrap
4. **Sector correlation protection** — during monitoring
5. **Intraday re-entry** — afternoon pass
6. **Post-trade learning** — lessons on every exit
7. **Weekly patterns + confidence tracking** — learning loop
8. **Monthly health check** — adaptation

## Constraints

- Single AngelOne session per cron run (no parallel auth)
- 1.5s API delay between calls (hard requirement)
- Claude CLI ~5-10s per call (Max subscription, no cost)
- Pi resource: lightweight, no persistent daemon — cron-based only
- All state in JSON files (no database)
