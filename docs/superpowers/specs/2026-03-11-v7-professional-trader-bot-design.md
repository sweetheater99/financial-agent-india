# V7 Professional Trader Bot — Design Spec

**Date:** 2026-03-11
**Status:** Draft
**Capital:** 3-5L INR
**Risk profile:** Moderate (5-8% monthly target, 10% max drawdown)
**Availability:** Fully autonomous (no human intervention during market hours)

---

## Problem Statement

The current V6 bot scans 200 stocks every 5 minutes, runs heavy Claude analysis on each candidate, and rarely trades. In ~10 days of operation, it opened ~7 positions with ₹-548 realized P&L. Claude rejects 95%+ of candidates due to strict multi-indicator consensus requirements, and option chain fetches frequently fail on AngelOne.

The system is an over-engineered signal scanner pretending to be a trader. It needs to be redesigned around how a professional full-time trader actually operates.

---

## Design Principles

1. **Plan then execute.** All analysis happens before market opens. During market hours, the bot executes mechanically.
2. **Fixed watchlist, not a screener.** Trade 10 instruments you know intimately, not 200 strangers.
3. **Level-based, not indicator-based.** Price at a key level with volume = trade. Not 8 indicators agreeing.
4. **Fewer trades, sized correctly.** 2-4 trades/day max. Each one properly sized for risk.
5. **Income-first mindset.** Theta baseline for steady income. Directional trades for upside. Survival mode when losing.

---

## Instruments

**Fixed universe of 10 (actively trade 4-5 per day based on weekly review):**

| Instrument | Type | Lot size | Why |
|---|---|---|---|
| NIFTY | Index | 75 | Core — most liquid, always tradeable |
| BANKNIFTY | Index | 30 | Higher volatility, bigger moves |
| RELIANCE | Stock | 250 | Heavyweight, liquid options |
| HDFCBANK | Stock | 550 | Banking bellwether |
| ICICIBANK | Stock | 700 | 2nd most liquid bank |
| TCS | Stock | 175 | IT sector proxy |
| TATAMOTORS | Stock | 575 | High beta auto |
| BAJFINANCE | Stock | 125 | NBFC, big moves |
| SBIN | Stock | 750 | PSU bank, policy-sensitive |
| INFY | Stock | 300 | IT sector proxy (alt to TCS) |

Watchlist rotates on weekly review — drop worst performer, add a liquid name that's trending.

---

## Trading Style

**Hybrid: momentum in trends + mean reversion in ranges + theta baseline.**

| Market condition | Style | Instrument preference |
|---|---|---|
| Trending day | Momentum — ride breakouts | Buy CE/PE directionally |
| Rangebound day | Mean reversion — fade extremes | Sell credit spreads |
| Low VIX (14-20) | Theta — collect premium | Iron condors on Nifty |
| High VIX (>22) | Buy options (premium is cheap relative to movement) | Protective puts, directional buys |
| Choppy day | No trade | Sit out or theta only |

---

## Holding Period

**Intraday-biased with selective overnight carry.**

- Default: close all positions by 3:15 PM.
- Carry criteria: profit > 1.5%, trend intact, VIX < 20, DTE > 3 days.
- Carry requires a protective hedge (max cost ₹500 or 0.5% of position value).
- Never carry on: expiry day, event tomorrow (RBI/budget/earnings), VIX > 22.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    V7 TRADING BOT                    │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────┐    ┌──────────────────────────┐  │
│  │  STRATEGIST   │    │  LEVEL MEMORY            │  │
│  │  (Claude)     │    │  (Persistent key levels, │  │
│  │               │    │   OI walls, trade grades) │  │
│  │  • Pre-market │◄──►│                          │  │
│  │  • Open read  │    └──────────────────────────┘  │
│  │  • Check-ins  │                                  │
│  │  • Exceptions │    ┌──────────────────────────┐  │
│  │  • EOD review │    │  DATA FEED               │  │
│  └──────┬───────┘    │  (Kite primary,           │  │
│         │            │   AngelOne fallback)       │  │
│         │ playbook   │                           │  │
│         ▼            │  • 1/5/15-min candles      │  │
│  ┌──────────────┐    │  • Live OI (5-min refresh) │  │
│  │  EXECUTOR     │◄──┤  • Option chain + greeks   │  │
│  │  (Mechanical) │    │  • VIX real-time           │  │
│  │               │    │  • Margin status           │  │
│  │  • Level      │    └──────────────────────────┘  │
│  │    watcher     │                                  │
│  │  • Order       │    ┌──────────────────────────┐  │
│  │    manager     │◄──►│  RISK ENGINE             │  │
│  │  • Position    │    │                          │  │
│  │    tracker     │    │  • Per-trade sizing       │  │
│  │  • Carry       │    │  • Daily P&L limit        │  │
│  │    decision    │    │  • Margin utilization     │  │
│  │                │    │  • Correlation check      │  │
│  └──────────────┘    │  • Chop detector          │  │
│                       │  • Brokerage optimizer     │  │
│                       │  • F&O ban list check      │  │
│                       └──────────────────────────┘  │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │  THETA ENGINE (background)                    │  │
│  │  Weekly Nifty iron condor. Runs independently │  │
│  │  from directional trades. Own risk budget.    │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │  JOURNAL & EDGE TRACKER                       │  │
│  │  Grade every trade. Weekly strategy review.   │  │
│  │  Kill strategies with <40% win rate after 30  │  │
│  │  trades. Feed lessons back to Strategist.     │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## Component 1: Strategist (Claude)

Claude acts as the trading desk head. Sets the plan, adapts during the day, reviews at EOD.

### Call Schedule

| Time | Call | Model | Purpose |
|---|---|---|---|
| 8:45 AM | Pre-market playbook | Sonnet | Analyze global cues, set day plan |
| 9:45 AM | Opening read | Sonnet | Classify day type from first 30 min |
| 10:30 AM | Check-in 1 | Sonnet | Adapt plan if morning thesis broke |
| 1:00 PM | Check-in 2 | Sonnet | Afternoon session plan |
| On exception | Emergency | Sonnet | VIX spike, flash crash, unexpected event |
| 3:30 PM | EOD review | Haiku | Grade trades, extract lessons |

**Total: 4-6 Claude calls/day (down from 50+ in V6).**

### Pre-Market Playbook (8:45 AM)

**Inputs:**
- US market close (S&P 500, NASDAQ, Dow)
- GIFT Nifty (indicates expected gap)
- India VIX (previous close)
- FII/DII data (previous day flows)
- Events today and this week (RBI, budget, earnings, F&O expiry)
- Level memory (persistent key levels from recent days)
- Recent trade lessons (from journal)
- MTD P&L and pacing status
- F&O ban list

**Output — the playbook:**

```json
{
  "date": "2026-03-11",
  "market_context": {
    "us_close": "+0.3%",
    "gift_nifty": "24250 (+0.2%)",
    "vix": 17.8,
    "fii_dii": "FII -1200cr, DII +800cr",
    "events_today": [],
    "events_this_week": ["RBI policy Thu"],
    "fo_ban_list": ["DELTACORP", "IBULHSGFIN"]
  },

  "day_classification": "LIKELY_TREND_UP",

  "nifty_plan": {
    "bias": "bullish",
    "key_levels": {
      "resistance_1": 24350,
      "resistance_2": 24500,
      "support_1": 24150,
      "support_2": 24000,
      "opening_range": null
    },
    "setups": [
      {
        "id": "N1",
        "priority": 1,
        "type": "breakout_long",
        "trigger": "15-min candle close above 24350 with volume > 1.5x avg",
        "instrument": "NIFTY CE",
        "strike_logic": "slightly OTM, delta 0.40-0.50",
        "target": 24500,
        "stoploss": 24280,
        "max_risk_pct": 1.5
      },
      {
        "id": "N2",
        "priority": 2,
        "type": "support_bounce",
        "trigger": "15-min candle holds above 24150, RSI < 35 on 5-min",
        "instrument": "NIFTY CE",
        "strike_logic": "ATM, delta 0.50",
        "target": 24300,
        "stoploss": 24100,
        "max_risk_pct": 1.5
      }
    ],
    "no_trade_zone": "24200-24300"
  },

  "stock_plans": [
    {
      "symbol": "HDFCBANK",
      "priority": 3,
      "bias": "bullish",
      "reason": "holding above 1600, sector strength",
      "trigger": "15-min close above 1625 with volume",
      "instrument": "HDFCBANK CE",
      "strike_logic": "ATM, delta 0.45",
      "target": 1660,
      "stoploss": 1610,
      "max_risk_pct": 1.0
    }
  ],

  "theta_plan": {
    "action": "hold",
    "current_position": "NIFTY 23800PE-24600CE condor, 12 DTE",
    "adjustment_trigger": "Nifty above 24450 → tighten call wing"
  },

  "risk_budget": {
    "max_capital_at_risk_today_pct": 4.0,
    "max_trades_today": 4,
    "max_per_trade_risk_pct": 1.5,
    "survival_mode": false,
    "pacing": {
      "mtd_pnl_pct": 2.1,
      "monthly_target_pct": 5.0,
      "status": "on_track"
    }
  },

  "carry_rules": {
    "carry_if": "profit > 1.5%, VIX < 20, DTE > 3",
    "hedge_with": "OTM protective option, max cost 500",
    "never_carry": ["expiry_day", "event_tomorrow", "vix_above_22"]
  },

  "no_trade_conditions": [
    "VIX spikes above 22 intraday",
    "Nifty moves > 2% in either direction",
    "3 consecutive SL hits today",
    "Daily loss > 2% of capital"
  ]
}
```

**Constraints on playbook:**
- Max 2-3 setups per instrument (Plan A and Plan B, not an encyclopedia)
- Max 2-3 stock plans per day (pick the best, don't spread thin)
- Every setup has a specific price trigger (not "buy if bullish")
- No-trade zones explicitly defined
- Priority ranking determines execution order when multiple setups fire simultaneously

### Opening Read (9:45 AM)

After the first 30 minutes of price discovery, Claude updates the playbook:

**Inputs:**
- Opening range: first 30-min high and low
- Gap analysis: gap up/down/flat vs GIFT Nifty expectation
- Gap behavior: filling or extending?
- Volume in first 30 min vs 20-day average opening volume
- OI shift from previous day close (where are new positions being built?)

**Output:**
- Day type confirmation or override (planned trend day but opened choppy → switch to range mode)
- Opening range levels added to playbook
- Setups adjusted if levels were invalidated by gap
- Possible: "No good setups today, theta only"

### Check-ins (10:30 AM, 1:00 PM)

**Inputs:**
- Current P&L for the day
- Open positions and their status
- Whether morning setups fired or not
- Any levels that were tested/broken since last check
- OI changes since morning
- VIX movement

**Output:**
- Confirm or modify remaining setups
- Add new setup if a clear opportunity emerged
- Declare "no trade rest of day" if market is choppy
- Adjust theta engine if needed

### Exception Triggers (Claude emergency call)

The executor calls Claude when something unexpected happens that the playbook doesn't cover:

- VIX jumps > 2 points in 30 minutes
- Nifty moves > 1.5% from day's open
- Watchlist stock moves > 3% intraday (possible news)
- Margin utilization exceeds 70%
- 3 consecutive SL hits in the same day

Claude responds with one of:
- Flatten everything (crash scenario)
- Hold current positions, no new trades
- Adjust SLs on open positions
- Specific action for the exception

---

## Component 2: Executor (Mechanical Engine)

The fast loop. Runs every 3 minutes during market hours. No Claude calls. Pure rule execution.

### Tick Cycle (every 3 min, 9:15-3:30)

```
1. FETCH DATA
   - LTP batch quote for all watchlist instruments (1 API call)
   - 5-min candles for instruments with active setups
   - OI refresh every 5 min (not every tick)
   - VIX current value

2. CARRIED POSITION CHECK (9:15-9:30 only)
   - Gap against position > SL distance → exit in first 5 min
   - Gap in favor → tighten SL to lock gains
   - Flat → treat as normal position with updated levels

3. CHECK PLAYBOOK TRIGGERS
   For each setup (in priority order):
     - Has 15-min candle closed beyond trigger level?
     - Is volume confirming? (> threshold)
     - Is it inside a no-trade zone? → skip
     - Is any no-trade condition active? → skip all
     - Risk budget still available? → if not, skip
     - F&O ban list? → skip
     - Brokerage > 1% of trade value? → skip (trade too small)
     - If all pass → ENTER

4. MANAGE OPEN POSITIONS
   For each position:
     - Update P&L and peak price
     - SL hit → EXIT immediately. No override. No delay.
     - Moved 1:1 R:R → tighten SL to breakeven
     - Target hit → full exit
     - Trailing stop: peak price - 1.5x ATR(5min) on long,
       peak price + 1.5x ATR(5min) on short
     - Drifting: no SL or target, just trailing catches it

5. EXCEPTION DETECTION
   - VIX jump > 2pts in 30 min → Claude call
   - Nifty > 1.5% from open → Claude call
   - Stock > 3% move → Claude call
   - Margin > 70% → Claude call
   - 3 consecutive SL hits → Claude call

6. WIND DOWN (2:30-3:15)
   - Close all intraday positions
   - Evaluate carry criteria on winners
   - Place protective hedge orders on carried positions
   - Close anything that doesn't meet carry criteria
   - Place bracket SL orders on exchange for carried positions
```

### Order Execution

```
ENTRY:
  - Full position size on trigger (no scaling at 1-2 lot sizes)
  - Limit order: bid + ₹0.50 (buying) / ask - ₹0.50 (selling)
  - Not filled in 30 seconds → modify to market
  - Confirm fill, record actual entry price

EXIT:
  - SL: limit order at ask - ₹0.50 (selling) / bid + ₹0.50 (buying)
  - Not filled in 15 seconds → market order (SL exits are urgent)
  - Target: limit order at target price, let it fill passively

BRACKET ORDERS (for overnight carry):
  - Place real SL order on exchange via Kite bracket order API
  - Bot crash / Pi disconnect = SL still active
  - Non-negotiable for autonomous operation
```

### Strike Selection (mechanical, no Claude needed)

```
DIRECTIONAL TRADES (buying CE/PE):
  Risk budget = max_risk_pct × capital
  Example: 1.5% × 3,00,000 = ₹4,500

  Target delta: 0.40-0.50 (good leverage, manageable theta)
  Pick strike where: premium × lot_size ≤ risk budget
  Reject if:
    - Bid-ask spread > ₹2 (Nifty) / ₹5 (BankNifty) / ₹3 (stocks)
    - Premium < ₹10 (too far OTM, will decay to zero)
    - Premium > risk_budget / lot_size (can't afford it)

CREDIT SPREADS (mean reversion / selling premium):
  Sell strike: delta 0.25-0.30
  Buy strike: 2-3 strikes further OTM (hedge leg)
  Net credit must be > ₹15 per lot
  Max loss = (strike width - net credit) × lot_size ≤ risk budget

HEDGES (for overnight carry):
  Buy 3-4 strikes OTM from current price
  Max cost: ₹500 or 0.5% of position value (whichever is less)
```

### Expiry Day Behavior (Thursday)

Separate playbook rules override normal operation:

- No new positions after 1:00 PM
- Only trade ATM ± 1 strike (everything else decays to zero)
- Tighter SL: half of normal distance
- No overnight carry
- Theta engine: close condor by Wednesday if profit > 40%
- Gamma awareness: moves near ATM strikes are amplified and fast

---

## Component 3: Data Feed

**Primary: Kite Connect (Zerodha)**
- Quotes: batch LTP for all instruments (10 req/sec limit)
- Historical: 1-min, 5-min, 15-min candles (3 req/sec limit)
- Option chain: instruments list + quotes for near-money strikes
- Margin: margin calculator API for position sizing
- Orders: bracket orders for exchange-level SL

**Fallback: AngelOne SmartAPI**
- Used when Kite token is expired (with daily Telegram reminder to refresh)
- Limited to: LTP quotes, daily candles
- Not reliable for: option chains, historical intraday data

**Rate limit management:**
- Kite: 3 req/sec historical, 10 req/sec quotes
- 3-min tick cycle with batch quotes = well within limits
- OI refresh every 5 min = ~2-3 extra calls per cycle
- Total: ~10-15 API calls per 3-min tick = safe

**Stale data handling:**
- No fresh LTP for 60 seconds → assume connection lost
- Don't open new positions on stale data
- Existing positions: bracket SL orders on exchange protect them
- Alert on Telegram immediately
- Retry connection every 30 seconds

---

## Component 4: Risk Engine

Runs on every trade decision. No Claude needed — pure rules.

### Per-Trade Sizing

```
risk_amount = capital × max_risk_pct_for_trade
             (e.g., 3,00,000 × 1.5% = ₹4,500)

For options (buying):
  max_lots = floor(risk_amount / (premium × lot_size))
  Usually 1-2 lots at ₹3-5L capital

For spreads (selling):
  max_lots = floor(risk_amount / (max_loss_per_lot))

For futures:
  Not used at ₹3-5L capital (margin too high)
```

### Conviction-Based Sizing

| Conviction | Risk per trade | When |
|---|---|---|
| A+ setup | 2.0% of capital | Multiple confirmations, trend + level + volume + OI |
| B setup | 1.5% of capital | Good trigger, one confirmation missing |
| C setup | 0.75% of capital | Marginal, but playbook says trade |

Conviction is set by the Strategist in the playbook for each setup.

### Daily Limits

| Guard | Threshold | Action |
|---|---|---|
| Daily loss | > 2% of capital | Block all new entries rest of day |
| Consecutive SL hits | 3 in a day | Block all + Claude exception call |
| Margin utilization | > 70% | Block new entries until margin frees |
| Max trades/day | 4 | No more entries (exits still allowed) |
| Correlation | 2+ positions same direction on correlated instruments | Block 3rd position |

### Monthly Pacing

```
First half of month (1st-15th):
  Trade normally per playbook

Second half if AHEAD of target:
  Reduce max_risk_pct by 25%
  Protect gains — don't give back a good month

Second half if BEHIND target:
  Do NOT increase risk to "catch up" (revenge trading)
  Maintain normal sizing
  Accept the month's result

MTD drawdown > 5%:
  SURVIVAL MODE
  - Directional trades blocked
  - Theta engine only (wider wings, smaller size)
  - Continue until next month or drawdown recovers to < 3%

MTD drawdown > 8%:
  FULL STOP
  - No trading for rest of month
  - Review and reset for next month
```

### Drawdown Recovery Plan

| Drawdown | Action |
|---|---|
| 0-3% | Normal trading |
| 3-5% | Reduce position sizes by 25% |
| 5-8% | Survival mode — theta only |
| 8-10% | Full stop for rest of month |
| > 10% | Full stop. Manual review required before restarting. |

### F&O Ban List Check

Before every trade:
- Fetch current F&O ban list from NSE (symbols where OI > 95% of MWPL)
- If symbol is in ban → skip, no new positions allowed
- Existing positions in banned symbols: can only close, not add

### Brokerage Optimization

- ₹20 flat per order (Zerodha)
- Minimum trade value: brokerage must be < 1% of trade value
- At ₹20 brokerage, minimum trade value = ₹2,000
- Prefer fewer larger trades over many small ones
- Round-trip cost (entry + exit) = ₹40 + STT + exchange + GST
- For a ₹5,000 option trade: total cost ~₹80-100 (1.5-2%)
- For a ₹15,000 trade: total cost ~₹120-150 (0.8-1.0%) — much better

### Chop Detection

Bot identifies choppy/no-edge conditions and reduces activity:

```
Chop signals:
  - 3+ whipsaws through a level in first hour (level tested, broken, reclaimed)
  - Opening range < 0.3% of price (very narrow — no conviction)
  - Volume in first hour < 50% of 20-day average
  - VIX flat (< 0.5pt movement) with price going nowhere

If chop detected:
  - 10:30 AM check-in: Claude likely calls "no trade rest of day"
  - Executor: only theta engine operates
  - If already in a position: tighten SL, accept the chop
```

---

## Component 5: Theta Engine

Independent background income strategy. Own risk budget (max 3% of capital at risk).

### Weekly Iron Condor on Nifty

```
ENTRY (Monday or Tuesday):
  Condition: VIX 14-20
  If VIX < 14: premiums too low, skip this week
  If VIX > 20: too volatile, skip this week

  Short CE: delta 0.20 (~300-400pts OTM)
  Short PE: delta 0.20 (~300-400pts OTM)
  Long CE: +200pts further OTM (hedge)
  Long PE: -200pts further OTM (hedge)

  Expiry: next week Thursday (7-12 DTE at entry)
  Target credit: ₹30-50 per lot
  Max risk: (wing width - credit) × lot_size ≤ 3% of capital

DAILY MANAGEMENT:
  Monitor short strike deltas:
    Delta > 0.35 → tighten hedge (buy closer protection)
    Delta > 0.45 → close threatened side, keep profitable side
    Delta > 0.50 → close entire condor (trade has gone wrong)

  Profit management:
    50% of credit captured → close entire condor (take profit)
    This usually happens by Wednesday on a quiet week

TIME MANAGEMENT:
  Close by Wednesday EOD if profit < 50%
  Never hold a condor to Thursday expiry (gamma risk too high)

ADJUSTMENT (if Nifty trends toward one wing):
  Option A: close threatened side at a loss, keep profitable side
  Option B: roll threatened side further OTM (only if net credit still positive)
  Never: hold and hope

SURVIVAL MODE (MTD drawdown > 5%):
  Theta engine = only active strategy
  Wider wings: delta 0.15 instead of 0.20
  Smaller size: 50% of normal lot count
  More conservative profit target: 40% of credit
```

---

## Component 6: Journal & Edge Tracker

### Daily Journal (3:30 PM, Haiku)

For each trade today:
- **Entry quality**: A (trigger + confirmation) / B (trigger, weak confirmation) / C (FOMO/forced)
- **Exit quality**: A (plan followed exactly) / B (minor deviation) / C (panic exit / held too long)
- **Lesson**: one sentence. "HDFCBANK breakout failed because sector was weak — check sector first."

Day summary:
- Trades: X wins, Y losses
- P&L breakdown: ₹X (directional) + ₹Y (theta) = ₹Z total
- Day type prediction accuracy: was the morning call correct?
- Best trade and worst trade

Saved to Obsidian vault: `~/Documents/Obsidian/trading-journal/YYYY-MM-DD.md`

### Weekly Review (Sunday, Sonnet)

**Performance attribution:**

| Dimension | Analysis |
|---|---|
| By strategy | Momentum win%, mean reversion win%, theta win% |
| By instrument | Which names made money, which lost |
| By time of day | Morning trades (9:45-11) vs midday (11-1) vs afternoon (1-2:30) |
| By setup type | Breakout vs support bounce vs fade vs credit spread |
| By day type | How accurate were day classifications? |

**Decisions:**
- Strategy with < 40% win rate after 30+ trades → disable it
- Instrument consistently losing → remove from active watchlist, replace
- Time slot consistently losing → add to no-trade window
- Setup type losing → remove from playbook templates

**Watchlist rotation:**
- Drop worst performing instrument from active list
- Add a liquid name that's been trending well
- Keep universe at 10, active at 4-5

**Level memory update:**
- Levels that held 2+ times this week → strengthen (higher priority next week)
- Levels that broke cleanly → remove or flip (old resistance = new support)
- New levels emerging from this week's price action → add to memory

### Monthly Report (1st of month, Sonnet)

- **P&L report**: gross P&L, transaction costs, net P&L, effective return %
- **Tax estimate**: turnover, estimated tax liability, advance tax due
- **Drawdown analysis**: max drawdown during month, recovery time
- **Strategy allocation for next month**: shift capital toward what's working
- **Capital recommendation**: grow (add funds) / maintain / reduce (withdraw)
- **Withdrawal**: recommend withdrawing 50% of net profit (income goal)
- **Compounding**: remaining 50% stays in trading capital

### Edge Tracking (persistent)

After 50+ trades, the system should know:

```json
{
  "overall_win_rate": 0.55,
  "by_strategy": {
    "momentum_breakout": {"trades": 45, "win_rate": 0.58, "avg_rr": 1.8},
    "mean_reversion": {"trades": 30, "win_rate": 0.52, "avg_rr": 1.3},
    "theta_condor": {"trades": 12, "win_rate": 0.75, "avg_rr": 0.6}
  },
  "by_instrument": {
    "NIFTY": {"trades": 35, "net_pnl": 12000},
    "BANKNIFTY": {"trades": 20, "net_pnl": -3000},
    "HDFCBANK": {"trades": 15, "net_pnl": 5000}
  },
  "by_time": {
    "9:45-11:00": {"trades": 40, "win_rate": 0.60},
    "11:00-13:00": {"trades": 25, "win_rate": 0.48},
    "13:00-14:30": {"trades": 20, "win_rate": 0.55}
  }
}
```

This data feeds back into the Strategist — Claude uses it to weight the playbook toward what actually works.

---

## Component 7: Level Memory (Persistent Store)

Not a new concept, but important enough to spec separately. Levels are the foundation of every trade.

### What gets stored

```json
{
  "NIFTY": {
    "levels": [
      {
        "price": 24000,
        "type": "support",
        "strength": 3,
        "source": "tested 3x in last 5 sessions, held each time",
        "last_tested": "2026-03-10",
        "created": "2026-03-05"
      },
      {
        "price": 24500,
        "type": "resistance",
        "strength": 2,
        "source": "weekly high, OI wall at 24500CE (50L OI)",
        "last_tested": "2026-03-08",
        "created": "2026-03-06"
      }
    ],
    "oi_walls": {
      "call_max_oi_strike": 24500,
      "put_max_oi_strike": 24000,
      "pcr": 1.1
    }
  }
}
```

### How levels are maintained

- **Added by**: Strategist (pre-market analysis), Opening read (opening range), Weekly review
- **Strengthened**: level tested and held → strength + 1
- **Weakened**: level broken on closing basis → strength - 1
- **Removed**: strength drops to 0, or level not tested for 10 sessions
- **Flipped**: resistance broken cleanly → becomes support (and vice versa)
- **OI walls**: refreshed daily from option chain data

---

## Cron Schedule (Pi)

```cron
# V7 Professional Trader Bot
43 8  * * 1-5  ~/financial-agent-india/scripts/v7_premarket.sh
13 9  * * 1-5  ~/financial-agent-india/scripts/v7_opening_read.sh
*/3 9-15 * * 1-5  ~/financial-agent-india/scripts/v7_executor.sh
28 10 * * 1-5  ~/financial-agent-india/scripts/v7_checkin.sh
58 12 * * 1-5  ~/financial-agent-india/scripts/v7_checkin.sh
33 15 * * 1-5  ~/financial-agent-india/scripts/v7_eod.sh
50 8  * * *    ~/financial-agent-india/scripts/kite_token_check.sh
3  10 * * 0    ~/financial-agent-india/scripts/v7_weekly_review.sh
7  10 1 * *    ~/financial-agent-india/scripts/v7_monthly_report.sh
```

---

## Multiple Timeframe Confirmation

The Strategist sets levels from daily charts. The Executor confirms entries on intraday charts.

| Timeframe | Used for | By whom |
|---|---|---|
| Daily | Key levels, overall trend, support/resistance | Strategist (pre-market) |
| Hourly | Session trend direction | Strategist (check-ins) |
| 15-min | Entry trigger confirmation (candle close beyond level) | Executor |
| 5-min | Trailing stop calculation (ATR), momentum confirmation | Executor |
| 1-min | Order execution timing (not for decisions) | Executor |

---

## Position Rolling

When an option is losing value to theta but the thesis is still valid:

```
Conditions for rolling:
  - Position in loss (20-40% of premium lost)
  - Original thesis still valid (level holding, trend intact)
  - DTE < 5 (theta accelerating)
  - Next week expiry available with similar strike

Action:
  - Close current position (take the loss)
  - Open same strike or nearest equivalent in next week expiry
  - Net cost = (new premium - old premium exit)
  - Only roll once per trade. If rolled position also fails → exit.

Do not roll if:
  - Loss > 50% (thesis is wrong, not just time decay)
  - VIX has spiked (new premium will be expensive)
  - Thesis has changed (level broken, trend reversed)
```

---

## Seasonal Awareness

The Strategist should adjust aggression based on market seasonality:

| Period | Behavior | Reason |
|---|---|---|
| Budget week (Feb) | Reduce size 50%, no theta | High volatility, unpredictable |
| F&O expiry week | Tighter exits, no carry | Gamma risk, settlement pressure |
| Election results | No trading day-of | Extreme gaps possible |
| April-May | Normal to aggressive | FII rebalancing creates trends |
| June-July pre-monsoon | Conservative, prefer theta | Historically choppy |
| Oct-Nov Diwali | Normal | Muhurat trading is ceremonial only |
| Dec-Jan | Conservative | Low volumes, holiday season |

---

## Margin Awareness

Before every entry, check margin impact:

```
current_margin_used = sum of margin for all open positions
new_trade_margin = estimated margin for proposed trade
total_after = current_margin_used + new_trade_margin
available_margin = capital (or broker margin if using leverage)

if total_after / available_margin > 0.70:
  → BLOCK entry (leave 30% buffer for MTM swings)

For spreads: margin = max loss (much lower than naked positions)
For bought options: margin = premium paid (no additional margin)
For sold options (condor legs): margin = SPAN + exposure
```

Use Kite margin calculator API for accurate margin estimates.

---

## Telegram Integration

All alerts go to Ops Hub → Stocks topic.

| Event | Format | Urgency |
|---|---|---|
| Pre-market playbook | Full plan summary | Low (informational) |
| Trade entry | Symbol, strike, price, SL, target, risk | High |
| Trade exit | Symbol, P&L, exit reason | High |
| Check-in update | Plan unchanged / modified | Low |
| Exception | What happened, what bot is doing | High |
| EOD summary | Day's P&L, open carried positions | Medium |
| Weekly report | Performance breakdown | Low |
| Error/disconnect | Connection lost, SL orders status | Critical |

---

## Migration from V6

### What to keep
- Kite + AngelOne data layer (kite_data.py, connect.py)
- Transaction cost model (paper_trade.py cost functions)
- Telegram alerting infrastructure
- Obsidian journal integration
- Risk manager drawdown tracking (risk_manager.py)
- VIX tier classification (config.py)

### What to replace
- Screener (screener.py) → fixed watchlist + level memory
- Claude intel batch evaluation (claude_intel.py) → Strategist with playbook format
- Paper trade open/monitor loop (paper_trade.py) → Executor with level watching
- 200-stock scan every 5 min → 10 instruments, price watching only
- Multi-indicator consensus (RSI + ADX + OBV + VWAP + OI + news + regime) → price at level + volume confirmation

### What to add
- Level memory (persistent store)
- Theta engine (independent condor strategy)
- Edge tracker (strategy performance attribution)
- Multiple timeframe data (currently daily only)
- Bracket orders for exchange-level SL
- Margin calculator integration
- F&O ban list check
- Chop detection
- Monthly pacing and survival mode
- Position rolling logic

### Transition plan
- Run V7 in paper mode alongside V6 for 2-4 weeks
- Compare: trades taken, P&L, drawdown
- Switch to V7 paper-only once confident
- Go live with minimum size after 50+ paper trades with positive expectancy

---

## Success Criteria

After 3 months of paper trading:

| Metric | Target |
|---|---|
| Monthly return | 3-5% average |
| Max monthly drawdown | < 10% |
| Win rate | > 50% |
| Average R:R | > 1.5:1 |
| Trades per day | 2-4 average |
| Theta income | Positive in 8/12 weeks |
| Days with no trades (chop) | 20-30% of trading days |
| Survival mode activations | < 2 per quarter |

---

## Open Questions

1. **Exact watchlist**: The 10 names listed are a starting point. Should be validated against current lot sizes and margin requirements at ₹3-5L.
2. **Paper vs live timeline**: How long to paper trade before going live?
3. **Capital allocation split**: What % for directional vs theta? Starting suggestion: 70% directional budget, 30% theta budget.
4. **Kite token automation**: Can we automate the daily Kite OAuth refresh via Playwright to remove the manual step?
