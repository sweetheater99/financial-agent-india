# V6: Claude-First Architecture Design

> **Goal:** Streamline the rule engine to let Claude drive all trade decisions. Remove 45+ pre-Claude filters and allocation modifiers. Rules compute signals and enforce safety limits — Claude makes every entry and exit decision within those limits.

## Architecture

**Before (V5):** Screener → 45+ rule filters → Claude opinion → 8-layer allocation cascade → open position. Rules trigger exits → Claude can override.

**After (V6):** Screener → Claude decision (with full context) → safety cap → open position. Rules compute exit signals → Claude decides → mechanical SL as backstop.

Three layers only:

| Layer | Role | Overridable? |
|-------|------|-------------|
| **Data Layer** | Screener scoring, regime detection, global intel, indicators, exit signal computation | N/A (provides context) |
| **Claude Layer** | All entry/exit decisions, allocation sizing, conviction | Only by Safety Layer |
| **Safety Layer** | Stoploss, circuit breaker, position limits, portfolio heat, event calendar | Never |

## 1. Entry Flow

```
Screener top N candidates (ranked by score)
    ↓
Fetch context: regime, VIX, FII/DII, PCR, max pain, macro, supertrend
    ↓
Safety pre-check (hard limits only):
  - Circuit breaker active? → block ALL
  - Portfolio heat > 6%? → block ALL
  - Max concurrent positions reached? → block ALL
  - Event calendar (budget/RBI day)? → block ALL
  - Capital < minimum? → block ALL
    ↓
Claude evaluates each candidate:
  - Sees: score, categories, RSI, volume, supertrend, news, sector
  - Sees: regime, VIX, PCR, max pain, FII/DII, macro, Nifty LTP
  - Sees: full portfolio (all open positions, sector exposure, directional bias)
  - Sees: recent trade lessons
  - Sees: recent daily journal entries (last 2-3 days)
  - Returns: TRADE/SKIP, conviction, allocation_adj (0.5-1.5), reasoning
    ↓
Safety post-check (cap allocation):
  - Max single position: 12% of capital
  - Cash reserve: 40% minimum
  - Max same sector: 2
    ↓
Store structured entry thesis on position
    ↓
Open position
```

**Removed:** Regime directional blocks, CASH regime hard-block, RSI/volume/news hard filters, PCR gate, max pain gate, supertrend disagreement penalty, macro BLOCK_BULLISH/REDUCE, 8-layer allocation cascade (regime × sector_rotation × breadth × correlation × drawdown × event × heat).

**Kept as context for Claude:** All the same data — regime, VIX tier, FII/DII, PCR, supertrend, RSI, volume. Nothing auto-blocks except safety limits.

## 2. Exit Flow

```
Every monitoring tick, for each open position:
    ↓
Compute exit signals (rule math, NO auto-trigger):
  - Trailing SL distance (ATR-based, with activation threshold)
  - Target distance
  - Time pressure (days held vs max hold)
  - Theta decay (options: DTE, premium erosion rate)
  - Partial profit threshold reached?
  - Auction window active?
    ↓
Mechanical SL check (ALWAYS fires, never reaches Claude):
  - Fixed stoploss hit? → EXIT immediately
  - Gap through SL? → EXIT immediately
    ↓
Any non-SL signal fired?
  - No → HOLD (skip Claude call, save latency)
  - Yes → Claude evaluates:
    - Sees: position state + all computed signals
    - Sees: original entry thesis + what would invalidate it
    - Sees: previous assessment for this position (continuity)
    - Sees: market context (regime, VIX, sector)
    - Returns: EXIT/HOLD/PARTIAL, reasoning
    ↓
Store Claude's assessment on position (for next tick continuity)
    ↓
Execute Claude's decision
```

**Key change from V5:** Rules no longer trigger exits. They compute signals and pass them to Claude. Claude gets a holistic view:

```
POSITION SIGNALS:
- Trailing stop: TRIGGERED (LTP ₹245 <= trail ₹246)
- Target: 2.1% away
- Time: day 7/10 (70% elapsed)
- Theta: -1.8%/day (DTE 4)
- Auction window: No

ENTRY THESIS:
- "Entered on strong OI buildup + volume breakout in TRENDING_UP regime"
- Invalidation: "Price below ₹238 or regime shift to VOLATILE"

PREVIOUS ASSESSMENT (5 min ago):
- HOLD: "Trailing stop triggered but RSI showing bullish divergence..."
```

## 3. Structured Entry Thesis

Every position stores a structured thesis at entry:

```json
{
  "entry_thesis": {
    "reasoning": "Strong OI buildup with 2.5x volume breakout...",
    "conviction": "high",
    "key_conditions": {
      "regime": "TRENDING_UP",
      "vix": 16.2,
      "fii_net": -2300,
      "sector": "Metals"
    },
    "invalidation": "Price below ₹238 or regime shift to VOLATILE",
    "expected_hold": "5-7 trading days",
    "target_scenario": "Breakout continuation toward ₹270 resistance"
  }
}
```

This feeds into:
- Exit evaluation: "Has the original thesis played out or been invalidated?"
- Post-trade review: "Was the entry thesis correct?"
- Daily journal: human-readable reasoning for every position

## 4. Position-Level Claude Memory

Store Claude's last assessment per position in `monitor_state.json`:

```json
{
  "position_assessments": {
    "MCX": {
      "timestamp": "2026-03-09T12:15:00+05:30",
      "action": "HOLD",
      "reasoning": "Trailing stop triggered but RSI showing bullish divergence...",
      "signals_at_assessment": ["trailing_stop_near"]
    }
  }
}
```

Fed back into the next evaluation prompt as "PREVIOUS ASSESSMENT." Prevents flip-flopping (HOLD → EXIT → HOLD on consecutive ticks) and gives Claude continuity across ticks.

Cleared when position closes.

## 5. Portfolio-Level Entry Prompt

`evaluate_entry()` prompt includes full portfolio awareness:

```
PORTFOLIO STATE:
- Open: 4 positions (3 bullish EQ, 1 bearish PUT)
- Sectors: IT (2), Metals (1), Pharma (1)
- Directional bias: 75% bullish
- Unrealized P&L: ₹+450 (+0.45%)
- Capital deployed: 52% (₹48,000 available)
- Today's realized: ₹-200 (1 loss)

RECENT LESSONS:
- [B] BPCL (exit_timing): "Held too long through trailing stop, should have exited on first trigger"
- [A] MCX (entry_timing): "Strong volume breakout in trending regime, textbook entry"
```

Claude can reason: "Already 3 bullish, adding a 4th increases correlation risk" or "Only 52% deployed, room for another position."

## 6. Daily Trading Journal → Obsidian Vault

At EOD wrap (3:20 PM), append to `~/Documents/Obsidian/daily/YYYY-MM-DD.md`:

```markdown
## Trading Journal

### Entries
- **MCX** (EQ, bullish) @ ₹2,557 — "Strong OI buildup with volume breakout in TRENDING_UP regime. VIX 16.2 supports risk-on."

### Exits
- **BPCL** (PUT, bearish) @ ₹42.50 — SL hit, P&L: -₹380 (-3.2%)
  - Lesson [C]: "Entered against rising DII support, bearish thesis weakened by institutional buying"

### Portfolio
- Realized: ₹-380 | Unrealized: ₹+450 | Net: ₹+70
- Open: MCX (bullish +1.2%), INFY (bullish +0.3%)

### Market Context
- Regime: TRENDING_UP | VIX: 16.2 | FII: -2,300cr | PCR: 0.85
```

**Morning briefing reads last 2-3 daily journal entries** — Claude learns from its own recent trading history. Creates a continuous improvement loop.

**Implementation:** Use the Obsidian vault helper (`Vault().write_draft()` to `_drafts/daily/`) or append directly to `daily/` via the EOD wrap in `paper_trade_cron.sh`. Since this is structured data the bot generates (not user notes), direct write to `daily/` is appropriate — or a dedicated `daily/trading/` subdirectory to keep it separate from personal daily notes.

## 7. Claude-Down Fallback

Inverts current fail-open to **fail-safe**:

| Scenario | Behavior |
|----------|----------|
| Entry evaluation fails | **Block entry** — no positions without Claude |
| Exit evaluation fails | **Execute the triggered exit** — honor rule-computed signal |
| Claude intermittent | Track `claude_consecutive_failures` in monitor_state.json |

### Graduated Escalation

| Consecutive Failures | Response |
|---------------------|----------|
| 1 | Retry next 5-min tick, no alert |
| 3 | Telegram: "⚠️ Claude intermittent — entries paused" |
| 5+ | Full lockdown: entries blocked, exits on autopilot, Telegram with diagnosis |
| Recovery | Auto-resume, Telegram: "✅ Claude back online" |

## 8. Claude System Prompt Update

Claude's system prompt absorbs removed rules as soft guidance:

```
You are a full-time F&O desk trader for Indian markets (NSE). You receive
market data and candidates — you decide what to trade.

ROLE:
- You make ALL entry and exit decisions
- You see the full portfolio, not just individual positions
- You learn from recent trade lessons and daily journal

GUIDELINES (weigh against signal quality — not hard rules):
- Regime matters: TRENDING_UP favors bullish, TRENDING_DOWN favors bearish,
  SIDEWAYS favors range-bound. Strong signals can override weak regime.
- VIX > 22: elevated risk, favor smaller positions or high-conviction only
- VIX > 28: crisis, sit out unless exceptional setup
- Heavy FII selling (>5000cr): bearish pressure, but DII absorption moderates
- PCR < 0.7: excessive bullish sentiment, contrarian bearish
- PCR > 1.3: excessive bearish sentiment, contrarian bullish
- Supertrend disagreement: weakens conviction, doesn't disqualify
- Friday afternoon: weekend risk for F&O, prefer closing marginal positions
- Afternoon entries (12:45+): only high conviction, max 2 new positions

HARD CONSTRAINTS (you cannot override):
- Never recommend allocation > 1.5x base
- If unsure, SKIP — missed trades cost nothing
- Never override a stoploss exit
- Respect position limits (max 8, max 2/sector)
```

## 9. Config Cleanup

### Remove (~40 constants):
- `US_CRASH_BLOCK_PCT`, `US_SEVERE_CRASH_PCT`, `US_MILD_RED_PCT`, `NASDAQ_IT_CRASH_PCT`
- `GIFT_GAP_REDUCE_PCT`, `GIFT_GAP_BLOCK_PCT`
- `FII_HEAVY_SELL_CRORES`, `FII_EXTREME_SELL_CRORES`, `DII_SUPPORT_MODERATE_PCT`
- `PCR_EUPHORIA`, `PCR_EXTREME_CALL`, `PCR_EXTREME_PUT`
- `SUPERTREND_DISAGREE_REDUCTION`
- `CPR_NARROW_PCT`, `CPR_WIDE_PCT`
- `SECTOR_ROTATION_BOOST`, `SECTOR_ROTATION_CUT`
- `OI_UNUSUAL_CAUTIONARY_CUT`
- All `ALLOC_*_MAX` percentage caps
- `DRAWDOWN_DAILY_HALT`, `DRAWDOWN_WEEKLY_REDUCE`, `DRAWDOWN_MONTHLY_HALT` (keep circuit breaker)
- `CONSECUTIVE_LOSS_PAUSE`
- `CASH_TRIGGER_*` constants

### Keep:
- `DAILY_LOSS_CIRCUIT_BREAKER_PCT` — safety
- `CASH_RESERVE_PCT`, `MAX_STRATEGY_ALLOC_PCT` — safety caps
- `MAX_CONCURRENT_POSITIONS`, `MAX_SAME_SECTOR`, `MAX_PORTFOLIO_HEAT_PCT` — position limits
- All SL/target/trailing ATR constants — used for signal computation
- `API_DELAY` — rate limiting
- Event dates (budget, RBI) — compliance
- `PAPER_TRADE_SLIPPAGE_PCT` — realism

## 10. Files Changed

| File | Change |
|------|--------|
| `paper_trade.py` | Remove pre-Claude filters in `open_positions()`, restructure exit flow in `monitor_positions()`, store entry thesis, position-level memory |
| `claude_intel.py` | New system prompt, portfolio-level entry prompt, exit signal prompt (replaces trigger-based), Claude-down tracking |
| `smart_monitor.py` | Add position assessments to state, Claude failure tracking, daily journal writer |
| `config.py` | Remove ~40 dead constants |
| `global_intel.py` | `compute_hard_gate()` returns context dict instead of block/reduce actions |
| `risk_guardrails.py` | Remove graduated drawdown multipliers, keep circuit breaker + portfolio heat |
| `scripts/paper_trade_cron.sh` | Add EOD journal write step |

## 11. New Market Intelligence Signals

Based on analysis of top Indian F&O traders (PR Sundar, Nitin Murarka, Reyaansh, professional desk traders), these signals are missing from the current system and should be added to Claude's context.

### 11.1 VWAP (Volume Weighted Average Price)

Currently not tracked at all. Every serious Indian intraday trader uses VWAP as the institutional benchmark.

**Data:** Compute VWAP from intraday candles (SmartAPI `getCandleData` with 5-min interval for the day).

**Feed to Claude:**
```
VWAP: ₹2,551 | LTP: ₹2,571 (+0.8% above VWAP) — bullish confirmation
```

**How pros use it:**
- Price above VWAP = bullish bias, below = bearish
- Don't enter against VWAP direction unless strong conviction
- VWAP acts as dynamic intraday support/resistance
- Combine with OI for high-probability setups

**Implementation:** New function `compute_vwap(smart_api, token)` → fetch today's 5-min candles, compute cumulative (price × volume) / cumulative volume. Cache per symbol per session.

### 11.2 Real-Time OI Change (Delta OI)

We track static OI categories from the screener (Long Built Up, Short Built Up, etc.) but don't monitor intraday OI changes on open positions. Pro traders watch OI changes every 15-30 minutes.

**The four patterns:**

| OI Change | Price Change | Signal | Strength |
|-----------|-------------|--------|----------|
| OI up | Price up | Long buildup | Strong bullish |
| OI up | Price down | Short buildup | Strong bearish |
| OI down | Price up | Short covering | Weak bullish |
| OI down | Price down | Long unwinding | Weak bearish |

**Feed to Claude:**
```
MCX: OI changed +12% since entry → Long buildup (confirms bullish thesis)
BPCL: OI changed -8% with price falling → Long unwinding (thesis weakening)
```

**Implementation:** Store OI at entry on each position (`oi_at_entry`). During monitoring, fetch current OI via option chain or futures data. Compute delta and classify the pattern. Pass to Claude's exit evaluation prompt.

### 11.3 Expiry Day Behavior

Currently no differentiation between expiry day and regular trading day. Pro traders have strict expiry rules because gamma explosion makes ATM options move 2-3x faster.

**Rules for Claude's system prompt:**
- Expiry day: gamma is elevated, ATM options are hypersensitive — widen SL or reduce size
- No new option positions after 2 PM on expiry — theta decay makes timing impossible
- OTM options on expiry: never average, never hold hoping for last-minute spike
- Max pain gravity: price gravitates to max pain strike on expiry (we compute max pain already)
- 3 PM cutoff: close all uncertain option positions before final 30 minutes
- Weekly expiry = Thursday, monthly expiry = last Thursday of month

**Implementation:** New function `is_expiry_day()` and `get_expiry_type()` (weekly/monthly). Feed into Claude prompt: "Today is weekly expiry. ATM gamma elevated. Reduce option position sizes."

### 11.4 OI-Based Support/Resistance Levels

Highest put OI strike = strong support. Highest call OI strike = strong resistance. Pro traders use these as dynamic levels that update throughout the day.

**Feed to Claude:**
```
OI S/R Levels:
- Max Put OI: 23,800 PE (support) — current price 23,912
- Max Call OI: 24,200 CE (resistance)
- Sudden OI spike: 24,000 CE gained 85,000 contracts since morning → new resistance forming
```

**How pros use it:**
- Don't enter bullish above highest call OI (resistance ceiling)
- Don't enter bearish below highest put OI (support floor)
- Watch for sudden OI spikes (50k+ contracts) as new S/R formation
- When support/resistance breaks (OI unwinds), trend acceleration expected

**Implementation:** Already have `oi_analysis.py` with option chain data. Add `get_oi_support_resistance()` → returns max put OI strike, max call OI strike, and any large OI changes since last check.

### 11.5 Gap Handling Rules

No overnight gap logic currently. When Nifty gaps >1% at open, pro traders follow specific protocols.

**Rules for Claude's system prompt:**
- Gap up/down >1%: wait 15-30 minutes for price to settle before new entries
- Gap-up + fade (price falls back) = potential reversal/short setup
- Gap-up + hold (price sustains above gap level) = trend continuation
- Existing positions: if gap goes through SL, immediate exit (already implemented via gap-through-SL); if gap moves toward target, consider partial profit booking on the gap
- Gap fills are common in first 60 minutes — don't chase the gap direction

**Implementation:** Compute gap % in morning briefing by comparing today's open to yesterday's close. Store in monitor state. Pass to Claude: "Nifty opened with +1.5% gap up. Wait for 9:30 AM settlement."

### 11.6 No-Trade Zones (Time-Based Discipline)

Pro traders avoid specific time windows where false signals are common or risk is elevated.

| Window | Rule | Reason |
|--------|------|--------|
| 9:15-9:30 AM | No new entries | Auction volatility, prices settling |
| 1:00-2:00 PM | Lighter sizing | Low volume, false breakouts |
| 2:00 PM+ on expiry | No new option entries | Theta decay, gamma risk |
| 3:15-3:30 PM | No new entries, close uncertain | Closing auction volatility |
| 30 min before RBI/news | No new entries | Event volatility incoming |

**Implementation:** Already have auction window buffer (9:15-9:30, 3:15-3:30). Extend to include post-lunch low-volume awareness and expiry-day afternoon cutoff. Feed time context to Claude: "Current time: 1:15 PM — post-lunch low-volume period. Lower conviction on new entries."

### 11.7 Position Adding/Scaling Rules

Currently only exit logic (partial profit, full exit). No logic for adding to winning positions. Pro traders scale in and out strategically.

**Rules for Claude's system prompt:**
- **Add to winners only**: never add to losing positions (hard rule)
- **Minimum profit threshold**: only add if position is at least +1x ATR from entry
- **Scale out in 3 tranches**: 25% at first target, 50% at second target, hold 25% for runners — currently only 50/50 split
- **Add on pullback**: after breakout, add at first pullback to VWAP/support if thesis holds
- **Max adds**: maximum 1 add per position (don't pyramid indefinitely)

**Feed to Claude:**
```
MCX is +2.3% from entry (>1x ATR). Original thesis: metals breakout.
Thesis still valid (OI confirming, VWAP holding). Consider adding on pullback to ₹2,560.
```

**Implementation:** Claude can recommend ADD action (alongside HOLD/EXIT/PARTIAL). Paper trade engine processes ADD by increasing position size within safety caps.

### 11.8 IV Crush Awareness

Options premium can collapse after known events (earnings, RBI, budget) even if direction is correct. Pro traders adjust around events.

**Rules for Claude's system prompt:**
- Avoid buying options before known events when IV is inflated (earnings, RBI, budget)
- Prefer selling options before events (capture IV crush)
- After event passes: IV drops sharply, bought options lose value even if direction is right
- Track whether current VIX is elevated relative to recent range (proxy for IV percentile)

**Feed to Claude:**
```
RBI policy meeting tomorrow. VIX elevated at 18.5 (above 20-day avg of 15.2).
IV likely to crush post-announcement. Avoid buying new options.
```

**Implementation:** Already have event calendar (RBI dates, budget dates) in config. Add "days until next major event" to Claude prompt context. Compare current VIX to 20-day average for IV elevation signal.

## 12. Updated Claude System Prompt

Incorporating all new intelligence, the full system prompt becomes:

```
You are a full-time F&O desk trader for Indian markets (NSE). You receive
market data and candidates — you decide what to trade.

ROLE:
- You make ALL entry and exit decisions
- You see the full portfolio, not just individual positions
- You learn from recent trade lessons and daily journal

MARKET MICROSTRUCTURE (use these signals):
- VWAP: Don't enter against VWAP direction unless exceptional setup
- OI changes: Long buildup (OI up + price up) confirms bullish;
  short buildup (OI up + price down) confirms bearish;
  unwinding/covering are weak signals
- OI S/R: Max put OI = support, max call OI = resistance. Don't fight these levels.
- Max pain: Price gravitates to max pain near expiry

EXPIRY DAY RULES:
- Gamma is elevated — ATM options move 2-3x faster
- No new option positions after 2 PM on expiry
- Never average OTM options on expiry day
- Close uncertain option positions before 3 PM
- Max pain gravity strongest on expiry day

GAP HANDLING:
- Gap >1%: wait 15-30 min for settlement before new entries
- Gap + fade = potential reversal; gap + hold = continuation
- Existing positions: consider partial profit on favorable gap

TIME AWARENESS:
- 9:15-9:30: prices settling, avoid new entries
- 1:00-2:00 PM: low volume, reduce conviction on new signals
- 2:00 PM+ on expiry: no new option entries
- 3:15-3:30: closing auction, no new entries
- Before RBI/budget: avoid buying options (IV inflated)

POSITION MANAGEMENT:
- Add to winners only (never losers), only if +1x ATR from entry
- Scale out: 25% first target, 50% second, hold 25% for runners
- Add on pullback to VWAP/support if thesis holds
- Maximum 1 add per position

IV AWARENESS:
- Before known events: prefer selling options over buying
- VIX elevated vs 20-day avg = IV inflated, bought options expensive
- After event: IV crush makes direction right but trade wrong

GUIDELINES (weigh against signal quality — not hard rules):
- Regime matters: TRENDING_UP favors bullish, TRENDING_DOWN favors bearish,
  SIDEWAYS favors range-bound. Strong signals can override weak regime.
- VIX > 22: elevated risk, favor smaller positions or high-conviction only
- VIX > 28: crisis, sit out unless exceptional setup
- Heavy FII selling (>5000cr): bearish pressure, but DII absorption moderates
- PCR < 0.7: excessive bullish sentiment, contrarian bearish
- PCR > 1.3: excessive bearish sentiment, contrarian bullish
- Supertrend disagreement: weakens conviction, doesn't disqualify
- Friday afternoon: weekend risk for F&O, prefer closing marginal positions
- Afternoon entries (12:45+): only high conviction, max 2 new positions

HARD CONSTRAINTS (you cannot override):
- Never recommend allocation > 1.5x base
- If unsure, SKIP — missed trades cost nothing
- Never override a stoploss exit
- Respect position limits (max 8, max 2/sector)
- Never add to losing positions
```

## Constraints

- Single AngelOne session per cron run (no parallel auth)
- 1.5s API delay between calls
- Claude CLI ~5-10s per call (Max subscription)
- Pi: lightweight, cron-based, all state in JSON
- Obsidian vault at `~/Documents/Obsidian/` (iCloud sync)
- Vault daily notes: append only, never overwrite existing content
- VWAP computation requires intraday candle data (additional API calls — budget within rate limits)
- OI change tracking requires storing OI at entry + fetching current OI during monitoring

## 13. Production Hardening

### 13.1 Kill Switch (V6_ENABLED flag)

Config flag to instantly revert to V5 rule-based logic if V6 underperforms:

```python
# config.py
V6_CLAUDE_FIRST = True  # Set False to revert to V5 rule-based decisions
```

**Behavior when disabled:**
- Entry flow reverts to pre-Claude filters + allocation cascade
- Exit flow reverts to rule-triggered exits with Claude override
- All V6-specific code paths (entry thesis, position memory, journal) skip gracefully
- No code changes needed — just config toggle

**Implementation:** Wrap V6 code paths in `if config.V6_CLAUDE_FIRST:` guards. V5 code stays in place (dead but functional) until V6 is proven over 30+ trading days.

### 13.2 Decision Replay Mode

Log every Claude input/output for prompt iteration and debugging:

```
data/paper_trades/claude_decisions/
  2026-03-09_09-18_ENTRY_MCX.json
  2026-03-09_11-45_EXIT_BPCL.json
```

Each file stores:
```json
{
  "timestamp": "2026-03-09T09:18:32+05:30",
  "type": "entry",
  "symbol": "MCX",
  "prompt_sent": "<full prompt text>",
  "claude_response": "<raw response>",
  "parsed_action": "TRADE",
  "parsed_conviction": "high",
  "execution_result": "opened",
  "portfolio_snapshot": { "capital": 100000, "positions": 3 }
}
```

**Uses:**
- Replay past decisions to test prompt changes: "Would the new prompt still TRADE MCX on March 9?"
- Debug bad trades: see exactly what Claude saw and said
- Build a golden test set: 20-30 real decisions → run against new prompts → compare
- Track confidence calibration over time

**Implementation:** `claude_intel.py` saves decision files after every `evaluate_entry()` and `evaluate_exit()` call. Retention: 30 days (auto-cleanup in EOD wrap).

### 13.3 NSE Holiday Calendar

Avoid wasted runs, false alerts, and incorrect expiry detection on market holidays:

```python
# config.py
NSE_HOLIDAYS_2026 = [
    "2026-01-26",  # Republic Day
    "2026-03-10",  # Maha Shivaratri
    "2026-03-17",  # Holi
    "2026-04-10",  # Ram Navami
    "2026-04-14",  # Dr. Ambedkar Jayanti / Good Friday
    "2026-04-21",  # Mahavir Jayanti
    "2026-05-01",  # Maharashtra Day
    "2026-06-25",  # Eid-ul-Fitr
    "2026-07-06",  # Muharram
    "2026-08-15",  # Independence Day
    "2026-08-26",  # Janmashtami
    "2026-10-02",  # Gandhi Jayanti / Dussehra
    "2026-10-21",  # Diwali Laxmi Pujan
    "2026-10-22",  # Diwali Balipratipada
    "2026-11-04",  # Guru Nanak Jayanti
    "2026-11-25",  # Christmas
]
```

**Impact on cron:** `paper_trade_cron.sh` checks holiday list early and exits. No API calls, no Claude calls, no Telegram noise.

**Impact on expiry detection:** If last Thursday is a holiday, expiry shifts to Wednesday. `is_expiry_day()` must check against holiday list.

**Maintenance:** Update annually. Fetch from NSE website or hard-code (only ~15 dates/year).

### 13.4 Cron Overlap Lock

Prevent two cron ticks from running simultaneously and corrupting state:

```bash
# paper_trade_cron.sh — add at top, after cd
LOCKFILE="/tmp/paper_trade.lock"
exec 200>"$LOCKFILE"
flock -n 200 || { echo "[SKIP] Previous tick still running" >> "$LOG"; exit 0; }
```

Uses `flock` (available on Raspberry Pi) for file-based locking. If previous tick is still running, new tick exits silently. Lock auto-releases when process exits.

### 13.5 Portfolio Backup & Corruption Recovery

Protect against corrupted `portfolio.json`:

```python
def save_portfolio(portfolio: dict) -> None:
    """Save with atomic write + backup."""
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    content = json.dumps(portfolio, indent=2, default=str)

    # Backup current file before overwriting
    if PORTFOLIO_FILE.exists():
        BACKUP_FILE = PORTFOLIO_FILE.with_suffix('.json.bak')
        shutil.copy2(PORTFOLIO_FILE, BACKUP_FILE)

    # Atomic write: write to temp, then rename
    tmp = PORTFOLIO_FILE.with_suffix('.json.tmp')
    tmp.write_text(content)
    tmp.rename(PORTFOLIO_FILE)

def load_portfolio() -> dict:
    """Load with corruption recovery."""
    for path in [PORTFOLIO_FILE, PORTFOLIO_FILE.with_suffix('.json.bak')]:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, ValueError) as e:
                logger.error("Corrupted %s: %s", path, e)
                continue
    return _empty_portfolio()
```

Same pattern for `monitor_state.json`.

### 13.6 Telegram Message Splitting

Handle messages exceeding 4096 character limit:

```python
def _telegram_send(msg: str, **kwargs) -> None:
    MAX_LEN = 4000  # Leave buffer for HTML tags
    if len(msg) <= MAX_LEN:
        _send_single(msg, **kwargs)
    else:
        # Split at last newline before limit
        chunks = []
        while msg:
            if len(msg) <= MAX_LEN:
                chunks.append(msg)
                break
            split_at = msg.rfind('\n', 0, MAX_LEN)
            if split_at == -1:
                split_at = MAX_LEN
            chunks.append(msg[:split_at])
            msg = msg[split_at:].lstrip('\n')
        for i, chunk in enumerate(chunks):
            header = f"({i+1}/{len(chunks)})\n" if len(chunks) > 1 else ""
            _send_single(header + chunk, **kwargs)
```

### 13.7 Claude JSON Response Validation

Validate Claude's response before accessing fields:

```python
def _validate_entry_response(parsed: dict) -> dict:
    """Normalize and validate Claude entry response."""
    defaults = {
        "action": "SKIP",
        "conviction": "medium",
        "allocation_adj": 1.0,
        "reasoning": "No reasoning provided"
    }
    result = {}
    for key, default in defaults.items():
        val = parsed.get(key, default)
        if val is None:
            val = default
        result[key] = val

    # Clamp allocation_adj
    try:
        result["allocation_adj"] = max(0.5, min(1.5, float(result["allocation_adj"])))
    except (TypeError, ValueError):
        result["allocation_adj"] = 1.0

    # Normalize action
    result["action"] = result["action"].upper() if isinstance(result["action"], str) else "SKIP"
    if result["action"] not in ("TRADE", "SKIP"):
        result["action"] = "SKIP"

    return result
```

Same pattern for exit responses (validate EXIT/HOLD/PARTIAL/ADD).

### 13.8 V5 → V6 Migration

Script to migrate existing portfolio and state files:

```python
# scripts/migrate_v5_to_v6.py
def migrate():
    portfolio = load_portfolio()

    # Add entry_thesis to existing positions
    for pos in portfolio.get("positions", []):
        if "entry_thesis" not in pos:
            pos["entry_thesis"] = {
                "reasoning": f"Pre-V6 position opened on {pos.get('entry_date', 'unknown')}",
                "conviction": "medium",
                "key_conditions": {},
                "invalidation": "Manual review required",
                "expected_hold": "unknown",
                "target_scenario": "unknown"
            }

    # Add version
    portfolio["schema_version"] = 6

    save_portfolio(portfolio)

    # Migrate monitor_state.json
    state = _load_state()
    state.setdefault("position_assessments", {})
    state.setdefault("claude_consecutive_failures", 0)
    state.setdefault("claude_lockdown_active", False)
    state.setdefault("schema_version", 6)
    _save_state(state)

    print("Migration complete")
```

Run once before deploying V6 code. Backward-compatible — V5 code ignores new fields.

### 13.9 Tick Duration Monitoring

Alert if a cron tick takes too long (risk of overlap):

```bash
# paper_trade_cron.sh — wrap main execution
TICK_START=$(date +%s)

# ... all existing logic ...

TICK_END=$(date +%s)
TICK_DURATION=$((TICK_END - TICK_START))
echo "[TIMING] Tick completed in ${TICK_DURATION}s" >> "$LOG"

if [ "$TICK_DURATION" -gt 240 ]; then  # > 4 minutes
    MSG="⚠️ Slow tick: ${TICK_DURATION}s (limit 300s)"
    # ... send via Telegram ...
fi
```

### 13.10 Log Rotation

Prevent `cron.log` from filling Pi's SD card:

```bash
# paper_trade_cron.sh — add after LOG= assignment
# Rotate log if > 5MB
if [ -f "$LOG" ] && [ "$(stat -f%z "$LOG" 2>/dev/null || stat -c%s "$LOG" 2>/dev/null)" -gt 5242880 ]; then
    mv "$LOG" "${LOG}.old"
    echo "--- Log rotated $(TZ=Asia/Kolkata date) ---" > "$LOG"
fi
```

Also clean up decision replay files older than 30 days in EOD wrap:
```bash
find data/paper_trades/claude_decisions/ -name "*.json" -mtime +30 -delete 2>/dev/null
```

### 13.11 State Cleanup on Position Close

Remove stale entries from `monitor_state.json` when positions close:

```python
# smart_monitor.py
def cleanup_closed_positions(portfolio: dict, state: dict) -> None:
    """Remove state for positions that are no longer open."""
    open_symbols = {pos["symbol"] for pos in portfolio.get("positions", [])
                    if pos.get("status") == "open"}

    # Clean last_check
    state["last_check"] = {k: v for k, v in state.get("last_check", {}).items()
                           if k in open_symbols}

    # Clean position_assessments
    state["position_assessments"] = {k: v for k, v in state.get("position_assessments", {}).items()
                                      if k in open_symbols}
```

Called in EOD wrap and after every `close_position()`.

### 13.12 EOD Wrap Timing Fix

Move EOD wrap from 3:20 PM to 3:35 PM to capture all end-of-day activity:

```bash
# paper_trade_cron.sh — change:
# OLD: if [ "$HOUR" -eq 15 ] && [ "$MIN" -ge 18 ] && [ "$MIN" -le 25 ]
# NEW:
if [ "$HOUR" -eq 15 ] && [ "$MIN" -ge 33 ] && [ "$MIN" -le 40 ]; then
    echo "[EOD] End-of-day wrap" >> "$LOG"
    # ... EOD logic ...
fi
```

Also extend cron window: `*/5 9-15 * * 1-5` → ensure it covers 3:35 PM (already does since 15:35 is within hour 15).
