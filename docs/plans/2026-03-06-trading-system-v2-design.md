# Design: Trading System V2 — Multi-Strategy F&O Bot

Date: 2026-03-06
Status: Approved
Supersedes: 2026-03-05-bear-put-spread-youtube-intel-design.md

## Goal

Build a solid, conservative trading system that generates 2-3% monthly returns on 1L+ capital. Multi-strategy, regime-aware, fully automated for personal use. Results are the product — if P&L is consistently green, everything else follows.

## Philosophy

- Capital preservation first, returns second
- 2% max risk per trade, no exceptions
- Boring but profitable > flashy but risky
- Every trade has a reason the system can explain
- Survive drawdowns, compound through them
- 4-6 high-quality trades per month, not 20 mediocre ones

---

## Part 1: Market Regime Detection

Every morning at 8:45 AM, before any trades:

### Regime Classification

| Regime | Conditions | Confidence Required |
|--------|-----------|-------------------|
| TRENDING_UP | ADX(10) > 20 for 3+ bars, price > 20 EMA, rising 5+ days | >70% |
| TRENDING_DOWN | ADX(10) > 20 for 3+ bars, price < 20 EMA, falling 5+ days | >70% |
| SIDEWAYS | ADX(10) < 20 for 3+ bars, Nifty range < 400pts for 7+ days, BB width at 20-day low | >70% |
| VOLATILE | VIX > 20 OR daily range > 2x 20-day average | Any |
| UNCERTAIN | No regime > 70% probability | — |
| CASH | Drawdown 8% monthly, or 5 consecutive losses, or VIX > 25 | Auto-triggered |

### Regime -> Strategy Map

| Regime | Equity Longs | Bull Call Spread | Bear Put Spread | Iron Condor | Momentum Buy | Bull Put Credit | Bear Call Credit |
|--------|-------------|-----------------|----------------|------------|-------------|----------------|-----------------|
| TRENDING_UP | Yes | Yes (low IV) | No | No | Yes (calls) | Yes (high IV) | No |
| TRENDING_DOWN | No (close existing) | No | Yes (low IV) | No | Yes (puts) | No | Yes (high IV) |
| SIDEWAYS | Yes (reduced) | No | No | Yes (VIX 12-18) | No | Yes (high IV) | Yes (high IV) |
| VOLATILE | No (tight stops) | No | No | No | Yes (small) | No | No |
| UNCERTAIN | 50% size only | No | No | No | No | No | No |
| CASH | No new trades | No | No | No | No | No | No |

**IV routing rule**: When IV percentile > 70%, debit spreads (B, C) are blocked. Credit spreads (F1, F2) activate instead. When IV < 50%, prefer debit spreads. Between 50-70%, either is valid based on signal strength.

### VIX Tiers

| VIX | Action |
|-----|--------|
| < 12 | Extreme complacency. Reduce all sizes 50%. Watch for reversal |
| 12-18 | Normal. Full allocation. Iron condors allowed |
| 18-22 | Elevated. Reduce 25%. No iron condors |
| 22-28 | High. Reduce 50%. No new short premium |
| > 28 | Crisis. Close all short premium. Momentum buys only or sit out |

### VIX Spike Override

If VIX rises > 20% in a single day: immediately close ALL short premium positions (iron condors, credit spreads). Override all other rules.

---

## Part 2: Strategies

### Strategy A: Equity Longs (Existing — Minor Updates)

Existing code in paper_trade.py. Changes:

- Add earnings blackout check (no new positions within 5 days of results)
- Add sector correlation limit (max 2 stocks from same sector)
- Add consecutive loss pause (3 consecutive equity losses -> pause 3 trading days)
- Add portfolio hedge trigger (when 3+ equity longs open, buy 1 cheap OTM Nifty monthly put as insurance, budget 500-1500)

### Strategy B: Bull Call Spread (New)

Mirror image of bear put spread. Same code structure.

**Entry:**
- Screener signal: bullish (LongBuildUp, ShortCovering, PercPriceGainers)
- Regime: TRENDING_UP
- Use when: moderate conviction, expecting quick 1-5% move within 1-7 days

**Parameters:**
| Parameter | Value |
|-----------|-------|
| Long leg | ATM call or 1-strike ITM |
| Short leg | 2-4 strikes OTM from long leg |
| Expiry | Next month (30-45 DTE minimum) |
| Max debit (risk) | 2% of current capital |
| Target | Underlying rises by 2.0x ATR |
| Stoploss | Underlying drops by 1.5x ATR |
| Profit cap | 80% of max profit reached |
| Time exit | 5 trading days (not 7 — tighter for spreads) |

### Strategy C: Bear Put Spread (New)

**Entry:**
- Screener signal: bearish (ShortBuildUp, LongUnwinding)
- Regime: TRENDING_DOWN

**Parameters:** Same as bull call spread but reversed direction.

| Parameter | Value |
|-----------|-------|
| Long leg | ATM put or 1-strike ITM |
| Short leg | 2-4 strikes OTM from long leg |
| Expiry | Next month (30-45 DTE minimum) |
| Max debit (risk) | 2% of current capital |
| Target | Underlying drops by 2.0x ATR |
| Stoploss | Underlying rises by 1.5x ATR |
| Profit cap | 80% of max profit reached |
| Time exit | 5 trading days |

### Strategy D: Iron Condor on Nifty (New)

**Entry (ALL must pass):**
- Regime: SIDEWAYS with >70% confidence
- ADX(10) < 20 for 3+ consecutive bars
- Nifty range < 400pts for 7+ trading days
- VIX between 12-18
- VIX rate of change < 15% today
- No macro event within 48 hours (RBI, Budget, Fed, expiry)
- Not Thursday/Friday (weekend gap risk)
- Max loss < 2% of capital

**Structure:**
- Sell OTM call (200-300 points above Nifty)
- Sell OTM put (200-300 points below Nifty)
- Buy further OTM call (100-200 points above short call)
- Buy further OTM put (100-200 points below short put)
- Target credit: 50-70 per lot minimum

**Exit:**
- Target: 50% of max profit (take the easy money, don't hold for last 50%)
- Stoploss: 2x credit received (if collected 5000, close at 10000 loss)
- Short strike delta exceeds 0.30: close threatened side immediately
- Time exit: close 5 days before expiry (gamma risk)
- VIX spike > 20% in 1 day: close immediately

**Frequency:** 0-2 per month. Most months won't qualify. That's correct.

### Strategy E: Momentum Options Buying on Nifty (New)

**Entry:**
- Nifty breaks 20-day high (buy CE) or 20-day low (buy PE)
- Volume on breakout bar > 1.5x average
- Wait for pullback/retest (don't chase the breakout candle)
- IV percentile < 80%
- Regime: TRENDING_UP (for calls), TRENDING_DOWN (for puts), VOLATILE (either)

**Parameters:**
| Parameter | Value |
|-----------|-------|
| Strike | ATM or 1-strike ITM |
| Expiry | Monthly only (never weekly — theta kills weeklies) |
| Max premium (risk) | 1% of capital (smaller than spreads) |
| Stoploss | 30-40% of premium paid |
| Target | 80-100% of premium (double your money) |
| Time exit | 3 trading days if < 15% profit |
| R:R minimum | 1:2 (won't enter if target/stop < 2) |

**Win rate:** 35-40%. Compensated by 1:2+ R:R.

### Strategy F: Credit Spreads on Nifty — Bull Put & Bear Call (New)

**Purpose:** Fill the high-IV gap. When IV percentile > 70%, debit spreads (B, C) are skipped. Credit spreads *profit* from elevated IV — sell expensive premium and let it decay.

#### F1: Bull Put Spread (Bullish + High IV)

**Entry (ALL must pass):**
- Screener signal: bullish (LongBuildUp, ShortCovering, PercPriceGainers)
- Regime: TRENDING_UP or SIDEWAYS
- IV percentile > 50% (preferably > 70% — that's the sweet spot)
- VIX between 15-25 (elevated but not crisis)
- No macro event within 48 hours
- Max loss < 2% of capital

**Structure:**
- Sell OTM put (100-200 points below Nifty)
- Buy further OTM put (100-200 points below short put as protection)
- Net credit received upfront

**Parameters:**
| Parameter | Value |
|-----------|-------|
| Short leg | 1-2 strikes OTM put (delta ~0.25-0.30) |
| Long leg | 2-4 strikes further OTM from short leg |
| Expiry | Monthly (30-45 DTE) — more time value to collect |
| Max loss (width - credit) | 2% of capital |
| Target | 50% of max profit (credit received) |
| Stoploss | 2x credit received |
| Time exit | Close 5 days before expiry |
| Short strike breach | Close if underlying drops below short strike |

#### F2: Bear Call Spread (Bearish + High IV)

**Entry (ALL must pass):**
- Screener signal: bearish (ShortBuildUp, LongUnwinding)
- Regime: TRENDING_DOWN or SIDEWAYS
- IV percentile > 50% (preferably > 70%)
- VIX between 15-25
- No macro event within 48 hours
- Max loss < 2% of capital

**Structure:**
- Sell OTM call (100-200 points above Nifty)
- Buy further OTM call (100-200 points above short call as protection)
- Net credit received upfront

**Parameters:** Mirror of bull put spread but reversed direction.
| Parameter | Value |
|-----------|-------|
| Short leg | 1-2 strikes OTM call (delta ~0.25-0.30) |
| Long leg | 2-4 strikes further OTM from short leg |
| Expiry | Monthly (30-45 DTE) |
| Max loss (width - credit) | 2% of capital |
| Target | 50% of max profit |
| Stoploss | 2x credit received |
| Time exit | Close 5 days before expiry |
| Short strike breach | Close if underlying rises above short strike |

#### Credit Spread Exit Rules (Both F1 and F2)
- **Target**: 50% of max profit reached — take profit early, don't hold for last 50%
- **Stoploss**: Loss reaches 2x credit received
- **Short strike breach**: Underlying crosses the short strike — close immediately
- **VIX spike > 20% in 1 day**: Close immediately (same as iron condor)
- **Time exit**: Close 5 trading days before expiry (gamma risk)
- **Delta exit**: Short strike delta exceeds 0.35 — close before it goes ITM

**Frequency:** 1-3 per month. Replaces debit spreads when IV is elevated.

**Key difference from iron condor:** Credit spreads are directional (you have a view). Iron condor is non-directional (you want price to stay in range). Credit spreads work in trending + high IV; iron condor only works in sideways + moderate IV.

### Strategy G: Portfolio Hedge — OTM Nifty Put (New)

When 3+ equity longs are open, buy 1 cheap OTM Nifty monthly put as crash insurance. Budget: Rs 500-1500. This is a cost, not a profit center. Accept losing this premium most months.

---

## Part 3: Capital Allocation

```
Total capital (dynamic, recalculated daily from current portfolio value):

  Equity longs:           max 40% (3-4 positions @ 10-15K)
  Vertical spreads:       max 30% (debit + credit combined — 2-3 spreads @ 10-15K)
    - Debit (bull call, bear put): when IV < 70%
    - Credit (bull put, bear call): when IV > 50%
    - Never both debit AND credit on same underlying simultaneously
  Iron condor:            max 10% (0-1 position)
  Momentum options:       max 15% (1-3 directional buys @ 1% each)
  Cash reserve:           min 5% always

Position sizing: 2% max risk per trade on any strategy.
At 1L capital: max loss per trade = 2,000.
For credit spreads: max loss = spread width - credit received. Must be < 2% of capital.
```

---

## Part 4: Pre-Trade Filters (Applied to ALL Strategies)

### Liquidity Gate (Options Strategies Only)

Both legs must pass:
- Open interest > 500 contracts
- Today's volume > 100 contracts
- Bid-ask spread < 5% of mid-price
- Combined entry slippage < 15% of max profit

If any check fails: skip the trade.

### Stock Whitelist (Stock Options Only)

Maintain a whitelist of 15-20 stocks with historically liquid options. Refresh weekly by scanning OI across strikes. Likely list: RELIANCE, HDFCBANK, ICICIBANK, TCS, INFY, SBIN, TATAMOTORS, BAJFINANCE, AXISBANK, INDUSINDBK, MARUTI, LT, WIPRO, HCLTECH, BHARTIARTL.

Nifty index options: always liquid, no whitelist needed.

### Earnings Blackout

No new spreads/options positions within 5 trading days of earnings announcement. Close existing spread positions before earnings.

Source: BSE corporate announcements or moneycontrol earnings calendar.

### F&O Ban Check

Before every stock F&O entry: check NSE F&O ban list. If stock is in ban (OI > 95% MWPL), skip. Index options are immune to F&O ban.

### IV Percentile Filter

- Debit spreads (bull call, bear put): don't enter when IV percentile > 70%
- Credit strategies (iron condor): don't enter when VIX < 12 or > 18
- Momentum buying: reduce size 50% when IV > 80th percentile

### Physical Delivery Safety

Hard rule: close ALL stock option positions 3 trading days before expiry (E-3). No exceptions. This avoids:
- Physical delivery margin (ramps to 50% of contract value)
- STT trap on exercise (0.15% of intrinsic value)
- Pin risk near short strike

Index options (Nifty): cash-settled, no delivery risk. Can hold closer to expiry but still close 2 days before (gamma risk).

---

## Part 5: Execution Logic

### Spread Leg Execution Order

NSE has no atomic multi-leg orders. Execute sequentially:

1. Place the SHORT (sell) leg first — it's typically harder to fill
2. Use limit order at mid-price with small buffer
3. Once short leg fills, immediately place LONG (buy) leg
4. If long leg doesn't fill within 60 seconds, close short leg and abort
5. Never use market orders

### Exit Execution

Close both legs simultaneously:
1. Place buy-to-close on short leg
2. Place sell-to-close on long leg
3. If one fills and other doesn't within 60 seconds, market-close the remaining leg (accept slippage on exit to avoid naked exposure)

---

## Part 6: Risk Management

### Per-Trade Limits

| Rule | Limit |
|------|-------|
| Max risk per trade | 2% of current capital |
| Max correlated positions | 2 same-sector stocks |
| Max concurrent positions | 6-8 total across all strategies |
| Max portfolio delta | 80% of capital (not too directionally exposed) |

### Drawdown Circuit Breakers

| Trigger | Action |
|---------|--------|
| Daily P&L < -3% | Close ALL positions. No new trades today |
| Weekly P&L < -5% | Reduce all position sizes by 50% next week |
| Monthly P&L < -8% | Halt all trading. Review strategy |
| Total drawdown > 15% | FULL STOP. Paper trade for 30 days before resuming |
| 3 consecutive losses (same strategy) | Pause that strategy for 3 trading days |

### Correlation Guard

Before opening any new position:
- Calculate: if ALL open positions hit max loss simultaneously, what's total damage?
- If total > 10% of capital: don't open the new position
- This assumes correlation = 1.0 during a crash (realistic per research)

### Kill Switch

Telegram `/kill` command: market-close all positions, cancel all pending orders, halt bot.

Auto-kill triggers:
- API heartbeat missed > 120 seconds with open positions
- Margin utilization > 90%
- Any unhandled exception
- Never auto-restart after kill. Manual intervention required.

### Reconciliation

Every 10 minutes during market hours:
- Fetch positions from broker API
- Compare with bot's internal position tracker
- If mismatch: alert via Telegram, halt new trades, log discrepancy

---

## Part 7: YouTube Intel Integration

### Mode A: Daily Market Intel (pre-market, 8:45 AM)

Search YouTube for "Nifty analysis today", "market outlook India" (last 24h). Pull transcripts from top 2-3 results. Claude haiku classifies:
```json
{
    "market_bias": "bullish | bearish | sideways",
    "key_levels": {"nifty_support": 22000, "nifty_resistance": 22500},
    "sectors_bullish": ["Banking", "IT"],
    "sectors_bearish": ["Auto", "Pharma"],
    "events_today": ["RBI policy", "US Fed minutes"],
    "confidence": "high | medium | low"
}
```
Feed into regime classification as additional signal. Cache 12 hours.

### Mode B: Pre-Trade Stock Research

After screener picks candidates, before opening positions. Search "{SYMBOL} stock analysis" (last 7 days). Claude haiku extracts sentiment + key levels + red flags.

Integration: YouTube strongly contradicts signal -> skip trade. Confirms -> no change. No data -> proceed normally.

### Channel Allowlist

PR Sundar, CA Rachana Ranade, Power of Stocks, Pranjal Kamra, Nitin Bhatia. Configurable in config.py.

### Implementation

New file: `youtube_intel.py`
- Uses yt-dlp for search + youtube-transcript-api for transcripts
- Claude haiku for classification
- File-based cache (data/youtube_cache/)
- Hindi transcripts supported (Claude handles Hindi well)

---

## Part 7.5: Data Sources and Calculations

Every component needs specific data. This section maps each data dependency to its source so nothing is ambiguous during implementation.

### IV Percentile Calculation

**Problem**: The system needs IV percentile to route between debit and credit spreads. SmartAPI provides current IV per strike but no historical IV.

**Solution (phased)**:
1. **Phase 1 (proxy)**: Use India VIX as IV percentile proxy. VIX has a well-known range — map VIX level to approximate percentile: VIX < 12 = bottom 10%, 12-15 = 25th, 15-18 = 50th, 18-22 = 70th, 22+ = 90th+. This is imprecise but directionally correct and avoids needing historical data.
2. **Phase 2 (real)**: Build a daily IV collector that runs at 3:35 PM. Store Nifty ATM IV (average of ATM CE and PE implied vol from option chain) to `data/iv_history.json`. After 30 days of data, compute rolling 30-day IV percentile. Until 30 days accumulated, fall back to VIX proxy.

**VIX data source**: Fetch India VIX via SmartAPI using the VIX instrument token (symbol: "India VIX", exchange: NSE). Alternatively, use `getCandleData()` for NIFTY VIX index.

### Greeks (Delta) Calculation

**Problem**: Delta monitoring is needed for iron condor and credit spread exits ("close if short delta > 0.30"). SmartAPI option chain may return Greeks but it's unreliable/missing for many strikes.

**Solution**: Calculate delta ourselves using Black-Scholes:
```
For calls: delta = N(d1)
For puts:  delta = N(d1) - 1

where d1 = [ln(S/K) + (r + sigma^2/2) * T] / (sigma * sqrt(T))

S = underlying price (Nifty spot)
K = strike price
r = risk-free rate (use 91-day T-bill rate, ~6.5-7%, hardcode as 0.065)
sigma = implied volatility (from option chain, annualized)
T = time to expiry in years (DTE / 365)
N() = standard normal CDF (scipy.stats.norm.cdf or math.erf approximation)
```

**Fallback**: If IV is unavailable for a strike, use moneyness proxy: `proxy_delta = max(0, 1 - abs(strike - spot) / (spot * 0.05))`. Less accurate but avoids blocking trades.

### Paper Trade Slippage Model

**Problem**: Paper trades fill at exact LTP. Live trades face slippage (bid-ask spread, market impact). Paper P&L without slippage creates false confidence.

**Solution**: Add configurable slippage to all paper trade entries and exits:
```
PAPER_TRADE_SLIPPAGE_PCT = 0.005  # 0.5%

Entry: buy at LTP * (1 + slippage), sell at LTP * (1 - slippage)
Exit:  buy-to-close at LTP * (1 + slippage), sell-to-close at LTP * (1 - slippage)
```
This makes paper P&L ~1% worse per round trip — closer to reality. For spreads, apply slippage to each leg independently.

### F&O Ban List Source

The design says "check F&O ban list before every stock F&O trade" but doesn't specify the data source.

**Source**: NSE publishes the ban list daily at ~6 PM for the next trading day. URL pattern: `https://nsearchives.nseindia.com/content/fo/fo_secban.csv`. Alternatively, SmartAPI's instrument master marks banned stocks. For simplicity:
- Download `fo_secban.csv` once daily at 8:45 AM (pre-market)
- Parse CSV for list of banned symbols
- Cache for the day
- Index options (Nifty) are never banned — skip check for index trades

### Earnings Calendar Source

The design says "no positions within 5 trading days of earnings" but doesn't specify data source.

**Source options**:
1. **BSE corporate announcements API** — free, official, but hard to parse
2. **MoneyControl earnings calendar** — web scrape, fragile
3. **Manual maintenance** — for 15-20 whitelisted stocks, manually update quarterly. At 4 earnings per year per stock = ~80 dates/year. Store in `data/earnings_calendar.json`.

**Recommendation**: Start with manual maintenance for whitelisted stocks. It's 20 minutes of work per quarter and 100% reliable. Automate later if needed.

### Technical Indicator Dependencies

Regime detection needs ADX, EMA, and Bollinger Bands. These are NOT in the current codebase.

**Library**: Use `ta` library (v0.11.0, works on Python 3.11):
- **ADX(10)**: `ta.trend.ADXIndicator(high, low, close, window=10).adx()`
- **EMA(20)**: `ta.trend.EMAIndicator(close, window=20).ema_indicator()`
- **Bollinger Bands**: `ta.volatility.BollingerBands(close, window=20, window_dev=2).bollinger_wband()`

**Data needed**: 50+ daily candles for Nifty (for ADX warm-up). Fetch via SmartAPI `getCandleData()` for NIFTY 50 index, last 60 trading days.

### First-Run Warmup (Cold Start)

**Problem**: Regime detection needs ADX(10) which needs 20+ candles to warm up. On first run, all indicators return None and the bot can't classify any regime.

**Solution**:
1. On first run (or when `data/nifty_candles.json` is missing/stale), fetch 60 trading days of Nifty daily candles via SmartAPI `getCandleData()`.
2. Store in `data/nifty_candles.json` with timestamps.
3. On subsequent runs, fetch only new candles since last stored date and append.
4. If indicators still return None (insufficient data), default to regime = UNCERTAIN (50% size, equity only).
5. VIX history: same pattern — store daily VIX closes in `data/vix_history.json`. First run fetches 60 days.

**Fallback regime when data is insufficient**: UNCERTAIN. This is the safest default — only allows equity longs at 50% size. No spreads, no iron condors, no momentum. The bot gradually unlocks strategies as data accumulates.

### Spread Position Schema

Current portfolio.json positions are equity-only. Spread positions need additional fields.

**Spread position structure** (extends existing position dict):
```json
{
    "id": "spread_20260310_NIFTY_BULL_CALL",
    "strategy": "bull_call_spread | bear_put_spread | bull_put_credit | bear_call_credit | iron_condor",
    "spread_type": "debit | credit",
    "spread_direction": "bullish | bearish | neutral",
    "symbol": "NIFTY",
    "underlying_at_entry": 22500,
    "entry_date": "2026-03-10",
    "expiry": "2026-03-26",
    "quantity": 75,
    "long_leg": {
        "strike": 22400,
        "option_type": "CE",
        "entry_premium": 180,
        "token": "12345"
    },
    "short_leg": {
        "strike": 22600,
        "option_type": "CE",
        "entry_premium": 90,
        "token": "12346"
    },
    "net_debit": 6750,
    "net_credit": 0,
    "max_profit": 8250,
    "max_loss": 6750,
    "spread_width": 200,
    "target_underlying": 22900,
    "stoploss_underlying": 22200,
    "status": "open",
    "regime_at_entry": "TRENDING_UP",
    "iv_percentile_at_entry": 45
}
```

For **iron condors**, the position has 4 legs:
```json
{
    "strategy": "iron_condor",
    "call_spread": {"short_strike": 22800, "long_strike": 22900, ...},
    "put_spread": {"short_strike": 22200, "long_strike": 22100, ...},
    "net_credit": 4500,
    "max_loss": 3000
}
```

Both spread and equity positions live in the same `portfolio.json` → `open_positions` list. The monitor loop checks `pos.get("strategy")` to route to the correct exit logic.

### Signal → Regime → Strategy Decision Tree

Complete routing logic for opening new positions:

```
1. Fetch regime (regime.py)
2. If regime == CASH → do nothing, return
3. Fetch screener candidates
4. For each candidate:
   a. Get signal direction (bullish/bearish from screener score)
   b. Get IV percentile (VIX proxy or real)
   c. Route:

   BULLISH signal:
     regime=TRENDING_UP + IV<70%  → bull call spread (debit)
     regime=TRENDING_UP + IV>=50% → bull put spread (credit)
     regime=TRENDING_UP + IV<50%  → equity long
     regime=SIDEWAYS + IV>=50%    → bull put spread (credit)
     regime=SIDEWAYS + IV<50%     → equity long (reduced size)
     regime=VOLATILE              → skip (or momentum call if breakout)
     regime=UNCERTAIN             → equity long at 50% size only

   BEARISH signal:
     regime=TRENDING_DOWN + IV<70%  → bear put spread (debit)
     regime=TRENDING_DOWN + IV>=50% → bear call spread (credit)
     regime=SIDEWAYS + IV>=50%      → bear call spread (credit)
     regime=SIDEWAYS + IV<50%       → skip (no bearish play in sideways without options)
     regime=VOLATILE                → skip (or momentum put if breakout)
     regime=UNCERTAIN               → skip

   NO SIGNAL (index-only):
     regime=SIDEWAYS + VIX 12-18 + all condor checks → iron condor
     regime=TRENDING + breakout detected → momentum options

5. Before opening: run pre-trade filters (liquidity, F&O ban, earnings blackout, capital allocation, correlation guard)
6. If passes all filters → execute trade
```

### Capital Allocation Constants

Add to config.py:
```python
# Capital allocation limits (% of current portfolio value)
ALLOC_EQUITY_MAX = 0.40          # max 40% in equity longs
ALLOC_SPREADS_MAX = 0.30         # max 30% in all vertical spreads (debit + credit)
ALLOC_IRON_CONDOR_MAX = 0.10     # max 10% in iron condors
ALLOC_MOMENTUM_MAX = 0.15        # max 15% in momentum options
ALLOC_CASH_MIN = 0.05            # min 5% always in cash
MAX_CONCURRENT_POSITIONS = 8     # hard cap across all strategies
MAX_SAME_SECTOR = 2              # max 2 positions in same sector
MAX_SIMULTANEOUS_LOSS_PCT = 10   # don't open if all positions losing simultaneously > 10%
```

### Dependencies

Add to requirements.txt:
```
ta>=0.11.0              # ADX, EMA, Bollinger Bands for regime detection (pandas-ta broken on 3.11)
yt-dlp                  # YouTube video search for youtube_intel.py
youtube-transcript-api  # YouTube transcript extraction
```

Already available (transitively installed): `pandas`, `numpy`, `scipy`.

---

## Part 8: System Infrastructure

### NSE Holidays 2026

16 trading holidays. Hardcode as a set in config.py:
```python
NSE_HOLIDAYS_2026 = {
    date(2026, 1, 26),  # Republic Day
    date(2026, 2, 26),  # Maha Shivaratri
    date(2026, 3, 10),  # Holi
    date(2026, 3, 30),  # Id-Ul-Fitr (Eid)
    date(2026, 4, 2),   # Ram Navami
    date(2026, 4, 3),   # Good Friday
    date(2026, 4, 14),  # Dr. Ambedkar Jayanti
    date(2026, 5, 1),   # Maharashtra Day
    date(2026, 6, 5),   # Id-Ul-Adha (Bakri Eid)
    date(2026, 7, 6),   # Muharram
    date(2026, 8, 15),  # Independence Day
    date(2026, 8, 19),  # Janmashtami (tentative)
    date(2026, 10, 2),  # Mahatma Gandhi Jayanti
    date(2026, 10, 20), # Dussehra
    date(2026, 11, 9),  # Diwali (Laxmi Pujan)
    date(2026, 11, 30), # Guru Nanak Jayanti
}
```
Update is_market_open(), _trading_days_between(), _add_trading_days() to check this set.

### Event Calendar

Maintain a list of known market events:
- RBI policy dates (bi-monthly)
- F&O expiry dates (last Thursday of month)
- US Fed meeting dates
- Budget day
- Nifty 50 component earnings dates

Before any trade: check if event is within 48 hours. If yes: no iron condors, tighten stops, reduce size.

### Daily Bot Schedule

```
08:45  YouTube market intel (Mode A)
08:50  Regime detection (VIX, ADX, EMA, Bollinger)
08:55  Login to AngelOne (generate TOTP, authenticate)
09:00  Check F&O ban list, earnings calendar, holidays
09:15  Market opens — monitor existing positions
09:20  Run screener for new candidates
09:25  Pre-trade filters on candidates (liquidity, IV, earnings, YouTube Mode B)
09:30  Open new positions (if any pass all filters)
09:30-15:15  Monitor every 10 minutes:
        - Position P&L vs stops/targets
        - Reconciliation (broker vs bot state)
        - Iron condor delta monitoring (every 30 min)
        - Drawdown circuit breaker checks
15:15  Last chance to close positions needing exit
15:25  No new orders after this
15:30  Market close
15:35  Daily P&L summary via Telegram
15:40  Logout from AngelOne
```

### Static IP (for future live trading)

Paper trading on Pi: fine, no static IP needed.
Live trading: AWS Lightsail Mumbai (350-700/month), static IP included. Pi as backup monitor.

---

## Part 9: Monitoring

### Telegram Alerts

| Severity | Examples | Alert |
|----------|---------|-------|
| CRITICAL | API down with open positions, margin > 90%, drawdown > 3% daily | Immediate Telegram + attempt auto-close |
| WARNING | Rate limit approaching, partial fill, margin > 70% | Telegram within 5 min |
| INFO | Trade opened/closed, daily P&L | Daily summary at 15:35 |

### Weekly Performance Report (Telegram, Saturday morning)

- Total P&L (week, month, inception)
- Win rate (overall + per strategy)
- Running expected value
- Drawdown status
- Strategy performance breakdown
- Costs breakdown

### Strategy Health Checks

- After 20+ trades: check if running EV is positive
- After 40+ trades: if EV negative, STOP and re-evaluate
- Win rate declining over rolling 15-trade window: flag for review
- If costs > 1% of capital/month: reduce trade frequency

---

## Part 10: What Can Kill This (And How We Handle It)

### Execution Risks
| Risk | Mitigation |
|------|-----------|
| Wide bid-ask on OTM strikes | Liquidity whitelist, OI > 500, bid-ask < 5% |
| Leg risk (one fill, other doesn't) | Short leg first, 60-second abort timer |
| Physical delivery margin trap | Hard exit at E-3, no exceptions |
| IV crush on debit spreads | No entry when IV > 70th percentile |
| STT on exercise | Never hold to expiry, always sell on exchange |

### Credit Spread Risks
| Risk | Mitigation |
|------|-----------|
| Short strike breached (goes ITM) | Close immediately when underlying crosses short strike |
| VIX drops after selling (credit shrinks less than expected) | VIX > 50% percentile entry requirement ensures meaningful premium |
| Assignment risk on short leg | NSE options are European-style — no early assignment possible |
| Margin spike on short leg | Credit spread margin = max loss (width - credit). Fully defined risk |
| Holding through earnings | No credit spreads within 5 days of earnings (same blackout rule) |

### Regime Risks
| Risk | Mitigation |
|------|-----------|
| False breakout | Require 2-3 closes beyond range + volume > 1.5x |
| ADX lag (3 weeks) | ADX(10) not ADX(14), 3-bar confirmation |
| VIX false safety | Track VIX rate of change, not just level |
| Overnight gap kills condor | Max loss per condor = 2%, no condors over weekends |
| Correlation blow-up | Size assuming all positions lose simultaneously |

### Small Account Risks
| Risk | Mitigation |
|------|-----------|
| Costs eat 1% monthly | Max 4-6 trades/month, limit orders only |
| No diversification at 1L | Accept 1-2 positions max, use Nifty for spreads |
| 5 consecutive losses = -10% | 2% risk per trade makes this survivable (3.4 months recovery) |
| Psychology (override, tinker) | 48-hour cooling period, weekly reviews only |

---

## Files Changed

| File | Change |
|------|--------|
| `paper_trade.py` | Add spread strategies (bull call, bear put, bull put credit, bear call credit), iron condor, momentum buying. Regime-based strategy selection. IV-routed spread selection. New exit logic. Capital allocation. Drawdown limits. Reconciliation loop |
| `agent_with_options.py` | Add `select_spread_strikes()`, IV percentile calculation, liquidity checks |
| `screener.py` | Wire YouTube market intel. Add earnings blackout check. Add F&O ban check |
| `youtube_intel.py` | **New** — market intel + stock intel + caching |
| `regime.py` | **New** — regime detection (ADX, EMA, VIX, Bollinger). Returns regime + confidence |
| `risk_manager.py` | **New** — drawdown limits, correlation guard, portfolio delta, kill switch |
| `config.py` | NSE_HOLIDAYS_2026, VIX tiers, strategy constants, YouTube config, event calendar |
| `tests/` | Tests for spread logic, regime detection, risk manager, YouTube intel |

---

## Success Criteria

After 6 months of paper trading:
- Monthly returns: 1.5-3.5% average (some months flat, that's fine)
- Max drawdown: < 15%
- Win rate: > 50% (across all strategies)
- Sharpe ratio: > 1.0
- Running EV: positive after 40+ trades
- Zero blown-up months (no single month worse than -5%)
