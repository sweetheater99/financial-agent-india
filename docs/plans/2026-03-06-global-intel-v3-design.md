# Trading System V3: Global Macro Intelligence + Strategy Enhancements

## Goal

Add global macro awareness, market microstructure signals, new strategies, and operational resilience to the paper trading bot. V2 trades blind to overnight US moves, FII flows, Twitter sentiment, and misses opportunities in BankNifty and weekly theta harvesting.

## Architecture

V3 adds three new modules and extends existing ones:

```
PRE-MARKET DATA COLLECTION
===========================

  global_intel.py          x_intel.py          oi_analysis.py
  +-----------------+   +----------------+   +------------------+
  | US Markets      |   | X/Twitter      |   | PCR computation  |
  |  (yfinance)     |   | (Python port)  |   | Max Pain calc    |
  | GIFT Nifty      |   | Cookie auth    |   | OI buildup       |
  |  (yfinance)     |   | Claude Haiku   |   | From existing    |
  | Asia Markets    |   | classification |   | option chain     |
  |  (yfinance)     |   +-------+--------+   +--------+---------+
  | FII/DII flows   |          |                      |
  |  (moneycontrol) |          |                      |
  +--------+--------+          |                      |
           |                   |                      |
           +-------------------+----------------------+
                               |
                               v
                    +---------------------+
                    | MacroContext dict    |
                    | (hard + soft gates) |
                    +----------+----------+
                               |
                               v
              +--------------------------------+
              | open_positions() pipeline      |
              |                                |
              | 1. Regime detection (existing) |
              | 2. Macro hard gates (NEW)      |
              | 3. Supertrend confirm (NEW)    |
              | 4. CPR day-type filter (NEW)   |
              | 5. Candidate filtering         |
              | 6. Position opening            |
              +--------------------------------+
                               |
              +--------------------------------+
              | _try_index_strategies()        |
              |                                |
              | Iron condor (Nifty + BankNifty)|
              | Momentum (Nifty + BankNifty)   |
              | Weekly theta harvest (NEW)     |
              +--------------------------------+
```

---

## Feature 1: US Markets + GIFT Nifty (Hard Gates)

**Source**: yfinance (free, no API key, pip-installable)

```python
import yfinance as yf
sp500 = yf.Ticker("^GSPC").history(period="2d")
nasdaq = yf.Ticker("^IXIC").history(period="2d")
gift_nifty = yf.Ticker("^NSEI")  # or alternative GIFT Nifty source
```

### Hard Gate Thresholds

| Signal | Condition | Action |
|--------|-----------|--------|
| US crash | S&P 500 down > 2% | BLOCK all bullish entries |
| US severe crash | S&P 500 down > 3% | BLOCK all entries |
| US IT crash | Nasdaq down > 3% | BLOCK IT sector bullish specifically |
| US mild red | S&P 500 down 1-2% | REDUCE bullish allocation 50% |
| GIFT gap down | GIFT Nifty gap > 1.5% below prev close | REDUCE position sizes 50% |
| GIFT crash gap | GIFT Nifty gap > 2.5% below prev close | BLOCK all bullish entries |
| US/GIFT green | Up > 1.5% | Log as tailwind (no override) |

### Caching

6-hour file cache (US data doesn't change intraday). Cache key: `us_markets_{date}`.

---

## Feature 2: X/Twitter Sentiment (Soft Signal)

**Source**: Python port of x-research scraper. Cookie-based auth using existing `TWITTER_COOKIES` from `~/.config/env/global.env`.

### Python Scraper Scope

Minimal port (~150 lines). Only `search()` function needed:

```python
def search_x(query: str, since_hours: int = 12, min_likes: int = 50, limit: int = 20) -> list[dict]:
    """Search X using cookie auth. Returns list of {text, likes, user, timestamp}."""
```

Uses Twitter's search API endpoint with cookie header (same as agent-twitter-client).

### Search Queries (India market focused)

```python
X_SEARCH_QUERIES = [
    "Nifty today",
    "GIFT Nifty",
    "India VIX",
    "FII selling OR FII buying",
    "RBI rate",
    "war India OR sanctions India",
]
```

### Classification

Aggregate tweets -> Claude Haiku -> JSON:

```python
{
    "sentiment": "bullish | bearish | neutral | crisis",
    "confidence": "high | medium | low",
    "key_themes": ["FII selling", "tariff fears"],
    "tweet_count": 15
}
```

### Integration

**Soft signal only** — never auto-blocks. Used as:
- Contradiction filter (like YouTube Mode B): if X says "crisis" but screener says "bullish", log warning
- Included in Telegram pre-market summary
- 1-hour cache

---

## Feature 3: FII/DII Flow Data (Hard Gate)

**Source**: Moneycontrol FII/DII activity page (free, no login, updated daily ~7 PM).

Scrape via `requests` + BeautifulSoup. Fallback: NSDL daily FPI data.

### Hard Gate Thresholds

| FII Activity | Condition | Action |
|--------------|-----------|--------|
| FII heavy selling | Net sell > 5,000 cr | REDUCE bullish allocation 50% |
| FII extreme selling | Net sell > 10,000 cr | BLOCK all bullish equity entries |
| DII support | DII buying while FII selling | Moderate FII sell signal (reduce threshold by 25%) |
| FII buying | Net buy > 3,000 cr | Log as tailwind |

### Data Structure

```python
{
    "fii_net_crores": -4200,
    "dii_net_crores": 3100,
    "fii_equity_buy": 12000,
    "fii_equity_sell": 16200,
    "date": "2026-03-05",
}
```

### Caching

12-hour cache (FII/DII updates once daily after market close).

---

## Feature 4: Nifty PCR (Put-Call Ratio)

**Source**: Computed from option chain already fetched in `open_positions()`. Zero additional API calls.

```python
def compute_nifty_pcr(chain: list[dict]) -> float:
    put_oi = sum(s.get("PE", {}).get("openInterest", 0) for s in chain)
    call_oi = sum(s.get("CE", {}).get("openInterest", 0) for s in chain)
    return put_oi / call_oi if call_oi > 0 else 1.0
```

### Gate Logic

| PCR | Interpretation | Action |
|-----|---------------|--------|
| > 1.3 | Extreme put buying (contrarian bullish) | Log, boost bullish confidence |
| 0.8 - 1.3 | Neutral | No action |
| < 0.7 | Extreme call buying (contrarian bearish) | REDUCE bullish allocation 25% |
| < 0.5 | Euphoria | BLOCK new bullish entries |

---

## Feature 5: Max Pain + OI Analysis

**Source**: Same option chain data.

### Max Pain Computation

```python
def compute_max_pain(chain: list[dict], lot_size: int = 75) -> float:
    """Find strike where total option writer pain is minimized."""
    strikes = sorted(set(s["strikePrice"] for s in chain))
    min_pain = float("inf")
    max_pain_strike = strikes[len(strikes) // 2]

    for test_strike in strikes:
        total_pain = 0
        for s in chain:
            sp = s["strikePrice"]
            # Call writers' pain
            call_oi = s.get("CE", {}).get("openInterest", 0)
            if test_strike > sp:
                total_pain += (test_strike - sp) * call_oi * lot_size
            # Put writers' pain
            put_oi = s.get("PE", {}).get("openInterest", 0)
            if test_strike < sp:
                total_pain += (sp - test_strike) * put_oi * lot_size

        if total_pain < min_pain:
            min_pain = total_pain
            max_pain_strike = test_strike

    return max_pain_strike
```

### Usage

- Iron condor strike selection: center condor around max pain
- Telegram alerts: report max pain vs current spot
- OI buildup at key strikes: identify support/resistance from options market

---

## Feature 6: Asia Markets Context (Soft Signal)

**Source**: yfinance

```python
hang_seng = yf.Ticker("^HSI").history(period="2d")   # Hong Kong
nikkei = yf.Ticker("^N225").history(period="2d")      # Japan
shanghai = yf.Ticker("000001.SS").history(period="2d") # China
```

**Soft signal only** — logged in Telegram pre-market summary. No hard gates (GIFT Nifty already captures India-specific Asia impact).

6-hour cache.

---

## Feature 7: Trailing Stop Loss for Equity

**Current**: Fixed ATR target (+5%) and stoploss (-3%) with 7-day max hold.

**Enhancement**: After position reaches +2% unrealized, activate trailing stop.

```python
# In monitor_positions(), for equity positions:
unrealized_pct = ((current_price - entry_price) / entry_price) * 100

if unrealized_pct >= TRAILING_SL_ACTIVATION_PCT:  # 2.0%
    # Track high water mark
    high_water = max(entry_price, pos.get("high_water_mark", current_price))
    pos["high_water_mark"] = max(high_water, current_price)

    trailing_stop = pos["high_water_mark"] - (1.5 * atr)
    trailing_stop = max(trailing_stop, entry_price)  # never trail below entry

    if current_price <= trailing_stop:
        # Exit: trailing stop hit
        close_position(pos, current_price, "trailing_stop")
```

### Constants

```python
TRAILING_SL_ACTIVATION_PCT = 2.0   # activate after +2%
TRAILING_SL_ATR_MULT = 1.5        # trail at 1.5x ATR below high water mark
```

---

## Feature 8: System Health Alerting

### Cron Wrapper Enhancement

```bash
#!/bin/bash
# paper_trade_cron.sh

RESULT=$(cd ~/financial-agent-india && source venv/bin/activate && python paper_trade.py "$1" 2>&1)
EXIT_CODE=$?

# Always send heartbeat
OPEN_COUNT=$(echo "$RESULT" | grep -oP 'Open: \K\d+' || echo "?")
CAPITAL=$(echo "$RESULT" | grep -oP 'Capital: .+?(?=\|)' || echo "?")

if [ $EXIT_CODE -eq 0 ]; then
    # Success heartbeat (compact one-liner)
    MSG="[heartbeat] $1 OK | Open: $OPEN_COUNT | $CAPITAL"
else
    # Error alert
    MSG="[ALERT] paper_trade.py $1 FAILED (exit $EXIT_CODE)\n$(echo "$RESULT" | tail -5)"
fi

curl -s -X POST "https://api.telegram.org/bot$BOT_TOKEN/sendMessage" \
    -d chat_id="$CHAT_ID" -d parse_mode="HTML" --data-urlencode "text=$MSG"
```

### Watchdog Cron

Separate 2-hour cron that checks if paper_trade last ran within 2 hours:

```bash
# Check last modification time of portfolio.json
LAST_MOD=$(stat -c %Y ~/financial-agent-india/data/paper_trades/portfolio.json)
NOW=$(date +%s)
DIFF=$(( (NOW - LAST_MOD) / 3600 ))

if [ $DIFF -ge 2 ]; then
    # Alert: bot hasn't run in 2+ hours
fi
```

---

## Feature 9: Supertrend Confirmation

**Source**: `ta` library (`ta.trend.SuperTrend` or manual computation).

Supertrend(10, 3) on daily Nifty candles. Used as entry confirmation:

```python
from ta.trend import STCIndicator  # or manual computation

def compute_supertrend(df, period=10, multiplier=3):
    """Returns 'buy' or 'sell' signal based on Supertrend."""
    # ATR-based trailing stop that flips on price crossing
    ...
```

### Integration

- If screener says "bullish" but Supertrend says "sell" -> log warning, reduce confidence
- If both agree -> full conviction
- Added as a field in the candidate dict: `supertrend_signal: "buy" | "sell"`

Not a hard gate — soft confirmation that adjusts position sizing by 25%.

---

## Feature 10: CPR (Central Pivot Range) for Day Classification

**Computation** (from previous day's candle):

```python
pivot = (prev_high + prev_low + prev_close) / 3
bc = (prev_high + prev_low) / 2  # bottom of central pivot
tc = (pivot - bc) + pivot          # top of central pivot
cpr_width = abs(tc - bc)
cpr_width_pct = (cpr_width / prev_close) * 100
```

### Day Type Classification

| CPR Width % | Day Type | Impact |
|-------------|----------|--------|
| < 0.3% | Narrow CPR (trending day expected) | Favor directional trades, skip iron condors |
| 0.3% - 0.8% | Normal CPR | No modification |
| > 0.8% | Wide CPR (sideways day expected) | Favor iron condors, reduce directional |

### Integration

- Narrow CPR day + TRENDING regime -> strong directional conviction
- Wide CPR day + SIDEWAYS regime -> strong iron condor conviction
- Mismatch (narrow CPR + SIDEWAYS) -> UNCERTAIN, reduce sizes

---

## Feature 11: Weekly Theta Harvesting (Strategy H)

New strategy that specifically targets weekly expiry theta decay.

### Strategy Rules

- **When**: Friday or Monday (targeting Tuesday weekly expiry)
- **What**: Sell OTM strangle on Nifty (sell OTM call + sell OTM put)
- **Strikes**: ~1 standard deviation away from spot (~200-300 points OTM for Nifty)
- **Exit**: Target 50% of premium collected, or Tuesday 2 PM (1.5 hours before expiry)
- **Risk**: Max loss capped at 2% of capital per trade
- **VIX filter**: Only when VIX > 14 (enough premium to sell)
- **Regime filter**: Not in VOLATILE or CASH regime
- **Macro filter**: No macro events within 48 hours

### Position Structure

```python
{
    "instrument": "WEEKLY_THETA",
    "symbol": "NIFTY",
    "strategy": "short_strangle",
    "call_strike": 24800,
    "put_strike": 24200,
    "call_premium": 45,
    "put_premium": 38,
    "total_credit": 83,  # per lot
    "lot_size": 75,
    "lots": 1,
    "expiry": "2026-03-10",
    "target_pct": 0.50,
    "max_loss": 2000,
}
```

### Exit Logic

```python
def check_weekly_theta_exit(pos, current_call_premium, current_put_premium):
    total_current = current_call_premium + current_put_premium
    total_entry = pos["total_credit"]

    # Target: 50% premium decay
    if total_current <= total_entry * 0.50:
        return "target"

    # Stop loss: premium doubles (loss = credit collected)
    if total_current >= total_entry * 2.0:
        return "stoploss"

    # Time exit: Tuesday 2 PM
    if is_expiry_day() and current_hour >= 14:
        return "time_exit"

    return None
```

---

## Feature 12: BankNifty Support

Add BankNifty as a second index for iron condors, momentum, and weekly theta.

### Token/Symbol Setup

```python
BANKNIFTY_TOKEN = "99926009"  # AngelOne symbol token for BankNifty
BANKNIFTY_LOT_SIZE = 30       # current lot size (was 15 before Nov 2024)
```

### Strategy Applicability

| Strategy | Nifty | BankNifty |
|----------|-------|-----------|
| Iron condor | Yes | Yes (wider wings due to higher vol) |
| Momentum options | Yes | Yes (higher premiums) |
| Weekly theta | Yes | Yes (Tuesday expiry) |
| Equity/spreads | Yes | N/A (index only) |

### BankNifty-Specific Adjustments

- BankNifty VIX is ~1.3x Nifty VIX — adjust VIX thresholds proportionally
- Wider strikes for condors (BankNifty moves more in absolute points)
- Higher minimum premium thresholds (BankNifty options are pricier)
- BankNifty weekly also expires Tuesday (same as Nifty)

### Capital Allocation Update

```python
# Updated allocations with BankNifty
ALLOC_EQUITY_MAX = 0.35        # reduced from 0.40
ALLOC_SPREADS_MAX = 0.25       # reduced from 0.30
ALLOC_IRON_CONDOR_MAX = 0.15   # increased from 0.10 (Nifty + BankNifty)
ALLOC_MOMENTUM_MAX = 0.10      # reduced from 0.15
ALLOC_WEEKLY_THETA_MAX = 0.10  # NEW
ALLOC_CASH_MIN = 0.05          # unchanged
```

---

## MacroContext Output (Complete)

```python
{
    # US Markets (Feature 1)
    "sp500_pct_change": -1.8,
    "nasdaq_pct_change": -2.3,

    # GIFT Nifty (Feature 1)
    "gift_nifty_gap_pct": -1.2,
    "gift_nifty_ltp": 24350.0,

    # X/Twitter (Feature 2)
    "x_sentiment": "bearish",
    "x_confidence": "medium",
    "x_key_themes": ["FII selling", "tariff fears"],
    "x_tweet_count": 15,

    # FII/DII (Feature 3)
    "fii_net_crores": -4200,
    "dii_net_crores": 3100,

    # Asia (Feature 6)
    "hang_seng_pct": -0.8,
    "nikkei_pct": -1.1,

    # Computed gates
    "hard_gate": "REDUCE_50",
    "hard_gate_reason": "S&P 500 -1.8% overnight",
    "fetched_at": "2026-03-09T08:45:00+05:30",
}
```

---

## New Files

| File | Lines (est) | Purpose |
|------|-------------|---------|
| `global_intel.py` | ~250 | US markets, GIFT Nifty, Asia markets, FII/DII scraping, MacroContext assembly |
| `x_intel.py` | ~180 | Python X/Twitter scraper (cookie auth + search) + Claude Haiku classification |
| `oi_analysis.py` | ~120 | PCR, max pain, OI buildup analysis from option chain |

## Modified Files

| File | Changes |
|------|---------|
| `paper_trade.py` | Import macro/OI modules, add hard gate logic, trailing SL, Supertrend/CPR filters, weekly theta strategy, BankNifty index strategies |
| `config.py` | New constants for all features (thresholds, allocations, BankNifty tokens) |
| `agent_with_options.py` | BankNifty option chain support, strangle strike selection |
| `scripts/paper_trade_cron.sh` | Heartbeat + failure alerts |

## New Dependencies

| Package | Purpose | Pi-compatible |
|---------|---------|---------------|
| `yfinance` | US/Asia/GIFT market data | Yes |
| `beautifulsoup4` | FII/DII scraping from moneycontrol | Yes |

## Testing Strategy

Each feature gets its own test file with mocked external calls:
- `tests/test_global_intel.py` — mock yfinance, test hard gate logic
- `tests/test_x_intel.py` — mock requests, test sentiment classification
- `tests/test_oi_analysis.py` — test PCR, max pain with synthetic chain data
- `tests/test_trailing_sl.py` — test trailing stop activation and movement
- `tests/test_weekly_theta.py` — test strangle entry/exit logic
- `tests/test_banknifty.py` — test BankNifty strategy wiring
- `tests/test_supertrend_cpr.py` — test indicator computation and gate logic

---

## Priority Order for Implementation

1. **Global intel** (US + GIFT + FII/DII) — highest impact, enables hard gates
2. **PCR + Max Pain + OI** — zero API cost, uses existing data
3. **Supertrend + CPR** — low effort indicator additions
4. **X/Twitter sentiment** — Python scraper port
5. **Trailing stop loss** — quick win for equity exits
6. **Weekly theta harvesting** — new strategy
7. **BankNifty support** — second index
8. **Asia markets** — soft signal, lowest priority
9. **System health alerts** — cron wrapper, operational
