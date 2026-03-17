"""V7 Market Intelligence — unified data gathering for premarket playbook.

Aggregates:
- US markets (S&P, Nasdaq) via yfinance
- GIFT Nifty gap via global_intel
- India VIX via data feed
- FII/DII flows via MoneyControl scraper
- Company news + sentiment via DuckDuckGo + Claude
- NSE corporate actions (earnings, dividends, board meetings)
- Economic calendar (RBI, budget, macro events)
- Time intelligence (expiry, gap, IV)
- F&O ban list from NSE
"""
from __future__ import annotations

import json
import logging
import time as time_mod
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

CACHE_DIR = Path("data/v7/intel_cache")


def _cache_get(key: str, max_age_hours: float = 12) -> dict | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            data = json.loads(path.read_text())
            cached_at = datetime.fromisoformat(data["_cached_at"])
            if datetime.now() - cached_at < timedelta(hours=max_age_hours):
                return data.get("result")
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
    return None


def _cache_set(key: str, result: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps({
        "_cached_at": datetime.now().isoformat(),
        "result": result,
    }, indent=2, default=str))


# ---------------------------------------------------------------------------
# US Markets
# ---------------------------------------------------------------------------

def fetch_us_close() -> dict:
    """S&P 500, Nasdaq overnight close."""
    try:
        from global_intel import fetch_us_markets
        return fetch_us_markets()
    except Exception as e:
        log.warning("US markets fetch failed: %s", e)
        return {"sp500_pct_change": 0.0, "nasdaq_pct_change": 0.0}


# ---------------------------------------------------------------------------
# GIFT Nifty
# ---------------------------------------------------------------------------

def fetch_gift_nifty(prev_nifty_close: float) -> str:
    """GIFT Nifty gap as a formatted string."""
    if prev_nifty_close <= 0:
        return "unavailable (no prev close)"
    try:
        from global_intel import fetch_gift_nifty_gap
        data = fetch_gift_nifty_gap(prev_nifty_close)
        gap = data.get("gift_nifty_gap_pct", 0)
        ltp = data.get("gift_nifty_ltp", 0)
        if ltp > 0:
            return f"{ltp:.0f} ({gap:+.2f}% gap)"
        return f"{gap:+.2f}% gap"
    except Exception as e:
        log.warning("GIFT Nifty fetch failed: %s", e)
        return "unavailable"


# ---------------------------------------------------------------------------
# FII/DII
# ---------------------------------------------------------------------------

def fetch_fii_dii_flows() -> str:
    """FII/DII net flows as formatted string."""
    try:
        from global_intel import fetch_fii_dii
        data = fetch_fii_dii()
        fii = data.get("fii_net", 0)
        dii = data.get("dii_net", 0)
        if fii != 0 or dii != 0:
            return f"FII: Rs.{fii:+,.0f}Cr, DII: Rs.{dii:+,.0f}Cr"
        return "unavailable"
    except Exception as e:
        log.warning("FII/DII fetch failed: %s", e)
        return "unavailable"


# ---------------------------------------------------------------------------
# Company News (for watchlist stocks)
# ---------------------------------------------------------------------------

def fetch_watchlist_news(symbols: list[str]) -> dict[str, list[str]]:
    """Fetch recent news headlines for watchlist stocks via DuckDuckGo.

    Returns: {symbol: [headline1, headline2, ...]}
    Cached for 6 hours.
    """
    today_str = date.today().isoformat()
    cache_key = f"news_{today_str}"
    cached = _cache_get(cache_key, max_age_hours=6)
    if cached:
        return cached

    result = {}
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        log.warning("duckduckgo-search not installed, skipping news")
        return {}

    for symbol in symbols:
        try:
            with DDGS() as ddgs:
                articles = list(ddgs.news(
                    f"{symbol} NSE stock India", max_results=3
                ))
            headlines = []
            for a in articles:
                title = a.get("title", "")
                if title:
                    headlines.append(title)
            result[symbol] = headlines
            time_mod.sleep(0.5)  # rate limit
        except Exception as e:
            log.debug("News fetch failed for %s: %s", symbol, e)
            result[symbol] = []

    _cache_set(cache_key, result)
    return result


def classify_news_batch(news: dict[str, list[str]]) -> dict[str, dict]:
    """Use Claude Haiku to classify news sentiment for all stocks at once.

    Returns: {symbol: {sentiment, has_earnings_soon, has_policy_impact, key_headline}}
    """
    # Filter to stocks with actual headlines
    with_news = {s: h for s, h in news.items() if h}
    if not with_news:
        return {}

    prompt_lines = [
        "Classify news sentiment for these Indian stocks. "
        "For each, return: sentiment (positive/neutral/negative), "
        "has_earnings_soon (bool), has_policy_impact (bool), "
        "key_headline (most important headline or empty string).\n"
    ]
    for symbol, headlines in with_news.items():
        prompt_lines.append(f"=== {symbol} ===")
        for h in headlines:
            prompt_lines.append(f"- {h}")
        prompt_lines.append("")

    prompt_lines.append(
        'Respond with JSON only: {"SYMBOL": {"sentiment": "...", '
        '"has_earnings_soon": bool, "has_policy_impact": bool, '
        '"key_headline": "..."}, ...}'
    )

    try:
        import config
        client = config.get_anthropic_client()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": "\n".join(prompt_lines)}],
        )
        text = response.content[0].text
        # Extract JSON from response
        import re
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return json.loads(match.group())
    except Exception as e:
        log.warning("News classification failed: %s", e)

    return {}


# ---------------------------------------------------------------------------
# NSE Corporate Actions
# ---------------------------------------------------------------------------

def fetch_nse_corporate_actions(symbols: list[str]) -> list[str]:
    """Fetch upcoming corporate actions for watchlist stocks.

    Scrapes NSE website for dividends, splits, rights issues, bonus
    happening in the next 7 days.

    Returns list of event strings like:
      "RELIANCE: Ex-dividend Rs.10 on 2026-03-15"
      "TCS: Board meeting on 2026-03-14 (quarterly results)"
    """
    today = date.today()
    cache_key = f"corp_actions_{today.isoformat()}"
    cached = _cache_get(cache_key, max_age_hours=12)
    if cached:
        return cached

    events = []

    # Method 1: DuckDuckGo search for corporate actions
    try:
        from duckduckgo_search import DDGS
        for symbol in symbols:
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.news(
                        f"{symbol} NSE corporate action dividend results board meeting 2026",
                        max_results=2,
                    ))
                for r in results:
                    title = r.get("title", "")
                    if any(kw in title.lower() for kw in [
                        "dividend", "bonus", "split", "rights", "board meet",
                        "results", "earnings", "quarterly", "AGM",
                    ]):
                        events.append(f"{symbol}: {title}")
                time_mod.sleep(0.5)
            except Exception:
                pass
    except ImportError:
        pass

    _cache_set(cache_key, events)
    return events


# ---------------------------------------------------------------------------
# Economic Calendar
# ---------------------------------------------------------------------------

def get_economic_events() -> tuple[list[str], list[str]]:
    """Get economic events for today and this week.

    Combines hardcoded dates (RBI, budget) with web-fetched macro events.

    Returns: (events_today, events_this_week)
    """
    today = date.today()
    week_end = today + timedelta(days=5)
    events_today = []
    events_this_week = []

    # Hardcoded major events from config
    try:
        from config import RBI_POLICY_DATES, BUDGET_BLOCK_DATES, NSE_HOLIDAYS_2026

        for d_str in RBI_POLICY_DATES:
            d = date.fromisoformat(d_str)
            if d == today:
                events_today.append("RBI Monetary Policy Decision")
            elif today < d <= week_end:
                events_this_week.append(f"RBI Policy on {d.strftime('%a %b %d')}")

        for d_str in BUDGET_BLOCK_DATES:
            d = date.fromisoformat(d_str)
            if d == today:
                events_today.append("Union Budget — high volatility expected")
            elif today < d <= week_end:
                events_this_week.append(f"Union Budget on {d.strftime('%a %b %d')}")

        for d, desc in NSE_HOLIDAYS_2026.items():
            if d == today:
                events_today.append(f"NSE Holiday: {desc}")
            elif today < d <= week_end:
                events_this_week.append(f"NSE Holiday {d.strftime('%a %b %d')}: {desc}")
    except Exception as e:
        log.debug("Config events lookup failed: %s", e)

    # Expiry awareness
    try:
        from time_intel import is_expiry_day
        expiry = is_expiry_day(today)
        if expiry:
            events_today.append(f"{expiry.capitalize()} expiry day — elevated volatility, pin risk")
        # Check if tomorrow is expiry
        tomorrow = today + timedelta(days=1)
        tomorrow_expiry = is_expiry_day(tomorrow)
        if tomorrow_expiry:
            events_today.append(f"Tomorrow is {tomorrow_expiry} expiry — consider carry implications")
    except Exception:
        pass

    # Fetch recent macro events via DuckDuckGo
    cache_key = f"macro_events_{today.isoformat()}"
    cached = _cache_get(cache_key, max_age_hours=8)
    if cached:
        events_today.extend(cached.get("today", []))
        events_this_week.extend(cached.get("week", []))
    else:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.news(
                    "India economy RBI CPI GDP PMI manufacturing data release",
                    max_results=5,
                ))
            macro_today = []
            macro_week = []
            for r in results:
                title = r.get("title", "")
                if any(kw in title.lower() for kw in [
                    "cpi", "gdp", "pmi", "inflation", "rbi", "rate cut",
                    "rate hike", "fiscal deficit", "trade deficit", "iip",
                ]):
                    macro_week.append(title)
            _cache_set(cache_key, {"today": macro_today, "week": macro_week})
            events_this_week.extend(macro_week)
        except Exception:
            pass

    return events_today, events_this_week


# ---------------------------------------------------------------------------
# F&O Ban List
# ---------------------------------------------------------------------------

def fetch_fo_ban_list() -> list[str]:
    """Fetch current NSE F&O ban list.

    Tries DuckDuckGo for today's ban list. Falls back to empty.
    """
    today = date.today()
    cache_key = f"fo_ban_{today.isoformat()}"
    cached = _cache_get(cache_key, max_age_hours=18)
    if cached is not None:
        return cached

    ban_list = []
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(
                f"NSE F&O ban list {today.strftime('%d %B %Y')}",
                max_results=3,
            ))
        # Parse ban list from snippets
        common_ban_stocks = [
            "AARTIIND", "ABFRL", "BALRAMCHIN", "BIOCON", "DELTACORP",
            "GNFC", "GRANULES", "HINDCOPPER", "IBULHSGFIN", "IDEA",
            "INDIACEM", "INDUSTOWER", "IRCTC", "MANAPPURAM", "NATIONALUM",
            "NAVINFLUOR", "PEL", "RBLBANK", "SAL", "ZEEL",
        ]
        for r in results:
            snippet = r.get("body", "").upper()
            for stock in common_ban_stocks:
                if stock in snippet and stock not in ban_list:
                    ban_list.append(stock)
    except Exception as e:
        log.debug("F&O ban list fetch failed: %s", e)

    _cache_set(cache_key, ban_list)
    return ban_list


# ---------------------------------------------------------------------------
# Unified Premarket Intelligence
# ---------------------------------------------------------------------------

def gather_premarket_intel(
    data_feed=None,
    edge_tracker=None,
    risk_engine=None,
    state=None,
    data_dir: Path = Path("data/v7"),
) -> dict:
    """Gather ALL market intelligence for premarket playbook generation.

    This is the single entry point that cmd_premarket should call.
    Returns the exact kwargs needed by strategist.generate_premarket_playbook().
    """
    from v7.config_v7 import WATCHLIST

    stock_symbols = [w["symbol"] for w in WATCHLIST if w["type"] == "stock"]
    all_symbols = [w["symbol"] for w in WATCHLIST]

    # 1. US markets
    us_close = fetch_us_close()

    # 2. GIFT Nifty
    prev_nifty_close = 0.0
    if data_feed:
        try:
            prices = data_feed.get_batch_ltp(["NIFTY"])
            prev_nifty_close = prices.get("NIFTY", 0.0)
        except Exception:
            pass
    gift_nifty = fetch_gift_nifty(prev_nifty_close)

    # 3. VIX
    prev_vix = 0.0
    if data_feed:
        try:
            prev_vix = data_feed.get_vix()
        except Exception:
            pass

    # 4. FII/DII
    fii_dii = fetch_fii_dii_flows()

    # 5. Economic events
    events_today, events_this_week = get_economic_events()

    # 6. Corporate actions for watchlist stocks
    corp_actions = fetch_nse_corporate_actions(stock_symbols)
    if corp_actions:
        events_today.extend([f"[Corp Action] {e}" for e in corp_actions[:5]])

    # 7. Company news + sentiment
    news = fetch_watchlist_news(stock_symbols)
    news_classified = classify_news_batch(news)
    # Add material news to events
    for symbol, data in news_classified.items():
        if data.get("has_earnings_soon"):
            events_today.append(f"{symbol}: earnings expected soon")
        if data.get("has_policy_impact"):
            events_today.append(f"{symbol}: policy impact detected")
        headline = data.get("key_headline", "")
        sentiment = data.get("sentiment", "neutral")
        if headline and sentiment != "neutral":
            events_today.append(f"{symbol} ({sentiment}): {headline}")

    # 8. F&O ban list
    fo_ban_list = fetch_fo_ban_list()

    # 9. Level memory
    level_memory = {}
    try:
        from v7.level_memory import LevelMemory
        lm = LevelMemory(data_dir=data_dir)
        level_memory = lm.summary_for_prompt()
    except Exception:
        pass

    # 10. Computed levels (pivots from previous day candles)
    #     Kite primary for all symbols, yfinance fallback for indices only
    computed_levels = {}
    try:
        from v7.computed_levels import LevelComputer
        from v7.config_v7 import WATCHLIST
        lc = LevelComputer()

        # Try Kite first for all watchlist symbols
        kite_ok = False
        try:
            from kite_data import fetch_candles_kite
            kite_ok = True
        except Exception:
            pass

        for item in WATCHLIST:
            symbol = item["symbol"]
            try:
                if kite_ok:
                    candles = fetch_candles_kite(symbol, interval="ONE_DAY", days=5)
                    if candles and len(candles) >= 2:
                        prev = candles[-2]  # [timestamp, open, high, low, close, volume]
                        levels = lc.compute_pivots(symbol, prev[2], prev[3], prev[4])
                        computed_levels[symbol] = levels.to_dict()
                        continue

                # yfinance fallback (indices only)
                if item["type"] == "index":
                    import yfinance as yf
                    ticker = "^NSEI" if symbol == "NIFTY" else "^NSEBANK"
                    hist = yf.Ticker(ticker).history(period="5d")
                    if len(hist) >= 2:
                        prev = hist.iloc[-2]
                        levels = lc.compute_pivots(symbol, prev["High"], prev["Low"], prev["Close"])
                        computed_levels[symbol] = levels.to_dict()
            except Exception as e:
                log.debug("Computed levels failed for %s: %s", symbol, e)
    except Exception as e:
        log.debug("Computed levels init failed: %s", e)

    # 11. Edge tracker stats
    edge_stats = {}
    if edge_tracker:
        try:
            edge_stats = edge_tracker.get_stats()
        except Exception:
            pass

    # 12. Risk state
    risk_state = {"mtd_pnl_pct": 0.0, "pacing": "on_track", "survival_mode": False}
    if risk_engine:
        try:
            risk_state = risk_engine.get_state_summary()
        except Exception:
            pass

    # 13. Recent lessons from edge tracker
    recent_lessons = []
    if edge_tracker:
        try:
            kill = edge_tracker.kill_candidates()
            if kill:
                recent_lessons.append(f"Kill candidates (low edge): {kill}")
        except Exception:
            pass

    log.info(
        "Premarket intel: US=%s, GIFT=%s, VIX=%.1f, FII/DII=%s, "
        "events_today=%d, events_week=%d, news=%d stocks, fo_ban=%d, "
        "computed_levels=%d",
        us_close, gift_nifty, prev_vix, fii_dii,
        len(events_today), len(events_this_week),
        len(news_classified), len(fo_ban_list),
        len(computed_levels),
    )

    # 14. OI pipeline context
    oi_context = {}
    try:
        from v7.oi_pipeline import OIPipeline
        oi = OIPipeline(data_dir=data_dir)
        oi_context = oi.format_for_prompt(["NIFTY", "BANKNIFTY"])
    except Exception as e:
        log.debug("OI pipeline context failed: %s", e)

    return {
        "us_close": us_close,
        "gift_nifty": gift_nifty,
        "prev_vix": prev_vix,
        "fii_dii": fii_dii,
        "events_today": events_today,
        "events_this_week": events_this_week,
        "level_memory": level_memory,
        "edge_tracker": edge_stats,
        "risk_state": risk_state,
        "fo_ban_list": fo_ban_list,
        "recent_lessons": recent_lessons,
        "computed_levels": computed_levels,
        "oi_context": oi_context,
    }
