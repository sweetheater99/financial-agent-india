"""X/Twitter intelligence for Trading System V3.

Searches X for India market sentiment, classifies via Claude Haiku.
Soft signal only — never auto-blocks, used as contradiction filter.

Requires TWITTER_COOKIES env var: "auth_token=XXX; ct0=YYY"
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import requests

import config
from utils import parse_claude_json

logger = logging.getLogger("paper_trade")

CACHE_DIR = Path("data/x_intel_cache")


# ── Cookie auth ───────────────────────────────────────────────────────────────

def _parse_cookies(cookie_str: str) -> dict:
    """Parse 'key=val; key2=val2' into dict."""
    if not cookie_str:
        return {}
    result = {}
    for part in cookie_str.split(";"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            result[k.strip()] = v.strip()
    return result


def _get_cookies() -> dict:
    cookie_str = os.environ.get("TWITTER_COOKIES", "")
    if not cookie_str:
        env_file = Path.home() / ".config" / "env" / "global.env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("TWITTER_COOKIES="):
                    cookie_str = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    return _parse_cookies(cookie_str)


# ── X Search ──────────────────────────────────────────────────────────────────

def search_x(query: str, since_hours: int = 12, min_likes: int = None, limit: int = 20) -> list[dict]:
    """Search X/Twitter for recent tweets matching query.

    Returns list of {text, likes, user, timestamp}.
    Requires TWITTER_COOKIES env var.
    """
    min_likes = min_likes or config.X_MIN_LIKES
    cookies = _get_cookies()
    if not cookies.get("auth_token") or not cookies.get("ct0"):
        logger.debug("X search: no Twitter cookies configured")
        return []

    headers = {
        "Authorization": "Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA",
        "X-Csrf-Token": cookies["ct0"],
        "Cookie": f"auth_token={cookies['auth_token']}; ct0={cookies['ct0']}",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    }

    since_str = (datetime.utcnow() - timedelta(hours=since_hours)).strftime("%Y-%m-%d")
    full_query = f"{query} since:{since_str} -is:retweet min_faves:{min_likes}"

    url = "https://api.twitter.com/2/search/adaptive.json"
    params = {
        "q": full_query,
        "count": limit,
        "tweet_mode": "extended",
        "result_type": "recent",
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code != 200:
            logger.debug("X search failed: HTTP %d", resp.status_code)
            return []

        data = resp.json()
        tweets_raw = data.get("globalObjects", {}).get("tweets", {})
        users_raw = data.get("globalObjects", {}).get("users", {})

        tweets = []
        for tid, tweet in tweets_raw.items():
            user_id = tweet.get("user_id_str", "")
            user = users_raw.get(user_id, {})
            tweets.append({
                "text": tweet.get("full_text", ""),
                "likes": tweet.get("favorite_count", 0),
                "user": user.get("screen_name", "unknown"),
                "timestamp": tweet.get("created_at", ""),
            })

        return sorted(tweets, key=lambda t: t["likes"], reverse=True)[:limit]

    except Exception as e:
        logger.debug("X search error: %s", e)
        return []


# ── Claude classification ─────────────────────────────────────────────────────

def _call_claude_haiku(system_prompt: str, user_message: str) -> dict | None:
    client = config.get_anthropic_client()
    response = client.messages.create(
        model=config.CLAUDE_MODEL_LIGHT,
        max_tokens=512,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    text = response.content[0].text
    return parse_claude_json(text)


def classify_x_sentiment(tweet_texts: list[str]) -> dict | None:
    """Classify aggregate X/Twitter sentiment from tweet texts.

    Returns {sentiment, confidence, key_themes} or None.
    """
    if not tweet_texts:
        return None

    combined = "\n---\n".join(tweet_texts[:20])
    system_prompt = (
        "You are an Indian stock market analyst. Analyze these recent X/Twitter posts "
        "about Indian markets and classify the overall sentiment. "
        "Respond with ONLY valid JSON:\n"
        '{"sentiment": "bullish | bearish | neutral | crisis", '
        '"confidence": "high | medium | low", '
        '"key_themes": ["theme1", "theme2"]}'
    )

    try:
        return _call_claude_haiku(system_prompt, combined[:4000])
    except Exception:
        return None


# ── Caching ───────────────────────────────────────────────────────────────────

def _get_cache(cache_key: str, max_age_hours: float) -> dict | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            cached_at = datetime.fromisoformat(data.get("_cached_at", "2000-01-01"))
            if datetime.now() - cached_at < timedelta(hours=max_age_hours):
                return data.get("result")
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _set_cache(cache_key: str, result: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    cache_file.write_text(json.dumps({
        "_cached_at": datetime.now().isoformat(),
        "result": result,
    }, indent=2))


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_x_sentiment() -> dict | None:
    """Fetch X/Twitter market sentiment. Cached per config.X_CACHE_HOURS.

    Returns {sentiment, confidence, key_themes, tweet_count} or None.
    """
    cache_key = f"x_sentiment_{datetime.now().strftime('%Y-%m-%d_%H')}"
    cached = _get_cache(cache_key, config.X_CACHE_HOURS)
    if cached:
        return cached

    all_texts = []
    for query in config.X_SEARCH_QUERIES:
        tweets = search_x(query, since_hours=12, limit=10)
        all_texts.extend(t["text"] for t in tweets)

    if not all_texts:
        return None

    result = classify_x_sentiment(all_texts)
    if result:
        result["tweet_count"] = len(all_texts)
        _set_cache(cache_key, result)
    return result
