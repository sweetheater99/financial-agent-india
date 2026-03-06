"""YouTube intelligence module for market and stock analysis.

Mode A: Daily Market Intel (pre-market) — searches for Nifty/market analysis videos,
    extracts transcripts, and uses Claude haiku to classify market bias, levels, sectors.

Mode B: Pre-Trade Stock Research — searches for stock-specific analysis videos,
    extracts transcripts, and classifies sentiment, levels, red flags.
"""

import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi

import config
from utils import parse_claude_json

# ── Constants ────────────────────────────────────────────────────────────────

CACHE_DIR = Path("data/youtube_cache")

MARKET_SEARCH_QUERIES = [
    "Nifty analysis today",
    "market outlook India today",
]

TRANSCRIPT_MAX_CHARS = 4000

# ── Low-level helpers ────────────────────────────────────────────────────────


def _run_yt_dlp_search(query, max_results=3):
    """Run yt-dlp search and return video metadata."""
    cmd = [
        "yt-dlp", f"ytsearch{max_results}:{query}",
        "--dump-json", "--flat-playlist",
        "--no-download", "--quiet",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []

    if result.returncode != 0:
        return []

    videos = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        try:
            data = json.loads(line)
            videos.append({
                "id": data.get("id", ""),
                "title": data.get("title", ""),
                "upload_date": data.get("upload_date", ""),
                "channel": data.get("channel", data.get("uploader", "")),
                "url": data.get("url", data.get("webpage_url", "")),
            })
        except json.JSONDecodeError:
            continue
    return videos


def search_youtube_videos(query, max_results=3, max_age_days=1):
    """Search YouTube for videos matching query, filtering by age.

    Returns list of dicts: {id, title, upload_date, channel, url}.
    """
    videos = _run_yt_dlp_search(query, max_results=max_results)

    if max_age_days is not None:
        cutoff = (datetime.now() - timedelta(days=max_age_days)).strftime("%Y%m%d")
        videos = [v for v in videos if v.get("upload_date", "00000000") >= cutoff]

    return videos


def extract_transcript(video_id):
    """Extract transcript from a YouTube video.

    Tries English manual, then Hindi manual, then any auto-generated.
    Returns transcript text (str) or None.
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Priority: English manual > Hindi manual > any auto-generated
        for lang in ["en", "hi"]:
            try:
                transcript = transcript_list.find_manually_created_transcript([lang])
                entries = transcript.fetch()
                return " ".join(e["text"] for e in entries)
            except Exception:
                continue

        # Fall back to auto-generated
        for transcript in transcript_list:
            if transcript.is_generated:
                entries = transcript.fetch()
                return " ".join(e["text"] for e in entries)

        return None
    except Exception:
        return None


# ── Claude classification ────────────────────────────────────────────────────


def _call_claude_haiku(system_prompt, user_message):
    """Call Claude haiku for classification. Returns parsed JSON dict."""
    client = config.get_anthropic_client()
    response = client.messages.create(
        model=config.CLAUDE_MODEL_LIGHT,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    text = response.content[0].text
    return parse_claude_json(text)


def classify_market_intel(transcript_text):
    """Classify a transcript into market intelligence.

    Returns dict with market_bias, key_levels, sectors, events, confidence, summary.
    Returns None on failure.
    """
    system_prompt = (
        "You are an Indian stock market analyst. Analyze the following YouTube video "
        "transcript and extract market intelligence. Respond with ONLY valid JSON in "
        "this exact format:\n"
        '{"market_bias": "bullish | bearish | sideways", '
        '"key_levels": {"nifty_support": <number>, "nifty_resistance": <number>}, '
        '"sectors_bullish": [<list of sector names>], '
        '"sectors_bearish": [<list of sector names>], '
        '"events_today": [<list of events>], '
        '"confidence": "high | medium | low", '
        '"summary": "<2-3 sentence summary>"}'
    )
    try:
        return _call_claude_haiku(system_prompt, transcript_text[:TRANSCRIPT_MAX_CHARS])
    except Exception:
        return None


def classify_stock_intel(symbol, transcript_text):
    """Classify a transcript into stock-specific intelligence.

    Returns dict with sentiment, key_levels, red_flags, catalyst, confidence.
    Returns None on failure.
    """
    system_prompt = (
        f"You are an Indian stock market analyst. Analyze the following YouTube video "
        f"transcript about {symbol} and extract stock-specific intelligence. "
        f"Respond with ONLY valid JSON in this exact format:\n"
        '{"sentiment": "bullish | bearish | neutral", '
        '"key_levels": {"support": <number>, "resistance": <number>}, '
        '"red_flags": [<list of concerns>], '
        '"catalyst": "<main catalyst or empty string>", '
        '"confidence": "high | medium | low"}'
    )
    try:
        return _call_claude_haiku(system_prompt, transcript_text[:TRANSCRIPT_MAX_CHARS])
    except Exception:
        return None


# ── Caching ──────────────────────────────────────────────────────────────────


def _get_cache(cache_key, max_age_hours=12):
    """Read from file cache if fresh enough. Returns cached result or None."""
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


def _set_cache(cache_key, result):
    """Write result to file cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    cache_file.write_text(json.dumps({
        "_cached_at": datetime.now().isoformat(),
        "result": result,
    }, indent=2))


# ── Orchestration (uncached) ─────────────────────────────────────────────────


def _fetch_market_intel_uncached():
    """Search YouTube for market analysis, extract transcripts, classify.

    Returns market intel dict or None.
    """
    all_transcripts = []
    seen_ids = set()

    for query in MARKET_SEARCH_QUERIES:
        videos = search_youtube_videos(query, max_results=3, max_age_days=1)
        for video in videos:
            if video["id"] in seen_ids:
                continue
            seen_ids.add(video["id"])
            transcript = extract_transcript(video["id"])
            if transcript:
                all_transcripts.append(transcript)
            if len(all_transcripts) >= 3:
                break
        if len(all_transcripts) >= 3:
            break

    if not all_transcripts:
        return None

    combined = "\n\n---\n\n".join(all_transcripts)
    return classify_market_intel(combined)


def _fetch_stock_intel_uncached(symbol):
    """Search YouTube for stock-specific analysis, extract transcripts, classify.

    Returns stock intel dict or None.
    """
    query = f"{symbol} stock analysis"
    videos = search_youtube_videos(query, max_results=3, max_age_days=7)

    all_transcripts = []
    for video in videos:
        transcript = extract_transcript(video["id"])
        if transcript:
            all_transcripts.append(transcript)
        if len(all_transcripts) >= 2:
            break

    if not all_transcripts:
        return None

    combined = "\n\n---\n\n".join(all_transcripts)
    return classify_stock_intel(symbol, combined)


# ── Public API (cached) ──────────────────────────────────────────────────────


def fetch_market_intel():
    """Mode A: Fetch daily market intel with file-based caching.

    Returns market intel dict or None.
    """
    cache_key = f"market_{datetime.now().strftime('%Y-%m-%d')}"
    cached = _get_cache(cache_key, max_age_hours=config.YT_CACHE_HOURS)
    if cached is not None:
        return cached

    result = _fetch_market_intel_uncached()
    if result is not None:
        _set_cache(cache_key, result)
    return result


def fetch_stock_intel(symbol):
    """Mode B: Fetch stock-specific intel with file-based caching.

    Returns stock intel dict or None.
    """
    cache_key = f"stock_{symbol}_{datetime.now().strftime('%Y-%m-%d')}"
    cached = _get_cache(cache_key, max_age_hours=config.YT_CACHE_HOURS)
    if cached is not None:
        return cached

    result = _fetch_stock_intel_uncached(symbol)
    if result is not None:
        _set_cache(cache_key, result)
    return result
