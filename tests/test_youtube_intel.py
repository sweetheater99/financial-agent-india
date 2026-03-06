"""Tests for youtube_intel.py — YouTube intelligence module."""

import sys
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_search_youtube_returns_list():
    from youtube_intel import search_youtube_videos

    with patch("youtube_intel._run_yt_dlp_search") as mock:
        mock.return_value = [
            {
                "id": "abc123",
                "title": "Nifty Analysis Today",
                "upload_date": "20260306",
                "channel": "PR Sundar",
                "url": "https://youtube.com/watch?v=abc123",
            },
        ]
        results = search_youtube_videos("Nifty analysis today", max_results=3)
        assert len(results) == 1
        assert results[0]["id"] == "abc123"


def test_search_youtube_filters_old_videos():
    from youtube_intel import search_youtube_videos

    with patch("youtube_intel._run_yt_dlp_search") as mock:
        mock.return_value = [
            {
                "id": "old1",
                "title": "Old Video",
                "upload_date": "20250101",
                "channel": "Someone",
                "url": "https://youtube.com/watch?v=old1",
            },
            {
                "id": "new1",
                "title": "New Video",
                "upload_date": datetime.now().strftime("%Y%m%d"),
                "channel": "PR Sundar",
                "url": "https://youtube.com/watch?v=new1",
            },
        ]
        results = search_youtube_videos("Nifty analysis", max_results=3, max_age_days=1)
        assert len(results) == 1
        assert results[0]["id"] == "new1"


def test_search_youtube_empty_on_failure():
    from youtube_intel import search_youtube_videos

    with patch("youtube_intel._run_yt_dlp_search") as mock:
        mock.return_value = []
        results = search_youtube_videos("nonexistent query")
        assert results == []


def test_extract_transcript_returns_text():
    from youtube_intel import extract_transcript

    with patch("youtube_intel.YouTubeTranscriptApi") as mock_api:
        mock_transcript = MagicMock()
        mock_transcript.fetch.return_value = [
            {"text": "Market is looking bullish today."},
            {"text": "Nifty support at 22000."},
        ]
        mock_list = MagicMock()
        mock_list.find_manually_created_transcript.return_value = mock_transcript
        mock_api.list_transcripts.return_value = mock_list

        text = extract_transcript("abc123")
        assert "bullish" in text.lower()
        assert "22000" in text


def test_extract_transcript_returns_none_on_failure():
    from youtube_intel import extract_transcript

    with patch("youtube_intel.YouTubeTranscriptApi") as mock_api:
        mock_api.list_transcripts.side_effect = Exception("No transcripts")
        result = extract_transcript("bad_id")
        assert result is None


def test_classify_market_intel_structure():
    from youtube_intel import classify_market_intel

    with patch("youtube_intel._call_claude_haiku") as mock:
        mock.return_value = {
            "market_bias": "bullish",
            "key_levels": {"nifty_support": 22000, "nifty_resistance": 22500},
            "sectors_bullish": ["Banking"],
            "sectors_bearish": ["Pharma"],
            "events_today": [],
            "confidence": "medium",
            "summary": "Market looking positive",
        }
        result = classify_market_intel("Some transcript text about the market")
        assert result["market_bias"] in ("bullish", "bearish", "sideways")
        assert "nifty_support" in result["key_levels"]
        assert "confidence" in result


def test_classify_stock_intel_structure():
    from youtube_intel import classify_stock_intel

    with patch("youtube_intel._call_claude_haiku") as mock:
        mock.return_value = {
            "sentiment": "bullish",
            "key_levels": {"support": 1250, "resistance": 1350},
            "red_flags": [],
            "catalyst": "sector rotation",
            "confidence": "high",
        }
        result = classify_stock_intel("RELIANCE", "Some transcript about Reliance")
        assert result["sentiment"] in ("bullish", "bearish", "neutral")
        assert "support" in result["key_levels"]


def test_classify_market_intel_returns_none_on_failure():
    from youtube_intel import classify_market_intel

    with patch("youtube_intel._call_claude_haiku") as mock:
        mock.side_effect = Exception("API error")
        result = classify_market_intel("Some text")
        assert result is None


def test_classify_stock_intel_returns_none_on_failure():
    from youtube_intel import classify_stock_intel

    with patch("youtube_intel._call_claude_haiku") as mock:
        mock.side_effect = Exception("API error")
        result = classify_stock_intel("RELIANCE", "Some text")
        assert result is None


def test_cache_set_and_get(tmp_path):
    from youtube_intel import _get_cache, _set_cache
    import youtube_intel

    original = youtube_intel.CACHE_DIR
    youtube_intel.CACHE_DIR = tmp_path

    try:
        _set_cache("test_key", {"market_bias": "bullish"})
        result = _get_cache("test_key", max_age_hours=12)
        assert result is not None
        assert result["market_bias"] == "bullish"
    finally:
        youtube_intel.CACHE_DIR = original


def test_cache_expires(tmp_path):
    from youtube_intel import _get_cache
    import youtube_intel

    original = youtube_intel.CACHE_DIR
    youtube_intel.CACHE_DIR = tmp_path

    try:
        cache_file = tmp_path / "old_key.json"
        cache_file.write_text(json.dumps({
            "_cached_at": (datetime.now() - timedelta(hours=24)).isoformat(),
            "result": {"market_bias": "old"},
        }))
        result = _get_cache("old_key", max_age_hours=12)
        assert result is None
    finally:
        youtube_intel.CACHE_DIR = original


def test_fetch_market_intel_uses_cache(tmp_path):
    from youtube_intel import fetch_market_intel
    import youtube_intel

    original = youtube_intel.CACHE_DIR
    youtube_intel.CACHE_DIR = tmp_path

    try:
        with patch("youtube_intel._fetch_market_intel_uncached") as mock:
            mock.return_value = {"market_bias": "bullish", "confidence": "medium"}
            r1 = fetch_market_intel()
            r2 = fetch_market_intel()
            assert mock.call_count == 1
            assert r1 == r2
    finally:
        youtube_intel.CACHE_DIR = original


def test_fetch_market_intel_returns_none_on_failure(tmp_path):
    from youtube_intel import fetch_market_intel
    import youtube_intel

    original = youtube_intel.CACHE_DIR
    youtube_intel.CACHE_DIR = tmp_path

    try:
        with patch("youtube_intel._fetch_market_intel_uncached") as mock:
            mock.return_value = None
            result = fetch_market_intel()
            assert result is None
    finally:
        youtube_intel.CACHE_DIR = original


def test_fetch_stock_intel_uses_cache(tmp_path):
    from youtube_intel import fetch_stock_intel
    import youtube_intel

    original = youtube_intel.CACHE_DIR
    youtube_intel.CACHE_DIR = tmp_path

    try:
        with patch("youtube_intel._fetch_stock_intel_uncached") as mock:
            mock.return_value = {"sentiment": "bullish", "confidence": "high"}
            r1 = fetch_stock_intel("RELIANCE")
            r2 = fetch_stock_intel("RELIANCE")
            assert mock.call_count == 1
            assert r1 == r2
    finally:
        youtube_intel.CACHE_DIR = original


def test_fetch_stock_intel_returns_none_on_failure(tmp_path):
    from youtube_intel import fetch_stock_intel
    import youtube_intel

    original = youtube_intel.CACHE_DIR
    youtube_intel.CACHE_DIR = tmp_path

    try:
        with patch("youtube_intel._fetch_stock_intel_uncached") as mock:
            mock.return_value = None
            result = fetch_stock_intel("RELIANCE")
            assert result is None
    finally:
        youtube_intel.CACHE_DIR = original


def test_fetch_market_intel_uncached_orchestration():
    """Test that _fetch_market_intel_uncached correctly orchestrates search -> transcript -> classify."""
    from youtube_intel import _fetch_market_intel_uncached

    with patch("youtube_intel.search_youtube_videos") as mock_search, \
         patch("youtube_intel.extract_transcript") as mock_transcript, \
         patch("youtube_intel.classify_market_intel") as mock_classify:

        mock_search.return_value = [
            {"id": "v1", "title": "Nifty Analysis", "upload_date": "20260306",
             "channel": "PR Sundar", "url": "https://youtube.com/watch?v=v1"},
        ]
        mock_transcript.return_value = "Market is bullish today. Nifty support at 22000."
        mock_classify.return_value = {
            "market_bias": "bullish",
            "key_levels": {"nifty_support": 22000, "nifty_resistance": 22500},
            "sectors_bullish": ["Banking"],
            "sectors_bearish": [],
            "events_today": [],
            "confidence": "medium",
            "summary": "Market bullish",
        }

        result = _fetch_market_intel_uncached()
        assert result is not None
        assert result["market_bias"] == "bullish"
        mock_search.assert_called()
        mock_transcript.assert_called_once_with("v1")
        mock_classify.assert_called_once()


def test_fetch_stock_intel_uncached_orchestration():
    """Test that _fetch_stock_intel_uncached correctly orchestrates search -> transcript -> classify."""
    from youtube_intel import _fetch_stock_intel_uncached

    with patch("youtube_intel.search_youtube_videos") as mock_search, \
         patch("youtube_intel.extract_transcript") as mock_transcript, \
         patch("youtube_intel.classify_stock_intel") as mock_classify:

        mock_search.return_value = [
            {"id": "v2", "title": "RELIANCE Stock Analysis", "upload_date": "20260306",
             "channel": "CA Rachana Ranade", "url": "https://youtube.com/watch?v=v2"},
        ]
        mock_transcript.return_value = "Reliance looking strong with support at 1250."
        mock_classify.return_value = {
            "sentiment": "bullish",
            "key_levels": {"support": 1250, "resistance": 1350},
            "red_flags": [],
            "catalyst": "sector rotation",
            "confidence": "high",
        }

        result = _fetch_stock_intel_uncached("RELIANCE")
        assert result is not None
        assert result["sentiment"] == "bullish"
        mock_search.assert_called_once()
        # Should search with symbol in query
        call_args = mock_search.call_args
        assert "RELIANCE" in call_args[0][0]


def test_fetch_market_intel_uncached_no_transcripts():
    """Returns None when no transcripts are found."""
    from youtube_intel import _fetch_market_intel_uncached

    with patch("youtube_intel.search_youtube_videos") as mock_search, \
         patch("youtube_intel.extract_transcript") as mock_transcript:

        mock_search.return_value = [
            {"id": "v1", "title": "Video", "upload_date": "20260306",
             "channel": "Someone", "url": "url"},
        ]
        mock_transcript.return_value = None

        result = _fetch_market_intel_uncached()
        assert result is None
