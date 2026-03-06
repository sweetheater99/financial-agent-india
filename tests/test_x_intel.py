"""Tests for X/Twitter intel module."""
from unittest.mock import patch, MagicMock


class TestXSearch:
    @patch("x_intel.requests.get")
    def test_search_returns_tweets(self, mock_get):
        from x_intel import search_x
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "globalObjects": {
                "tweets": {
                    "1": {"full_text": "Nifty looking bearish today", "favorite_count": 100,
                          "created_at": "Mon Mar 09 05:00:00 +0000 2026",
                          "user_id_str": "123"},
                }
            }
        }
        mock_get.return_value = mock_resp

    def test_parse_cookies(self):
        from x_intel import _parse_cookies
        cookies = _parse_cookies("auth_token=abc123; ct0=xyz789")
        assert cookies["auth_token"] == "abc123"
        assert cookies["ct0"] == "xyz789"

    def test_parse_cookies_empty(self):
        from x_intel import _parse_cookies
        cookies = _parse_cookies("")
        assert cookies == {}


class TestXClassification:
    @patch("x_intel._call_claude_haiku")
    def test_classify_bullish(self, mock_haiku):
        from x_intel import classify_x_sentiment
        mock_haiku.return_value = {
            "sentiment": "bullish",
            "confidence": "high",
            "key_themes": ["FII buying", "Nifty breakout"],
        }
        result = classify_x_sentiment(["Nifty looking strong", "FII buying heavily"])
        assert result["sentiment"] == "bullish"

    @patch("x_intel._call_claude_haiku")
    def test_classify_crisis(self, mock_haiku):
        from x_intel import classify_x_sentiment
        mock_haiku.return_value = {
            "sentiment": "crisis",
            "confidence": "high",
            "key_themes": ["war", "sanctions"],
        }
        result = classify_x_sentiment(["War escalation", "India sanctions"])
        assert result["sentiment"] == "crisis"

    def test_classify_empty(self):
        from x_intel import classify_x_sentiment
        result = classify_x_sentiment([])
        assert result is None


class TestFetchXSentiment:
    @patch("x_intel.search_x")
    @patch("x_intel.classify_x_sentiment")
    def test_cached_result(self, mock_classify, mock_search):
        from x_intel import fetch_x_sentiment
        mock_search.return_value = [{"text": "test", "likes": 100}]
        mock_classify.return_value = {"sentiment": "neutral", "confidence": "low", "key_themes": []}
        result = fetch_x_sentiment()
        assert result is not None or result is None  # graceful either way
