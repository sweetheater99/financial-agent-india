"""
Loads credentials from .env file.

SmartAPI credentials are read from environment variables.
Anthropic auth is resolved in this order:
  1. ANTHROPIC_API_KEY env var / .env  (explicit API key → uses SDK directly)
  2. Claude Code CLI  (Max subscription → uses `claude -p` under the hood)
"""

import json
import os
import shutil
import subprocess
import sys
from dotenv import load_dotenv

load_dotenv()

ANGELONE_API_KEY = os.getenv("ANGELONE_API_KEY")
ANGELONE_CLIENT_ID = os.getenv("ANGELONE_CLIENT_ID")
ANGELONE_PASSWORD = os.getenv("ANGELONE_PASSWORD")
ANGELONE_TOTP_SECRET = os.getenv("ANGELONE_TOTP_SECRET")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Centralized defaults — change here to update all modules at once
CLAUDE_MODEL = "claude-sonnet-4-20250514"
API_DELAY = 1.5  # seconds between SmartAPI calls to avoid rate-limiting

# Whether claude CLI is available (for Max subscription users)
CLAUDE_CLI_AVAILABLE = shutil.which("claude") is not None


def get_anthropic_client():
    """
    Create an Anthropic client using the best available auth method.
    Returns a real Anthropic client if API key is set, otherwise a
    ClaudeCLIClient wrapper that uses the `claude` CLI (Max subscription).
    """
    if ANTHROPIC_API_KEY:
        import anthropic
        return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    if CLAUDE_CLI_AVAILABLE:
        return ClaudeCLIClient()

    print("No Anthropic credentials found.")
    print("Either set ANTHROPIC_API_KEY in .env, or install/login to Claude Code (Max subscription).")
    sys.exit(1)


class ClaudeCLIClient:
    """
    Drop-in wrapper that calls the `claude` CLI instead of the Anthropic SDK.
    Uses your Max subscription auth — no API key needed.
    """

    def __init__(self):
        self.messages = self._Messages()

    class _Messages:
        def create(self, *, model=None, max_tokens=1024,
                   system=None, messages=None, **kwargs):
            # Build the user prompt from messages
            user_text = ""
            for msg in (messages or []):
                if msg["role"] == "user":
                    user_text += msg["content"] + "\n"

            cmd = ["claude", "-p", "--output-format", "json", "--no-session-persistence"]

            # Map model names to claude CLI model aliases
            model_name = model or CLAUDE_MODEL
            model_map = {
                "claude-sonnet-4-20250514": "sonnet",
                "claude-opus-4-20250514": "opus",
                "claude-haiku-3-5-20241022": "haiku",
            }
            cmd.extend(["--model", model_map.get(model_name, model_name)])

            if system:
                cmd.extend(["--system-prompt", system])

            cmd.append(user_text.strip())

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
            )

            if result.returncode != 0:
                raise RuntimeError(f"claude CLI failed: {result.stderr.strip()}")

            # Parse the JSON output from claude CLI
            try:
                output = json.loads(result.stdout)
                text = output.get("result", result.stdout)
            except json.JSONDecodeError:
                text = result.stdout.strip()

            return _CLIResponse(text)


class _CLIResponse:
    """Mimics anthropic.types.Message enough for agent.py to work."""

    def __init__(self, text: str):
        self.content = [_TextBlock(text)]


class _TextBlock:
    def __init__(self, text: str):
        self.text = text


def validate() -> bool:
    """Check that all required credentials are present. Prints what's missing."""
    missing = []
    if not ANGELONE_API_KEY:
        missing.append("ANGELONE_API_KEY")
    if not ANGELONE_CLIENT_ID:
        missing.append("ANGELONE_CLIENT_ID")
    if not ANGELONE_PASSWORD:
        missing.append("ANGELONE_PASSWORD")
    if not ANGELONE_TOTP_SECRET:
        missing.append("ANGELONE_TOTP_SECRET")
    if not ANTHROPIC_API_KEY and not CLAUDE_CLI_AVAILABLE:
        missing.append("ANTHROPIC_API_KEY (or Claude Code CLI)")

    if missing:
        print("Missing credentials:")
        for var in missing:
            print(f"  - {var}")
        print("\nFor AngelOne: copy .env.example to .env and fill in your values.")
        print("For Anthropic: set ANTHROPIC_API_KEY in .env, or just use Claude Code (Max).")
        return False
    return True


def is_market_open(_now=None) -> tuple[bool, str]:
    """Check if NSE market is currently open.

    NSE trading hours: Mon-Fri 9:15 AM – 3:30 PM IST (UTC+5:30).
    Returns (is_open, status_message).
    """
    from datetime import datetime, timezone, timedelta

    ist = timezone(timedelta(hours=5, minutes=30))
    now = _now if _now is not None else datetime.now(ist)

    # Weekend check
    if now.weekday() >= 5:  # 5=Saturday, 6=Sunday
        day_name = "Saturday" if now.weekday() == 5 else "Sunday"
        return (False, f"Market closed ({day_name})")

    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

    if now < market_open:
        return (False, f"Market closed (opens at 9:15 AM IST, current time {now.strftime('%H:%M')} IST)")
    if now >= market_close:
        return (False, f"Market closed (after 3:30 PM IST, current time {now.strftime('%H:%M')} IST)")

    return (True, "Market is open")


if __name__ == "__main__":
    if validate():
        auth_method = "API key" if ANTHROPIC_API_KEY else "Claude Code CLI (Max subscription)"
        print(f"All credentials loaded. Anthropic auth: {auth_method}")
    else:
        sys.exit(1)
