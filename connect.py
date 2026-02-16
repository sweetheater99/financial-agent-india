"""
Authenticates with AngelOne SmartAPI and returns a session.

Usage:
    # As a module
    from connect import get_session
    smart_api = get_session()

    # As a script (to test your credentials)
    python connect.py
"""

import logging
import sys
import pyotp
from SmartApi import SmartConnect
import config

logger = logging.getLogger("paper_trade")


def refresh_session(smart_api: SmartConnect) -> None:
    """Re-authenticate an existing SmartConnect session in-place using fresh TOTP."""
    totp = pyotp.TOTP(config.ANGELONE_TOTP_SECRET)
    totp_code = totp.now()

    session = smart_api.generateSession(
        clientCode=config.ANGELONE_CLIENT_ID,
        password=config.ANGELONE_PASSWORD,
        totp=totp_code,
    )

    if session.get("status") is False:
        raise RuntimeError(f"Session refresh failed: {session.get('message')}")

    logger.info("Session refreshed for %s", config.ANGELONE_CLIENT_ID)


def get_session() -> SmartConnect:
    """
    Authenticate with AngelOne SmartAPI using credentials from .env.
    Returns an authenticated SmartConnect object ready to make API calls.
    """
    if not config.validate():
        sys.exit(1)

    # Generate the current TOTP code from the secret
    print("Generating TOTP...")
    try:
        totp = pyotp.TOTP(config.ANGELONE_TOTP_SECRET)
        totp_code = totp.now()
    except Exception as e:
        print(f"Failed to generate TOTP. Is your TOTP_SECRET correct?")
        print(f"  Error: {e}")
        print(f"  The secret should be a base32 string (letters A-Z, digits 2-7)")
        sys.exit(1)

    # Create SmartAPI connection
    smart_api = SmartConnect(api_key=config.ANGELONE_API_KEY)

    # Authenticate
    print("Authenticating with AngelOne SmartAPI...")
    try:
        session = smart_api.generateSession(
            clientCode=config.ANGELONE_CLIENT_ID,
            password=config.ANGELONE_PASSWORD,
            totp=totp_code,
        )
    except Exception as e:
        print(f"Authentication failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your ANGELONE_CLIENT_ID (format: A12345678)")
        print("  2. Check your ANGELONE_PASSWORD (your trading PIN)")
        print("  3. Make sure TOTP is enabled in the AngelOne app")
        print("  4. Check your API key at smartapi.angelbroking.com")
        sys.exit(1)

    if session.get("status") is False:
        msg = session.get("message", "Unknown error")
        print(f"Login failed: {msg}")
        if "Invalid" in msg:
            print("  Double-check your client ID and password in .env")
        sys.exit(1)

    # Pull tokens from the session
    auth_token = session["data"]["jwtToken"]
    feed_token = smart_api.getfeedToken()

    print("Login successful!")
    print(f"Session token: {auth_token[:20]}...")
    print(f"Feed token: {feed_token}")
    print(f"Connected as client: {config.ANGELONE_CLIENT_ID}")

    return smart_api


if __name__ == "__main__":
    smart_api = get_session()
    print("\nConnection test passed. You're good to go.")
