"""
Kite Connect OAuth login helper.

Run daily before market hours to get a fresh access token.
Opens browser for Zerodha login, catches redirect, saves token.

Usage:
    python kite_auth.py              # opens browser
    python kite_auth.py --token XYZ  # manual token paste
"""

import argparse
import json
import sys
import webbrowser
import threading
from flask import Flask, request

from kiteconnect import KiteConnect
import config
from kite_data import save_access_token

app = Flask(__name__)
_request_token = None
_server_done = threading.Event()


@app.route("/")
def callback():
    """Catch the OAuth redirect and extract request_token."""
    global _request_token
    _request_token = request.args.get("request_token")
    status = request.args.get("status")

    if status == "success" and _request_token:
        _server_done.set()
        return "<h2>Login successful! You can close this tab.</h2>"
    else:
        return f"<h2>Login failed: {request.args}</h2>", 400


def login_with_browser():
    """Open browser for Zerodha login, wait for redirect."""
    global _request_token

    kite = KiteConnect(api_key=config.KITE_API_KEY)
    login_url = kite.login_url()

    print(f"Opening browser for Zerodha login...")
    print(f"URL: {login_url}")

    # Start Flask server to catch redirect
    server = threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=5555, use_reloader=False),
        daemon=True,
    )
    server.start()

    webbrowser.open(login_url)

    print("Waiting for login redirect (timeout: 120s)...")
    if not _server_done.wait(timeout=120):
        print("Timeout waiting for login. Use --token flag to paste manually.")
        sys.exit(1)

    return exchange_token(kite, _request_token)


def exchange_token(kite: KiteConnect, request_token: str) -> str:
    """Exchange request_token for access_token."""
    print(f"Exchanging request token...")
    data = kite.generate_session(request_token, api_secret=config.KITE_API_SECRET)
    access_token = data["access_token"]

    save_access_token(access_token)
    print(f"Access token saved. Valid until tomorrow.")
    return access_token


def login_with_manual_token(request_token: str):
    """Exchange a manually pasted request_token."""
    kite = KiteConnect(api_key=config.KITE_API_KEY)
    return exchange_token(kite, request_token)


def main():
    parser = argparse.ArgumentParser(description="Kite Connect daily login")
    parser.add_argument("--token", help="Manually paste request_token from redirect URL")
    args = parser.parse_args()

    if args.token:
        login_with_manual_token(args.token)
    else:
        login_with_browser()

    # Verify token works
    kite = KiteConnect(api_key=config.KITE_API_KEY)
    kite.set_access_token(
        json.loads(open(config.KITE_ACCESS_TOKEN_FILE).read())["access_token"]
    )
    profile = kite.profile()
    print(f"Logged in as: {profile['user_name']} ({profile['user_id']})")


if __name__ == "__main__":
    main()
