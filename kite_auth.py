"""
Kite Connect OAuth login helper — headless TOTP-based.

Run daily before market hours to get a fresh access token.
Uses requests + pyotp for fully automated login (no browser needed).

Usage:
    python kite_auth.py              # headless TOTP login
    python kite_auth.py --token XYZ  # manual request_token paste
"""

import argparse
import json
import sys

import pyotp
import requests
from kiteconnect import KiteConnect

import config
from kite_data import save_access_token


def login_headless() -> str:
    """Automated headless login using TOTP — no browser needed."""
    session = requests.Session()

    # Step 1: POST user_id + password to Kite login
    print("Logging in...")
    resp = session.post(
        "https://kite.zerodha.com/api/login",
        data={
            "user_id": config.KITE_USER_ID,
            "password": config.KITE_PASSWORD,
        },
    )
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "success":
        raise RuntimeError(f"Login failed: {data}")

    request_id = data["data"]["request_id"]

    # Step 2: Submit TOTP
    print("Submitting TOTP...")
    totp = pyotp.TOTP(config.KITE_TOTP_SECRET)
    resp = session.post(
        "https://kite.zerodha.com/api/twofa",
        data={
            "user_id": config.KITE_USER_ID,
            "request_id": request_id,
            "twofa_value": totp.now(),
            "twofa_type": "totp",
        },
    )
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "success":
        raise RuntimeError(f"TOTP failed: {data}")

    # Step 3: Follow redirect chain to get request_token
    print("Got request_token, exchanging for access_token...")
    kite = KiteConnect(api_key=config.KITE_API_KEY)

    # First redirect: login URL → /connect/finish
    resp = session.get(kite.login_url(), allow_redirects=False)
    redirect_url = resp.headers.get("Location", "")

    # Follow up to 5 redirects manually to find request_token
    for _ in range(5):
        if "request_token=" in redirect_url:
            break
        if resp.status_code not in (301, 302, 303, 307):
            break
        resp = session.get(redirect_url, allow_redirects=False)
        redirect_url = resp.headers.get("Location", redirect_url)

    if "request_token=" not in redirect_url:
        raise RuntimeError(
            f"No request_token in redirect chain. Last URL: {redirect_url[:200]}"
        )

    from urllib.parse import urlparse, parse_qs
    params = parse_qs(urlparse(redirect_url).query)
    request_token = params["request_token"][0]

    return exchange_token(kite, request_token)


def exchange_token(kite: KiteConnect, request_token: str) -> str:
    """Exchange request_token for access_token."""
    data = kite.generate_session(request_token, api_secret=config.KITE_API_SECRET)
    access_token = data["access_token"]
    save_access_token(access_token)
    print("Access token saved. Valid until tomorrow.")
    return access_token


def main():
    parser = argparse.ArgumentParser(description="Kite Connect daily login")
    parser.add_argument("--token", help="Manually paste request_token from redirect URL")
    args = parser.parse_args()

    if args.token:
        kite = KiteConnect(api_key=config.KITE_API_KEY)
        exchange_token(kite, args.token)
    else:
        login_headless()

    # Verify token works
    kite = KiteConnect(api_key=config.KITE_API_KEY)
    kite.set_access_token(
        json.loads(open(config.KITE_ACCESS_TOKEN_FILE).read())["access_token"]
    )
    profile = kite.profile()
    print(f"Logged in as: {profile['user_name']} ({profile['user_id']})")


if __name__ == "__main__":
    main()
