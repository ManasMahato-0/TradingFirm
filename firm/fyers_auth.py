"""Fyers login for the dashboard: the same flow as scripts/fyers_login.py, split into two steps so a
web page can drive it. The access token goes straight into .env and is never returned, printed,
logged or shown. DATA ONLY: this module cannot place orders.
"""
from __future__ import annotations

import hashlib
import secrets
import time
from datetime import date, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from firm.config import IST, ROOT

ENV_PATH = ROOT / ".env"


def _session(env: dict, state: str):
    from fyers_apiv3 import fyersModel

    return fyersModel.SessionModel(
        client_id=env["FYERS_CLIENT_ID"], secret_key=env["FYERS_SECRET_KEY"], redirect_uri=env["FYERS_REDIRECT_URI"],
        response_type="code", grant_type="authorization_code", state=state)


def _env(env_path: Path) -> dict:
    from dotenv import dotenv_values

    return dict(dotenv_values(env_path))


def begin(env_path: Path = ENV_PATH) -> tuple[str, str]:
    """(login_url, state). Keep `state` and pass it to finish(): it ties the pasted address to THIS attempt."""
    env = _env(env_path)
    missing = [k for k in ("FYERS_CLIENT_ID", "FYERS_SECRET_KEY", "FYERS_REDIRECT_URI") if not env.get(k)]
    if missing:
        raise ValueError(f"Missing in .env: {', '.join(missing)}")
    state = secrets.token_urlsafe(16)
    return _session(env, state).generate_authcode(), state


def finish(redirected_url: str, expected_state: str | None, env_path: Path = ENV_PATH) -> tuple[bool, str]:
    """Exchange the address you were sent back to for a token and save it. Returns (ok, message);
    the message never contains the token or the one-time code."""
    if not expected_state:
        return False, "Start the login first (press 'Open Fyers login'), then paste the address you land on."
    params = parse_qs(urlparse(redirected_url.strip()).query)
    code = (params.get("auth_code") or [None])[0]
    if not code:
        return False, "That address has no login code. Copy the WHOLE address bar after logging in and paste it here."
    if (params.get("state") or [None])[0] != expected_state:
        return False, "That address is not from this login attempt. Press 'Open Fyers login' again and use the new one."
    env = _env(env_path)
    try:
        session = _session(env, expected_state)
        session.set_token(code)
        resp = session.generate_token()
    except Exception as e:
        return False, f"Fyers did not accept the login ({type(e).__name__}). Try again."
    token = resp.get("access_token") if isinstance(resp, dict) else None
    if not token:
        reason = str(resp.get("message", "no token returned")) if isinstance(resp, dict) else "no token returned"
        return False, f"Fyers said: {reason}"
    from dotenv import set_key

    set_key(str(env_path), "FYERS_ACCESS_TOKEN", token)
    save_refresh_token(resp, env_path)
    return True, "Logged in. The token is saved (it is not shown anywhere)."


# ------------------------------------------------------------ renewing the daily token without a browser
# Fyers hands out a refresh token (good for REFRESH_DAYS) with each browser login. With that token and the
# account PIN, a program can ask for a fresh daily access token. After REFRESH_DAYS a browser login is needed.

REFRESH_DAYS = 15
RENEW_URL = "https://api-t1.fyers.in/api/v3/validate-refresh-token"
MIN_GAP_SECONDS = 300
_last_attempt: float | None = None


def _today() -> date:
    return datetime.now(IST).date()


def save_refresh_token(resp: dict, env_path: Path = ENV_PATH) -> bool:
    """Keep the refresh token from a browser login. Returns False if the response had none."""
    token = resp.get("refresh_token") if isinstance(resp, dict) else None
    if not token:
        return False
    from dotenv import set_key

    set_key(str(env_path), "FYERS_REFRESH_TOKEN", token)
    set_key(str(env_path), "FYERS_REFRESH_SAVED", _today().isoformat())
    return True


def save_pin(pin: str, env_path: Path = ENV_PATH) -> tuple[bool, str]:
    pin = (pin or "").strip()
    if not (pin.isdigit() and len(pin) == 4):
        return False, "The Fyers PIN is 4 digits."
    from dotenv import set_key

    set_key(str(env_path), "FYERS_PIN", pin)
    return True, "PIN saved on this computer (in .env). It is never shown or sent anywhere except to Fyers."


def forget_pin(env_path: Path = ENV_PATH) -> None:
    from dotenv import unset_key

    unset_key(str(env_path), "FYERS_PIN")


def renew_status(env_path: Path = ENV_PATH, today: date | None = None) -> tuple[str, int | None]:
    """('on', days_left) | ('no_pin', None) | ('no_refresh', None) | ('expired', 0).
    no_refresh = no browser login has happened since this feature existed."""
    env = _env(env_path)
    if not env.get("FYERS_PIN"):
        return "no_pin", None
    if not env.get("FYERS_REFRESH_TOKEN"):
        return "no_refresh", None
    saved = env.get("FYERS_REFRESH_SAVED")
    try:
        left = REFRESH_DAYS - ((today or _today()) - date.fromisoformat(saved)).days
    except (TypeError, ValueError):
        return "on", None
    return ("on", left) if left > 0 else ("expired", 0)


def auto_renew(env_path: Path = ENV_PATH, post=None, force: bool = False) -> tuple[bool, str]:
    """Get a new daily access token using the refresh token and PIN. Messages never contain secrets.
    Tries at most once per MIN_GAP_SECONDS unless forced."""
    global _last_attempt
    state, _ = renew_status(env_path)
    if state != "on":
        return False, {"no_pin": "Auto-renew is off (no PIN saved).", "no_refresh": "Auto-renew needs one browser login first.",
                       "expired": "The 15-day refresh login has run out: log in once in the browser."}[state]
    if not force and _last_attempt is not None and time.monotonic() - _last_attempt < MIN_GAP_SECONDS:
        return False, "Renewal was tried a moment ago."
    _last_attempt = time.monotonic()
    env = _env(env_path)
    app_hash = hashlib.sha256(f"{env['FYERS_CLIENT_ID']}:{env['FYERS_SECRET_KEY']}".encode()).hexdigest()
    if post is None:
        import requests

        post = requests.post
    try:
        r = post(RENEW_URL, json={"grant_type": "refresh_token", "appIdHash": app_hash,
                                  "refresh_token": env["FYERS_REFRESH_TOKEN"], "pin": env["FYERS_PIN"]}, timeout=20)
        body = r.json()
    except Exception as e:
        return False, f"Could not reach Fyers to renew ({type(e).__name__})."
    token = body.get("access_token") if isinstance(body, dict) else None
    if not token:
        reason = str(body.get("message", "no token returned"))[:120] if isinstance(body, dict) else "no token returned"
        return False, f"Fyers refused the renewal: {reason}"
    from dotenv import set_key

    set_key(str(env_path), "FYERS_ACCESS_TOKEN", token)
    return True, "Renewed the Fyers login automatically."


def token_status() -> tuple[bool, str]:
    """Does Fyers answer with the saved token? If not, try to renew it quietly. Any failure means 'please log in'."""
    from firm.data.fyers import FyersData
    from firm.live.sandbox_live import probe_token

    return probe_token(FyersData, renew=auto_renew)
