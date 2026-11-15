"""Fyers login for the dashboard: the same flow as scripts/fyers_login.py, split into two steps so a
web page can drive it. The access token goes straight into .env and is never returned, printed,
logged or shown. DATA ONLY: this module cannot place orders.
"""
from __future__ import annotations

import secrets
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from firm.config import ROOT

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
    return True, "Logged in. The token is saved (it is not shown anywhere)."


def token_status() -> tuple[bool, str]:
    """Does Fyers answer with the saved token? Any failure means 'please log in'."""
    from firm.data.fyers import FyersData
    from firm.live.sandbox_live import probe_token

    return probe_token(FyersData)
