"""Daily Fyers login: gets an access token and saves it to .env.

Run once per trading day (tokens expire daily):

    .venv\\Scripts\\python scripts\\fyers_login.py

1. Your browser opens the Fyers login page. Log in normally.
2. Fyers redirects to your redirect URL (the page may fail to load; that is fine).
3. Copy the full URL from the address bar and paste it here.

Keep that URL private: it contains a one-time login code.
"""
from __future__ import annotations

import secrets
import sys
import webbrowser
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ENV_PATH = ROOT / ".env"


def main() -> int:
    from dotenv import dotenv_values, set_key
    from fyers_apiv3 import fyersModel

    env = dotenv_values(ENV_PATH)
    missing = [k for k in ("FYERS_CLIENT_ID", "FYERS_SECRET_KEY", "FYERS_REDIRECT_URI") if not env.get(k)]
    if missing:
        print(f"Missing in .env: {', '.join(missing)}")
        return 1

    state = secrets.token_urlsafe(16)
    session = fyersModel.SessionModel(
        client_id=env["FYERS_CLIENT_ID"],
        secret_key=env["FYERS_SECRET_KEY"],
        redirect_uri=env["FYERS_REDIRECT_URI"],
        response_type="code",
        grant_type="authorization_code",
        state=state,
    )
    login_url = session.generate_authcode()
    print("Opening Fyers login in your browser...")
    print("If it does not open, visit this URL:\n", login_url, "\n")
    webbrowser.open(login_url)

    redirected = input("After logging in, paste the full URL you were redirected to:\n> ").strip()
    params = parse_qs(urlparse(redirected).query)
    auth_code = (params.get("auth_code") or [None])[0]
    if not auth_code:
        print("No auth_code found in that URL. Copy the whole address bar and try again.")
        return 1
    if (params.get("state") or [None])[0] != state:
        print("State mismatch: that URL is not from this login attempt. Run the script again.")
        return 1

    session.set_token(auth_code)
    resp = session.generate_token()
    token = resp.get("access_token") if isinstance(resp, dict) else None
    if not token:
        print(f"Token request failed: {resp.get('message', resp) if isinstance(resp, dict) else resp}")
        return 1

    set_key(str(ENV_PATH), "FYERS_ACCESS_TOKEN", token)
    print("Access token saved to .env.")
    from firm.fyers_auth import save_refresh_token

    if save_refresh_token(resp):
        print("Refresh token saved too (add FYERS_PIN from the dashboard to turn on daily auto-renew).")

    # Quick check that data actually flows.
    from firm.data.fyers import FyersData

    df = FyersData(env["FYERS_CLIENT_ID"], token).history(
        "RELIANCE", "1d", date.today() - timedelta(days=10), date.today()
    )
    print(f"Test fetch OK: {len(df)} daily bars for RELIANCE, last close {df['close'].iloc[-1]:.2f}"
          if len(df) else "Login worked but test fetch returned no data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
