"""Daily Fyers token renewal from the refresh token + PIN. Fake network, temp .env; secrets never leak."""
from __future__ import annotations

import hashlib
from datetime import date

import pytest

from firm import fyers_auth
from firm.live.sandbox_live import probe_token

BASE = "FYERS_CLIENT_ID=ABC-100\nFYERS_SECRET_KEY=shh\nFYERS_REDIRECT_URI=http://127.0.0.1/cb\nFYERS_ACCESS_TOKEN=OLD\n"


class Resp:
    def __init__(self, body):
        self.body = body

    def json(self):
        return self.body


def env_file(tmp_path, extra=""):
    p = tmp_path / ".env"
    p.write_text(BASE + extra)
    return p


def test_pin_must_be_four_digits_and_is_saved(tmp_path):
    p = env_file(tmp_path)
    assert not fyers_auth.save_pin("12a4", p)[0] and not fyers_auth.save_pin("123", p)[0]
    assert "FYERS_PIN" not in p.read_text()
    ok, msg = fyers_auth.save_pin("1234", p)
    assert ok and "1234" not in msg and "1234" in p.read_text()
    fyers_auth.forget_pin(p)
    assert "1234" not in p.read_text()


def test_a_browser_login_keeps_the_refresh_token(tmp_path):
    p = env_file(tmp_path)
    assert not fyers_auth.save_refresh_token({"access_token": "x"}, p)
    assert fyers_auth.save_refresh_token({"access_token": "x", "refresh_token": "REFRESH1"}, p)
    text = p.read_text()
    assert "REFRESH1" in text and "FYERS_REFRESH_SAVED" in text


def test_status_walks_through_the_setup_steps(tmp_path):
    p = env_file(tmp_path)
    assert fyers_auth.renew_status(p) == ("no_pin", None)
    fyers_auth.save_pin("1234", p)
    assert fyers_auth.renew_status(p) == ("no_refresh", None)
    fyers_auth.save_refresh_token({"refresh_token": "R"}, p)
    saved = date.fromisoformat(fyers_auth._env(p)["FYERS_REFRESH_SAVED"])
    assert fyers_auth.renew_status(p, saved) == ("on", 15)
    assert fyers_auth.renew_status(p, date.fromordinal(saved.toordinal() + 14)) == ("on", 1)
    assert fyers_auth.renew_status(p, date.fromordinal(saved.toordinal() + 15)) == ("expired", 0)


def _real_auto_renew():
    import importlib.util

    spec = importlib.util.spec_from_file_location("real_fyers_auth", fyers_auth.__file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_renewal_end_to_end_with_a_fake_fyers(tmp_path):
    real = _real_auto_renew()
    p = env_file(tmp_path, "FYERS_PIN=4321\nFYERS_REFRESH_TOKEN=REFRESH1\nFYERS_REFRESH_SAVED=" + date.today().isoformat() + "\n")
    sent = {}

    def post(url, json, timeout):
        sent.update(url=url, **json)
        return Resp({"s": "ok", "access_token": "NEWTOKEN-999"})

    ok, msg = real.auto_renew(p, post=post)
    assert ok and "NEWTOKEN" not in msg and "4321" not in msg
    assert sent["url"].endswith("/validate-refresh-token") and sent["grant_type"] == "refresh_token"
    assert sent["appIdHash"] == hashlib.sha256(b"ABC-100:shh").hexdigest()
    assert sent["pin"] == "4321" and sent["refresh_token"] == "REFRESH1"
    text = p.read_text()
    assert "NEWTOKEN-999" in text and "OLD" not in text and "REFRESH1" in text


def test_refused_or_unreachable_renewal_leaves_the_env_alone_and_leaks_nothing(tmp_path):
    real = _real_auto_renew()
    p = env_file(tmp_path, "FYERS_PIN=4321\nFYERS_REFRESH_TOKEN=REFRESH1\nFYERS_REFRESH_SAVED=" + date.today().isoformat() + "\n")
    before = p.read_text()
    ok, msg = real.auto_renew(p, post=lambda *a, **k: Resp({"s": "error", "message": "invalid pin"}), force=True)
    assert not ok and "invalid pin" in msg and "4321" not in msg
    def boom(*a, **k):
        raise ConnectionError("net down 4321 REFRESH1")
    ok, msg = real.auto_renew(p, post=boom, force=True)
    assert not ok and "4321" not in msg and "REFRESH1" not in msg
    assert p.read_text() == before


def test_renewal_is_not_attempted_without_pin_or_after_expiry(tmp_path):
    real = _real_auto_renew()
    calls = []
    post = lambda *a, **k: calls.append(1) or Resp({})
    assert not real.auto_renew(env_file(tmp_path), post=post)[0]
    old = env_file(tmp_path, "FYERS_PIN=1111\nFYERS_REFRESH_TOKEN=R\nFYERS_REFRESH_SAVED=2020-01-01\n")
    assert not real.auto_renew(old, post=post)[0]
    assert calls == []


def test_it_does_not_hammer_fyers(tmp_path):
    real = _real_auto_renew()
    p = env_file(tmp_path, "FYERS_PIN=4321\nFYERS_REFRESH_TOKEN=R\nFYERS_REFRESH_SAVED=" + date.today().isoformat() + "\n")
    calls = []
    post = lambda *a, **k: calls.append(1) or Resp({"s": "error", "message": "no"})
    real.auto_renew(p, post=post)
    real.auto_renew(p, post=post)
    assert len(calls) == 1


def test_probe_renews_once_then_succeeds():
    state = {"ok": False}

    class F:
        def nse_cash_status(self):
            if not state["ok"]:
                raise RuntimeError("Bad request")
            return "open"

    def renew():
        state["ok"] = True
        return True, "renewed"

    ok, msg = probe_token(F, renew=renew)
    assert ok and "open" in msg


def test_probe_reports_the_original_failure_when_renewal_fails():
    class F:
        def nse_cash_status(self):
            raise RuntimeError("Bad request")

    ok, msg = probe_token(F, renew=lambda: (False, "no pin"))
    assert not ok and "Bad request" in msg
    assert not probe_token(F)[0]
