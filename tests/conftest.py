import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _never_touch_the_real_fyers_login(monkeypatch):
    """No test may renew the real Fyers token or read the real .env PIN. Tests of auto_renew pass their own
    env file and fake `post`, and restore the real function themselves."""
    import firm.fyers_auth as fa

    monkeypatch.setattr(fa, "auto_renew", lambda *a, **k: (False, "disabled in tests"))
    monkeypatch.setattr(fa, "_last_attempt", None)
