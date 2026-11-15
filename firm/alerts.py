"""Tell the owner when something needs them, instead of waiting for them to look.

Two things for now: the daily Fyers login (a reminder at 08:45, again at 09:00 and 09:10 if still not
logged in) and the practice-trading program being down on a trading morning. Each goes out as a Windows
notification, and as a phone push too if NTFY_TOPIC is set in .env (ntfy.sh; the topic name acts like a
password, and only generic text is ever sent). Everything is injected, so tests never touch the real world.
"""
from __future__ import annotations

import os
import subprocess
from datetime import date, datetime, time as dtime
from typing import Callable

REMIND_AT = (dtime(8, 45), dtime(9, 0), dtime(9, 10))
SLOT_WINDOW_MIN = 15                     # a slot missed by more than this (laptop was asleep) is skipped, not sent late
SANDBOX_CHECK = (dtime(9, 5), dtime(9, 30))

_TOAST = r"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
$t = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent('ToastText02')
$n = $t.GetElementsByTagName('text')
$n.Item(0).AppendChild($t.CreateTextNode($env:TF_TITLE)) | Out-Null
$n.Item(1).AppendChild($t.CreateTextNode($env:TF_MSG)) | Out-Null
$app = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app).Show([Windows.UI.Notifications.ToastNotification]::new($t))
"""


def notify(title: str, message: str, run: Callable = subprocess.run, post: Callable | None = None,
           env: dict | None = None) -> list[str]:
    """Windows notification, plus ntfy if configured. Returns which channels were attempted. Never raises."""
    env = os.environ if env is None else env
    sent = []
    try:
        run(["powershell", "-NoProfile", "-NonInteractive", "-Command", _TOAST], timeout=20, capture_output=True,
            env={**os.environ, "TF_TITLE": title, "TF_MSG": message}, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        sent.append("windows")
    except Exception:
        pass
    topic = (env.get("NTFY_TOPIC") or "").strip()
    if topic:
        try:
            if post is None:
                import requests

                post = requests.post
            post(f"https://ntfy.sh/{topic}", data=message.encode("utf-8"), headers={"Title": title}, timeout=10)
            sent.append("phone")
        except Exception:
            pass
    return sent


class Watch:
    """Call check() every tick. It decides whether anything is worth saying, and says each thing once."""

    def __init__(self, probe: Callable[[], tuple[bool, str]], notifier: Callable[[str, str], object],
                 sandbox_running: Callable[[], bool], is_trading_day: Callable[[date], bool],
                 enabled: Callable[[], bool] = lambda: True, log: Callable[[str], None] = lambda m: None) -> None:
        self.probe, self.notifier, self.sandbox_running = probe, notifier, sandbox_running
        self.is_trading_day, self.enabled, self.log = is_trading_day, enabled, log
        self._sent: set[tuple[date, str]] = set()
        self._logged_in_on: date | None = None

    def _say(self, day: date, key: str, title: str, message: str) -> str:
        self._sent.add((day, key))
        self._sent = {s for s in self._sent if s[0] == day}
        self.notifier(title, message)
        self.log(f"{title}: {message}")
        return key

    def check(self, now: datetime) -> list[str]:
        """Returns the alerts sent on this call (usually none)."""
        day, t = now.date(), now.time()
        if not self.enabled() or not self.is_trading_day(day):
            return []
        out = []
        if self._logged_in_on != day:
            for slot in REMIND_AT:
                key = f"login-{slot:%H%M}"
                late = (now - datetime.combine(day, slot, tzinfo=now.tzinfo)).total_seconds() / 60
                if (day, key) in self._sent or not 0 <= late < SLOT_WINDOW_MIN:
                    continue
                ok, _ = self.probe()
                if ok:
                    self._logged_in_on = day
                    break
                out.append(self._say(day, key, "Fyers login needed",
                                     "Log in to Fyers now (dashboard, Home page) so today's practice trading can run."
                                     if slot < REMIND_AT[-1] else
                                     "Last reminder: no Fyers login yet and the market opens at 09:15."))
                break
        lo, hi = SANDBOX_CHECK
        if lo <= t <= hi and (day, "sandbox-down") not in self._sent and not self.sandbox_running():
            out.append(self._say(day, "sandbox-down", "Practice trading is not running",
                                 "Start it from the dashboard (Settings, Programs) before 09:15."))
        return out
