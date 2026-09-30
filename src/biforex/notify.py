"""Telegram notifications. Never raises: a Telegram outage must not affect trading."""

from __future__ import annotations

import logging
import os
import time

import requests

log = logging.getLogger(__name__)


def net_error(exc: Exception) -> str:
    """Short description of a network error. Never the URL: it would contain the bot token."""
    text = str(exc)
    for marker in ("NameResolutionError", "timed out", "ConnectTimeout", "unreachable network",
                   "Connection refused", "Connection aborted", "RemoteDisconnected"):
        if marker in text:
            return f"{type(exc).__name__} ({marker})"
    return type(exc).__name__


class Notifier:
    """Does nothing; used when Telegram is not configured."""

    enabled = False

    def send(self, text: str, key: str | None = None, every_s: float = 0) -> None:
        pass

    def commands(self) -> list[str]:
        """New commands (e.g. "/status") sent by the owner since the last call."""
        return []


class Telegram(Notifier):
    enabled = True

    def __init__(self, token: str, chat_id: str, prefix: str = ""):
        self.api = f"https://api.telegram.org/bot{token}"
        self.url = f"{self.api}/sendMessage"
        self.chat_id = chat_id
        self.prefix = prefix
        self._last: dict[str, float] = {}
        self._offset: int | None = None

    def send(self, text: str, key: str | None = None, every_s: float = 0) -> None:
        """key + every_s: send a message with this key at most once per every_s seconds."""
        if key is not None:
            now = time.monotonic()
            if now - self._last.get(key, -1e18) < every_s:
                return
            self._last[key] = now
        try:
            resp = requests.post(self.url, timeout=10, json={
                "chat_id": self.chat_id, "text": f"{self.prefix}{text}",
                "disable_web_page_preview": True})
            if resp.status_code != 200:
                log.warning("telegram: HTTP %d %s", resp.status_code, resp.text[:200])
        except requests.RequestException as exc:
            log.warning("telegram unreachable: %s", net_error(exc))


    def commands(self) -> list[str]:
        """Messages starting with "/" from the owner's chat; everyone else is ignored."""
        params = {"timeout": 0, "allowed_updates": '["message"]'}
        if self._offset is not None:
            params["offset"] = self._offset
        try:
            data = requests.get(f"{self.api}/getUpdates", params=params, timeout=10).json()
        except (requests.RequestException, ValueError) as exc:
            log.warning("telegram unreachable: %s", net_error(exc))
            return []
        out = []
        for u in data.get("result", []):
            self._offset = u["update_id"] + 1
            msg = u.get("message") or {}
            text = (msg.get("text") or "").strip()
            if str(msg.get("chat", {}).get("id")) == str(self.chat_id) and text.startswith("/"):
                out.append(text.split()[0].split("@")[0].lower())
        return out


def from_env(prefix: str = "") -> Notifier:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if token and chat_id:
        return Telegram(token, chat_id, prefix)
    if token:
        log.warning("TELEGRAM_BOT_TOKEN set but TELEGRAM_CHAT_ID missing: "
                    "run scripts/telegram_setup.py")
    return Notifier()
