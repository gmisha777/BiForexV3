"""Find your Telegram chat_id, save it to config/.env and send a test message.

Before running: create a bot with @BotFather, put TELEGRAM_BOT_TOKEN=... into config/.env,
open your bot in Telegram and press Start (or send it any message).

    python scripts/telegram_setup.py
"""

from __future__ import annotations

import os
import sys

import requests

from biforex.config import PROJECT_ROOT, load_settings
from biforex.notify import Telegram

ENV = PROJECT_ROOT / "config" / ".env"


def save_env(key: str, value: str) -> None:
    lines = ENV.read_text(encoding="utf-8").splitlines() if ENV.exists() else []
    lines = [ln for ln in lines if not ln.startswith(f"{key}=")]
    lines.append(f"{key}={value}")
    ENV.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    load_settings()  # loads config/.env
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        sys.exit("TELEGRAM_BOT_TOKEN is missing in config/.env (get it from @BotFather)")
    me = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10).json()
    if not me.get("ok"):
        sys.exit(f"token rejected by Telegram: {me.get('description')}")
    print(f"bot: @{me['result']['username']}")

    updates = requests.get(f"https://api.telegram.org/bot{token}/getUpdates",
                           timeout=10).json().get("result", [])
    chats = {}
    for u in updates:
        msg = u.get("message") or u.get("my_chat_member") or {}
        chat = msg.get("chat")
        if chat and chat.get("type") == "private":
            chats[str(chat["id"])] = chat.get("username") or chat.get("first_name", "")
    if not chats:
        sys.exit(f"no messages yet: open https://t.me/{me['result']['username']}, press Start, "
                 "then run this script again")
    if len(chats) > 1:
        print("several private chats wrote to the bot:", chats)
        sys.exit("keep only yours: set TELEGRAM_CHAT_ID in config/.env by hand")

    chat_id, name = next(iter(chats.items()))
    save_env("TELEGRAM_CHAT_ID", chat_id)
    print(f"chat {chat_id} ({name}) saved to {ENV}")
    Telegram(token, chat_id).send("✅ BiForexV3: Telegram connected. You will receive closed "
                                  "trades, errors and trading halts here.")
    print("test message sent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
