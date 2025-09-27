from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from app.bot.bot import BotConfig, TelegramBot


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _get_env(name: str, *, required: bool = True) -> str:
    value = os.getenv(name)
    if required and not value:
        raise RuntimeError(f"Environment variable {name} is required")
    return value or ""


async def main() -> None:
    load_dotenv()

    token = _get_env("TELEGRAM_BOT_TOKEN")
    api_key = _get_env("GOOGLE_API_KEY")
    admin_group_id_str = _get_env("ADMIN_GROUP_ID")
    try:
        admin_group_id = int(admin_group_id_str)
    except ValueError as exc:  # pragma: no cover
        raise RuntimeError("ADMIN_GROUP_ID must be integer") from exc

    data_dir = Path(_get_env("DATA_DIRECTORY", required=False) or "data")
    db_path = Path(_get_env("DATABASE_PATH", required=False) or "chattractive.db")
    model_dir_env = _get_env("AUDIO_MODEL_DIR", required=False) or None
    model_dir = Path(model_dir_env) if model_dir_env else None
    voice_device = _get_env("VOICE_DEVICE", required=False) or "cpu"
    model_name = _get_env("GEMINI_MODEL", required=False) or "gemini-2.0-flash-exp"

    config = BotConfig(
        token=token,
        admin_group_id=admin_group_id,
        data_dir=data_dir,
        db_path=db_path,
        model_dir=model_dir,
        model_name=model_name,
        voice_device=voice_device,
    )

    bot = TelegramBot(config, api_key)
    try:
        await bot.start()
    finally:
        await bot.close()


if __name__ == "__main__":
    asyncio.run(main())
