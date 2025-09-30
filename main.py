from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from chattractive.bot.bot import BotConfig, TelegramBot
from load_model import ensure_model_present, missing_required_files, resolve_model_dir


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

    model_dir_env = _get_env("AUDIO_MODEL_DIR", required=False)

    if model_dir_env:
        model_dir = Path(model_dir_env)
        missing_files = missing_required_files(model_dir)
        if missing_files:
            missing_str = ", ".join(missing_files)
            raise RuntimeError(
                f"AUDIO_MODEL_DIR points to {model_dir} but is missing required files: {missing_str}"
            )
    else:
        model_dir = resolve_model_dir()
        ensure_model_present(model_dir)

    voice_device = _get_env("VOICE_DEVICE", required=False) or "cpu"
    voice_language = _get_env("VOICE_LANGUAGE", required=False) or "ru"
    model_name = _get_env("GEMINI_MODEL", required=False) or "gemini-2.0-flash-exp"

    config = BotConfig(
        token=token,
        admin_group_id=admin_group_id,
        data_dir=data_dir,
        db_path=db_path,
        model_dir=model_dir,
        model_name=model_name,
        voice_device=voice_device,
        voice_language=voice_language,
    )

    bot = TelegramBot(config, api_key)
    try:
        await bot.start()
    finally:
        await bot.close()


if __name__ == "__main__":
    asyncio.run(main())

