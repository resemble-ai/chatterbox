"""Telegram bot glue code."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import FSInputFile, KeyboardButton, Message, ReplyKeyboardMarkup

from app.AI.chat_service import ChatTurn, GeminiChatService
from app.audio.voice_service import VoiceSynthesizer
from app.db.storage import ChatDatabase


logger = logging.getLogger(__name__)


RESTART_BUTTON = "Restart"
ENABLE_MANUAL_BUTTON = "Enable manual"
DISABLE_MANUAL_BUTTON = "Auto mode"
ENABLE_VOICE_BUTTON = "Enable voice"
DISABLE_VOICE_BUTTON = "Text only"


def _build_keyboard(*, manual_mode: bool, voice_enabled: bool) -> ReplyKeyboardMarkup:
    manual_button = DISABLE_MANUAL_BUTTON if manual_mode else ENABLE_MANUAL_BUTTON
    voice_button = DISABLE_VOICE_BUTTON if voice_enabled else ENABLE_VOICE_BUTTON
    keyboard = ReplyKeyboardMarkup(
        resize_keyboard=True,
        keyboard=[
            [KeyboardButton(text=RESTART_BUTTON)],
            [KeyboardButton(text=manual_button)],
            [KeyboardButton(text=voice_button)],
        ],
    )
    return keyboard


def _split_message(text: str, max_len: int = 4000) -> List[str]:
    '''Split long Telegram replies into chunks respecting word boundaries when possible.'''
    if not text:
        return []

    chunks: List[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break

        split_idx = max(remaining.rfind('\n', 0, max_len), remaining.rfind(' ', 0, max_len))
        if split_idx <= 0:
            split_idx = max_len

        chunk = remaining[:split_idx].rstrip()
        if not chunk:
            chunk = remaining[:max_len]
            split_idx = max_len

        chunks.append(chunk)
        remaining = remaining[split_idx:].lstrip()

    return chunks


@dataclass
class BotConfig:
    token: str
    admin_group_id: int
    data_dir: Path
    db_path: Path
    model_dir: Optional[Path] = None
    model_name: str = "gemini-2.0-flash-exp"
    voice_device: str = "cpu"
    voice_language: str = "ru"


class TelegramBot:
    def __init__(self, config: BotConfig, api_key: str) -> None:
        self._config = config
        self._bot = Bot(token=config.token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
        self._dispatcher = Dispatcher()
        self._db = ChatDatabase(config.db_path)
        self._chat_service = GeminiChatService(
            api_key=api_key,
            data_dir=config.data_dir,
            model=config.model_name,
        )
        self._voice = VoiceSynthesizer(model_dir=config.model_dir, device=config.voice_device, language=config.voice_language)
        self._register_handlers()

    @property
    def dispatcher(self) -> Dispatcher:
        return self._dispatcher

    async def close(self) -> None:
        await self._bot.session.close()
        self._db.close()

    def _register_handlers(self) -> None:
        dp = self._dispatcher

        @dp.message(Command("start"))
        async def cmd_start(message: Message) -> None:
            chat_id = message.chat.id
            self._db.ensure_chat(chat_id, message.from_user.id if message.from_user else None)
            manual = self._db.is_manual_mode(chat_id)
            voice = self._db.is_voice_enabled(chat_id)
            keyboard = _build_keyboard(manual_mode=manual, voice_enabled=voice)
            await message.answer(
                "Привет! Я чат-бот, который отвечает на вопросы по материалам из папки данных."
                " Используй меню, чтобы переключаться между автоответом и ручным режимом.",
                reply_markup=keyboard,
            )

        @dp.message(F.text == RESTART_BUTTON)
        async def restart_dialog(message: Message) -> None:
            chat_id = message.chat.id
            self._db.ensure_chat(chat_id, message.from_user.id if message.from_user else None)
            self._db.clear_chat(chat_id)
            await message.answer("История очищена. Можем начать сначала!", reply_markup=self._current_keyboard(chat_id))

        @dp.message(F.text.in_({ENABLE_MANUAL_BUTTON, DISABLE_MANUAL_BUTTON}))
        async def toggle_manual(message: Message) -> None:
            chat_id = message.chat.id
            self._db.ensure_chat(chat_id, message.from_user.id if message.from_user else None)
            manual = self._db.is_manual_mode(chat_id)
            new_state = not manual
            self._db.set_manual_mode(chat_id, new_state)
            text = (
                "Ручной режим включён. Сообщения будут передаваться администраторам."
                if new_state
                else "Автоответ снова активен."
            )
            await message.answer(text, reply_markup=self._current_keyboard(chat_id))

        @dp.message(F.text.in_({ENABLE_VOICE_BUTTON, DISABLE_VOICE_BUTTON}))
        async def toggle_voice(message: Message) -> None:
            chat_id = message.chat.id
            self._db.ensure_chat(chat_id, message.from_user.id if message.from_user else None)
            voice_enabled = self._db.is_voice_enabled(chat_id)
            new_state = not voice_enabled
            self._db.set_voice_enabled(chat_id, new_state)
            await message.answer(
                "Голосовые ответы включены." if new_state else "Теперь отвечаю только текстом.",
                reply_markup=self._current_keyboard(chat_id),
            )

        @dp.message(F.chat.id == self._config.admin_group_id)
        async def handle_admin_reply(message: Message) -> None:
            if not message.reply_to_message:
                return
            mapping = self._db.resolve_manual_reply(message.reply_to_message.message_id)
            if not mapping:
                return
            chat_id, _, _ = mapping
            if message.text:
                await self._bot.send_message(chat_id, message.text)
                self._db.add_message(chat_id, "assistant", message.text)
            elif message.voice:
                await self._bot.send_voice(chat_id, message.voice.file_id)
                self._db.add_message(chat_id, "assistant", "[Голосовое сообщение]")
            else:
                await message.copy_to(chat_id)
                self._db.add_message(chat_id, "assistant", "[Ответ администратора]")
            self._db.remove_manual_record(message.reply_to_message.message_id)

        @dp.message()
        async def handle_message(message: Message) -> None:
            chat_id = message.chat.id
            user_id = message.from_user.id if message.from_user else None
            self._db.ensure_chat(chat_id, user_id)
            if not message.text:
                await message.answer("Пожалуйста, отправьте текстовое сообщение.")
                return
            if self._db.is_manual_mode(chat_id):
                forwarded = await message.forward(self._config.admin_group_id)
                self._db.register_manual_forward(
                    admin_message_id=forwarded.message_id,
                    chat_id=chat_id,
                    user_id=user_id or 0,
                    user_message_id=message.message_id,
                )
                self._db.add_message(chat_id, "user", message.text)
                await message.answer("Отправил сообщение администраторам. Они скоро ответят.")
                return

            history_records = self._db.get_history(chat_id)
            history_turns = [
                ChatTurn(role="user" if role == "user" else "model", content=content)
                for role, content in history_records
            ]
            loop = asyncio.get_running_loop()
            reply_text, documents = await loop.run_in_executor(
                None,
                lambda: self._chat_service.answer(history_turns, message.text),
            )
            self._db.add_message(chat_id, "user", message.text)
            self._db.add_message(chat_id, "assistant", reply_text)

            for chunk in _split_message(reply_text):
                if chunk:
                    await message.answer(chunk)

            if self._db.is_voice_enabled(chat_id):
                audio_path = Path("tmp_audio") / f"reply_{chat_id}_{message.message_id}.wav"
                generated_path = await loop.run_in_executor(
                    None,
                    lambda: self._voice.synthesize(reply_text, audio_path),
                )
                if generated_path:
                    await self._bot.send_voice(chat_id, FSInputFile(str(generated_path)))
                    try:
                        generated_path.unlink()
                    except OSError:
                        logger.debug("Failed to remove temporary voice file %s", generated_path)

    def _current_keyboard(self, chat_id: int) -> ReplyKeyboardMarkup:
        manual = self._db.is_manual_mode(chat_id)
        voice = self._db.is_voice_enabled(chat_id)
        return _build_keyboard(manual_mode=manual, voice_enabled=voice)

    async def start(self) -> None:
        await self._dispatcher.start_polling(self._bot)



