"""High level conversational service built on top of Gemini API."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from google import genai

from .knowledge_base import DocumentChunk, LocalKnowledgeBase


logger = logging.getLogger(__name__)


@dataclass
class ChatTurn:
    role: str
    content: str


def _format_documents(documents: Sequence[DocumentChunk]) -> str:
    parts: List[str] = []
    for idx, doc in enumerate(documents, start=1):
        parts.append(f"[Источник {idx}: {doc.source}]\n{doc.text}")
    return "\n\n".join(parts)


class GeminiChatService:
    """Conversational interface around Gemini with document retrieval."""

    def __init__(
        self,
        *,
        api_key: str,
        data_dir: Path,
        system_prompt: Optional[str] = None,
        model: str = "gemini-2.0-flash-exp",
        history_limit: int = 12,
    ) -> None:
        if not api_key:
            raise ValueError("Gemini API key must be provided")
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._kb = LocalKnowledgeBase(data_dir)
        self._system_prompt = system_prompt or (
            "Ты помощник, отвечающий на вопросы об ИТМО."
            " Используй предоставленные фрагменты и предыдущий диалог."
            " Если не нашёл точного ответа, честно сообщи об этом."
        )
        self._history_limit = max(2, history_limit)

    def build_prompt(
        self,
        history: Sequence[ChatTurn],
        user_message: str,
        documents: Sequence[DocumentChunk],
    ) -> List[dict]:
        payload: List[dict] = []
        payload.append(
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "[SYSTEM]\n"
                            f"{self._system_prompt}\n\n"
                            "Сначала изучи контекст, затем ответь на вопрос"
                            " коротко и понятно."
                        )
                    }
                ],
            }
        )

        for turn in history[-self._history_limit :]:
            payload.append({"role": turn.role, "parts": [{"text": turn.content}]})

        if documents:
            payload.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "[КОНТЕКСТ]\n"
                                f"{_format_documents(documents)}\n\n"
                                "Используй только факты из контекста."
                            )
                        }
                    ],
                }
            )

        payload.append(
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "[ВОПРОС]\n"
                            f"{user_message}\n\n"
                            "Формат ответа: вежливое пояснение на русском языке"
                            " с отсылками к источникам вида [Источник N]."
                        )
                    }
                ],
            }
        )
        return payload

    def _generate(self, payload: List[dict]) -> str:
        response = self._client.models.generate_content(model=self._model, contents=payload)
        text = getattr(response, "text", None)
        if not text:
            text = "Извини, не удалось получить ответ от модели."
        return text.strip()

    def answer(
        self,
        history: Sequence[ChatTurn],
        user_message: str,
        *,
        top_k: int = 4,
    ) -> Tuple[str, List[DocumentChunk]]:
        documents = self._kb.search(user_message, top_k=top_k)
        payload = self.build_prompt(history, user_message, documents)
        reply = self._generate(payload)
        return reply, list(documents)
