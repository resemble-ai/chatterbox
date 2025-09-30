"""High level conversational service built on top of Gemini API."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from google import genai
from google.genai.errors import ClientError
from requests.exceptions import RequestException

from .knowledge_base import DocumentChunk, LocalKnowledgeBase


logger = logging.getLogger(__name__)


@dataclass
class ChatTurn:
    role: str
    content: str


def _format_documents(documents: Sequence[DocumentChunk]) -> str:
    parts: List[str] = []
    for idx, doc in enumerate(documents, start=1):
        parts.append(f"Документ {idx} ({doc.source})\n{doc.text}")
    return "\n\n".join(parts)


def _retry_delay_seconds(details: Any) -> Optional[float]:
    """Best-effort extraction of retry delay (seconds) from Gemini error payload."""
    entries: Sequence[Any] = ()
    if isinstance(details, dict):
        error_payload = details.get("error")
        if isinstance(error_payload, dict):
            entries = error_payload.get("details") or ()
        else:
            entries = details.get("details") or ()
    elif isinstance(details, list):
        entries = details
    else:
        return None

    for entry in entries or ():
        if not isinstance(entry, dict):
            continue
        retry_info = entry.get("retryDelay")
        if retry_info is not None:
            if isinstance(retry_info, (int, float)) and retry_info > 0:
                return float(retry_info)
            if isinstance(retry_info, str):
                cleaned = retry_info.strip().lower()
                if cleaned.endswith('s'):
                    cleaned = cleaned[:-1]
                try:
                    value = float(cleaned)
                except ValueError:
                    pass
                else:
                    if value > 0:
                        return value
        seconds = entry.get("seconds")
        nanos = entry.get("nanos")
        if seconds is not None or nanos is not None:
            try:
                total = float(seconds or 0) + float(nanos or 0) / 1_000_000_000
            except (TypeError, ValueError):
                continue
            if total > 0:
                return total
    return None

def _quota_hint(details: Any) -> str:
    """Extract a human-friendly quota hint from Gemini error details."""
    if not isinstance(details, dict):
        return ""
    error_payload = details.get("error")
    if isinstance(error_payload, dict):
        entries = error_payload.get("details") or ()
    else:
        entries = details.get("details") or ()
    for entry in entries or ():
        if not isinstance(entry, dict):
            continue
        if entry.get("@type") != "type.googleapis.com/google.rpc.QuotaFailure":
            continue
        violations = entry.get("violations") or ()
        for violation in violations:
            if not isinstance(violation, dict):
                continue
            metric = violation.get("quotaMetric")
            limit = violation.get("quotaValue")
            if metric and limit:
                return f" Limit {metric} allows {limit} requests."
        break
    return ""

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
            "Ты — дружелюбный ассистент LISA INFO. Отвечай только на русском языке в мужском роде и помогай кратко и по делу. "
            "Если вопрос требует данных из базы знаний, используй факты из документов без перечисления источников. "
            "Когда точного ответа нет, честно сообщи об этом и предложи дальнейшие шаги. "
            "Держи каждый ответ короче 2000 символов, начинай с конкретных действий и избегай лишней воды."
        )
        self._history_limit = max(2, history_limit)

    def build_prompt(
        self,
        history: Sequence[ChatTurn],
        user_message: str,
        documents: Sequence[DocumentChunk],
    ) -> List[dict]:
        payload: List[dict] = []

        system_text = (
            "[SYSTEM]\n"
            f"{self._system_prompt}\n\n"
            "Всегда отвечай на русском языке. Если информации недостаточно, объясни это и предложи, что можно сделать дальше."
            "\nИспользуй для форматирование только эти символы, которые понимает Telegram: **жирный**, __курсив__, ~~зачёркнутый~~, `моно`, ```моно-блок``` и ничего сверх этого."
            "\nKeep replies focused on concrete actions, stay under 2000 characters, and avoid filler."
        )
        payload.append({"role": "user", "parts": [{"text": system_text}]})

        for turn in history[-self._history_limit :]:
            payload.append({"role": turn.role, "parts": [{"text": turn.content}]})

        if documents:
            docs_text = (
                "[DOCUMENTS]\n"
                f"{_format_documents(documents)}\n\n"
                "Используй эти выдержки по смыслу, но не перечисляй источники в ответе."
            )
            payload.append({"role": "user", "parts": [{"text": docs_text}]})

        user_text = (
            "[USER]\n"
            f"{user_message}\n\n"
            "Ответ сформируй по-русски, без явного перечисления источников."
        )
        payload.append({"role": "user", "parts": [{"text": user_text}]})

        return payload

    def _generate(self, payload: List[dict]) -> str:
        max_attempts = 3
        response = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model, contents=payload
                )
                break
            except ClientError as exc:
                if exc.code == 429:
                    retry_delay = _retry_delay_seconds(getattr(exc, "details", None))
                    if retry_delay and attempt < max_attempts:
                        logger.warning(
                            "Gemini quota reached, retrying in %.2fs (attempt %d/%d)",
                            retry_delay,
                            attempt,
                            max_attempts,
                        )
                        time.sleep(retry_delay)
                        continue
                    quota_hint = _quota_hint(getattr(exc, "details", None))
                    logger.warning("Gemini quota exhausted: %s", exc)
                    message = (
                        "Gemini API quota exhausted for model "
                        f"{self._model}. Wait for the quota to reset or upgrade your Gemini plan."
                    )
                    if quota_hint:
                        message += quota_hint
                    return message
                logger.exception("Gemini client error: %s", exc)
                return (
                    "Gemini returned a client error. Check the request payload and logs, then try again."
                )
            except RequestException as exc:
                logger.warning("Gemini request failed: %s", exc)
                return "Gemini request failed. Check your network connection and try again."
            except Exception as exc:
                logger.exception("Gemini request crashed: %s", exc)
                return "Gemini request crashed unexpectedly. Please review the logs and try again."
        if response is None:
            return "Gemini did not return a response. Please retry shortly."
        text_response = getattr(response, "text", None)
        if not text_response:
            text_response = "Gemini returned an empty response. Please retry or contact support if the issue persists."
        return text_response.strip()

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
