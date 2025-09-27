import json
from typing import List, Dict, Any, Optional

from google import genai

from utilities.consts import (
    GOOGLE_API_KEY,
    GeminiModelsEnum,
    SupportedLanguagesCodesEnum,
    MIN_COUNT,
)
from utilities.prompts import PROMPTS, PromptType


class AskGemini:
    """Wrapper for Gemini model interactions.

    Обёртка для взаимодействия с моделью Gemini.
    """

    def __init__(self,
                 system_prompt: str = "",
                 user_context: str = "",
                 model: GeminiModelsEnum = GeminiModelsEnum.gemini_2_5_flash,
                 file_parts: Optional[list] = None):
        """Initialize Gemini client and context.

        Инициализировать клиент Gemini и контекст.

        Pipeline:

            1. Validate API key.
               Проверить API ключ.

            2. Create client and store parameters.
               Создать клиента и сохранить параметры.

        Args:

            system_prompt (str):
                Global system instructions.
                Глобальные системные инструкции.

            user_context (str):
                Additional user context.
                Дополнительный пользовательский контекст.

            model (GeminiModelsEnum):
                Gemini model to use.
                Используемая модель Gemini.

            file_parts (Optional[list]):
                List of file descriptors.
                Список описаний файлов.

        Raises:

            ValueError:
                Missing API key.
                Отсутствует API ключ.
        """

        # Step 1: Ensure API key is available
        # Шаг 1: Убедиться, что API ключ доступен
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set in environment")

        # Step 2: Initialize client and store settings
        # Шаг 2: Инициализировать клиента и сохранить настройки
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model = str(model)
        self.system_prompt = system_prompt.strip()
        self.user_context = (user_context or "").strip()
        # file_parts: list of {"file_uri": str, "mime_type": str}
        self.file_parts = file_parts or []

    def _gen(self,
             role: str = 'user',
             parts: List[Dict[str, Any]] = None,
             response_schema: Optional[Dict[str, Any]] = None,
             response_mime_type: Optional[str] = None):
        """Send prompt parts to Gemini model.

        Отправить части запроса модели Gemini.

        Pipeline:

            1. Build request payload.
               Сформировать полезную нагрузку.

            2. Call model and return response.
               Вызвать модель и вернуть ответ.

        Args:

            role (str):
                Role of the sender.
                Роль отправителя.

            parts (List[Dict[str, Any]]):
                Content parts for the model.
                Части контента для модели.

        Returns:
            Any:
                Response from Gemini.
                Ответ от Gemini.

        Raises:

            Exception:
                Propagated client errors.
                Ошибки клиента пробрасываются.
        """

        # Step 1: Build payload for the request
        # Шаг 1: Сформировать полезную нагрузку запроса
        payload = [{"role": role, "parts": parts}]

        # Step 2: Send request to Gemini and return response
        # Шаг 2: Отправить запрос Gemini и вернуть ответ
        config: Dict[str, Any] = {}
        if response_schema:
            config["response_schema"] = response_schema
            config["response_mime_type"] = response_mime_type or "application/json"

        return self.client.models.generate_content(
            model=self.model,
            contents=payload,
            **({"config": config} if config else {})
        )

    @staticmethod
    def _validate_review_payload(data: Dict[str, Any], tips_limit: int, slide_text: Optional[str] = None) -> Dict[str, Any]:
        """Validate strictly the structured JSON payload.

        Строго валидирует структурированный JSON-ответ.

        Args:
            data: Parsed object from structured output.
            tips_limit: Max tips to keep.

        Returns:
            Dict[str, Any]: Normalized dict with required keys.

        Raises:
            ValueError: If required fields are missing or of wrong type.
        """

        if not isinstance(data, dict):
            raise ValueError("Structured output is not an object")
        if "feedback" not in data or "tips" not in data:
            raise ValueError("Structured output missing required keys: feedback, tips")
        feedback = data["feedback"]
        tips_val = data["tips"]
        if not isinstance(feedback, str):
            raise ValueError("Field 'feedback' must be a string")
        if not isinstance(tips_val, list):
            raise ValueError("Field 'tips' must be an array")
        norm_tips: List[Dict[str, str]] = []
        for t in tips_val:
            if not isinstance(t, dict):
                raise ValueError("Each tip must be an object with title and text")
            title = t.get("title")
            text = t.get("text")
            if not isinstance(title, str) or not isinstance(text, str):
                raise ValueError("Tip fields 'title' and 'text' must be strings")
            norm_tips.append({"title": title.strip(), "text": text.strip()})
        # Optional: mains/negative phrases (arrays of strings, up to 5)
        pos = data.get("mains")
        neg = data.get("negative")
        # Required scores block
        scores_in = data.get("scores")
        def _norm_phrases(x):
            if isinstance(x, list):
                out = []
                for v in x:
                    if isinstance(v, str):
                        s = v.strip()
                        if s:
                            out.append(s)
                return out[:5]
            return []
        pos_n = _norm_phrases(pos)
        neg_n = _norm_phrases(neg)
        # Ensure minimum counts: mains >= 1 always; negative >= MIN_COUNT when configured
        try:
            min_required = max(0, min(5, int(MIN_COUNT)))
        except Exception:
            min_required = 1
        if not pos_n:
            raise ValueError("Model did not return at least one main thought (mains)")
        if min_required > 0 and len(neg_n) < min_required:
            raise ValueError("Model returned fewer negative formulations than required")
        # Validate scores
        if not isinstance(scores_in, dict):
            raise ValueError("Missing required 'scores' object")
        def _to_score(v):
            if isinstance(v, (int, float)):
                iv = int(v)
            else:
                try:
                    iv = int(str(v))
                except Exception:
                    raise ValueError("Score must be integer 0..100")
            if iv < 0 or iv > 100:
                raise ValueError("Score must be in 0..100")
            return iv
        required_scores = ["overall", "goal", "structure", "clarity", "delivery"]
        scores: Dict[str, int] = {}
        for k in required_scores:
            if k not in scores_in:
                raise ValueError(f"Missing score: {k}")
            scores[k] = _to_score(scores_in[k])
        return {
            "feedback": feedback.strip(),
            "tips": norm_tips[:tips_limit],
            "mains": pos_n,
            "negative": neg_n,
            "scores": scores,
        }

    def review_slide(
            self, slide_index: int, polished_text: str) -> Dict[str, Any]:
        """Generate feedback for a single slide.

        Сгенерировать отзыв для отдельного слайда.

        Pipeline:

            1. Attach optional files.
               Прикрепить необязательные файлы.

            2. Compose prompt with system data and slide text.
               Сформировать запрос из системных данных и текста слайда.

            3. Call Gemini model.
               Вызвать модель Gemini.

            4. Parse and normalize response.
               Разобрать и нормализовать ответ.

        Args:

            slide_index (int):
                Position of the slide.
                Номер слайда.

            polished_text (str):
                Prepared transcription of the slide.
                Подготовленный текст слайда.

        Returns:

            Dict[str, Any]:
                Feedback and up to three tips.
                Отзыв и до трёх советов.

        Raises:

            Exception:
                Propagated Gemini client errors.
                Пробрасываемые ошибки клиента Gemini.
        """

        # Step 1: Attach files if provided
        # Шаг 1: Прикрепить файлы при наличии
        parts = []
        for f in self.file_parts:
            uri = f.get("file_uri")
            mt = f.get("mime_type")
            if uri and mt:
                parts.append({"file_data": {"file_uri": uri, "mime_type": mt}})

        # Step 2: Add prompt sections
        # Шаг 2: Добавить части запроса
        parts += [
            {"text": f"[SYSTEM]\n{self.system_prompt}"},
            {"text": f"[CONTEXT]\n{self.user_context}"},
            {"text": f"[SLIDE {slide_index}]\n{polished_text}"},
            {"text": f"[REQUIREMENTS]\n{PROMPTS[PromptType.REVIEW_SLIDE]}"},
        ]

        # Structured output: enforce schema and return strictly
        min_count = max(0, min(5, int(MIN_COUNT)))
        schema = {
            "type": "object",
            "properties": {
                "feedback": {"type": "string", "description": "Отзыв по слайду"},
                # Минимум 1 основная мысль всегда
                "mains": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                # Минимум для неудачных — из переменной окружения
                "negative": {"type": "array", "items": {"type": "string"}, "minItems": min_count},
                "scores": {
                    "type": "object",
                    "properties": {
                        "overall": {"type": "integer", "minimum": 0, "maximum": 100},
                        "goal": {"type": "integer", "minimum": 0, "maximum": 100},
                        "structure": {"type": "integer", "minimum": 0, "maximum": 100},
                        "clarity": {"type": "integer", "minimum": 0, "maximum": 100},
                        "delivery": {"type": "integer", "minimum": 0, "maximum": 100},
                    },
                    "required": ["overall", "goal", "structure", "clarity", "delivery"]
                },
                "tips": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "text": {"type": "string"}
                        },
                        "required": ["title", "text"]
                    }
                }
            },
            "required": ["feedback", "tips", "mains", "negative", "scores"]
        }
        parts.append({
            "text": (
                "[FORMAT]\n"
                "Всегда отвечай на русском языке. Верни строго JSON по схеме:\n"
                "{feedback: string, mains: string[], negative: string[], scores: {overall: number, goal: number, structure: number, clarity: number, delivery: number}, tips: [{title: string, text: string}]}.\n"
                f"Где mains — это основные мысли (минимум 1), negative — неудачные формулировки (минимум {min_count}). Поля scores — целые числа 0..100. При недостатке материала сформулируй обобщённые варианты по контексту.\n"
                "Не добавляй ничего вне JSON."
            )
        })
        res_struct = self._gen(parts=parts, response_schema=schema, response_mime_type="application/json")
        parsed = getattr(res_struct, 'parsed', None)
        return self._validate_review_payload(parsed, tips_limit=3, slide_text=polished_text)

    def summarize(
            self,
            per_slide_findings: List[Dict[str, Any]],
            transcripts: Optional[List[str]] = None) -> Dict[str, Any]:

        # Step 1: Build snippets from slide findings
        # Шаг 1: Сформировать фрагменты из данных по слайдам
        slide_snippets = []
        for i, item in enumerate(per_slide_findings, start=1):
            fb = (item or {}).get("feedback", "").strip()
            tips = (item or {}).get("tips", [])
            tip_texts: List[str] = []
            if isinstance(tips, list):
                for t in tips:
                    if isinstance(t, dict):
                        title = str(t.get("title", "")).strip()
                        text = str(t.get("text", "")).strip()
                        if title or text:
                            tip_texts.append(f"{title}: {text}".strip(": "))
                    else:
                        s = str(t).strip()
                        if s:
                            tip_texts.append(s)
            tips_str = "; ".join([s for s in tip_texts if s])
            if fb or tips_str:
                slide_snippets.append(f"Slide {i}: {fb} Tips: {tips_str}")

        # Step 2: Prepare transcript note if provided
        # Шаг 2: Подготовить заметку по транскриптам при наличии
        transcript_note = ""
        if transcripts:
            transcript_note = (
                "\n\nTRANSCRIPTS:\n" +
                "\n".join([t[:500] for t in transcripts if t])
            )

        # Step 3: Attach files and assemble prompt
        # Шаг 3: Прикрепить файлы и собрать запрос
        parts = []
        for f in self.file_parts:
            uri = f.get("file_uri")
            mt = f.get("mime_type")
            if uri and mt:
                parts.append({"file_data": {"file_uri": uri, "mime_type": mt}})

        parts += [
            {"text": f"[SYSTEM]\n{self.system_prompt}"},
            {"text": f"[CONTEXT]\n{self.user_context}"},
            {"text": f"[PER_SLIDE]\n" + "\n".join(slide_snippets)},
            {"text": transcript_note},
            {"text": f"[REQUIREMENTS]\n{PROMPTS[PromptType.SUMMARIZE]}"},
        ]

        # Structured output for summary: feedback + mains + scores + tips
        summary_schema = {
            "type": "object",
            "properties": {
                "feedback": {"type": "string", "description": "Общий отзыв по презентации"},
                "mains": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "scores": {
                    "type": "object",
                    "properties": {
                        "overall": {"type": "integer", "minimum": 0, "maximum": 100},
                        "goal": {"type": "integer", "minimum": 0, "maximum": 100},
                        "structure": {"type": "integer", "minimum": 0, "maximum": 100},
                        "clarity": {"type": "integer", "minimum": 0, "maximum": 100},
                        "delivery": {"type": "integer", "minimum": 0, "maximum": 100},
                    },
                    "required": ["overall", "goal", "structure", "clarity", "delivery"]
                },
                "tips": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "text": {"type": "string"}
                        },
                        "required": ["title", "text"]
                    }
                }
            },
            "required": ["feedback", "mains", "scores", "tips"]
        }
        parts.append({
            "text": (
                "[FORMAT]\n"
                "Всегда отвечай на русском языке. Верни строго JSON по схеме:\n"
                "{feedback: string, mains: string[], scores: {overall: number, goal: number, structure: number, clarity: number, delivery: number}, tips: [{title: string, text: string}]}.\n"
                "Не добавляй ничего вне JSON."
            )
        })
        res_struct = self._gen(parts=parts, response_schema=summary_schema, response_mime_type="application/json")
        parsed = getattr(res_struct, 'parsed', None)
        if not isinstance(parsed, dict):
            raise ValueError("Invalid structured summary output")
        feedback = str(parsed.get("feedback", "")).strip()
        mains_list = []
        for s in (parsed.get("mains") or []):
            if isinstance(s, str) and s.strip():
                mains_list.append(s.strip())
        if not mains_list:
            raise ValueError("Summary mains missing")
        tips_norm: List[Dict[str, str]] = []
        for t in (parsed.get("tips") or []):
            if isinstance(t, dict):
                title = str(t.get("title", "")).strip()
                text = str(t.get("text", "")).strip()
                if title or text:
                    tips_norm.append({"title": title, "text": text})
        tips_norm = tips_norm[:5]
        sc_in = parsed.get("scores") or {}
        scores = {k: int(sc_in.get(k)) for k in ["overall", "goal", "structure", "clarity", "delivery"]}
        return {"feedback": feedback, "mains": mains_list[:5], "scores": scores, "tips": tips_norm}

    def restore_transcribed_text(
            self,
            transcribed_text: str,
            language: SupportedLanguagesCodesEnum = (
                SupportedLanguagesCodesEnum.RU
            ),
    ):

        if self.client is None:
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY is not set in environment")
            self.client = genai.Client(api_key=GOOGLE_API_KEY)

        # Step 2: Validate input text
        # Шаг 2: Проверить входной текст
        if not transcribed_text:
            raise ValueError("Transcribed text is empty.")

        # Step 3: Build prompt parts
        # Шаг 3: Собрать части запроса
        parts = [
            {"text": PROMPTS[PromptType.RESTORE].replace("{language}", str(language))},
            {"text": transcribed_text},
        ]

        # Step 4: Request refinement from Gemini
        # Шаг 4: Запросить улучшение у Gemini
        response = self._gen(parts=parts)

        # Step 5: Return refined text
        # Шаг 5: Вернуть улучшенный текст
        transcribed_text = (response.text or "").strip()
        return transcribed_text
