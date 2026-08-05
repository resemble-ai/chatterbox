import gc
import random
import os
import threading
import time
from contextlib import contextmanager

import numpy as np
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
from chatterbox.tts import ChatterboxTTS
from chatterbox.tts_turbo import ChatterboxTurboTTS
import gradio as gr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T3_MODEL = os.getenv("CHATTERBOX_MULTILINGUAL_T3_MODEL", "v2")

# --- Model catalogue ---
# Nothing here is instantiated at import time: `build` only runs when a model is
# actually needed, so the process starts with an empty GPU.
MODEL_SPECS = {
    "multilingual-v2": {
        "label": "Multilingual v2 (23 languages)",
        "multilingual": True,
        "build": lambda device: ChatterboxMultilingualTTS.from_pretrained(device, t3_model="v2"),
    },
    "multilingual-v3": {
        "label": "Multilingual v3 (23 languages)",
        "multilingual": True,
        "build": lambda device: ChatterboxMultilingualTTS.from_pretrained(device, t3_model="v3"),
    },
    "english": {
        "label": "English (Chatterbox)",
        "multilingual": False,
        "build": lambda device: ChatterboxTTS.from_pretrained(device),
    },
    "turbo": {
        "label": "Turbo (English, paralinguistic tags)",
        "multilingual": False,
        "build": lambda device: ChatterboxTurboTTS.from_pretrained(device, nano=False),
    },
    "nano": {
        "label": "Nano (English, smallest)",
        "multilingual": False,
        "build": lambda device: ChatterboxTurboTTS.from_pretrained(device, nano=True),
    },
}


def label_for(key: str | None) -> str:
    return MODEL_SPECS.get(key, {}).get("label", str(key))


def get_spec(key: str) -> dict:
    if key not in MODEL_SPECS:
        raise gr.Error(f"Unknown model '{key}'. Available: {', '.join(MODEL_SPECS)}")
    return MODEL_SPECS[key]


def _default_model_key() -> str:
    """Which model the picker starts on.

    CHATTERBOX_MODEL wins; otherwise fall back to the older
    CHATTERBOX_MULTILINGUAL_T3_MODEL setting so existing launch scripts keep
    selecting the checkpoint they always did.
    """
    explicit = os.getenv("CHATTERBOX_MODEL")
    if explicit:
        if explicit in MODEL_SPECS:
            return explicit
        print(f"WARNING: unknown CHATTERBOX_MODEL={explicit!r}; ignoring it.")
    key = f"multilingual-{T3_MODEL}"
    return key if key in MODEL_SPECS else "multilingual-v2"


def _idle_timeout_seconds() -> float:
    """Minutes of inactivity before the resident model is released. 0 disables."""
    raw = os.getenv("CHATTERBOX_IDLE_TIMEOUT_MINUTES", "20")
    try:
        minutes = float(raw)
    except ValueError:
        print(f"WARNING: invalid CHATTERBOX_IDLE_TIMEOUT_MINUTES={raw!r}; using 20.")
        minutes = 20.0
    return max(0.0, minutes) * 60.0


DEFAULT_MODEL_KEY = _default_model_key()
IDLE_TIMEOUT_SECONDS = _idle_timeout_seconds()

print(f"🚀 Running on device: {DEVICE}")
print(f"Default model: {label_for(DEFAULT_MODEL_KEY)} (loaded on first use)")
if IDLE_TIMEOUT_SECONDS:
    print(f"Idle unload after {IDLE_TIMEOUT_SECONDS / 60:.0f} minutes of inactivity.")
else:
    print("Idle unload disabled (CHATTERBOX_IDLE_TIMEOUT_MINUTES=0).")

LANGUAGE_CONFIG = {
    "ar": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ar_f/ar_prompts2.flac",
        "text": "في الشهر الماضي، وصلنا إلى معلم جديد بمليارين من المشاهدات على قناتنا على يوتيوب."
    },
    "da": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/da_m1.flac",
        "text": "Sidste måned nåede vi en ny milepæl med to milliarder visninger på vores YouTube-kanal."
    },
    "de": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/de_f1.flac",
        "text": "Letzten Monat haben wir einen neuen Meilenstein erreicht: zwei Milliarden Aufrufe auf unserem YouTube-Kanal."
    },
    "el": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/el_m.flac",
        "text": "Τον περασμένο μήνα, φτάσαμε σε ένα νέο ορόσημο με δύο δισεκατομμύρια προβολές στο κανάλι μας στο YouTube."
    },
    "en": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac",
        "text": "Last month, we reached a new milestone with two billion views on our YouTube channel."
    },
    "es": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/es_f1.flac",
        "text": "El mes pasado alcanzamos un nuevo hito: dos mil millones de visualizaciones en nuestro canal de YouTube."
    },
    "fi": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fi_m.flac",
        "text": "Viime kuussa saavutimme uuden virstanpylvään kahden miljardin katselukerran kanssa YouTube-kanavallamme."
    },
    "fr": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fr_f1.flac",
        "text": "Le mois dernier, nous avons atteint un nouveau jalon avec deux milliards de vues sur notre chaîne YouTube."
    },
    "he": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/he_m1.flac",
        "text": "בחודש שעבר הגענו לאבן דרך חדשה עם שני מיליארד צפיות בערוץ היוטיוב שלנו."
    },
    "hi": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/hi_f1.flac",
        "text": "पिछले महीने हमने एक नया मील का पत्थर छुआ: हमारे YouTube चैनल पर दो अरब व्यूज़।"
    },
    "it": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/it_m1.flac",
        "text": "Il mese scorso abbiamo raggiunto un nuovo traguardo: due miliardi di visualizzazioni sul nostro canale YouTube."
    },
    "ja": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ja/ja_prompts1.flac",
        "text": "先月、私たちのYouTubeチャンネルで二十億回の再生回数という新たなマイルストーンに到達しました。"
    },
    "ko": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ko_f.flac",
        "text": "지난달 우리는 유튜브 채널에서 이십억 조회수라는 새로운 이정표에 도달했습니다."
    },
    "ms": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ms_f.flac",
        "text": "Bulan lepas, kami mencapai pencapaian baru dengan dua bilion tontonan di saluran YouTube kami."
    },
    "nl": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/nl_m.flac",
        "text": "Vorige maand bereikten we een nieuwe mijlpaal met twee miljard weergaven op ons YouTube-kanaal."
    },
    "no": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/no_f1.flac",
        "text": "Forrige måned nådde vi en ny milepæl med to milliarder visninger på YouTube-kanalen vår."
    },
    "pl": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pl_m.flac",
        "text": "W zeszłym miesiącu osiągnęliśmy nowy kamień milowy z dwoma miliardami wyświetleń na naszym kanale YouTube."
    },
    "pt": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pt_m1.flac",
        "text": "No mês passado, alcançámos um novo marco: dois mil milhões de visualizações no nosso canal do YouTube."
    },
    "ru": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ru_m.flac",
        "text": "В прошлом месяце мы достигли нового рубежа: два миллиарда просмотров на нашем YouTube-канале."
    },
    "sv": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sv_f.flac",
        "text": "Förra månaden nådde vi en ny milstolpe med två miljarder visningar på vår YouTube-kanal."
    },
    "sw": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sw_m.flac",
        "text": "Mwezi uliopita, tulifika hatua mpya ya maoni ya bilioni mbili kweny kituo chetu cha YouTube."
    },
    "tr": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/tr_m.flac",
        "text": "Geçen ay YouTube kanalımızda iki milyar görüntüleme ile yeni bir dönüm noktasına ulaştık."
    },
    "zh": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/zh_f2.flac",
        "text": "上个月，我们达到了一个新的里程碑. 我们的YouTube频道观看次数达到了二十亿次，这绝对令人难以置信。"
    },
}

# --- UI Helpers ---
def default_audio_for_ui(lang: str) -> str | None:
    return LANGUAGE_CONFIG.get(lang, {}).get("audio")


def default_text_for_ui(lang: str) -> str:
    return LANGUAGE_CONFIG.get(lang, {}).get("text", "")


def get_supported_languages_display() -> str:
    """Generate a formatted display of all supported languages."""
    language_items = []
    for code, name in sorted(SUPPORTED_LANGUAGES.items()):
        language_items.append(f"**{name}** (`{code}`)")
    
    # Split into 2 lines
    mid = len(language_items) // 2
    line1 = " • ".join(language_items[:mid])
    line2 = " • ".join(language_items[mid:])
    
    return f"""
### 🌍 Supported Languages ({len(SUPPORTED_LANGUAGES)} total)
{line1}

{line2}
"""


class ModelManager:
    """Owns the single resident TTS model.

    Nothing is loaded at import time. The first Generate — from the UI or the
    API — loads whichever model is selected, and a background reaper releases it
    once it has been idle long enough, handing the VRAM back to the system.
    """

    def __init__(self, device: str, idle_timeout: float, check_interval: float = 15.0):
        self.device = device
        self.idle_timeout = idle_timeout
        self._lock = threading.RLock()
        self._model = None
        self._key = None
        self._active = 0  # in-flight generations; the reaper never cuts under one
        self._last_used = time.monotonic()
        self._stop = threading.Event()
        if idle_timeout > 0:
            threading.Thread(
                target=self._reap_forever,
                args=(check_interval,),
                name="model-idle-reaper",
                daemon=True,
            ).start()

    @contextmanager
    def use(self, key: str):
        """Yield the requested model, loading or switching to it as needed."""
        model = self._acquire(key)
        try:
            yield model
        finally:
            with self._lock:
                self._active = max(0, self._active - 1)
                self._last_used = time.monotonic()

    def _acquire(self, key: str):
        spec = get_spec(key)
        with self._lock:
            if self._model is not None and self._key != key:
                print(f"Switching model: {label_for(self._key)} -> {spec['label']}")
                self._drop_locked()
            if self._model is None:
                print(f"Loading {spec['label']} on {DEVICE}...")
                started = time.monotonic()
                model = spec["build"](self.device)
                if hasattr(model, "to") and str(getattr(model, "device", self.device)) != self.device:
                    model.to(self.device)
                self._model = model
                self._key = key
                print(f"Loaded {spec['label']} in {time.monotonic() - started:.1f}s.")
            self._active += 1
            self._last_used = time.monotonic()
            return self._model

    def reload(self, key: str) -> str:
        """Release whatever is resident and load `key`, even when `key` is already
        the current model — that is what makes "Apply / Reload" able to restart
        the selected model without switching away and back."""
        with self._lock:
            if self._active:
                return "⏳ A generation is running — try again when it finishes."
            self._drop_locked()
        with self.use(key):
            pass
        return f"✅ {label_for(key)} loaded."

    def unload(self) -> str:
        with self._lock:
            if self._active:
                return "⏳ A generation is running — try again when it finishes."
            if self._model is None:
                return "Nothing loaded."
            label = label_for(self._key)
            self._drop_locked()
            return f"✅ Unloaded {label}."

    def _drop_locked(self):
        """Full release: drop every reference, collect, then hand the cached
        blocks back to the driver so the VRAM actually leaves the process."""
        if self._model is None:
            return
        label = label_for(self._key)
        self._model = None
        self._key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print(f"Released {label}.")

    def _reap_forever(self, check_interval: float):
        while not self._stop.wait(check_interval):
            with self._lock:
                if self._model is None or self._active:
                    continue
                idle_for = time.monotonic() - self._last_used
                if idle_for >= self.idle_timeout:
                    print(f"Idle for {idle_for / 60:.1f} min - unloading.")
                    self._drop_locked()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "key": self._key,
                "active": self._active,
                "idle_for": time.monotonic() - self._last_used,
            }


MANAGER = ModelManager(DEVICE, IDLE_TIMEOUT_SECONDS)


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    if minutes and secs:
        return f"{minutes}m {secs}s"
    return f"{minutes}m" if minutes else f"{secs}s"


def _vram_note() -> str:
    if DEVICE != "cuda":
        return ""
    return f" · {torch.cuda.memory_reserved() / 1024 ** 3:.1f} GB VRAM"


def model_status(selected_key: str) -> str:
    """One-line summary of what is resident, for the status bar under the picker."""
    selected_key = selected_key or DEFAULT_MODEL_KEY
    snap = MANAGER.snapshot()

    if snap["key"] is None:
        return (
            f"⚪ **Nothing loaded** — Generate will load *{label_for(selected_key)}* first "
            "(one-time wait while it reads from disk)."
        )

    label = label_for(snap["key"])
    if snap["active"]:
        return f"🔵 **{label}** — generating…{_vram_note()}"
    if snap["key"] != selected_key:
        return (
            f"🟡 **{label}** loaded{_vram_note()} — *{label_for(selected_key)}* is selected; "
            "Generate or Apply will switch."
        )
    if MANAGER.idle_timeout <= 0:
        return f"🟢 **{label}** loaded{_vram_note()} · idle unload disabled"
    remaining = MANAGER.idle_timeout - snap["idle_for"]
    return f"🟢 **{label}** loaded{_vram_note()} · unloads in {_format_duration(remaining)}"


def apply_model(model_choice: str) -> str:
    """Load the selected model, restarting it if it is already the resident one."""
    key = model_choice or DEFAULT_MODEL_KEY
    gr.Info(MANAGER.reload(key))
    return model_status(key)


def unload_model(model_choice: str = None) -> str:
    """Release the resident model now and free its VRAM."""
    gr.Info(MANAGER.unload())
    return model_status(model_choice)

def set_seed(seed: int):
    """Sets the random seed for reproducibility across torch, numpy, and random."""
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    
def resolve_audio_prompt(language_id: str, provided_path: str | None) -> str | None:
    """
    Decide which audio prompt to use:
    - If user provided a path (upload/mic/url), use it.
    - Else, fall back to language-specific default (if any).
    """
    if provided_path and str(provided_path).strip():
        return provided_path
    return LANGUAGE_CONFIG.get(language_id, {}).get("audio")


def generate_tts_audio(
    text_input: str,
    language_id: str,
    audio_prompt_path_input: str = None,
    exaggeration_input: float = 0.5,
    temperature_input: float = 0.8,
    seed_num_input: int = 0,
    cfgw_input: float = 0.5,
    model_choice: str = None
) -> tuple[int, np.ndarray]:
    """
    Generate high-quality speech audio from text using Chatterbox with optional reference audio styling.
    The multilingual models cover 23 languages; the english, turbo and nano models are English-only.

    This tool synthesizes natural-sounding speech from input text. When a reference audio file
    is provided, it captures the speaker's voice characteristics and speaking style. The generated audio
    maintains the prosody, tone, and vocal qualities of the reference speaker, or uses default voice if no reference is provided.

    The selected model is loaded on demand, so the first call after startup — or after an idle
    unload — pays a one-off load cost instead of failing.

    Args:
        text_input (str): The text to synthesize into speech (maximum 300 characters)
        language_id (str): The language code for synthesis (eg. en, fr, de, es, it, pt, hi). Only used by the multilingual models.
        audio_prompt_path_input (str, optional): File path or URL to the reference audio file that defines the target voice style. Defaults to None.
        exaggeration_input (float, optional): Controls speech expressiveness (0.25-2.0, neutral=0.5, extreme values may be unstable). Defaults to 0.5. Ignored by turbo/nano.
        temperature_input (float, optional): Controls randomness in generation (0.05-5.0, higher=more varied). Defaults to 0.8.
        seed_num_input (int, optional): Random seed for reproducible results (0 for random generation). Defaults to 0.
        cfgw_input (float, optional): CFG/Pace weight controlling generation guidance (0.2-1.0). Defaults to 0.5, 0 for language transfer. Ignored by turbo/nano.
        model_choice (str, optional): Which model to synthesize with — one of multilingual-v2, multilingual-v3, english, turbo, nano. Defaults to the server's configured model.

    Returns:
        tuple[int, np.ndarray]: A tuple containing the sample rate (int) and the generated audio waveform (numpy.ndarray)
    """
    key = model_choice or DEFAULT_MODEL_KEY
    spec = get_spec(key)

    if seed_num_input != 0:
        set_seed(int(seed_num_input))

    print(f"Generating audio for text: '{text_input[:50]}...'")

    # Handle optional audio prompt. The per-language defaults are multilingual demo
    # clips, so only fall back to them for the multilingual models — the others
    # have a built-in voice that is a better default than a foreign-language clip.
    if spec["multilingual"]:
        chosen_prompt = audio_prompt_path_input or default_audio_for_ui(language_id)
    else:
        chosen_prompt = audio_prompt_path_input

    generate_kwargs = {"temperature": temperature_input}
    if chosen_prompt:
        generate_kwargs["audio_prompt_path"] = chosen_prompt
        print(f"Using audio prompt: {chosen_prompt}")
    else:
        print("No audio prompt provided; using default voice.")

    if spec["multilingual"]:
        generate_kwargs["language_id"] = language_id
    if key != "turbo" and key != "nano":
        # Turbo and Nano ignore these and log a warning when they are non-zero.
        generate_kwargs["exaggeration"] = exaggeration_input
        generate_kwargs["cfg_weight"] = cfgw_input

    # `use()` loads the model if it was never loaded or was released while idle,
    # and switches checkpoints when the picker moved — Generate no longer fails
    # with "TTS model is not loaded".
    with MANAGER.use(key) as current_model:
        try:
            wav = current_model.generate(
                text_input[:300],  # Truncate text to max chars
                **generate_kwargs
            )
        except AssertionError as e:
            raise gr.Error(f"{spec['label']} rejected that reference audio: {e}") from e
        sample_rate = current_model.sr

    print("Audio generation complete.")
    return (sample_rate, wav.squeeze(0).numpy())

with gr.Blocks() as demo:
    gr.Markdown(
        f"""
        # Chatterbox Multilingual Demo
        Generate high-quality multilingual speech from text with reference audio styling, supporting 23 languages.

        Models load on first use and are released after
        {IDLE_TIMEOUT_SECONDS / 60:.0f} minutes idle, so the GPU sits empty when you aren't generating.
        Generate always loads whatever is selected — use **Apply / Reload** to warm it up ahead of time
        or restart the model you're already on.
        """
        if IDLE_TIMEOUT_SECONDS else
        """
        # Chatterbox Multilingual Demo
        Generate high-quality multilingual speech from text with reference audio styling, supporting 23 languages.

        Models load on first use. Idle unloading is disabled — use **Unload now** to free the VRAM by hand.
        """
    )
    
    # Display supported languages
    gr.Markdown(get_supported_languages_display())
    with gr.Row():
        with gr.Column():
            initial_lang = "fr"

            model_choice = gr.Dropdown(
                choices=[(spec["label"], key) for key, spec in MODEL_SPECS.items()],
                value=DEFAULT_MODEL_KEY,
                label="Model",
                info="Loaded on first use, released after the idle timeout",
            )
            with gr.Row():
                apply_btn = gr.Button("Apply / Reload model")
                unload_btn = gr.Button("Unload now")
            status_md = gr.Markdown(model_status(DEFAULT_MODEL_KEY))

            text = gr.Textbox(
                value=default_text_for_ui(initial_lang),
                label="Text to synthesize (max chars 300)",
                max_lines=5
            )
            
            language_id = gr.Dropdown(
                choices=list(ChatterboxMultilingualTTS.get_supported_languages().keys()),
                value=initial_lang,
                label="Language",
                info="Select the language for text-to-speech synthesis"
            )
            
            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Reference Audio File (Optional)",
                value=default_audio_for_ui(initial_lang)
            )
            
            gr.Markdown(
                "💡 **Note**: Ensure that the reference clip matches the specified language tag. Otherwise, language transfer outputs may inherit the accent of the reference clip's language. To mitigate this, set the CFG weight to 0.",
                elem_classes=["audio-note"]
            )
            
            exaggeration = gr.Slider(
                0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=.5
            )
            cfg_weight = gr.Slider(
                0.2, 1, step=.05, label="CFG/Pace", value=0.5
            )

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="Temperature", value=.8)

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

        def on_language_change(lang, current_ref, current_text):
            return default_audio_for_ui(lang), default_text_for_ui(lang)

        language_id.change(
            fn=on_language_change,
            inputs=[language_id, ref_wav, text],
            outputs=[ref_wav, text],
            show_progress=False
        )

        def on_model_change(selected_key):
            """Selecting a model doesn't load it — that stays lazy. Just reflect
            what the choice means for the language picker and the status line."""
            key = selected_key or DEFAULT_MODEL_KEY
            multilingual = MODEL_SPECS.get(key, {}).get("multilingual", False)
            return (
                gr.Dropdown(
                    interactive=multilingual,
                    info=(
                        "Select the language for text-to-speech synthesis"
                        if multilingual
                        else "This model is English-only"
                    ),
                ),
                model_status(key),
            )

        model_choice.change(
            fn=on_model_change,
            inputs=[model_choice],
            outputs=[language_id, status_md],
            show_progress=False
        )

    apply_btn.click(
        fn=apply_model,
        inputs=[model_choice],
        outputs=[status_md],
        api_name="load_model",
    )

    unload_btn.click(
        fn=unload_model,
        inputs=[model_choice],
        outputs=[status_md],
        api_name="unload_model",
    )

    # A fresh page shouldn't trust the status baked in at startup.
    demo.load(
        fn=model_status,
        inputs=[model_choice],
        outputs=[status_md],
        show_progress=False
    )

    # Keeps the countdown and VRAM figure honest without touching the model.
    gr.Timer(10).tick(
        fn=model_status,
        inputs=[model_choice],
        outputs=[status_md],
        show_progress=False
    )

    run_btn.click(
        fn=generate_tts_audio,
        inputs=[
            text,
            language_id,
            ref_wav,
            exaggeration,
            temp,
            seed_num,
            cfg_weight,
            model_choice,
        ],
        outputs=[audio_output],
    ).then(
        fn=model_status,
        inputs=[model_choice],
        outputs=[status_md],
        show_progress=False
    )

demo.launch(mcp_server=True)
