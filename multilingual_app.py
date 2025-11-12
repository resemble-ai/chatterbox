import random
import numpy as np
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
import gradio as gr
from safetensors.torch import load_file as load_safetensors
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {DEVICE}")

# --- Custom T3 Model Configuration ---
CUSTOM_T3_MODELS = {
    "Default": None,
    "Czech (t3_cs)": "t3_cs",  # Path to your safetensors file
    # Add more custom models here:
    # "Another Language": "path/to/model.safetensors",
}

# --- Global Model Initialization ---
MODEL = None

LANGUAGE_CONFIG = {
    "ar": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ar_f/ar_prompts2.flac",
        "text": "في الشهر الماضي، وصلنا إلى معلم جديد بمليارين من المشاهدات على قناتنا على يوتيوب."
    },
    "cs": {  # Add Czech language
        "audio": None,
        "text": "Dobrý den, vítáme vás v našem testu syntézy řeči"
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
    # Combine base supported languages with any custom ones
    all_langs = dict(SUPPORTED_LANGUAGES)
    all_langs.update({"cs": "Czech"})  # Add custom languages here
    
    language_items = []
    for code, name in sorted(all_langs.items()):
        language_items.append(f"**{name}** (`{code}`)")
    
    # Split into 2 lines
    mid = len(language_items) // 2
    line1 = " • ".join(language_items[:mid])
    line2 = " • ".join(language_items[mid:])
    
    return f"""
### 🌍 Supported Languages ({len(all_langs)} total)
{line1}

{line2}
"""


def get_or_load_model():
    """Loads the ChatterboxMultilingualTTS model if it hasn't been loaded already."""
    global MODEL
    if MODEL is None:
        print("Model not loaded, initializing...")
        try:
            MODEL = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
            if hasattr(MODEL, 'to') and str(MODEL.device) != DEVICE:
                MODEL.to(DEVICE)
            print(f"Model loaded successfully. Internal device: {getattr(MODEL, 'device', 'N/A')}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    return MODEL


def switch_t3_model(model_choice: str):
    """Switch the T3 model to a custom version"""
    global MODEL
    
    if MODEL is None:
        MODEL = get_or_load_model()
    
    custom_path = CUSTOM_T3_MODELS.get(model_choice)
    
    if custom_path:
        # Check if path exists
        if not os.path.exists(custom_path):
            return f"❌ Error: Model path not found: {custom_path}"
        
        print(f"Loading custom T3 model from: {custom_path}")
        try:
            # Load the custom T3 state dict
            t3_state = load_safetensors(custom_path, device="cpu")
            MODEL.t3.load_state_dict(t3_state)
            MODEL.t3.to(DEVICE).eval()
            print(f"✓ Loaded custom T3 model: {model_choice}")
            return f"✓ Loaded: {model_choice}"
        except Exception as e:
            return f"❌ Error loading model: {str(e)}"
    else:
        print("Reloading default T3 model...")
        try:
            # Reload the entire model to get default T3
            MODEL = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
            print("✓ Loaded default T3 model")
            return "✓ Loaded: Default T3 model"
        except Exception as e:
            return f"❌ Error loading model: {str(e)}"


# Attempt to load the model at startup.
try:
    get_or_load_model()
except Exception as e:
    print(f"CRITICAL: Failed to load model on startup. Application may not function. Error: {e}")


def set_seed(seed: int):
    """Sets the random seed for reproducibility across torch, numpy, and random."""
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def generate_tts_audio(
    text_input: str,
    language_id: str,
    audio_prompt_path_input: str = None,
    exaggeration_input: float = 0.5,
    temperature_input: float = 0.8,
    seed_num_input: int = 0,
    cfgw_input: float = 0.5
) -> tuple[int, np.ndarray]:
    """Generate TTS audio with custom T3 model support"""
    current_model = get_or_load_model()

    if current_model is None:
        raise RuntimeError("TTS model is not loaded.")

    if seed_num_input != 0:
        set_seed(int(seed_num_input))

    print(f"Generating audio for text: '{text_input[:50]}...' in language: {language_id}")
    
    # Handle optional audio prompt
    chosen_prompt = audio_prompt_path_input or default_audio_for_ui(language_id)

    generate_kwargs = {
        "exaggeration": exaggeration_input,
        "temperature": temperature_input,
        "cfg_weight": cfgw_input,
    }
    if chosen_prompt:
        generate_kwargs["audio_prompt_path"] = chosen_prompt
        print(f"Using audio prompt: {chosen_prompt}")
    else:
        print("No audio prompt provided; using default voice.")
        
    wav = current_model.generate(
        text_input[:300],  # Truncate text to max chars
        language_id=language_id,
        **generate_kwargs
    )
    print("Audio generation complete.")
    return (current_model.sr, wav.squeeze(0).numpy())


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎙️ Chatterbox Multilingual Demo with Custom T3 Support
        Generate high-quality multilingual speech from text with reference audio styling and custom model support.
        """
    )
    
    # Display supported languages
    gr.Markdown(get_supported_languages_display())
    
    with gr.Row():
        with gr.Column():
            # Model Selection Section
            gr.Markdown("### 🔧 Model Configuration")
            t3_model_dropdown = gr.Dropdown(
                choices=list(CUSTOM_T3_MODELS.keys()),
                value="Default",
                label="T3 Model",
                info="Select which T3 model to use"
            )
            model_status = gr.Textbox(
                label="Model Status",
                value="Default model loaded",
                interactive=False,
                lines=1
            )
            load_t3_btn = gr.Button("🔄 Load Selected T3 Model", variant="secondary", size="sm")
            
            gr.Markdown("---")
            
            # TTS Controls
            initial_lang = "cs"  # Default to Czech for testing
            text = gr.Textbox(
                value=default_text_for_ui(initial_lang),
                label="Text to synthesize (max chars 300)",
                max_lines=5
            )
            
            # Get all supported languages including custom ones
            all_language_codes = list(SUPPORTED_LANGUAGES.keys()) + ["cs"]
            language_id = gr.Dropdown(
                choices=sorted(set(all_language_codes)),
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
                "💡 **Note**: Ensure that the reference clip matches the specified language tag. For custom languages, set CFG weight to 0 if experiencing accent issues.",
                elem_classes=["audio-note"]
            )
            
            exaggeration = gr.Slider(
                0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5)", value=.5
            )
            cfg_weight = gr.Slider(
                0.0, 1, step=.05, label="CFG/Pace (0 for language transfer)", value=0.5
            )

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="Temperature", value=.8)

            run_btn = gr.Button("🎬 Generate Speech", variant="primary", size="lg")

        with gr.Column():
            gr.Markdown("### 📊 Output")
            audio_output = gr.Audio(label="Generated Audio")
            
            gr.Markdown("""
            ---
            ### 💡 Tips
            
            **Custom T3 Models:**
            - Load your fine-tuned T3 models for new languages
            - Place `.safetensors` files in the working directory
            - Switch between models without restarting
            
            **Voice Cloning:**
            - Upload 5-10 seconds of clear reference audio
            - Single speaker, minimal background noise
            - Match reference language to target language
            
            **Parameters:**
            - **Exaggeration**: Controls emotion intensity
            - **Temperature**: Higher = more variation
            - **CFG Weight**: Set to 0 for language transfer without accent
            """)

    def on_language_change(lang, current_ref, current_text):
        return default_audio_for_ui(lang), default_text_for_ui(lang)

    language_id.change(
        fn=on_language_change,
        inputs=[language_id, ref_wav, text],
        outputs=[ref_wav, text],
        show_progress=False
    )
    
    load_t3_btn.click(
        fn=switch_t3_model,
        inputs=[t3_model_dropdown],
        outputs=[model_status]
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
        ],
        outputs=[audio_output],
    )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎙️  CHATTERBOX MULTILINGUAL TTS - CUSTOM T3 EDITION")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Available T3 models: {len(CUSTOM_T3_MODELS)}")
    for model_name in CUSTOM_T3_MODELS.keys():
        print(f"  - {model_name}")
    print("="*60 + "\n")
    
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(share=True)