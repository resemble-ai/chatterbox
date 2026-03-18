import random
import numpy as np
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
import gradio as gr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {DEVICE}")

# --- Global Model Initialization ---
MODEL = None

LANGUAGE_CONFIG = {
    "da": {
        "audio_options": {
            "mic": "voices/mic.wav",
            "nic": "voices/nic.wav"
        },
        "default_audio": "voices/mic.wav",  # Default to mic
        "text": "Sidste måned nåede vi en ny milepæl med to milliarder visninger på vores YouTube-kanal."
    },
    "en": {
        "audio": "voices/en_f1.flac",
        "text": "Last month, we reached a new milestone with two billion views on our YouTube channel."
    },
}

# --- UI Helpers ---
def default_audio_for_ui(lang: str, danish_voice: str = "mic") -> str | None:
    config = LANGUAGE_CONFIG.get(lang, {})
    if lang == "da" and "audio_options" in config:
        return config["audio_options"].get(danish_voice, config.get("default_audio"))
    return config.get("audio")


def default_text_for_ui(lang: str) -> str:
    return LANGUAGE_CONFIG.get(lang, {}).get("text", "")


def get_danish_voice_options() -> list[tuple[str, str]]:
    """Get the available Danish voice options for the dropdown."""
    return [("Mic", "mic"), ("Nic", "nic")]


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
### 🇩🇰 Supported Languages ({len(SUPPORTED_LANGUAGES)} total)
{line1}

{line2}
"""



def get_or_load_model():
    """Loads the ChatterboxMultilingualTTS model if it hasn't been loaded already,
    and ensures it's on the correct device."""
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
    temperature_input: float = 0.8,
    seed_num_input: int = 0,
    cfgw_input: float = 0.5
) -> tuple[int, np.ndarray]:
    """
    Generate high-quality speech audio from text using Chatterbox Multilingual model with optional reference audio styling.
    Supported languages: English, French, German, Spanish, Italian, Portuguese, and Hindi.
    
    This tool synthesizes natural-sounding speech from input text. When a reference audio file 
    is provided, it captures the speaker's voice characteristics and speaking style. The generated audio 
    maintains the prosody, tone, and vocal qualities of the reference speaker, or uses default voice if no reference is provided.

    Args:
        text_input (str): The text to synthesize into speech (maximum 300 characters)
        language_id (str): The language code for synthesis (eg. en, fr, de, es, it, pt, hi)
        audio_prompt_path_input (str, optional): File path or URL to the reference audio file that defines the target voice style. Defaults to None.
        exaggeration_input (float, optional): Controls speech expressiveness (0.25-2.0, neutral=0.5, extreme values may be unstable). Defaults to 0.5.
        temperature_input (float, optional): Controls randomness in generation (0.05-5.0, higher=more varied). Defaults to 0.8.
        seed_num_input (int, optional): Random seed for reproducible results (0 for random generation). Defaults to 0.
        cfgw_input (float, optional): CFG/Pace weight controlling generation guidance (0.2-1.0). Defaults to 0.5, 0 for language transfer. 

    Returns:
        tuple[int, np.ndarray]: A tuple containing the sample rate (int) and the generated audio waveform (numpy.ndarray)
    """
    current_model = get_or_load_model()
    exaggeration = 0.5

    if current_model is None:
        raise RuntimeError("TTS model is not loaded.")

    if seed_num_input != 0:
        set_seed(int(seed_num_input))

    print(f"Generating audio for text: '{text_input[:50]}...'")
    
    # Handle optional audio prompt
    chosen_prompt = audio_prompt_path_input or default_audio_for_ui(language_id)

    generate_kwargs = {
        "exaggeration": exaggeration,
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

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Danish Chatterbox Demo
        Generate high-quality danish speech from text with reference audio styling.
        """
    )
    
    # Display supported languages
    gr.Markdown(get_supported_languages_display())
    with gr.Row():
        with gr.Column():
            initial_lang = "da"
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
            
            danish_voice = gr.Dropdown(
                choices=get_danish_voice_options(),
                value="mic",
                label="Danish Voice Selection",
                info="Choose between different Danish voice options",
                visible=(initial_lang == "da")
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
            is_danish = (lang == "da")
            danish_voice_val = "mic" if is_danish else "mic"  # Default to mic
            return (
                default_audio_for_ui(lang, danish_voice_val), 
                default_text_for_ui(lang), 
                gr.update(visible=is_danish),  # Update Danish voice dropdown visibility
                danish_voice_val
            )

        def on_danish_voice_change(lang, danish_voice_val):
            if lang == "da":
                return default_audio_for_ui(lang, danish_voice_val)
            return gr.update()  # No change if not Danish

        language_id.change(
            fn=on_language_change,
            inputs=[language_id, ref_wav, text],
            outputs=[ref_wav, text, danish_voice, danish_voice],
            show_progress=False
        )

        danish_voice.change(
            fn=on_danish_voice_change,
            inputs=[language_id, danish_voice],
            outputs=[ref_wav],
            show_progress=False
        )

    run_btn.click(
        fn=generate_tts_audio,
        inputs=[
            text,
            language_id,
            ref_wav,
            temp,
            seed_num,
            cfg_weight,
        ],
        outputs=[audio_output],
    )

demo.launch() #mcp_server=True