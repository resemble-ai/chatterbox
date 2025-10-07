import torch
import torchaudio as ta
import psutil
import os
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


baseline = get_memory_mb()

before = get_memory_mb()

print(f"Before: {before:.1f}")

# Detect device (Mac with M1/M2/M3/M4)
device = "mps" if torch.backends.mps.is_available() else "cpu"
map_location = torch.device(device)

torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)

torch.load = patched_torch_load

after_load = get_memory_mb()
print(f"After load: {after_load:.1f}")

multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
# model = ChatterboxTTS.from_pretrained(device=device)
text = "Hoy anunciamos que Cognition está adquiriendo Windsurf"
text2 = "Hola, soy Scott, director ejecutivo de Cognition. Y soy Jeff, director ejecutivo de Windsurf. Hoy anunciamos la adquisición de Windsurf por parte de Cognition. Hemos estado desarrollando al ingeniero de software de IA, Devon, y Windsurf fue pionero en el IDE de Agentic. Estamos entusiasmados con las posibilidades que podemos lograr juntos. Para el equipo y los usuarios de Windsurf, esta es la combinación perfecta. Incorporar agentes remotos al IDE desbloqueará todo tipo de flujos de trabajo. Imagina planificar tareas en Windsurf, lanzar un equipo de Devons y revisar solicitudes de incorporación de cambios desde la comodidad de tu IDE. Las oportunidades para multiplicar por cien tu experiencia son infinitas. Y, por supuesto, volvemos a ser amigos de Anthropic. En Cognition, desarrollamos muchas de nuestras funciones, como DeepWiki y AskDevon, para mejorar la colaboración entre humanos y agentes. Incorporar el primer IDE de Agentic nos ayudará a llevar nuestra visión aún más lejos. Sin duda. Y créanlo o no, de todos los equipos de IA, el de Cognition era el que más respetábamos. Trabajar con el mejor equipo de ingeniería del sector supondrá un gran avance para nuestro producto y nuestro equipo de lanzamiento al mercado. Pero los mayores ganadores serán nuestros usuarios. El nuevo Cognition avanzará más rápido que nunca. Queremos redefinir la colaboración entre humanos y agentes. Y estamos deseando mostrarles lo que creamos."
text3 = "Hola, soy Scott, director ejecutivo de Cognition y soy Jeff, director ejecutivo de Windsurf."

text4 = "I'm excited to be here."

middle = get_memory_mb()

print(f"Middle: {middle:.1f}")

# Progress callback for testing
def my_progress_handler(stage, **kwargs):
    if stage == "preparing_conditionals":
        print(f"\n📂 Loading reference audio: {kwargs['audio_path']}")

    elif stage == "text_split":
        print(f"✂️  Split into {kwargs['total_chunks']} chunks")
        for i, word_count, preview in kwargs['chunk_previews']:
            print(f"   Chunk {i}: {word_count} words - '{preview}...'")

    elif stage == "chunk_start":
        print(f"\n[{kwargs['chunk_number']}/{kwargs['total_chunks']}] 🎙️  Generating ({kwargs['word_count']} words): {kwargs['text_preview']}...")

    elif stage == "chunk_complete":
        print(f"[{kwargs['chunk_number']}/{kwargs['total_chunks']}] ✓ Complete (audio shape: {kwargs['audio_shape']})")

    elif stage == "crossfading":
        print(f"\n🔀 Crossfading {kwargs['total_chunks']} chunks (overlap: {kwargs['overlap_duration']}s)...")

    elif stage == "complete":
        print(f"✅ Generation complete! Final audio shape: {kwargs['final_audio_shape']}\n")

# If you want to synthesize with a different voice, specify the audio prompt
# AUDIO_PROMPT_PATH = "earl-nightingale.wav"
AUDIO_PROMPT_PATH = "julia-whelan.wav"
# wav = model.generate(
#     text,
#     audio_prompt_path=AUDIO_PROMPT_PATH,
#     exaggeration=2.0,
#     cfg_weight=0.5
#     )
# Use generate_long for long text (text2 is 226 words)
# Use generate for short text (text3 is 14 words)
wav = multilingual_model.generate(
    text4,
    language_id="en",
    audio_prompt_path=AUDIO_PROMPT_PATH,
    exaggeration=0.2,
    cfg_weight=0.5,  # Use default CFG weight for better quality
    # repetition_penalty=1.2,  # Lower repetition penalty for more natural speech
)
after = get_memory_mb()

print(f"Gen: {before:.1f} → {after:.1f} MB (+{after - before:.1f})")
ta.save("test-2.wav", wav, multilingual_model.sr)
