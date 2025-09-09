import functools
import gradio as gr
import numpy as np
import random
import torch

from argparse import ArgumentParser
from pathlib import Path
from xml.parsers.expat import model
from time import  perf_counter_ns
from src.cache_utils import (
    load_conditionals_cache,
    save_conditionals_cache,
    get_cache_key,
    save_torchaudio_wav,
    init_conditional_memory_cache
)

from loguru import logger
import warnings
warnings.filterwarnings("ignore", message=" UserWarning: In 2.9, this function's implementation will be changed to use torchaudio.save_with_torchcodec` under the hood.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MODEL = None
MULTILINGUAL = False

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_model():
    global MODEL, MULTILINGUAL
    if MODEL is None:
        if MULTILINGUAL:
            logger.info("Loading Multilingual Model")
            from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS as Chatterbox
        else:
            logger.info("Loading English Model")
            from src.chatterbox.tts import ChatterboxTTS as Chatterbox
        MODEL = Chatterbox.from_pretrained(DEVICE)
        MODEL.t3.to(dtype=DTYPE)
        MODEL.conds.t3.to(dtype=DTYPE)
        torch.cuda.empty_cache()
    return MODEL

def generate(model, text,  language_id="en",audio_prompt_path=None, exaggeration=0.5, temperature=0.8, seed_num=0, cfgw=0, cache_uuid=0):

    logger.info(f"generate called for: \"{text}\", {Path(audio_prompt_path).stem if audio_prompt_path else "No ref audio"}, uuid: {cache_uuid}, exaggeration: {exaggeration}")  

    enable_memory_cache = True
    enable_disk_cache = True
    cache_voice = True
    device = DEVICE
    dtype = DTYPE

    if model is None:
        model = load_model()

    if seed_num != 0:
        set_seed(int(seed_num))

    func_start_time = perf_counter_ns()

    # Enhanced conditional preparation with configurable caching
    if audio_prompt_path is not None:
        # Generate cache key
        cache_key = get_cache_key(audio_prompt_path, cache_uuid, exaggeration)
        conditionals_loaded = False
        # Try to load from cache first (respecting cache flags)
        if cache_key and (enable_memory_cache or enable_disk_cache):
            if load_conditionals_cache(cache_key, model, device, dtype, enable_memory_cache, enable_disk_cache):
                conditionals_loaded = True
        # If not loaded from cache, prepare and optionally cache
        if not conditionals_loaded:
            model.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
            if dtype != torch.float32:
                model.conds.t3.to(dtype=dtype)
            # Save to cache if we have a cache key and caching is enabled
            if cache_key and (enable_memory_cache or enable_disk_cache):
                save_conditionals_cache(cache_key, model.conds, enable_memory_cache, enable_disk_cache)
        # Update in-memory cache tracking
        if cache_voice:
            model._cached_prompt_path = audio_prompt_path
    conditional_start_time = perf_counter_ns()
    logger.info(f"Conditionals prepared. Time: {(conditional_start_time - func_start_time) / 1_000_000:.4f}ms")
    #generate_start_time = perf_counter_ns()
    wav = model.generate(
        text,
        #audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        temperature=temperature,
        cfg_weight=cfgw,
        t3_params={
            #"initial_forward_pass_backend": "eager", # slower - default
            #"initial_forward_pass_backend": "cudagraphs", # speeds up set up
            "generate_token_backend": "cudagraphs-manual", # fastest - default
            # "generate_token_backend": "cudagraphs",
            # "generate_token_backend": "eager",
            # "generate_token_backend": "inductor",
            # "generate_token_backend": "inductor-strided",
            #"generate_token_backend": "cudagraphs-strided",
            "stride_length": 4, # "strided" options compile <1-2-3-4> iteration steps together, which improves performance by reducing memory copying issues in torch.compile
            "skip_when_1": True, # skips Top P when it's set to 1.0
            #"benchmark_t3": True, # Synchronizes CUDA to get the real it/s 
        }
    )
    #logger.info(f"Generation completed. Time: {(perf_counter_ns() - generate_start_time) / 1_000_000_000:.2f}s")
    # Log execution time
    func_end_time = perf_counter_ns()

    total_duration_s = (func_end_time - func_start_time)  / 1_000_000_000  # Convert nanoseconds to seconds
    wav_length = wav.shape[-1]   / model.sr

    logger.info(f"Generated audio: {wav_length:.2f}s {model.sr/1000:.2f}kHz in {total_duration_s:.2f}s. Speed: {wav_length / total_duration_s:.2f}x")
    wave_file = str(save_torchaudio_wav(wav.cpu(), model.sr, audio_path=audio_prompt_path, uuid=cache_uuid))
    del wav
    torch.cuda.empty_cache()
    return wave_file

    #return (model.sr, wav.squeeze(0).cpu().numpy())

### SkyrimNet Zonos Emulated
@functools.cache
def cpp_uuid_to_seed(uuid_64: int) -> int:
    return (abs(uuid_64) % (2 ** 31 - 1)) #+ 1
    
def generate_audio(
    model_choice = None,
    text= "On that first day from Saturalia, My missus gave for me, A big bowl of moon sugar!",
    language= "en",
    speaker_audio= None,
    prefix_audio= None,
    e1= None,
    e2= None,
    e3= None,
    e4= None,
    e5= None,
    e6= None,
    e7= None,
    e8= None,
    vq_single= None,
    fmax= None,
    pitch_std= None,
    speaking_rate= None,
    dnsmos_ovrl= None,
    speaker_noised: bool = None,
    cfg_scale= 0.3,
    top_p= 1.0,
    top_k= None,
    min_p= 0.5,
    linear= None,
    confidence= None,
    quadratic= None,
    uuid= -1,
    randomize_seed: bool = False,
    unconditional_keys: list = None,
    chunked=False,
):
    #logger.info(f"generate_audio called for: {Path(speaker_audio).stem } with {text}, uuid: {uuid}, exaggeration: {0.55}")  

    #seed_num = 0 if randomize_seed else cpp_uuid_to_seed(uuid)
    seed_num = cpp_uuid_to_seed(uuid)
    return generate(model=MODEL, text=text, language_id=language, audio_prompt_path=speaker_audio, seed_num=seed_num, cache_uuid=uuid,
                    exaggeration=0.55,
                    temperature=0.9,
                    cfgw=0
                    ), uuid

with gr.Blocks() as demo:
    model_state = gr.State(None)  # Loaded once per session/user

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
                label="Text to synthesize (max chars 300)",
                max_lines=5
            )
            ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File", value=None)

            exaggeration = gr.Slider(0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=.5)
            cfg_weight = gr.Slider(0.0, 1, step=.05, label="CFG/Pace", value=0.5)

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="temperature", value=.8)
            language_id = gr.Dropdown([
                "ar",
                "da",
                "de",
                "el",
                "en",
                "es",
                "fi",
                "fr",
                "he",
                "hi",
                "it",
                "ja",
                "ko",
                "ms",
                "nl",
                "no",
                "pl",
                "pt",
                "ru",
                "sv",
                "sw",
                "tr",
                "zh"], value="en", multiselect=False, label="Language", info="Language only for multilanguage model")
            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio", type="filepath", autoplay=True)

    demo.load(fn=load_model, inputs=[], outputs=model_state)

    run_btn.click(
        fn=generate,
        inputs=[
            model_state,
            text,
            language_id,
            ref_wav,
            exaggeration,
            temp,
            seed_num,
            cfg_weight,
        ],
        outputs=audio_output,
    )
    model_choice = gr.Textbox(visible=False)
    language = gr.Textbox(visible=False)
    speaker_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File", value=None, visible=False)
    prefix_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File", value=None, visible=False)
    emotion1 = gr.Number(visible=False)
    emotion2 = gr.Number(visible=False)
    emotion3 = gr.Number(visible=False)
    emotion4 = gr.Number(visible=False)
    emotion5 = gr.Number(visible=False)
    emotion6 = gr.Number(visible=False)
    emotion7 = gr.Number(visible=False)
    emotion8 = gr.Number(visible=False)
    vq_single = gr.Number(visible=False)
    fmax = gr.Number(visible=False)
    pitch_std = gr.Number(visible=False)
    speaking_rate = gr.Number(visible=False)
    dnsmos = gr.Number(visible=False)
    speaker_noised_checkbox = gr.Checkbox(visible=False)
    cfg_scale = gr.Number(visible=False)
    top_p = gr.Number(visible=False)
    min_k = gr.Number(visible=False)
    min_p = gr.Number(visible=False)
    linear = gr.Number(visible=False)
    confidence = gr.Number(visible=False)
    quadratic = gr.Number(visible=False)
    randomize_seed_toggle = gr.Checkbox(visible=False)
    unconditional_keys = gr.Textbox(visible=False)
    hidden_btn = gr.Button(visible=False)
    hidden_btn.click(fn=generate_audio, api_name="generate_audio", inputs=[
        model_choice,
        text,
        language,
        speaker_audio,
        prefix_audio,
        emotion1,
        emotion2,
        emotion3,
        emotion4,
        emotion5,
        emotion6,
        emotion7,
        emotion8,
        vq_single,
        fmax,
        pitch_std,
        speaking_rate,
        dnsmos,
        speaker_noised_checkbox,
        cfg_scale,
        top_p,
        min_k,
        min_p,
        linear,
        confidence,
        quadratic,
        seed_num,
        randomize_seed_toggle,
        unconditional_keys,
    ],
                     outputs=[audio_output, seed_num],
                     )
    
def parse_arguments():
    """Parse command line arguments"""
    parser = ArgumentParser()
    parser.add_argument('--share', action='store_true')
    parser.add_argument("--server", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=int, required=False)
    parser.add_argument("--inbrowser", action='store_true')
    parser.add_argument("--multilingual", action='store_true', default=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    MULTILINGUAL = args.multilingual
    model = load_model()
    init_conditional_memory_cache(model, DEVICE, DTYPE)
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(
        server_name=args.server, 
        server_port=args.port, 
        share=args.share, 
        inbrowser=args.inbrowser
    )
