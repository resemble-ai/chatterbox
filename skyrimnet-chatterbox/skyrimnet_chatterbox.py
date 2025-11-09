from argparse import ArgumentParser
from pathlib import Path
import random
from time import perf_counter_ns

import gradio as gr
from loguru import logger
import numpy as np
import torch

try:
    from cache_utils import (
        load_conditionals_cache,
        save_conditionals_cache,
        get_cache_key,
        save_torchaudio_wav,
        init_conditional_memory_cache,
        clear_output_directories,
        clear_cache_files
    )
    from chatterbox.shared_utils import validate_language_id
    from chatterbox.tensor_utils import initialize_model_dtype, safe_conditional_to_dtype
    from shared_config import get_tts_params, DEFAULT_CACHE_CONFIG, DEFAULT_TTS_PARAMS, SUPPORTED_LANGUAGE_CODES
except ImportError:
    from .cache_utils import (
        load_conditionals_cache,
        save_conditionals_cache,
        get_cache_key,
        save_torchaudio_wav,
        init_conditional_memory_cache,
        clear_output_directories,
        clear_cache_files
    )
    from .chatterbox.shared_utils import validate_language_id
    from .chatterbox.tensor_utils import initialize_model_dtype, safe_conditional_to_dtype
    from .shared_config import get_tts_params, DEFAULT_CACHE_CONFIG, DEFAULT_TTS_PARAMS, SUPPORTED_LANGUAGE_CODES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MODEL = None
MULTILINGUAL = False
IGNORE_PING = None
SILENCE_AUDIO_PATH = "assets/silence_100ms.wav"
# Cache flags - defaults that can be overridden by skyrimnet_config.txt
ENABLE_DISK_CACHE = DEFAULT_CACHE_CONFIG["ENABLE_DISK_CACHE"]
ENABLE_MEMORY_CACHE = DEFAULT_CACHE_CONFIG["ENABLE_MEMORY_CACHE"]
# Testing flag - when True, bypasses config loading and uses all API values
_FROM_GRADIO = False

def set_seed(seed: int):
    """Sets the random seed for reproducibility across torch, numpy, and random."""
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_model():
    global MODEL, MULTILINGUAL
    if MODEL is None:
        if MULTILINGUAL:
            logger.info("Loading Multilingual Model")
            try:
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS as Chatterbox
            except ImportError:
                from .chatterbox.mtl_tts import ChatterboxMultilingualTTS as Chatterbox
        else:
            logger.info("Loading English Model")
            try:
                from chatterbox.tts import ChatterboxTTS as Chatterbox
            except ImportError:
                from .chatterbox.tts import ChatterboxTTS as Chatterbox
        MODEL = Chatterbox.from_pretrained(DEVICE)
        initialize_model_dtype(MODEL, DTYPE)
        init_conditional_memory_cache(MODEL, DEVICE, DTYPE, supported_languages=SUPPORTED_LANGUAGE_CODES)
        torch.cuda.empty_cache()
    return MODEL


def generate(model, text,  language_id="en",audio_prompt_path=None, exaggeration=0.5, temperature=0.8, seed_num=0, cfgw=0, min_p=0.05, top_p=1.0, repetition_penalty=1.2):
    global MODEL, MULTILINGUAL
    
    language_id = validate_language_id(language_id, SUPPORTED_LANGUAGE_CODES)

    
    logger.info(f'generate called for: "{text}", {Path(audio_prompt_path).stem if audio_prompt_path else "No ref audio"}, exaggeration: {exaggeration}')  
    logger.info(f"Parameters - temp: {temperature}, min_p: {min_p}, top_p: {top_p}, rep_penalty: {repetition_penalty}, cfg_weight: {cfgw}")

    enable_memory_cache = ENABLE_MEMORY_CACHE
    enable_disk_cache = ENABLE_DISK_CACHE
    device = DEVICE
    dtype = DTYPE

    # Check if we need to switch to multilingual model
    if not language_id.startswith("en") and not MULTILINGUAL:
        logger.info(f"Non-English language '{language_id}' detected, switching to multilingual model")
        MULTILINGUAL = True
        MODEL = None
    
    if MODEL is None or model is None:
        model = load_model()


    exaggeration = float(exaggeration)
    temperature = float(temperature)
    cfgw = float(cfgw)
    min_p = float(min_p)
    top_p = float(top_p)
    repetition_penalty = float(repetition_penalty)

    func_start_time = perf_counter_ns()

    if audio_prompt_path is not None:
        cache_key = get_cache_key(audio_prompt_path, exaggeration)
        conditionals_loaded = False
        if cache_key and (enable_memory_cache or enable_disk_cache):
            if load_conditionals_cache(language_id, cache_key, model, device=device, dtype=dtype, enable_memory_cache=enable_memory_cache, enable_disk_cache=enable_disk_cache):
                conditionals_loaded = True
        if not conditionals_loaded:
            model.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
            if dtype != torch.float32:
                safe_conditional_to_dtype(model, dtype)
            if cache_key and (enable_memory_cache or enable_disk_cache):
                save_conditionals_cache(language_id, cache_key, model.conds, enable_memory_cache, enable_disk_cache)
    #conditional_start_time = perf_counter_ns()
    #logger.info(f"Conditionals prepared. Time: {(conditional_start_time - func_start_time) / 1_000_000:.4f}ms")
    #generate_start_time = perf_counter_ns()
    
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
    generate_args={
        "text": text,
        "exaggeration": exaggeration,
        "temperature": temperature,
        "cfg_weight": cfgw,
        "min_p": min_p,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "t3_params": t3_params,
    }
    if MULTILINGUAL:
        generate_args["language_id"] = language_id

    wav = model.generate(
        **generate_args
    )

    #logger.info(f"Generation completed. Time: {(perf_counter_ns() - generate_start_time) / 1_000_000_000:.2f}s")
    # Log execution time
    func_end_time = perf_counter_ns()

    total_duration_s = (func_end_time - func_start_time)  / 1_000_000_000  # Convert nanoseconds to seconds
    wav_length = wav.shape[-1]   / model.sr

    logger.info(f"Generated audio: {wav_length:.2f}s {model.sr/1000:.2f}kHz in {total_duration_s:.2f}s. Speed: {wav_length / total_duration_s:.2f}x")
    wave_file = str(save_torchaudio_wav(wav.cpu(), model.sr, audio_path=audio_prompt_path))
    del wav
    torch.cuda.empty_cache()
    return wave_file

    #return (model.sr, wav.squeeze(0).cpu().numpy())


### SkyrimNet Zonos Emulated   

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
    job_id= -1,
    randomize_seed: bool = False,
    unconditional_keys: list = None
    ):
    """Generate audio using configurable parameter system"""
    global IGNORE_PING

    if isinstance(speaker_audio, dict) and 'path' in speaker_audio:
        speaker_audio = speaker_audio['path']
    logger.info(f"inputs: text={text}, language={language}, speaker_audio={Path(speaker_audio).stem if speaker_audio else 'None'}, seed={job_id}")

    if text == "ping":
       if IGNORE_PING is None:
          IGNORE_PING = "pending"
       else:
          logger.info("Ping request received, sending silence audio.")
          return SILENCE_AUDIO_PATH, job_id

    # Build payload with API values (map SkyrimNet UI names to our parameter names)
    # Note: linear->temperature, confidence->repetition_penalty, quadratic->exaggeration, cfg_scale->cfg_weight
    payload_params = {
        'temperature': linear,
        'min_p': min_p,
        'top_p': top_p,
        'repetition_penalty': confidence,
        'cfg_weight': cfg_scale,
        'exaggeration': quadratic
    }
    
    # Use shared config function to resolve final parameters
    # override_flag=True when from Gradio web UI
    inference_kwargs = get_tts_params(payload_params=payload_params, override_flag=_FROM_GRADIO)
    
    logger.debug(f"Final parameters - temp: {inference_kwargs['temperature']}, min_p: {inference_kwargs['min_p']}, top_p: {inference_kwargs['top_p']}, rep_penalty: {inference_kwargs['repetition_penalty']}, cfg_weight: {inference_kwargs['cfg_weight']}, exaggeration: {inference_kwargs['exaggeration']}")
    

    return generate(
        model=MODEL, 
        text=text, 
        language_id=language, 
        audio_prompt_path=speaker_audio, 
        seed_num=job_id, 
        exaggeration=inference_kwargs['exaggeration'],
        temperature=inference_kwargs['temperature'],
        cfgw=inference_kwargs['cfg_weight'],
        min_p=inference_kwargs['min_p'],
        top_p=inference_kwargs['top_p'],
        repetition_penalty=inference_kwargs['repetition_penalty'],
    ), job_id

with gr.Blocks() as demo:
    model_state = gr.State(None)  # Loaded once per session/user

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
                label="Text to synthesize",
                lines=5,
            )
            ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File", value=None)

            exaggeration = gr.Slider(0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=0.55)
            cfg_weight = gr.Slider(0.0, 1, step=.05, label="CFG/Pace", value=DEFAULT_TTS_PARAMS['CFG_WEIGHT'])

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=20250527, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="temperature", value=DEFAULT_TTS_PARAMS['TEMPERATURE'])
                min_p = gr.Slider(0.00, 1.00, step=0.01, label="min_p || Newer Sampler. Recommend 0.02 > 0.1. Handles Higher Temperatures better. 0.00 Disables", value=DEFAULT_TTS_PARAMS['MIN_P'])
                top_p = gr.Slider(0.00, 1.00, step=0.01, label="top_p || Original Sampler. 1.0 Disables(recommended). Original 0.8", value=DEFAULT_TTS_PARAMS['TOP_P'])
                repetition_penalty = gr.Slider(1.00, 2.00, step=0.1, label="repetition_penalty", value=DEFAULT_TTS_PARAMS['REPETITION_PENALTY'])
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
            min_p,
            top_p,
            repetition_penalty,
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
    parser.add_argument('--share', action='store_true',help="Create a EXTERNAL facing public link using Gradio's servers")
    parser.add_argument("--server", type=str, default='0.0.0.0', help="Server address to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, required=False, default=7860, help="Port to run the server on (default: 7860)")
    parser.add_argument("--inbrowser", action='store_true', help="Open the UI in a new browser window")
    parser.add_argument("--multilingual", action='store_true', default=False, help="Use the multilingual model (requires more VRAM)")
    parser.add_argument("--clearoutput", action='store_true', help="Remove all folders in audio output directory and exit")
    parser.add_argument("--clearcache", action='store_true', help="Remove all .pt cache files and exit")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Handle cleanup arguments that exit immediately
    if args.clearoutput:
        logger.info("Clearing output directories...")
        count = clear_output_directories()
        logger.info(f"Cleared {count} output directories. Exiting.")
        exit(0)
    
    if args.clearcache:
        logger.info("Clearing cache files...")
        count = clear_cache_files()
        logger.info(f"Cleared {count} cache files. Exiting.")
        exit(0)
    
    MULTILINGUAL = args.multilingual
    set_seed(20250527)
    model = load_model()
    
    # Determine supported languages based on model type

    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(
        server_name=args.server, 
        server_port=args.port, 
        share=args.share, 
        inbrowser=args.inbrowser
    )
