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
    init_conditional_memory_cache,
    clear_output_directories,
    clear_cache_files
)

from loguru import logger

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MODEL = None
MULTILINGUAL = False
# Cache flags - defaults that can be overridden by skyrimnet_config.txt
ENABLE_DISK_CACHE = True
ENABLE_MEMORY_CACHE = True
_CONFIG_CACHE = None
_CONFIG_FILE_PATH = "skyrimnet_config.txt"

def load_skyrimnet_config():
    """Load configuration from skyrimnet_config.txt with error handling"""
    global _CONFIG_CACHE, ENABLE_MEMORY_CACHE, ENABLE_DISK_CACHE
    
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    
    # Default configuration
    default_config = {
        'temperature': 0.8,
        'min_p': 0.05, 
        'top_p': 1.0,
        'repetition_penalty': 2.0,
        'cfg_weight': 0.0,  # Speed optimized default
        'exaggeration': 0.55
    }
    
    global_flags = {
        'enable_memory_cache': ENABLE_MEMORY_CACHE,
        'enable_disk_cache': ENABLE_DISK_CACHE
    }
    
    config_mode = {
        'temperature': 'default',
        'min_p': 'default',
        'top_p': 'default', 
        'repetition_penalty': 'default',
        'cfg_weight': 'default',
        'exaggeration': 'default'
    }
    
    try:
        config_path = Path(_CONFIG_FILE_PATH)
        if not config_path.exists():
            logger.warning(f"Config file {_CONFIG_FILE_PATH} not found, using hardcoded defaults")
            _CONFIG_CACHE = (default_config, config_mode, global_flags)
            return _CONFIG_CACHE
            
        with open(config_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
                
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Handle global boolean flags
                if key in global_flags:
                    if value.lower() in ['true', 'yes', '1', 'on']:
                        global_flags[key] = True
                        # Update global variables
                        if key == 'enable_memory_cache':
                            ENABLE_MEMORY_CACHE = True
                        elif key == 'enable_disk_cache':
                            ENABLE_DISK_CACHE = True
                        logger.info(f"Setting {key} to True")
                    elif value.lower() in ['false', 'no', '0', 'off']:
                        global_flags[key] = False
                        # Update global variables
                        if key == 'enable_memory_cache':
                            ENABLE_MEMORY_CACHE = False
                        elif key == 'enable_disk_cache':
                            ENABLE_DISK_CACHE = False
                        logger.info(f"Setting {key} to False")
                    else:
                        logger.warning(f"Invalid boolean value '{value}' for {key}, using default")
                
                # Handle parameter modes
                elif key in config_mode:
                    if value.lower() == 'default':
                        config_mode[key] = 'default'
                    elif value.lower() == 'api':
                        config_mode[key] = 'api'
                    else:
                        try:
                            custom_value = float(value)
                            config_mode[key] = 'custom'
                            default_config[key] = custom_value
                            logger.info(f"Using custom {key} value: {custom_value}")
                        except ValueError:
                            logger.warning(f"Invalid value '{value}' for {key}, using default")
                            
        logger.info(f"Loaded config: {config_mode}")
        logger.info(f"Global flags: {global_flags}")
        _CONFIG_CACHE = (default_config, config_mode, global_flags)
        return _CONFIG_CACHE
        
    except Exception as e:
        logger.error(f"Error reading config file {_CONFIG_FILE_PATH}: {e}, using hardcoded defaults")
        _CONFIG_CACHE = (default_config, config_mode, global_flags)
        return _CONFIG_CACHE

def get_config_value(param_name, api_value, defaults, modes):
    """Get the appropriate value based on configuration mode"""
    mode = modes.get(param_name, 'default')
    
    if mode == 'api':
        return api_value if api_value is not None else defaults[param_name]
    else:  # 'default' or 'custom'
        return defaults[param_name]


def reload_config():
    """Force reload of configuration file"""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None
    return load_skyrimnet_config()

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

def generate(model, text,  language_id="en",audio_prompt_path=None, exaggeration=0.5, temperature=0.8, seed_num=0, cfgw=0, min_p=0.05, top_p=1.0, repetition_penalty=1.2,cache_uuid=0):

    logger.info(f"generate called for: \"{text}\", {Path(audio_prompt_path).stem if audio_prompt_path else "No ref audio"}, uuid: {cache_uuid}, exaggeration: {exaggeration}")  
    logger.info(f"Parameters - temp: {temperature}, min_p: {min_p}, top_p: {top_p}, rep_penalty: {repetition_penalty}, cfg_weight: {cfgw}")

    enable_memory_cache = ENABLE_MEMORY_CACHE
    enable_disk_cache = ENABLE_DISK_CACHE
    device = DEVICE
    dtype = DTYPE

    if model is None:
        model = load_model()

    if seed_num != 0:
        set_seed(int(seed_num))

    exaggeration = float(exaggeration)
    temperature = float(temperature)
    cfgw = float(cfgw)
    min_p = float(min_p)
    top_p = float(top_p)
    repetition_penalty = float(repetition_penalty)

    func_start_time = perf_counter_ns()

    if audio_prompt_path is not None:
        cache_key = get_cache_key(audio_prompt_path, cache_uuid, exaggeration)
        conditionals_loaded = False
        if cache_key and (enable_memory_cache or enable_disk_cache):
            if load_conditionals_cache(cache_key, model, device, dtype, enable_memory_cache, enable_disk_cache):
                conditionals_loaded = True
        if not conditionals_loaded:
            model.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
            if dtype != torch.float32:
                model.conds.t3.to(dtype=dtype)
            if cache_key and (enable_memory_cache or enable_disk_cache):
                save_conditionals_cache(cache_key, model.conds, enable_memory_cache, enable_disk_cache)
    conditional_start_time = perf_counter_ns()
    logger.info(f"Conditionals prepared. Time: {(conditional_start_time - func_start_time) / 1_000_000:.4f}ms")
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
    """Generate audio using configurable parameter system"""
    
    defaults, modes, flags = load_skyrimnet_config()
    
    # Map API parameters to our config system
    # Note: confidence maps to repetition_penalty in SkyrimNet UI
    api_temperature = linear if linear is not None else None
    api_min_p = min_p if min_p is not None else None
    api_top_p = top_p if top_p is not None else None  
    api_repetition_penalty = confidence if confidence is not None else None
    api_cfg_weight = cfg_scale if cfg_scale is not None else None
    api_exaggeration = quadratic if quadratic is not None else None
    
    final_temperature = get_config_value('temperature', api_temperature, defaults, modes)
    final_min_p = get_config_value('min_p', api_min_p, defaults, modes)
    final_top_p = get_config_value('top_p', api_top_p, defaults, modes)
    final_repetition_penalty = get_config_value('repetition_penalty', api_repetition_penalty, defaults, modes)
    final_cfg_weight = get_config_value('cfg_weight', api_cfg_weight, defaults, modes)
    final_exaggeration = get_config_value('exaggeration', api_exaggeration, defaults, modes)
    
    logger.debug(f"Final parameters - temp: {final_temperature}, min_p: {final_min_p}, top_p: {final_top_p}, rep_penalty: {final_repetition_penalty}, cfg_weight: {final_cfg_weight}, exaggeration: {final_exaggeration}")
    
    seed_num = cpp_uuid_to_seed(uuid)
    return generate(
        model=MODEL, 
        text=text, 
        language_id=language, 
        audio_prompt_path=speaker_audio, 
        seed_num=seed_num, 
        cache_uuid=uuid,
        exaggeration=final_exaggeration,
        temperature=final_temperature,
        cfgw=final_cfg_weight,
        min_p=final_min_p,
        top_p=final_top_p,
        repetition_penalty=final_repetition_penalty
    ), uuid

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
            cfg_weight = gr.Slider(0.0, 1, step=.05, label="CFG/Pace", value=0.0)

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="temperature", value=.8)
                min_p = gr.Slider(0.00, 1.00, step=0.01, label="min_p || Newer Sampler. Recommend 0.02 > 0.1. Handles Higher Temperatures better. 0.00 Disables", value=0.05)
                top_p = gr.Slider(0.00, 1.00, step=0.01, label="top_p || Original Sampler. 1.0 Disables(recommended). Original 0.8", value=1.00)
                repetition_penalty = gr.Slider(1.00, 2.00, step=0.1, label="repetition_penalty", value=2.0)
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
    
    # Load configuration at startup
    logger.info("Loading SkyrimNet configuration...")
    load_skyrimnet_config()
    
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
