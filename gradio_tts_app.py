from pathlib import Path

from src.simple_model_state import simple_manage_model_state

script_directory = Path(__file__).parent
import sys

sys.path.append(str(Path(script_directory).joinpath("src")))
import random
import numpy as np
import torch
import gradio as gr
import functools
from src.api import (tts, get_model, resolve_dtype, resolve_device)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "bfloat16"

@functools.cache
def cpp_uuid_to_seed(uuid_64: int) -> int:
    return (abs(uuid_64) % (2 ** 31 - 1)) + 1

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

@simple_manage_model_state("chatterbox")
def load_model():
    return get_model(model_name="chatterbox",
              device=resolve_device(DEVICE), dtype=resolve_dtype(DTYPE))

@functools.wraps(tts)
def tts_decorated(*args, _type=None, **kwargs):
    return tts(*args, **kwargs)

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
    uuid= -1,
    randomize_seed: bool = False,
    unconditional_keys: list = None,
):

    """
    Generates audio based on the provided UI parameters.
    We do NOT use language_id or ctc_loss even if the model has them.
    """

    # Handle C++ UUID seed format
    seed = 0 if randomize_seed else cpp_uuid_to_seed(uuid)

    # Direct return without intermediate variables, now with UUID for caching
    return [generate(
        model_state=model_state,
        text=text,
        audio_prompt_path=speaker_audio,
        seed_num=seed,
        cfgw=0.3,
        min_p=0.5,
        top_p=1.0,
        repetition_penalty=1.9,
        cache_uuid=uuid,  # Pass UUID for disk caching
        do_progress=False,
    ), uuid]

####

def generate(model_state, text, audio_prompt_path,
             exaggeration=0.78,
             temperature=0.9,
             seed_num=0,
             cfgw=0.25,
             min_p=0.18,
             top_p=1.0,
             repetition_penalty=1.9,
             device=DEVICE,
             dtype=DTYPE,
             chunked=True,
             cache_voice=True,
             max_cache_len=450,
             max_new_tokens=300,
             use_compilation=True,
             cache_uuid=-1,
             do_progress=False,
             ):
    print(f"""Generate using inputs: model_state = {model_state}, text = {text}, audio_prompt_path = {audio_prompt_path},
    exaggeration = {exaggeration}, 
    temperature = {temperature}, 
    seed_num = {seed_num}, 
    cfgw = {cfgw}, 
    min_p = {min_p}, 
    top_p = {top_p}, 
    repetition_penalty = {repetition_penalty}, 
    device = {device}, 
    dtype = {dtype}, 
    chunked = {chunked}, 
    cache_voice = {cache_voice}, 
    max_cache_len = {max_cache_len}, 
    max_new_tokens = {max_new_tokens}, 
    use_compilation = {use_compilation}, 
    cache_uuid = {cache_uuid}
    """)


    if seed_num == 0:
        seed_num = torch.randint(0, 2**31 - 1, (1,)).item()
    set_seed(seed_num)

    return tts_decorated(
        model_state=model_state,
        text=text,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        temperature=temperature,
        seed_num=seed_num,
        cfgw=cfgw,
        min_p=min_p,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        device=device,
        dtype=dtype,
        chunked=chunked,
        cache_voice=cache_voice,
        max_cache_len=max_cache_len,
        max_new_tokens=max_new_tokens,
        use_compilation=use_compilation,
        cache_uuid=cache_uuid,
        do_progress=do_progress,
    )


with gr.Blocks() as demo:
    delete_cache = (86400, 86400)
    model_state = gr.State(None)  # Loaded once per session/user

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
                label="Text to synthesize (max chars 300)", max_lines=5)
            ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File",
                               value=None)
            exaggeration = gr.Slider(0.25, 2, step=.05,
                                     label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=.7)
            cfg_weight = gr.Slider(0.0, 1, step=.05, label="CFG/Pace", value=0.3)
            #max_new_tokens = gr.Slider(100, 5000, step=25, label="max_new_tokens", value=500)
            #max_cache_len = gr.Slider(100, 5000, step=25, label="max_cache_len", value=1000)

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="temperature", value=.9)
                min_p = gr.Slider(0.00, 1.00, step=0.01,
                                  label="min_p || Newer Sampler. Recommend 0.02 > 0.1. Handles Higher Temperatures better. 0.00 Disables",
                                  value=0.05)
                top_p = gr.Slider(0.00, 1.00, step=0.01,
                                  label="top_p || Original Sampler. 1.0 Disables(recommended). Original 0.8",
                                  value=1.00)
                repetition_penalty = gr.Slider(1.00, 2.00, step=0.1, label="repetition_penalty", value=1.9)

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")


    run_btn.click(
       fn=generate,
       inputs=[
           model_state,
           text,
           ref_wav,
           exaggeration,
           temp,
           seed_num,
           cfg_weight,
           min_p,
           top_p,
           repetition_penalty,
           #max_cache_len,
           #max_new_tokens,
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
    min_k = gr.Number(visible=False)
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
    #demo.load(fn=load_model, inputs=[], outputs=model_state)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
parser.add_argument("--output_dir", type=str, default='./outputs')
args = parser.parse_args()

if __name__ == "__main__":
    if "demo" in locals():
        locals()["demo"].close()

    print("\nWarming up model...\n")
    warmup_output = generate_audio(speaker_audio=Path(script_directory).joinpath("assets","fishaudio_horror.wav"))

    gr.set_static_paths(paths=[Path.cwd().absolute()/"assets"])
    gr.cache_examples=True
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(
        server_name=args.server,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser
    )