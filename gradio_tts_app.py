from pathlib import Path

script_directory = Path(__file__).parent
import sys

sys.path.append(str(Path(script_directory).joinpath("src")))
import random
import numpy as np
import torch
import gradio as gr
import functools
from src.api import (tts, get_model, resolve_dtype, resolve_device)
from src.tts_webui.decorators import decorator_add_model_type

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "bfloat16" if torch.cuda.is_available() else "float32"


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_model():
    return get_model(model_name="chatterbox",
              device=resolve_device(DEVICE), dtype=resolve_dtype(DTYPE))

@functools.wraps(tts)
@decorator_add_model_type("chatterbox")
def tts_decorated(*args, _type=None, **kwargs):
    return tts(*args, **kwargs)

### SkyrimNet Zonos Emulated
def generate_audio(
    model_choice: str,
    text: str,
    language: str,
    speaker_audio: str,
    prefix_audio: str,
    e1: float,
    e2: float,
    e3: float,
    e4: float,
    e5: float,
    e6: float,
    e7: float,
    e8: float,
    vq_single: float,
    fmax: float,
    pitch_std: float,
    speaking_rate: float,
    dnsmos_ovrl: float,
    speaker_noised: bool,
    cfg_scale: float,
    top_p: float,
    top_k: float,
    min_p: float,
    linear: float,
    confidence: float,
    quadratic: float,
    seed: int,
    randomize_seed: bool,
    unconditional_keys: list,

):
    print(    model_choice,
    text,
    language,
    speaker_audio,
    prefix_audio,
    e1,
    e2,
    e3,
    e4,
    e5,
    e6,
    e7,
    e8,
    vq_single,
    fmax,
    pitch_std,
    speaking_rate,
    dnsmos_ovrl,
    speaker_noised,
    cfg_scale,
    top_p,
    top_k,
    min_p,
    linear,
    confidence,
    quadratic,
    seed,
    randomize_seed,
    unconditional_keys,
    )
    """
      Generates audio based on the provided UI parameters.
      We do NOT use language_id or ctc_loss even if the model has them.
      """
    #speaker_audio_path = ast.literal_eval(speaker_audio)["path"] if speaker_audio else None
    #print("Speaker audio path:", speaker_audio)


    l_repetition_penalty = 1.2
    l_exaggeration = 0.8


    #if randomize_seed:
    #    seed = torch.randint(0, 2**32 - 1, (1,)).item()
    #torch.manual_seed(seed)

    seed_num=seed >> 64 if randomize_seed is False else 0
    return [tts_decorated(
        model_state=model_state,
        text=text,
        audio_prompt_path=speaker_audio,
        exaggeration=l_exaggeration,
        temperature=0.9,
        seed_num=-seed_num,
        cfgw=cfg_scale,
        min_p=min_p,
        top_p=top_p,
        repetition_penalty=l_repetition_penalty,
        device=DEVICE,
        dtype=DTYPE,
        chunked=True,
        cache_voice=True,
        max_cache_len=700,
        max_new_tokens=500,
        use_compilation=True,
    ) , seed]

####

def generate(model_state, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw, min_p, top_p, repetition_penalty):
    # if seed_num != 0:
    #    set_seed(int(seed_num))

    return tts_decorated(model_state, text, exaggeration=exaggeration, temperature=temperature, cfg_weight=cfgw, min_p=min_p,
        top_p=top_p, repetition_penalty=repetition_penalty, audio_prompt_path=audio_prompt_path, # model
        model_name="just_a_placeholder", device=DEVICE, dtype=DTYPE, cpu_offload=False, # hyperparameters
        chunked=True, cache_voice=True, # streaming
        tokens_per_slice=1000, remove_milliseconds=100, remove_milliseconds_start=100, chunk_overlap_method="zero",
        # chunks
        desired_length=200, max_length=300, halve_first_chunk=False, seed=-1,  # for signature compatibility
        progress=gr.Progress(), streaming=False, # progress=gr.Progress(track_tqdm=True),
        use_compilation=True, max_new_tokens=400, max_cache_len=600,  # Affects the T3 speed, hence important
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
                                     label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=.5)
            cfg_weight = gr.Slider(0.0, 1, step=.05, label="CFG/Pace", value=0.5)

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="temperature", value=.8)
                min_p = gr.Slider(0.00, 1.00, step=0.01,
                                  label="min_p || Newer Sampler. Recommend 0.02 > 0.1. Handles Higher Temperatures better. 0.00 Disables",
                                  value=0.05)
                top_p = gr.Slider(0.00, 1.00, step=0.01,
                                  label="top_p || Original Sampler. 1.0 Disables(recommended). Original 0.8",
                                  value=1.00)
                repetition_penalty = gr.Slider(1.00, 2.00, step=0.1, label="repetition_penalty", value=1.2)

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
    demo.load(fn=load_model, inputs=[], outputs=model_state)

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
        ],
        outputs=audio_output,
    )
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

    #print("\nWarming up model...\n")
    #warmup_output = tts_decorated(model_state=model_state,device="cuda",dtype=DTYPE,text="Warm up", speaker_audio=Path(script_directory).joinpath("assets","dlc1seranavoice.wav"),use_compilation=True)

    gr.set_static_paths(paths=[Path.cwd().absolute()/"assets"])
    gr.cache_examples=True
    demo.queue(
        max_size=50,
        default_concurrency_limit=5,
    ).launch(
        server_name=args.server,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser
    )