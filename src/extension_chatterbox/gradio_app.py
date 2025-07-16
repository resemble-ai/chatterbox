import functools
import gradio as gr
from gradio_iconbutton import IconButton

from tts_webui.decorators import *
from tts_webui.decorators.decorator_save_wav import (
    decorator_save_wav_generator_accumulated,
)
from tts_webui.extensions_loader.decorator_extensions import (
    decorator_extension_inner,
    decorator_extension_outer,
    decorator_extension_inner_generator,
    decorator_extension_outer_generator,
)
from tts_webui.utils.list_dir_models import unload_model_button
from tts_webui.utils.randomize_seed import randomize_seed_ui
from tts_webui.utils.OpenFolderButton import OpenFolderButton
from tts_webui.utils.get_path_from_root import get_path_from_root

from src.api import (
    move_model_to_device_and_dtype,
    tts,
    tts_stream,
    interrupt,
    get_voices,
    vc,
    compile_t3,
    remove_t3_compilation,
    get_current_model,
)
from .memory import get_chatterbox_memory_usage


@functools.wraps(tts)
@decorator_extension_outer
@decorator_apply_torch_seed
@decorator_save_metadata
@decorator_save_wav
@decorator_add_model_type("chatterbox")
@decorator_add_base_filename
@decorator_add_date
@decorator_log_generation
@decorator_extension_inner
@log_function_time
def tts_decorated(*args, _type=None, **kwargs):
    return tts(*args, **kwargs)


@functools.wraps(tts_stream)
# @decorator_convert_audio_output_generator  # <-- This goes first/top
@decorator_extension_outer_generator
@decorator_apply_torch_seed_generator
@decorator_save_metadata_generator
@decorator_save_wav_generator_accumulated
@decorator_add_model_type_generator("chatterbox")
@decorator_add_base_filename_generator_accumulated
@decorator_add_date_generator
@decorator_log_generation_generator
@decorator_extension_inner_generator
@log_generator_time
def tts_generator_decorated(*args, **kwargs):
    yield from tts_stream(*args, **kwargs)


def ui():
    gr.HTML(
        """
  <h2 style="text-align: center;">Chatterbox TTS</h2>

  <div style="display: flex; flex-wrap: wrap; gap: 20px;">
    <div style="flex: 1; min-width: 300px;">
      <h2>Model sizes:</h2>
      <ul style="list-style-type: disc; padding-left: 20px; margin-top: 0;">
        <li><strong>t3_cfg.safetensors</strong> 2.13 GB</li>
        <li><strong>s3gen.safetensors</strong> 1.06 GB</li>
      </ul>
    </div>

    <div style="flex: 1; min-width: 300px;">
      <h2>Performance on NVIDIA RTX 3090</h2>
      <ul style="list-style-type: disc; padding-left: 20px;">
        <li><strong>VRAM</strong>: Float32: 5-7 GB, Bfloat16: 3-4 GB, CPU Offloading Passive: 0.7 GB of VRAM</li>
        <li><strong>Speed</strong>: ~32.04 iterations per second, 1:1 ratio</li>
      </ul>
    </div>
  </div>
                """
    )
    with gr.Tabs():
        with gr.Tab("TTS"):
            with gr.Row():
                chatterbox_tts()
        with gr.Tab("Voice Conversion"):
            with gr.Row():
                chatterbox_vc()


def chatterbox_tts():
    with gr.Column():
        text = gr.Textbox(label="Text", lines=3)
        with gr.Row():
            btn_interrupt = gr.Button("Interrupt next chunk", interactive=False)
            btn_stream = gr.Button("Streaming generation", variant="secondary")
            btn = gr.Button("Generate", variant="primary")
        btn_interrupt.click(
            fn=lambda: gr.Button("Interrupting..."),
            outputs=[btn_interrupt],
        ).then(fn=interrupt, outputs=[btn_interrupt])

        with gr.Row():
            voice_dropdown = gr.Dropdown(
                label="Saved voices", choices=["refresh to load the voices"]
            )
            IconButton("refresh").click(
                fn=lambda: gr.Dropdown(choices=get_voices()),
                outputs=[voice_dropdown],
            )
            OpenFolderButton(
                get_path_from_root("voices", "chatterbox"),
                api_name="chatterbox_open_voices_dir",
            )

        audio_prompt_path = gr.Audio(
            label="Reference Audio", type="filepath", value=None
        )

        voice_dropdown.change(
            lambda x: gr.Audio(value=x),
            inputs=[voice_dropdown],
            outputs=[audio_prompt_path],
        )

        exaggeration = gr.Slider(
            label="Exaggeration (Neutral = 0.5, extreme values can be unstable)",
            minimum=0,
            maximum=2,
            value=0.5,
        )
        cfg_weight = gr.Slider(
            label="CFG Weight/Pace", minimum=0.0, maximum=1, value=0.5
        )
        temperature = gr.Slider(label="Temperature", minimum=0.05, maximum=5, value=0.8)

        seed, randomize_seed_callback = randomize_seed_ui()

    with gr.Column():
        audio_out = gr.Audio(label="Audio Output")
        streaming_audio_output = gr.Audio(
            label="Audio Output (streaming)", streaming=True, autoplay=True
        )

        gr.Markdown("## Settings")

        with gr.Accordion("Chunking", open=True), gr.Group():
            chunked = gr.Checkbox(label="Split prompt into chunks", value=False)
            with gr.Row():
                desired_length = gr.Slider(
                    label="Desired length (characters)",
                    minimum=10,
                    maximum=1000,
                    value=200,
                    step=1,
                )
                max_length = gr.Slider(
                    label="Max length (characters)",
                    minimum=10,
                    maximum=1000,
                    value=300,
                    step=1,
                )
                halve_first_chunk = gr.Checkbox(
                    label="Halve first chunk size",
                    value=False,
                )
                cache_voice = gr.Checkbox(
                    label="Cache voice (not implemented)",
                    value=False,
                    visible=False,
                )
        # model
        with gr.Accordion("Model", open=False):
            with gr.Row():
                device = gr.Radio(
                    label="Device",
                    choices=["auto", "cuda", "mps", "cpu"],
                    value="auto",
                )
                dtype = gr.Radio(
                    label="Dtype",
                    choices=["float32", "float16", "bfloat16"],
                    value="float32",
                )
                cpu_offload = gr.Checkbox(label="CPU Offload", value=False)
                model_name = gr.Dropdown(
                    label="Model",
                    choices=["just_a_placeholder"],
                    value="just_a_placeholder",
                    visible=False,
                )
            
            with gr.Row():
                btn_move_model = gr.Button("Move to device and dtype")
                btn_move_model.click(
                    fn=lambda: gr.Button("Moving..."),
                    outputs=[btn_move_model],
                ).then(
                    fn=move_model_to_device_and_dtype,
                    inputs=[device, dtype, cpu_offload],
                ).then(
                    fn=lambda: gr.Button("Move to device and dtype"),
                    outputs=[btn_move_model],
                )
                unload_model_button("chatterbox")

            gr.Markdown("## Optimization")
            gr.Markdown("""
                        By reducing cache length, the model becomes faster, but maximum generation length is reduced. Gives an error if too low.
                        For fastest speeds, reduce prompt length and max new tokens. Fast: 330 max new tokens, 600 cache length.
                        """)
            with gr.Row():
                max_new_tokens = gr.Slider(
                    label="Max new tokens",
                    minimum=100,
                    maximum=1000,
                    value=1000,
                    step=10,
                )
                max_cache_len = gr.Slider(
                    label="Cache length",
                    minimum=200,
                    maximum=1500,
                    value=1500,
                    step=10,
                )
            use_compilation = gr.Checkbox(label="Use compilation", value=None, visible=True)

            with gr.Row(visible=False):
                btn_compile = gr.Button("Compile model", variant="primary")
                btn_compile.click(
                    fn=lambda: gr.Button("Compiling..."),
                    outputs=[btn_compile],
                ).then(
                    fn=lambda: compile_t3(get_current_model("chatterbox")),
                    inputs=[],
                    outputs=[],
                ).then(
                    fn=lambda: gr.Button("Compile model"),
                    outputs=[btn_compile],
                )
                btn_remove_compilation = gr.Button("Remove compilation")
                btn_remove_compilation.click(
                    fn=lambda: gr.Button("Removing compilation..."),
                    outputs=[btn_remove_compilation],
                ).then(
                    fn=lambda: remove_t3_compilation(get_current_model("chatterbox")),
                    inputs=[],
                    outputs=[],
                ).then(
                    fn=lambda: gr.Button("Remove compilation"),
                    outputs=[btn_remove_compilation],
                )
                def reset_torch_dynamo():
                    import torch
                    torch._dynamo.reset()
                gr.Button("Reset compilation", variant="stop").click(
                    fn=reset_torch_dynamo,
                    inputs=[],
                    outputs=[],
                )

            gr.Markdown("Memory usage:")
            gr.Button("Check memory usage").click(
                fn=get_chatterbox_memory_usage,
                outputs=[gr.Markdown()],
            )

        gr.Markdown(
            "Sliced audio streaming is deprecated due to artifacts, use chunking instead."
        )
        with gr.Accordion("Streaming (Advanced Settings)", open=False, visible=False):
            gr.Markdown(
                """
Streaming has issues due to Chatterbox producing artifacts.
Tokens per slice: 
* 1000 is recommended, it is the default maximum value, equivalent to disabling streaming.
* One second is around 23.5 tokens.
                        
Remove milliseconds:
* 25 - 65 is recommended.
* This removes the last 45 milliseconds of each slice to avoid artifacts.
* start: 15 - 35 is recommended.
* This removes the first 25 milliseconds of each slice to avoid artifacts.
                        
Chunk overlap method:
* zero means that each chunk is seen sparately by the audio generator. 
* full means that each chunk is appended and decoded as one long audio file.
                        
Thus **the challenge is to fix the seams** - with no overlap, the artifacts are high. With a very long overlap, such as a 0.5s crossfade, the audio starts to produce echo.
"""
            )
            with gr.Row():
                tokens_per_slice = gr.Slider(
                    label="Tokens per slice",
                    minimum=1,
                    maximum=1000,
                    value=1000,
                    step=1,
                )
                remove_milliseconds = gr.Slider(
                    label="Remove milliseconds",
                    minimum=0,
                    maximum=300,
                    value=45,
                    step=1,
                )
                remove_milliseconds_start = gr.Slider(
                    label="Remove milliseconds start",
                    minimum=0,
                    maximum=300,
                    value=25,
                    step=1,
                )
                chunk_overlap_method = gr.Radio(
                    label="Chunk overlap method",
                    choices=["zero", "full"],
                    value="zero",
                )

    inputs = {
        text: "text",
        exaggeration: "exaggeration",
        cfg_weight: "cfg_weight",
        temperature: "temperature",
        audio_prompt_path: "audio_prompt_path",
        seed: "seed",
        # model
        device: "device",
        dtype: "dtype",
        model_name: "model_name",
        # hyperparameters
        chunked: "chunked",
        cpu_offload: "cpu_offload",
        cache_voice: "cache_voice",
        # streaming
        tokens_per_slice: "tokens_per_slice",
        remove_milliseconds: "remove_milliseconds",
        remove_milliseconds_start: "remove_milliseconds_start",
        chunk_overlap_method: "chunk_overlap_method",
        # chunks
        desired_length: "desired_length",
        max_length: "max_length",
        halve_first_chunk: "halve_first_chunk",
        # compile
        use_compilation: "use_compilation",
        # optimization
        max_new_tokens: "max_new_tokens",
        max_cache_len: "max_cache_len",
    }

    generation_start = {
        "fn": lambda: [
            gr.Button("Generating...", interactive=False),
            gr.Button("Generating...", interactive=False),
            gr.Button("Interrupt next chunk", interactive=True, variant="stop"),
        ],
        "outputs": [btn, btn_stream, btn_interrupt],
    }
    generation_end = {
        "fn": lambda: [
            gr.Button("Generate", interactive=True, variant="primary"),
            gr.Button("Streaming generation", interactive=True, variant="secondary"),
            gr.Button("Interrupt next chunk", interactive=False, variant="stop"),
        ],
        "outputs": [btn, btn_stream, btn_interrupt],
    }

    btn.click(**randomize_seed_callback).then(**generation_start).then(
        **dictionarize_wraps(
            tts_decorated,
            inputs=inputs,
            outputs={
                "audio_out": audio_out,
                "metadata": gr.JSON(visible=False),
                "folder_root": gr.Textbox(visible=False),
            },
            api_name="chatterbox_tts",
        )
    ).then(**generation_end)

    btn_stream.click(**randomize_seed_callback).then(**generation_start).then(
        **dictionarize_wraps(
            tts_generator_decorated,
            inputs=inputs,
            outputs={
                "audio_out": streaming_audio_output,
                # "metadata": gr.JSON(visible=False),
                # "folder_root": gr.Textbox(visible=False),
            },
            api_name="chatterbox_tts_streaming",
        )
    ).then(**generation_end)


@functools.wraps(vc)
@decorator_extension_outer
@decorator_apply_torch_seed
@decorator_save_metadata
@decorator_save_wav
@decorator_add_model_type("chatterbox-vc")
@decorator_add_base_filename
@decorator_add_date
@decorator_log_generation
@decorator_extension_inner
@log_function_time
def vc_decorated(*args, **kwargs):
    return vc(*args, **kwargs)


def chatterbox_vc():
    with gr.Column():
        audio_in = gr.Audio(label="Input Audio", type="filepath", value=None)
        btn = gr.Button("Convert", variant="primary")
        audio_ref = gr.Audio(label="Audio Reference", type="filepath", value=None)
    with gr.Column():
        audio_out = gr.Audio(label="Output Audio")

    btn.click(fn=lambda: gr.Button("Converting..."), outputs=[btn]).then(
        **dictionarize_wraps(
            vc_decorated,
            inputs={audio_in: "audio_in", audio_ref: "audio_ref"},
            outputs={
                "audio_out": audio_out,
                "metadata": gr.JSON(visible=False),
                "folder_root": gr.Textbox(visible=False),
            },
            api_name="chatterbox_vc",
        )
    ).then(fn=lambda: gr.Button("Convert"), outputs=[btn])


if __name__ == "__main__":
    if "demo" in locals():
        locals()["demo"].close()
    with gr.Blocks() as demo:
        ui()

    demo.launch(
        server_port=7771,
    )
    # python -m workspace.extension_chatterbox.extension_chatterbox.gradio_app
