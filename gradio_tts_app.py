import random
import numpy as np
import torch
import gradio as gr
from chatterbox.tts import ChatterboxTTS

EXAMPLE_TEXTS = [
    "Harambe did nothing wrong.",
    "Hello there! I'm the open source Chatterbox TTS from Resemble AI."
]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_model():
    model = ChatterboxTTS.from_pretrained(DEVICE)
    return model


def generate(model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw, min_p, top_p, repetition_penalty):
    if model is None:
        model = ChatterboxTTS.from_pretrained(DEVICE)

    if seed_num != 0:
        set_seed(int(seed_num))

    wav = model.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        temperature=temperature,
        cfg_weight=cfgw,
        min_p=min_p,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    return (model.sr, wav.squeeze(0).numpy())


with gr.Blocks(title="Chatterbox TTS") as demo:
    gr.Markdown("# Chatterbox TTS\nGenerate expressive speech with your own voice sample.")
    model_state = gr.State(None)

    with gr.Tab("Synthesize"):
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(
                    label="Text to synthesize",
                    max_lines=5,
                    placeholder="Enter text here...",
                    value=EXAMPLE_TEXTS[0],
                )
                gr.Examples(EXAMPLE_TEXTS, inputs=text, label="Example prompts")
                ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File", value=None)
                exaggeration = gr.Slider(0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5)", value=.5)
                cfg_weight = gr.Slider(0.0, 1, step=.05, label="CFG/Pace", value=0.5)

                with gr.Accordion("Advanced", open=False):
                    seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                    temp = gr.Slider(0.05, 5, step=.05, label="temperature", value=.8)
                    min_p = gr.Slider(0.00, 1.00, step=0.01, label="min_p", value=0.05)
                    top_p = gr.Slider(0.00, 1.00, step=0.01, label="top_p", value=1.00)
                    repetition_penalty = gr.Slider(1.00, 2.00, step=0.1, label="repetition_penalty", value=1.2)

                run_btn = gr.Button("Generate", variant="primary")

            with gr.Column():
                audio_output = gr.Audio(label="Output Audio")

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

    with gr.Tab("About"):
        gr.Markdown(
            "Chatterbox is an open source TTS model by [Resemble AI](https://resemble.ai)."
        )

if __name__ == "__main__":
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(share=True)
