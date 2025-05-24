import random
import numpy as np
import torch
import gradio as gr
from chatterbox.tts import ChatterboxTTS


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP_REGISTER = None


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


model = ChatterboxTTS.from_pretrained(DEVICE)

def generate(text, audio_prompt_path, exaggeration, pace, temperature, seed_num):
    if seed_num != 0:
        set_seed(int(seed_num))

    # Only `.prepare_conditionals` when the audio prompt changes.
    global FP_REGISTER
    if audio_prompt_path is not None and FP_REGISTER != audio_prompt_path:
        model.prepare_conditionals(audio_prompt_path)
        FP_REGISTER = audio_prompt_path

    wav = model.generate(
        text,
        # audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        pace=pace,
        temperature=temperature,
    )
    return model.sr, wav.squeeze(0).numpy()


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(value="What does the fox say?", label="Text to synthesize")
            ref_wav = gr.Audio(sources="upload", type="filepath", label="Reference Audio File", value=None)
            exaggeration = gr.Slider(-5, 5, step=.05, label="exaggeration", value=.5)

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="temperature", value=.8)
                pace = gr.Slider(0.8, 1.2, step=.01, label="pace", value=1)

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

    run_btn.click(
        fn=generate,
        inputs=[
            text,
            ref_wav,
            exaggeration,
            pace,
            temp,
            seed_num,
        ],
        outputs=audio_output,
    )

if __name__ == "__main__":
    demo.launch()
