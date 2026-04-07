import random

import gradio as gr
import numpy as np
import torch

from chatterbox import ChatterboxInference
from chatterbox.mtl_tts import SUPPORTED_LANGUAGES


LANGUAGE_CHOICES = [(f"{name} ({code})", code) for code, name in sorted(SUPPORTED_LANGUAGES.items(), key=lambda x: x[1])]


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_model(model_type, repo_id, local_path):
    repo_id = repo_id.strip() if repo_id else None
    local_path = local_path.strip() if local_path else None

    if local_path:
        inference = ChatterboxInference.from_local(local_path, model_type=model_type)
    elif repo_id:
        inference = ChatterboxInference.from_pretrained(model_type=model_type, repo_id=repo_id)
    else:
        inference = ChatterboxInference.from_pretrained(model_type=model_type)

    return inference, f"Loaded: {model_type}" + (f" from {repo_id or local_path}" if repo_id or local_path else " (default)")


def generate(
    inference,
    text,
    language_id,
    audio_prompt_path,
    normalize_text,
    sentence_split,
    inter_sentence_silence_ms,
    exaggeration,
    cfg_weight,
    temperature,
    min_p,
    top_p,
    top_k,
    repetition_penalty,
    seed_num,
):
    if inference is None:
        raise gr.Error("Load a model first.")

    if seed_num != 0:
        set_seed(int(seed_num))

    wav = inference.generate(
        text,
        language_id=language_id or None,
        normalize_text=normalize_text,
        sentence_split=sentence_split,
        inter_sentence_silence_ms=int(inter_sentence_silence_ms),
        audio_prompt_path=audio_prompt_path or None,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
        min_p=min_p,
        top_p=top_p,
        top_k=int(top_k),
        repetition_penalty=repetition_penalty,
    )
    return (inference.sr, wav.squeeze(0).numpy())


with gr.Blocks(title="Chatterbox TTS") as demo:
    gr.Markdown("# Chatterbox TTS")
    inference_state = gr.State(None)

    # --- Model loading ---
    with gr.Group():
        gr.Markdown("### Model")
        with gr.Row():
            model_type = gr.Dropdown(
                choices=["base", "multilingual", "turbo"],
                value="multilingual",
                label="Model type",
            )
            repo_id = gr.Textbox(
                label="HuggingFace repo ID (optional)",
                placeholder="e.g. CoRal-project/roest-v3-chatterbox-500m",
            )
            local_path = gr.Textbox(
                label="Local checkpoint path (optional)",
                placeholder="/path/to/model_dir",
            )
        load_btn = gr.Button("Load Model", variant="secondary")
        load_status = gr.Textbox(label="Status", interactive=False, value="No model loaded.")

    # --- Generation ---
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Temperaturen i dag er 12 grader. Der er 3 æbler og 1 liter mælk tilbage i køleskabet.",
                label="Text to synthesize",
                max_lines=6,
            )
            language_id = gr.Dropdown(
                choices=[("None (base / turbo)", "")] + LANGUAGE_CHOICES,
                value="da",
                label="Language (multilingual only)",
            )
            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Reference audio (optional)",
            )

            with gr.Accordion("Text Processing", open=True):
                normalize_text = gr.Checkbox(value=True, label="Normalize text (number expansion, punctuation)")
                sentence_split = gr.Checkbox(value=True, label="Split into sentences")
                inter_sentence_silence_ms = gr.Slider(0, 500, step=10, value=100, label="Silence between sentences (ms)")

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")
            with gr.Accordion("Generation Parameters", open=False):
                gr.Markdown("*`exaggeration` and `cfg_weight` are ignored by turbo. `top_k` is turbo-only.*")
                exaggeration = gr.Slider(0.25, 2.0, step=0.05, value=0.5, label="Exaggeration")
                cfg_weight = gr.Slider(0.0, 1.0, step=0.05, value=0.5, label="CFG / Pace")
                temperature = gr.Slider(0.05, 2.0, step=0.05, value=0.8, label="Temperature")
                min_p = gr.Slider(0.00, 1.00, step=0.01, value=0.05, label="Min P")
                top_p = gr.Slider(0.00, 1.00, step=0.01, value=1.00, label="Top P")
                top_k = gr.Slider(0, 1000, step=10, value=1000, label="Top K (turbo only)")
                repetition_penalty = gr.Slider(1.00, 2.00, step=0.05, value=2.0, label="Repetition Penalty")
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")

    model_type.change(
        fn=lambda mt: gr.update(value=1.2 if mt == "turbo" else 2.0),
        inputs=model_type,
        outputs=repetition_penalty,
    )

    load_btn.click(
        fn=load_model,
        inputs=[model_type, repo_id, local_path],
        outputs=[inference_state, load_status],
    )

    run_btn.click(
        fn=generate,
        inputs=[
            inference_state,
            text,
            language_id,
            ref_wav,
            normalize_text,
            sentence_split,
            inter_sentence_silence_ms,
            exaggeration,
            cfg_weight,
            temperature,
            min_p,
            top_p,
            top_k,
            repetition_penalty,
            seed_num,
        ],
        outputs=audio_output,
    )

if __name__ == "__main__":
    demo.queue(max_size=50, default_concurrency_limit=1).launch(share=True)
