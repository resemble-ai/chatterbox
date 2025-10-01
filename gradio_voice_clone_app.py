"""Gradio demo focused on rapid voice cloning with Chatterbox."""
from __future__ import annotations

from typing import Sequence

import torch
import gradio as gr

from chatterbox.tts import ChatterboxTTS


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


model = ChatterboxTTS.from_pretrained(DEVICE)


def _flatten_reference_inputs(
    quick_take: str | None, extra_files: Sequence[str] | None
) -> list[str] | None:
    """Combine optional reference inputs into a list of file paths."""

    references: list[str] = []

    if quick_take:
        references.append(quick_take)

    if extra_files:
        # ``gr.File`` returns a sequence whose entries are already file paths when ``type="filepath"``.
        for path in extra_files:
            if path and isinstance(path, str):
                references.append(path)

    return references or None


def clone_voice(
    prompt_text: str,
    quick_take: str | None,
    reference_batch: Sequence[str] | None,
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    repetition_penalty: float,
    min_p: float,
    top_p: float,
):
    """Generate speech using an optional set of reference recordings."""

    if not prompt_text or not prompt_text.strip():
        raise gr.Error("Please provide some text for the model to speak.")

    references = _flatten_reference_inputs(quick_take, reference_batch)

    try:
        wav = model.generate(
            prompt_text.strip(),
            audio_prompt_path=references,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
        )
    except Exception as exc:  # pragma: no cover - surfaced through Gradio UI
        raise gr.Error(str(exc)) from exc

    return model.sr, wav.squeeze(0).numpy()


with gr.Blocks(title="Chatterbox Voice Cloner") as demo:
    gr.Markdown(
        """
        # 🔊 Chatterbox Voice Cloner

        1. Record or upload a clean sample of the voice you want to clone (10–15 seconds works well).
        2. Optionally stack extra takes for more stable conditioning.
        3. Enter the line you want the cloned voice to speak and press **Generate voice clone**.

        Leave the reference inputs empty to fall back to the built-in voice.
        """
    )

    with gr.Row():
        prompt_text = gr.Textbox(
            label="What should the cloned voice say?",
            placeholder="Type your script here…",
            lines=3,
        )

    with gr.Row():
        quick_take = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Quick reference take (record or upload)",
        )
        reference_batch = gr.File(
            file_types=["audio"],
            file_count="multiple",
            type="filepath",
            label="Optional extra takes for conditioning",
            info="Drop in several clean recordings of the target voice for better tone stability.",
        )

    with gr.Accordion("Advanced voice controls", open=False):
        exaggeration = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            step=0.05,
            value=0.5,
            label="Emotion exaggeration",
            info="Higher values add intensity and speed, lower values sound calmer.",
        )
        cfg_weight = gr.Slider(
            minimum=0.0,
            maximum=1.5,
            step=0.05,
            value=0.5,
            label="CFG weight",
            info="Lower values favour the reference pacing, higher values follow the text more strictly.",
        )
        temperature = gr.Slider(
            minimum=0.1,
            maximum=1.5,
            step=0.05,
            value=0.8,
            label="Temperature",
            info="Controls variation in pronunciation. Lower values sound safer and more monotone.",
        )
        repetition_penalty = gr.Slider(
            minimum=1.0,
            maximum=2.0,
            step=0.05,
            value=1.2,
            label="Repetition penalty",
            info="Discourage repeated phonemes in longer passages.",
        )
        min_p = gr.Slider(
            minimum=0.0,
            maximum=0.5,
            step=0.01,
            value=0.05,
            label="min_p",
            info="Lower bound for nucleus sampling; increase if outputs sound too noisy.",
        )
        top_p = gr.Slider(
            minimum=0.1,
            maximum=1.0,
            step=0.05,
            value=1.0,
            label="top_p",
            info="Upper bound for nucleus sampling; reduce slightly for more conservative diction.",
        )

    generate_btn = gr.Button("Generate voice clone", variant="primary")
    output_audio = gr.Audio(label="Cloned voice", type="numpy")

    generate_btn.click(
        fn=clone_voice,
        inputs=[
            prompt_text,
            quick_take,
            reference_batch,
            exaggeration,
            cfg_weight,
            temperature,
            repetition_penalty,
            min_p,
            top_p,
        ],
        outputs=output_audio,
    )


if __name__ == "__main__":
    demo.launch()

