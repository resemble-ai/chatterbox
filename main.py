import random
import numpy as np
import torch
import librosa
import gradio as gr

from chatterbox.tts import ChatterboxTTS
from chatterbox.vc import ChatterboxVC
from chatterbox.audio_editing import (
    splice_audios,
    trim_audio,
    insert_audio,
    delete_segment,
    crossfade,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXAMPLE_TEXTS = [
    "Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
    "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill.",
    "Hello there! I'm the open source Chatterbox TTS from Resemble AI.",
]


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_tts_model():
    return ChatterboxTTS.from_pretrained(DEVICE)


def load_vc_model():
    return ChatterboxVC.from_pretrained(DEVICE)


def tts_generate(model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw, min_p, top_p, repetition_penalty):
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


def vc_generate(model, audio, target_voice_path):
    if model is None:
        model = ChatterboxVC.from_pretrained(DEVICE)

    wav = model.generate(
        audio,
        target_voice_path=target_voice_path,
    )
    return model.sr, wav.squeeze(0).numpy()


def edit_splice(audio1, audio2):
    if audio1 is None or audio2 is None:
        raise gr.Error("Two audio files required")
    wav1, sr1 = librosa.load(audio1, sr=None)
    wav2, sr2 = librosa.load(audio2, sr=None)
    if sr1 != sr2:
        raise gr.Error("Sampling rates must match")
    joined = splice_audios([wav1, wav2])
    return sr1, joined


def edit_trim(audio, start_sec, end_sec):
    if audio is None:
        raise gr.Error("Audio file required")
    wav, sr = librosa.load(audio, sr=None)
    trimmed = trim_audio(wav, start_sec=float(start_sec), end_sec=float(end_sec), sr=sr)
    return sr, trimmed


def edit_insert(base_audio, insert_audio_file, position_sec):
    if base_audio is None or insert_audio_file is None:
        raise gr.Error("Need base and insert audio")
    base_wav, sr = librosa.load(base_audio, sr=None)
    ins_wav, sr2 = librosa.load(insert_audio_file, sr=sr)
    inserted = insert_audio(base_wav, ins_wav, float(position_sec), sr=sr)
    return sr, inserted


def edit_delete(audio, start_sec, end_sec):
    if audio is None:
        raise gr.Error("Audio file required")
    wav, sr = librosa.load(audio, sr=None)
    deleted = delete_segment(wav, float(start_sec), float(end_sec), sr=sr)
    return sr, deleted


def edit_crossfade(audio1, audio2, duration_sec):
    if audio1 is None or audio2 is None:
        raise gr.Error("Two audio files required")
    wav1, sr1 = librosa.load(audio1, sr=None)
    wav2, sr2 = librosa.load(audio2, sr=sr1)
    out = crossfade(wav1, wav2, float(duration_sec), sr=sr1)
    return sr1, out


with gr.Blocks(title="Chatterbox") as demo:
    gr.Markdown("# Chatterbox Main Interface")
    tts_state = gr.State(None)
    vc_state = gr.State(None)

    with gr.Tab("Text to Speech"):
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

                run_tts = gr.Button("Generate", variant="primary")

            with gr.Column():
                tts_output = gr.Audio(label="Output Audio")

        demo.load(fn=load_tts_model, inputs=[], outputs=tts_state)
        run_tts.click(
            fn=tts_generate,
            inputs=[
                tts_state,
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
            outputs=tts_output,
        )

    with gr.Tab("Voice Conversion"):
        with gr.Row():
            input_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Input audio file")
            target_voice = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Target voice audio file", value=None)
        run_vc = gr.Button("Convert", variant="primary")
        vc_output = gr.Audio(label="Output Audio")

        demo.load(fn=load_vc_model, inputs=[], outputs=vc_state)
        run_vc.click(
            fn=vc_generate,
            inputs=[vc_state, input_audio, target_voice],
            outputs=vc_output,
        )

    with gr.Tab("Audio Editing"):
        with gr.Tabs():
            with gr.Tab("Splice"):
                splice_a = gr.Audio(sources=["upload", "microphone"], type="filepath", label="First audio")
                splice_b = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Second audio")
                run_splice = gr.Button("Splice")
                splice_out = gr.Audio(label="Output Audio")
                run_splice.click(edit_splice, [splice_a, splice_b], splice_out)

            with gr.Tab("Trim"):
                trim_audio_in = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio to trim")
                trim_start = gr.Number(value=0.0, label="Start sec")
                trim_end = gr.Number(value=1.0, label="End sec")
                run_trim = gr.Button("Trim")
                trim_out = gr.Audio(label="Output Audio")
                run_trim.click(edit_trim, [trim_audio_in, trim_start, trim_end], trim_out)

            with gr.Tab("Insert"):
                base_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Base audio")
                insert_audio_file = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Insert audio")
                insert_pos = gr.Number(value=0.0, label="Position sec")
                run_insert = gr.Button("Insert")
                insert_out = gr.Audio(label="Output Audio")
                run_insert.click(edit_insert, [base_audio, insert_audio_file, insert_pos], insert_out)

            with gr.Tab("Delete"):
                del_audio_in = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio")
                del_start = gr.Number(value=0.0, label="Start sec")
                del_end = gr.Number(value=1.0, label="End sec")
                run_delete = gr.Button("Delete")
                delete_out = gr.Audio(label="Output Audio")
                run_delete.click(edit_delete, [del_audio_in, del_start, del_end], delete_out)

            with gr.Tab("Crossfade"):
                cross_a = gr.Audio(sources=["upload", "microphone"], type="filepath", label="First audio")
                cross_b = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Second audio")
                duration = gr.Number(value=0.01, label="Duration sec")
                run_cross = gr.Button("Crossfade")
                cross_out = gr.Audio(label="Output Audio")
                run_cross.click(edit_crossfade, [cross_a, cross_b, duration], cross_out)

if __name__ == "__main__":
    demo.queue(max_size=50, default_concurrency_limit=1).launch()
