from chatterbox.tts import ChatterboxTTS
import gradio as gr


model = ChatterboxTTS.from_local("checkpoints", "cuda")
def generate(text, audio_prompt_path, exaggeration, pace, temperature):
    wav = model.generate(
        text, audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        pace=pace,
        temperature=temperature,
    )
    return 24000, wav.squeeze(0).numpy()


demo = gr.Interface(
    generate,
    [
        gr.Textbox(value="What does the fox say?", label="Text to synthesize"),
        gr.Audio(sources="upload", type="filepath", label="Reference Audio File", value=None),
        gr.Slider(-5, 5, step=.05, label="exaggeration", value=.5),
        gr.Slider(0.8, 1.2, step=.01, label="pace", value=1),
        gr.Slider(0.05, 5, step=.05, label="temperature", value=.8),
    ],
    "audio",
)

if __name__ == "__main__":
    demo.launch()
