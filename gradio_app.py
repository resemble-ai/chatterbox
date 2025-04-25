from orator.tts import OratorTTS
import gradio as gr


model = OratorTTS.from_local("checkpoints", "cuda")
def generate(text, audio_prompt_path, emotion_adv):
    wav = model.generate(text, audio_prompt_path=audio_prompt_path, emotion_adv=emotion_adv)
    return 24000, wav.squeeze(0).numpy()

demo = gr.Interface(
    generate,
    [
        gr.Textbox(value="What does the fox say?", label="Text to synthesize"),
        gr.Audio(sources="upload", type="filepath", label="Input Audio File"),
        gr.Slider(0, 1, step=.05, label="emotion_adv", value=.5),
    ],
    "audio",
)

if __name__ == "__main__":
    demo.launch()
