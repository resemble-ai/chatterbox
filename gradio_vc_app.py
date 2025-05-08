from orator.vc import OratorVC
import gradio as gr


model = OratorVC.from_pretrained("cuda")
def generate(audio, target_voice_path):
    wav = model.generate(
        audio, target_voice_path=target_voice_path,
    )
    return 24000, wav.squeeze(0).numpy()


demo = gr.Interface(
    generate,
    [
        gr.Audio(sources="upload", type="filepath", label="Input audio file"),
        gr.Audio(sources="upload", type="filepath", label="Target voice audio file (if none, the default voice is used)", value=None),
    ],
    "audio",
)

if __name__ == "__main__":
    demo.launch()
