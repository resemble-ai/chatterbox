from chatterbox.vc import ChatterboxVC
import gradio as gr


model = ChatterboxVC.from_pretrained("cuda")
def generate(audio, target_voice_path):
    wav = model.generate(
        audio, target_voice_path=target_voice_path,
    )
    return model.sr, wav.squeeze(0).numpy()


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
