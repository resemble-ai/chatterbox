import gradio as gr
from gradio_app import ui

with gr.Blocks() as demo:
    ui()

demo.launch()
