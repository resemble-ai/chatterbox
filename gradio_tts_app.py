import random
import numpy as np
import torch
import gradio as gr
from chatterbox.tts import ChatterboxTTS

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

def generate(model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw):
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
    )
    return (model.sr, wav.squeeze(0).numpy())

with gr.Blocks(title="Chatterbox TTS") as demo:
    model_state = gr.State(None)  # Loaded once per session/user
    
    gr.Markdown("# Chatterbox TTS")
    gr.Markdown("Generate high-quality speech with emotion control using Resemble AI's open-source TTS model.")
    
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
                label="Text to synthesize (max chars 300)",
                max_lines=5,
                placeholder="Enter the text you want to convert to speech..."
            )
            ref_wav = gr.Audio(
                sources=["upload", "microphone"], 
                type="filepath", 
                label="Reference Audio File (Optional - for voice cloning)", 
                value=None
            )
            exaggeration = gr.Slider(
                0.25, 2, 
                step=0.05, 
                label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", 
                value=0.5,
                info="Higher values make speech more expressive"
            )
            cfg_weight = gr.Slider(
                0.0, 1, 
                step=0.05, 
                label="CFG/Pace", 
                value=0.5,
                info="Lower values = slower speech, higher values = faster speech"
            )
            
            with gr.Accordion("Advanced Options", open=False):
                seed_num = gr.Number(
                    value=0, 
                    label="Random seed (0 for random)",
                    info="Set to specific number for reproducible results"
                )
                temp = gr.Slider(
                    0.05, 5, 
                    step=0.05, 
                    label="Temperature", 
                    value=0.8,
                    info="Higher values = more variation, lower values = more consistent"
                )
            
            run_btn = gr.Button("🎤 Generate Speech", variant="primary", size="lg")
            
        with gr.Column():
            audio_output = gr.Audio(
                label="Generated Audio",
                show_download_button=True,
                interactive=False
            )
            
            gr.Markdown("""
            ### Tips:
            - **General Use**: Default settings work well for most cases
            - **Fast speakers**: Lower CFG weight to ~0.3
            - **Expressive speech**: Lower CFG (~0.3) + higher exaggeration (~0.7+)
            - **Voice cloning**: Upload a clear reference audio (3-10 seconds recommended)
            """)

    # Event handlers
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
        ],
        outputs=audio_output,
    )

if __name__ == "__main__":  # Fixed: was **name**
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(
        share=True,
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860
    )
