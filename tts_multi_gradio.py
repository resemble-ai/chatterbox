import random
import numpy as np
import torch
import gradio as gr
from chatterbox.tts import ChatterboxTTS
from transformers import BarkModel
import re
import os
from pathlib import Path


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CHARS_PER_CHUNK = 400  # Adjust this based on testing

# Custom Bark model configuration
CUSTOM_BARK_MODELS = {
    "Default (suno/bark-small)": None,
    "czt3": "C:/ChatterboxTraining/t3",
    # Add more custom models here:
    # "My Voice Clone": "C:/path/to/another/model",
}


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_model():
    """Load ChatterboxTTS with default Bark model"""
    print(f"Loading Chatterbox TTS on {DEVICE}...")
    model = ChatterboxTTS.from_pretrained(DEVICE)
    print("✓ Chatterbox TTS loaded with default Bark model")
    return model


def switch_bark_model(model, model_choice):
    """Switch the Bark model to a different version"""
    if model is None:
        model = ChatterboxTTS.from_pretrained(DEVICE)
    
    custom_path = CUSTOM_BARK_MODELS.get(model_choice)
    
    if custom_path:
        # Check if path exists
        if not os.path.exists(custom_path):
            return model, f"❌ Error: Model path not found: {custom_path}"
        
        print(f"Loading custom Bark model from: {custom_path}")
        try:
            # Replace the Bark model
            model.bark_model = BarkModel.from_pretrained(custom_path).to(DEVICE)
            print(f"✓ Loaded custom Bark model: {model_choice}")
            return model, f"✓ Loaded: {model_choice}"
        except Exception as e:
            return model, f"❌ Error loading model: {str(e)}"
    else:
        print("Loading default Bark model...")
        try:
            # Reload default model
            model.bark_model = BarkModel.from_pretrained("suno/bark-small").to(DEVICE)
            print("✓ Loaded default Bark model")
            return model, "✓ Loaded: Default Bark model"
        except Exception as e:
            return model, f"❌ Error loading model: {str(e)}"


def split_text_smart(text, max_chars=MAX_CHARS_PER_CHUNK):
    """Split text into chunks at sentence boundaries"""
    if len(text) <= max_chars:
        return [text]
    
    # Split by sentences (., !, ?)
    sentences = re.split(r'([.!?]+\s*)', text)
    
    chunks = []
    current_chunk = ""
    
    for i in range(0, len(sentences), 2):
        sentence = sentences[i]
        punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
        full_sentence = sentence + punctuation
        
        # If adding this sentence would exceed limit
        if len(current_chunk) + len(full_sentence) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = full_sentence
            else:
                # Single sentence is too long, split by words
                words = full_sentence.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk) + len(word) + 1 <= max_chars:
                        temp_chunk += word + " "
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = word + " "
                current_chunk = temp_chunk
        else:
            current_chunk += full_sentence
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def generate(model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw, min_p, top_p, repetition_penalty):
    if model is None:
        model = ChatterboxTTS.from_pretrained(DEVICE)

    if not text or not text.strip():
        return None
    
    # Set seed if specified
    if seed_num != 0:
        set_seed(int(seed_num))

    # Split text into chunks
    chunks = split_text_smart(text.strip())
    
    print(f"Processing {len(chunks)} chunk(s)...")
    
    try:
        audio_chunks = []
        
        for i, chunk in enumerate(chunks):
            print(f"Generating chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
            
            wav = model.generate(
                chunk,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfgw,
                min_p=min_p,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            
            # Convert to numpy and add to chunks
            audio_chunks.append(wav.squeeze(0).numpy())
        
        # Concatenate all audio chunks
        if len(audio_chunks) > 1:
            # Add small silence between chunks (0.1 seconds)
            silence = np.zeros(int(model.sr * 0.1))
            final_audio = audio_chunks[0]
            for chunk in audio_chunks[1:]:
                final_audio = np.concatenate([final_audio, silence, chunk])
            print(f"Successfully generated {len(chunks)} chunks!")
            return (model.sr, final_audio)
        else:
            return (model.sr, audio_chunks[0])
            
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        return None


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    model_state = gr.State(None)  # Loaded once per session/user

    gr.Markdown("""
    # 🎙️ Chatterbox TTS
    
    Advanced text-to-speech with custom Bark model support.
    
    **Note:** Long texts are automatically split into chunks for processing.
    Each chunk is limited to ~400 characters for optimal quality.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            # Model Selection
            gr.Markdown("### 🔧 Model Configuration")
            model_dropdown = gr.Dropdown(
                choices=list(CUSTOM_BARK_MODELS.keys()),
                value="Default (suno/bark-small)",
                label="Bark Model",
                info="Select which Bark model to use"
            )
            model_status = gr.Textbox(
                label="Model Status",
                value="Default model loaded",
                interactive=False,
                lines=1
            )
            load_model_btn = gr.Button("🔄 Load Selected Model", variant="secondary", size="sm")
            
            gr.Markdown("---")
            
            # Text Input
            gr.Markdown("### 📝 Text Input")
            text = gr.Textbox(
                value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
                label="Text to synthesize",
                lines=6,
                max_lines=12,
                placeholder="Enter text to convert to speech..."
            )
            
            # Reference Audio
            gr.Markdown("### 🎵 Voice Reference")
            ref_wav = gr.Audio(
                sources=["upload", "microphone"], 
                type="filepath", 
                label="Reference Audio File",
                value=None
            )
            
            # Main Controls
            gr.Markdown("### 🎛️ Voice Controls")
            exaggeration = gr.Slider(
                0.25, 2, 
                step=.05, 
                label="Exaggeration",
                info="Neutral = 0.5, extreme values can be unstable",
                value=.5
            )
            cfg_weight = gr.Slider(
                0.0, 1, 
                step=.05, 
                label="CFG Weight / Pace",
                info="Controls generation guidance",
                value=0.3
            )

            # Advanced Options
            with gr.Accordion("⚙️ Advanced Options", open=False):
                seed_num = gr.Number(
                    value=0, 
                    label="Random Seed",
                    info="0 for random, set number for reproducible results"
                )
                temp = gr.Slider(
                    0.05, 5, 
                    step=.05, 
                    label="Temperature",
                    info="Higher = more creative/variable",
                    value=.8
                )
                min_p = gr.Slider(
                    0.00, 1.00, 
                    step=0.01, 
                    label="Min P",
                    info="Newer sampler. 0.02-0.1 recommended. 0.00 disables",
                    value=0.05
                )
                top_p = gr.Slider(
                    0.00, 1.00, 
                    step=0.01, 
                    label="Top P",
                    info="Original sampler. 1.0 disables (recommended)",
                    value=1.00
                )
                repetition_penalty = gr.Slider(
                    1.00, 2.00, 
                    step=0.1, 
                    label="Repetition Penalty",
                    info="Reduces repeated phrases",
                    value=1.2
                )

            # Generate Button
            run_btn = gr.Button("🎬 Generate Speech", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### 🔊 Output")
            audio_output = gr.Audio(label="Generated Audio")
            
            gr.Markdown("""
            ---
            ### 💡 Tips
            
            **Model Selection:**
            - **Default**: Original Bark model from Suno AI
            - **Fine-tuned**: Your custom trained model
            
            **Voice Cloning:**
            - Upload 5-10 seconds of clear reference audio
            - Single speaker, minimal background noise
            
            **Text Processing:**
            - Texts over 400 chars are auto-chunked
            - Use proper punctuation for better prosody
            
            **Parameter Guide:**
            - **Exaggeration**: Controls emotion intensity
            - **Temperature**: Higher = more variation
            - **CFG Weight**: Affects pacing and adherence
            """)

    # Event Handlers
    demo.load(fn=load_model, inputs=[], outputs=model_state)
    
    load_model_btn.click(
        fn=switch_bark_model,
        inputs=[model_state, model_dropdown],
        outputs=[model_state, model_status]
    )

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
            min_p,
            top_p,
            repetition_penalty,
        ],
        outputs=audio_output,
    )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎙️  CHATTERBOX TTS - CUSTOM MODEL EDITION")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Available models: {len(CUSTOM_BARK_MODELS)}")
    for model_name in CUSTOM_BARK_MODELS.keys():
        print(f"  - {model_name}")
    print("="*60 + "\n")
    
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(share=True)