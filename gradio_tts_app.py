import random
import numpy as np
import torch
import gradio as gr
from chatterbox.tts import ChatterboxTTS
import re


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CHARS_PER_CHUNK = 400  # Adjust this based on testing


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_model():
    model = ChatterboxTTS.from_pretrained(DEVICE)
    return model


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


with gr.Blocks() as demo:
    model_state = gr.State(None)  # Loaded once per session/user

    gr.Markdown("""
    # Chatterbox TTS
    
    **Note:** Long texts are automatically split into chunks for processing.
    Each chunk is limited to ~400 characters for optimal quality.
    """)

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
                label="Text to synthesize (automatically chunked if too long)",
                lines=8,
                max_lines=15
            )
            ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File", value=None)
            exaggeration = gr.Slider(0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=.5)
            cfg_weight = gr.Slider(0.0, 1, step=.05, label="CFG/Pace", value=0.5)

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="temperature", value=.8)
                min_p = gr.Slider(0.00, 1.00, step=0.01, label="min_p || Newer Sampler. Recommend 0.02 > 0.1. Handles Higher Temperatures better. 0.00 Disables", value=0.05)
                top_p = gr.Slider(0.00, 1.00, step=0.01, label="top_p || Original Sampler. 1.0 Disables(recommended). Original 0.8", value=1.00)
                repetition_penalty = gr.Slider(1.00, 2.00, step=0.1, label="repetition_penalty", value=1.2)

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

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
            min_p,
            top_p,
            repetition_penalty,
        ],
        outputs=audio_output,
    )

if __name__ == "__main__":
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(share=True)
