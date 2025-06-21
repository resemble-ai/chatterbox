import random
import numpy as np
import torch
import gradio as gr
from chatterbox.tts import ChatterboxTTS
import time
import re
from typing import List
import logging
import sys
from optimum.bettertransformer import BetterTransformer
from mps_fast_patch import optimize_chatterbox_for_fast_mps

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model variable
GLOBAL_MODEL = None

# Device detection
if torch.backends.mps.is_available():
    DEVICE = "mps"
    logger.info("🚀 Apple Silicon MPS backend is available, will use GPU after CPU loading.")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    logger.info("🚀 CUDA GPU detected, will use GPU after CPU loading")
else:
    DEVICE = "cpu"
    logger.info("🚀 No GPU detected, running on CPU only")


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def split_text_into_chunks(text: str, max_chars: int = 250) -> List[str]:
    """
    Split text into chunks at sentence boundaries, respecting max character limit.
    """
    if len(text) <= max_chars:
        return [text]
    
    # Split by sentences first (period, exclamation, question mark)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If single sentence is too long, split by commas or spaces
        if len(sentence) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Split long sentence by commas
            parts = re.split(r'(?<=,)\s+', sentence)
            for part in parts:
                if len(part) > max_chars:
                    # Split by spaces as last resort
                    words = part.split()
                    word_chunk = ""
                    for word in words:
                        if len(word_chunk + " " + word) <= max_chars:
                            word_chunk += " " + word if word_chunk else word
                        else:
                            if word_chunk:
                                chunks.append(word_chunk.strip())
                            word_chunk = word
                    if word_chunk:
                        chunks.append(word_chunk.strip())
                else:
                    if len(current_chunk + " " + part) <= max_chars:
                        current_chunk += " " + part if current_chunk else part
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = part
        else:
            # Normal sentence processing
            if len(current_chunk + " " + sentence) <= max_chars:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if chunk.strip()]


def warmup_model(model):
    """
    Warm up the model with sample generation to ensure all kernels are compiled.
    Returns True if successful, False otherwise.
    """
    logger.info("🔥 Starting model warm-up...")
    try:
        start_time = time.time()
        # Use a realistic sentence for warm-up
        warmup_text = "Hello, this is a test to warm up the model for faster inference."
        
        # Generate audio
        wav = model.generate(
            warmup_text, 
            audio_prompt_path=None, 
            temperature=0.7, 
            cfg_weight=0.5,
            min_p=0.05,
            top_p=1.0,
            repetition_penalty=1.2
        )
        
        # Verify output
        if wav is None or wav.shape[-1] == 0:
            logger.error("❌ Warm-up failed: No audio generated")
            return False
            
        duration = wav.shape[-1] / model.sr
        end_time = time.time()
        logger.info(f"✅ Model warm-up completed in {end_time - start_time:.2f} seconds")
        logger.info(f"   Generated {duration:.2f} seconds of audio")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model warm-up failed: {e}", exc_info=True)
        return False


def transfer_model_to_gpu(model, device):
    """
    Transfers all components of the ChatterboxTTS model to the specified GPU device.
    """
    logger.info(f"📤 Transferring model components from CPU to {device}...")
    start_time = time.time()
    
    try:
        # Transfer each component
        logger.info("   - Transferring T3 model...")
        model.t3 = model.t3.to(device)
        
        logger.info("   - Transferring S3Gen model...")
        model.s3gen = model.s3gen.to(device)
        
        logger.info("   - Transferring Voice Encoder...")
        model.ve = model.ve.to(device)
        
        # Update device attribute
        model.device = device
        
        # Transfer conditionals if they exist
        if model.conds is not None:
            logger.info("   - Transferring conditionals...")
            model.conds = model.conds.to(device)
        
        # Apply MPS-specific optimizations if needed
        if device == 'mps':
            logger.info("   - Applying fast MPS optimizations...")
            model = optimize_chatterbox_for_fast_mps(model)
        
        # Synchronize to ensure transfers complete
        if device == 'cuda':
            torch.cuda.synchronize()
        elif device == 'mps':
            torch.mps.synchronize()
            
        end_time = time.time()
        logger.info(f"✅ Model components successfully transferred to {device} in {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"❌ Failed to transfer model to GPU: {e}", exc_info=True)
        raise
        
    return model


def load_and_prepare_model():
    """
    Load model on CPU, transfer to GPU, apply optimizations, and warm up.
    """
    logger.info("=" * 60)
    logger.info("🚀 INITIALIZING CHATTERBOX TTS MODEL")
    logger.info("=" * 60)
    
    # Step 1: Load model on CPU
    logger.info("📥 Step 1: Loading model on CPU...")
    start_time = time.time()
    model = ChatterboxTTS.from_pretrained("cpu")
    logger.info(f"✅ Model loaded on CPU in {time.time() - start_time:.2f} seconds")
    
    # Step 2: Transfer to GPU if available
    if DEVICE != "cpu":
        logger.info(f"📤 Step 2: Transferring model to {DEVICE}...")
        model = transfer_model_to_gpu(model, DEVICE)
    else:
        logger.info("ℹ️  Step 2: Skipping GPU transfer (CPU-only mode)")
    
    # Step 3: Apply BetterTransformer optimization
    logger.info("⚡ Step 3: Applying BetterTransformer optimization...")
    try:
        opt_start = time.time()
        model.t3 = BetterTransformer.transform(model.t3)
        model.s3gen = BetterTransformer.transform(model.s3gen)
        logger.info(f"✅ BetterTransformer applied in {time.time() - opt_start:.2f} seconds")
    except Exception as e:
        logger.warning(f"⚠️  Could not apply BetterTransformer: {e}")
    
    # Step 4: Warm up the model
    logger.info("🔥 Step 4: Warming up the model...")
    warmup_success = warmup_model(model)
    
    if not warmup_success:
        logger.error("❌ Model warm-up failed!")
        raise RuntimeError("Model warm-up failed. Please check the logs.")
    
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"✅ MODEL READY! Total initialization time: {total_time:.2f} seconds")
    logger.info("=" * 60)
    
    return model


def generate(text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw, min_p, top_p, repetition_penalty):
    """
    Generate speech from text using the global model.
    """
    if GLOBAL_MODEL is None:
        raise RuntimeError("Model not initialized!")

    if seed_num != 0:
        set_seed(int(seed_num))

    # Verify model is still on the correct device
    if DEVICE != "cpu" and GLOBAL_MODEL.device != DEVICE:
        logger.warning(f"Model device mismatch! Expected {DEVICE}, got {GLOBAL_MODEL.device}")
        GLOBAL_MODEL.device = DEVICE

    logger.info(f"🎤 Generating audio for text: '{text[:50]}...'")
    if audio_prompt_path:
        logger.info(f"   Using reference audio: {audio_prompt_path}")
    else:
        logger.info("   Using default voice")
    
    logger.info(f"   Parameters: temp={temperature}, cfg={cfgw}, exaggeration={exaggeration}")
    start_time = time.time()

    text_chunks = split_text_into_chunks(text)
    logger.info(f"   Processing {len(text_chunks)} text chunk(s)")

    generated_wavs = []
    for i, chunk in enumerate(text_chunks):
        logger.info(f"   Chunk {i+1}/{len(text_chunks)}: '{chunk[:50]}...'")
        chunk_start = time.time()
        
        wav = GLOBAL_MODEL.generate(
            chunk,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfgw,
            min_p=min_p,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        
        chunk_time = time.time() - chunk_start
        chunk_duration = wav.shape[-1] / GLOBAL_MODEL.sr
        logger.info(f"   ✓ Generated {chunk_duration:.2f}s in {chunk_time:.2f}s (RTF: {chunk_time/chunk_duration:.2f})")
        generated_wavs.append(wav)

    # Concatenate chunks with silence
    if len(generated_wavs) > 1:
        silence_samples = int(0.3 * GLOBAL_MODEL.sr)
        silence = torch.zeros(1, silence_samples, dtype=generated_wavs[0].dtype, device=generated_wavs[0].device)
        final_wav = torch.cat([torch.cat([wav, silence], dim=1) for wav in generated_wavs[:-1]] + [generated_wavs[-1]], dim=1)
    else:
        final_wav = generated_wavs[0]

    end_time = time.time()
    total_duration = final_wav.shape[-1] / GLOBAL_MODEL.sr
    total_time = end_time - start_time
    rtf = total_time / total_duration
    
    logger.info(f"✅ Total: Generated {total_duration:.2f}s of audio in {total_time:.2f}s (RTF: {rtf:.2f})")
    
    # Check if we meet the performance target
    if total_duration >= 60 and total_time < 20:
        logger.info("🎯 Performance target achieved: >1 minute of audio in <20 seconds!")
    
    return (GLOBAL_MODEL.sr, final_wav.squeeze(0).cpu().numpy())


# Initialize model before creating the interface
logger.info("🚀 Starting Chatterbox TTS Application...")
try:
    GLOBAL_MODEL = load_and_prepare_model()
except Exception as e:
    logger.error(f"Failed to initialize model: {e}", exc_info=True)
    raise

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🎤 Chatterbox TTS")
    gr.Markdown("Model loaded and warmed up. Ready for fast inference!")
    
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
                label="Text to synthesize",
                max_lines=5
            )
            ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File", value=None)
            exaggeration = gr.Slider(0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5)", value=.5)
            cfg_weight = gr.Slider(0.0, 1, step=.05, label="CFG/Pace", value=0.5)

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="temperature", value=.8)
                min_p = gr.Slider(0.00, 1.00, step=0.01, label="min_p", value=0.05)
                top_p = gr.Slider(0.00, 1.00, step=0.01, label="top_p", value=1.00)
                repetition_penalty = gr.Slider(1.00, 2.00, step=0.1, label="repetition_penalty", value=1.2)

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

    run_btn.click(
        fn=generate,
        inputs=[
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
    ).launch(share=False)
