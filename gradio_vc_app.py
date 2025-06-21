import random
import numpy as np
import torch
import gradio as gr
from chatterbox.vc import ChatterboxVC
import time
import logging
import sys
from optimum.bettertransformer import BetterTransformer

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


def warmup_model(model):
    """
    Warm up the model with a sample audio file to ensure all kernels are compiled.
    Returns True if successful, False otherwise.
    """
    logger.info("🔥 Starting model warm-up...")
    try:
        start_time = time.time()
        
        # Create a dummy audio for warm-up (1 second of silence)
        sample_rate = 16000
        duration = 1.0
        dummy_audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
        
        # Save to temporary file
        import tempfile
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, dummy_audio, sample_rate)
            tmp_path = tmp_file.name
        
        # Run voice conversion
        result = model.convert(tmp_path, tmp_path, temperature=0.7, cfg_weight=0.5)
        
        # Clean up
        import os
        os.unlink(tmp_path)
        
        if result is None or result[1].shape[-1] == 0:
            logger.error("❌ Warm-up failed: No audio generated")
            return False
            
        duration = result[1].shape[-1] / result[0]
        end_time = time.time()
        logger.info(f"✅ Model warm-up completed in {end_time - start_time:.2f} seconds")
        logger.info(f"   Generated {duration:.2f} seconds of audio")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model warm-up failed: {e}", exc_info=True)
        return False


def transfer_model_to_gpu(model, device):
    """
    Transfers all components of the ChatterboxVC model to the specified GPU device.
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
    logger.info("🚀 INITIALIZING CHATTERBOX VC MODEL")
    logger.info("=" * 60)
    
    # Step 1: Load model on CPU
    logger.info("📥 Step 1: Loading model on CPU...")
    start_time = time.time()
    model = ChatterboxVC.from_pretrained("cpu")
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


def convert(src_wav_path, ref_wav_path, temperature, seed_num, cfgw, min_p, top_p, repetition_penalty):
    """
    Convert voice using the global model.
    """
    if GLOBAL_MODEL is None:
        raise RuntimeError("Model not initialized!")

    if seed_num != 0:
        set_seed(int(seed_num))

    logger.info(f"🎤 Converting voice...")
    logger.info(f"   Source: {src_wav_path}")
    logger.info(f"   Reference: {ref_wav_path}")
    start_time = time.time()

    result = GLOBAL_MODEL.convert(
        src_wav_path,
        ref_wav_path,
        temperature=temperature,
        cfg_weight=cfgw,
        min_p=min_p,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    end_time = time.time()
    if result:
        duration = result[1].shape[-1] / result[0]
        total_time = end_time - start_time
        rtf = total_time / duration
        logger.info(f"✅ Generated {duration:.2f}s of audio in {total_time:.2f}s (RTF: {rtf:.2f})")
    else:
        logger.error("❌ Voice conversion failed")
        
    return result


# Initialize model before creating the interface
logger.info("🚀 Starting Chatterbox VC Application...")
try:
    GLOBAL_MODEL = load_and_prepare_model()
except Exception as e:
    logger.error(f"Failed to initialize model: {e}", exc_info=True)
    raise

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ Chatterbox Voice Conversion")
    gr.Markdown("Model loaded and warmed up. Ready for fast inference!")
    
    with gr.Row():
        with gr.Column():
            src_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Source Audio File")
            ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File")
            cfg_weight = gr.Slider(0.0, 1, step=.05, label="CFG", value=0.5)

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="temperature", value=.7)
                min_p = gr.Slider(0.00, 1.00, step=0.01, label="min_p", value=0.00)
                top_p = gr.Slider(0.00, 1.00, step=0.01, label="top_p", value=0.80)
                repetition_penalty = gr.Slider(1.00, 2.00, step=0.1, label="repetition_penalty", value=1.2)

            run_btn = gr.Button("Convert", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

    run_btn.click(
        fn=convert,
        inputs=[
            src_wav,
            ref_wav,
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
