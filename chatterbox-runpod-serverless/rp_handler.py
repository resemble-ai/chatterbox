import runpod
import torch
import torchaudio
import nltk
import os
import time
import base64
import tempfile
import requests
from io import BytesIO
import logging
from typing import Optional, Dict, Any, List
import traceback

# Import your ChatterboxTTS functionality
try:
    from chatterbox.tts import ChatterboxTTS
except ImportError as e:
    print(f"Warning: Could not import ChatterboxTTS: {e}")
    ChatterboxTTS = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model caching
tts_model = None
device = None

def decode_audio_prompt_base64(audio_base64: str) -> Optional[str]:
    """Decode base64 audio data and save to temporary file."""
    try:
        logger.info("Decoding base64 audio prompt...")
        
        # Decode base64 data
        audio_data = base64.b64decode(audio_base64)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_data)
            tmp_path = tmp_file.name
        
        logger.info(f"Audio prompt saved to: {tmp_path}")
        return tmp_path
    except Exception as e:
        logger.error(f"Failed to decode audio prompt: {e}")
        return None

def ensure_nltk_data():
    """Ensure NLTK punkt tokenizer is available."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
        logger.info("NLTK punkt tokenizer found")
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer...")
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            logger.info("NLTK punkt tokenizer downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download NLTK punkt tokenizer: {e}")
            raise

def initialize_model():
    """Initialize the ChatterboxTTS model. Called once when worker starts."""
    global tts_model, device
    
    if tts_model is not None:
        logger.info("Model already initialized")
        return tts_model
    
    logger.info("Initializing ChatterboxTTS model...")
    
    if ChatterboxTTS is None:
        raise RuntimeError("ChatterboxTTS module not available")
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"CUDA available - using GPU: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        logger.warning("CUDA not available - using CPU (slower)")
    
    try:
        # Initialize model with proper device mapping
        # If loading on CPU but model was saved on CUDA, we need to handle device mapping
        if device == "cpu":
            # Force map model weights from CUDA to CPU if needed
            logger.info("Loading model on CPU - will handle CUDA->CPU mapping if needed")
            # Try to set the mapping for torch.load operations
            old_load = torch.load
            def new_load_on_cpu(*args, **kwargs):
                # Always force map_location to 'cpu' when running on CPU
                kwargs['map_location'] = 'cpu'
                logger.debug("Forcing map_location='cpu' for torch.load")
                return old_load(*args, **kwargs)
            torch.load = new_load_on_cpu
            
        tts_model = ChatterboxTTS.from_pretrained(device=device)
        
        # Restore original torch.load if we modified it
        if device == "cpu":
            torch.load = old_load
            
        logger.info(f"Model loaded successfully on {device}")
        logger.info(f"Model sample rate: {tts_model.sr} Hz")
        
        # Ensure NLTK data is available
        ensure_nltk_data()
        
        return tts_model
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        logger.error(traceback.format_exc())
        
        # If first attempt failed and we're on CPU, try alternative approach
        if device == "cpu":
            logger.info("Retrying model loading with explicit CPU mapping...")
            try:
                # Try setting torch default device
                with torch.no_grad():
                    torch.set_default_tensor_type('torch.FloatTensor')
                    tts_model = ChatterboxTTS.from_pretrained(device=device)
                    logger.info(f"Model loaded successfully on {device} (retry)")
                    logger.info(f"Model sample rate: {tts_model.sr} Hz")
                    ensure_nltk_data()
                    return tts_model
            except Exception as e2:
                logger.error(f"Retry also failed: {e2}")
                logger.error(traceback.format_exc())
        
        raise

def split_text_into_chunks(text: str, max_chars_per_chunk: int = 300) -> List[str]:
    """Split text into manageable chunks using NLTK sentence tokenizer."""
    try:
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if adding this sentence exceeds the limit
            if len(current_chunk) + len(sentence) + 1 > max_chars_per_chunk and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
            else:
                current_chunk += sentence + " "
            
            # Add the last chunk
            if i == len(sentences) - 1 and current_chunk.strip():
                chunks.append(current_chunk.strip())
        
        logger.info(f"Text split into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting text: {e}")
        # Fallback: return original text as single chunk
        return [text]

def generate_audio_chunk(
    text_chunk: str,
    model: ChatterboxTTS,
    audio_prompt_path: Optional[str] = None,
    temperature: float = 0.8,
    cfg_weight: float = 0.5,
    exaggeration: float = 0.5
) -> Optional[torch.Tensor]:
    """Generate audio tensor for a single text chunk."""
    try:
        logger.info(f"Generating audio for chunk: {text_chunk[:50]}...")
        
        # Check if audio prompt exists
        effective_prompt_path = None
        if audio_prompt_path and os.path.exists(audio_prompt_path):
            effective_prompt_path = audio_prompt_path
        elif audio_prompt_path:
            logger.warning(f"Audio prompt path not found: {audio_prompt_path}")
        
        # Generate audio
        wav_tensor = model.generate(
            text_chunk,
            audio_prompt_path=effective_prompt_path,
            temperature=temperature,
            cfg_weight=cfg_weight,
            exaggeration=exaggeration
        )
        
        # Ensure tensor is on CPU and properly shaped
        wav_tensor_cpu = wav_tensor.cpu().float()
        
        # Ensure tensor is 2D: [channels, samples]
        if wav_tensor_cpu.ndim == 1:
            wav_tensor_cpu = wav_tensor_cpu.unsqueeze(0)
        elif wav_tensor_cpu.ndim > 2:
            logger.warning(f"Unexpected tensor shape {wav_tensor_cpu.shape}, attempting to fix")
            wav_tensor_cpu = wav_tensor_cpu.squeeze()
            if wav_tensor_cpu.ndim == 1:
                wav_tensor_cpu = wav_tensor_cpu.unsqueeze(0)
            elif wav_tensor_cpu.ndim != 2 or wav_tensor_cpu.shape[0] != 1:
                logger.error(f"Could not reshape tensor {wav_tensor.shape} to [1, N]")
                return None
        
        return wav_tensor_cpu
        
    except Exception as e:
        logger.error(f"Error generating audio chunk: {e}")
        logger.error(traceback.format_exc())
        return None

def text_to_speech_pipeline(
    text: str,
    model: ChatterboxTTS,
    max_chars_per_chunk: int = 300,
    inter_chunk_silence_ms: int = 350,
    audio_prompt_path: Optional[str] = None,
    temperature: float = 0.8,
    cfg_weight: float = 0.5,
    exaggeration: float = 0.5
) -> Optional[torch.Tensor]:
    """Convert text to speech with chunking support."""
    try:
        # Split text into chunks
        text_chunks = split_text_into_chunks(text, max_chars_per_chunk)
        
        if not text_chunks:
            logger.error("No text chunks to process")
            return None
        
        all_audio_tensors = []
        sample_rate = model.sr
        
        logger.info(f"Processing {len(text_chunks)} chunks at {sample_rate} Hz")
        
        for i, chunk_text in enumerate(text_chunks):
            logger.info(f"Processing chunk {i+1}/{len(text_chunks)}")
            
            chunk_tensor = generate_audio_chunk(
                chunk_text,
                model,
                audio_prompt_path,
                temperature,
                cfg_weight,
                exaggeration
            )
            
            if chunk_tensor is None:
                logger.warning(f"Skipping chunk {i+1} due to generation error")
                continue
            
            all_audio_tensors.append(chunk_tensor)
            
            # Add silence between chunks (except after the last chunk)
            if i < len(text_chunks) - 1 and inter_chunk_silence_ms > 0:
                silence_samples = int(sample_rate * inter_chunk_silence_ms / 1000.0)
                silence_tensor = torch.zeros(
                    (1, silence_samples),
                    dtype=chunk_tensor.dtype,
                    device=chunk_tensor.device
                )
                all_audio_tensors.append(silence_tensor)
        
        if not all_audio_tensors:
            logger.error("No audio tensors generated")
            return None
        
        # Concatenate all audio tensors
        logger.info("Concatenating audio tensors...")
        final_audio_tensor = torch.cat(all_audio_tensors, dim=1)
        
        logger.info(f"Final audio shape: {final_audio_tensor.shape}")
        return final_audio_tensor
        
    except Exception as e:
        logger.error(f"Error in text-to-speech pipeline: {e}")
        logger.error(traceback.format_exc())
        return None

def audio_tensor_to_base64(audio_tensor: torch.Tensor, sample_rate: int) -> str:
    """Convert audio tensor to base64 encoded WAV data."""
    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            torchaudio.save(tmp_file.name, audio_tensor, sample_rate)
            
            # Read back as binary data
            with open(tmp_file.name, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            # Clean up temporary file
            os.unlink(tmp_file.name)
            
            # Encode as base64
            return base64.b64encode(audio_data).decode('utf-8')
            
    except Exception as e:
        logger.error(f"Error converting audio to base64: {e}")
        raise

def validate_input(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize input parameters."""
    # Required parameters
    if 'text' not in job_input:
        raise ValueError("Missing required parameter: 'text'")
    
    text = job_input['text'].strip()
    if not text:
        raise ValueError("Text parameter cannot be empty")
    
    # Optional parameters with defaults
    validated_input = {
        'text': text,
        'audio_prompt_base64': job_input.get('audio_prompt_base64'),
        'max_chars_per_chunk': job_input.get('max_chars_per_chunk', 300),
        'inter_chunk_silence_ms': job_input.get('inter_chunk_silence_ms', 350),
        'temperature': job_input.get('temperature', 0.8),
        'cfg_weight': job_input.get('cfg_weight', 0.5),
        'exaggeration': job_input.get('exaggeration', 0.5),
        'output_format': job_input.get('output_format', 'wav')
    }
    
    # Validate parameter ranges
    if not (0 <= validated_input['temperature'] <= 2.0):
        raise ValueError("Temperature must be between 0.0 and 2.0")
    
    if not (0 <= validated_input['cfg_weight'] <= 1.0):
        raise ValueError("cfg_weight must be between 0.0 and 1.0")
    
    if not (0 <= validated_input['exaggeration'] <= 1.0):
        raise ValueError("exaggeration must be between 0.0 and 1.0")
    
    if validated_input['max_chars_per_chunk'] < 50:
        raise ValueError("max_chars_per_chunk must be at least 50")
    
    if validated_input['inter_chunk_silence_ms'] < 0:
        raise ValueError("inter_chunk_silence_ms must be non-negative")
    
    logger.info(f"Input validation successful for text length: {len(text)} characters")
    return validated_input

def handler(job):
    """Main handler function for RunPod serverless worker."""
    start_time = time.time()
    
    try:
        logger.info("=== ChatterboxTTS Handler Started ===")
        
        # Initialize model if not already done
        model = initialize_model()
        
        # Extract and validate input
        job_input = job.get('input', {})
        logger.info(f"Received input keys: {list(job_input.keys())}")
        
        validated_input = validate_input(job_input)
        
        # Process audio prompt if provided
        audio_prompt_path = None
        if validated_input['audio_prompt_base64']:
            audio_prompt_path = decode_audio_prompt_base64(validated_input['audio_prompt_base64'])
            if audio_prompt_path is None:
                logger.warning("Failed to decode audio prompt, proceeding without it")
        
        # Generate audio
        logger.info("Starting text-to-speech generation...")
        audio_tensor = text_to_speech_pipeline(
            text=validated_input['text'],
            model=model,
            max_chars_per_chunk=validated_input['max_chars_per_chunk'],
            inter_chunk_silence_ms=validated_input['inter_chunk_silence_ms'],
            audio_prompt_path=audio_prompt_path,
            temperature=validated_input['temperature'],
            cfg_weight=validated_input['cfg_weight'],
            exaggeration=validated_input['exaggeration']
        )
        
        if audio_tensor is None:
            raise RuntimeError("Failed to generate audio")
        
        # Convert to base64
        logger.info("Converting audio to base64...")
        audio_base64 = audio_tensor_to_base64(audio_tensor, model.sr)
        
        # Calculate metadata
        duration_seconds = audio_tensor.shape[1] / model.sr
        processing_time = time.time() - start_time
        
        # Clean up temporary files
        if audio_prompt_path and os.path.exists(audio_prompt_path):
            try:
                os.unlink(audio_prompt_path)
                logger.info("Cleaned up temporary audio prompt file")
            except Exception as e:
                logger.warning(f"Failed to clean up audio prompt file: {e}")
        
        # Prepare response
        response = {
            "status": "success",
            "audio_base64": audio_base64,
            "metadata": {
                "duration_seconds": round(duration_seconds, 2),
                "sample_rate": model.sr,
                "num_chunks": len(split_text_into_chunks(validated_input['text'], validated_input['max_chars_per_chunk'])),
                "processing_time_seconds": round(processing_time, 2),
                "text_length": len(validated_input['text']),
                "audio_shape": list(audio_tensor.shape)
            }
        }
        
        logger.info(f"=== Handler Completed Successfully in {processing_time:.2f}s ===")
        return response
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_message = str(e)
        logger.error(f"Handler failed after {processing_time:.2f}s: {error_message}")
        logger.error(traceback.format_exc())
        
        return {
            "status": "error",
            "error": error_message,
            "metadata": {
                "processing_time_seconds": round(processing_time, 2)
            }
        }

# Initialize model when worker starts (outside handler for caching)
if __name__ == '__main__':
    logger.info("Starting ChatterboxTTS Serverless Worker...")
    
    try:
        # Pre-initialize model for faster subsequent requests
        logger.info("Pre-initializing model...")
        initialize_model()
        logger.info("Model pre-initialization completed")
        
        # Start the serverless worker
        runpod.serverless.start({
            'handler': handler,
            'return_aggregate_stream': False  # We return complete results, not streaming
        })
        
    except Exception as e:
        logger.error(f"Failed to start worker: {e}")
        logger.error(traceback.format_exc())
        raise
