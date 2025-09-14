"""
FastAPI REST service for Chatterbox Multilingual TTS
"""

import asyncio
import logging
import base64
from typing import Optional, Dict, Any
import torch
import torchaudio
import soundfile as sf

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi import UploadFile, File, Form
from pydantic import BaseModel
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from database import audio_db, get_audio_prompt_path, cleanup_temp_file
from openai import OpenAI
import io
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chatterbox TTS Service (Multilingual)",
    description="REST-based text-to-speech service using Chatterbox Multilingual TTS",
    version="1.1.0"
)

# Global model instance
model: Optional[ChatterboxMultilingualTTS] = None

async def initialize_model():
    """Initialize the Chatterbox Multilingual TTS model"""
    global model
    try:
        logger.info("Initializing Chatterbox Multilingual TTS model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Initialize model in a thread to avoid blocking startup
        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(None, ChatterboxMultilingualTTS.from_pretrained, device)
        
        logger.info("Model initialized successfully")
        
        # Warm up the model with a short test generation
        logger.info("Warming up model...")
        try:
            warmup_start = asyncio.get_event_loop().time()
            _ = await loop.run_in_executor(
                None,
                lambda: model.generate("Hello", language_id="en", exaggeration=0.5, cfg_weight=0.5)
            )
            warmup_time = asyncio.get_event_loop().time() - warmup_start
            logger.info(f"Model warmup completed in {warmup_time:.2f}s")
        except Exception as warmup_error:
            logger.warning(f"Model warmup failed (this is usually fine): {warmup_error}")
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize model and database on startup"""
    # Initialize database connection
    try:
        await audio_db.connect()
    except Exception as e:
        logger.warning(f"Database connection failed (continuing without database): {e}")
    
    # Initialize TTS model
    await initialize_model()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    await audio_db.disconnect()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {"status": "healthy", "model_loaded": True}

# Pydantic models for request/response
class TTSRequest(BaseModel):
    text: str
    audio_prompt_path: Optional[str] = None
    actor_name: Optional[str] = None
    emotion: Optional[str] = None
    language_id: Optional[str] = "en"
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.7

class TTSResponse(BaseModel):
    message: str
    audio_duration: float
    sample_rate: int

class UploadResponse(BaseModel):
    id: str
    message: str

class TranscribeResponse(BaseModel):
    text: str
    language: Optional[str] = None

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Chatterbox TTS Service (Multilingual)",
        "endpoints": {
            "health": "/health",
            "actors": "/actors (GET)",
            "emotions": "/emotions (GET)",
            "generate": "/generate (POST)",
            "docs": "/docs"
        }
    }

@app.get("/actors")
async def list_actors():
    """Get list of available actors from database"""
    try:
        actors = await audio_db.list_actors()
        return {
            "actors": actors,
            "count": len(actors)
        }
    except Exception as e:
        logger.error(f"Error fetching actors: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch actors")

@app.get("/emotions")
async def list_emotions(actor_name: Optional[str] = None):
    """Get list of available emotions, optionally filtered by actor"""
    try:
        emotions = await audio_db.list_emotions(actor_name)
        return {
            "emotions": emotions,
            "count": len(emotions),
            "filtered_by_actor": actor_name
        }
    except Exception as e:
        logger.error(f"Error fetching emotions: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch emotions")

@app.post("/upload", response_model=UploadResponse)
async def upload_audio_prompt(
    actor_name: str = Form(...),
    emotion: str = Form(...),
    language_code: str = Form(...),
    transcription: str = Form(""),
    auto_transcribe: bool = Form(False),
    openai_api_key: Optional[str] = Form(None),
    file: UploadFile = File(...)
):
    """
    Upload a WAV file with transcription and metadata to store as an audio prompt in DB.
    Accessible from FastAPI docs as a form upload.
    """
    try:
        if file.content_type not in ("audio/wav", "audio/x-wav", "application/octet-stream"):
            raise HTTPException(status_code=400, detail="File must be a WAV audio")

        wav_bytes = await file.read()
        if not wav_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Auto-transcribe if requested and transcription empty
        if auto_transcribe and (not transcription or not transcription.strip()):
            try:
                api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise HTTPException(status_code=400, detail="Missing OpenAI API key for transcription")
                client = OpenAI(api_key=api_key)
                audio_file = (file.filename or "audio.wav", io.BytesIO(wav_bytes))
                result = client.audio.transcriptions.create(model="gpt-4o-transcribe", file=audio_file)
                text = getattr(result, 'text', None) or (result.get('text') if isinstance(result, dict) else None)
                if not text:
                    raise HTTPException(status_code=502, detail="Transcription failed")
                transcription = text
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Auto-transcription failed: {e}")
                raise HTTPException(status_code=500, detail="Auto-transcription failed")

        inserted_id = await audio_db.add_audio_prompt(
            actor_name=actor_name,
            emotion=emotion,
            transcription=transcription,
            language_code=language_code,
            wav_bytes=wav_bytes,
            original_file_name=file.filename,
            extra_metadata={"content_type": file.content_type}
        )
        if not inserted_id:
            raise HTTPException(status_code=500, detail="Failed to insert document")

        return UploadResponse(id=inserted_id, message="Audio prompt uploaded successfully")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    openai_api_key: Optional[str] = Form(None),
    model: str = Form("gpt-4o-transcribe")
):
    """
    Proxy transcription endpoint using OpenAI's gpt-4o-transcribe.
    Accepts audio upload and returns transcription text.
    """
    try:
        raw_bytes = await file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=400, detail="Missing OpenAI API key")

        client = OpenAI(api_key=api_key)
        audio_file = (file.filename or "audio.wav", io.BytesIO(raw_bytes))
        result = client.audio.transcriptions.create(
            model=model,
            file=audio_file
        )
        text = getattr(result, 'text', None) or (result.get('text') if isinstance(result, dict) else None)
        if not text:
            raise HTTPException(status_code=502, detail="Transcription failed")
        return TranscribeResponse(text=text)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail="Transcription error")

@app.get("/audio-prompt/{actor_name}/{emotion}")
async def get_audio_prompt_info(actor_name: str, emotion: str):
    """Get information about a specific audio prompt"""
    try:
        info = await audio_db.get_audio_prompt_info(actor_name, emotion)
        if not info:
            raise HTTPException(status_code=404, detail=f"No audio prompt found for actor '{actor_name}' with emotion '{emotion}'")
        return info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching audio prompt info: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch audio prompt info")

@app.post("/generate")
async def generate_tts(request: TTSRequest):
    """
    Generate complete TTS audio and return as WAV file
    
    - **text**: Text to synthesize (required)
    - **audio_prompt_path**: Path to reference audio for voice cloning (optional)
    - **actor_name**: Actor name for database audio prompt lookup (optional)
    - **emotion**: Emotion for database audio prompt lookup (optional)
    - **language_id**: Two-letter language code (e.g. "en", "fr", "zh")
    - **exaggeration**: Emotion intensity control (0.0-1.0+, default: 0.5)
    - **cfg_weight**: Classifier-free guidance weight (0.0-1.0, default: 0.5)
    - **temperature**: Sampling randomness (0.1-1.0, default: 0.7)
    
    Note: If both audio_prompt_path and actor_name/emotion are provided, audio_prompt_path takes precedence.
    
    Returns a WAV audio file
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    logger.info(f"Processing non-streaming TTS request: '{request.text[:50]}...'")
    
    # Get audio prompt path (from database or direct path)
    audio_prompt_path = await get_audio_prompt_path(
        request.actor_name, 
        request.emotion, 
        request.audio_prompt_path
    )
    
    # Require an audio prompt if no default voice (conds) is available
    if audio_prompt_path is None and getattr(model, "conds", None) is None:
        raise HTTPException(status_code=400, detail="No default voice configured. Provide audio_prompt_path or configure a default voice.")
    
    # Determine language: prefer DB language when using DB prompt; default to 'en' if missing
    language_source = "request"
    language_id_to_use = request.language_id or "en"
    used_db_prompt_flag = (
        request.actor_name is not None and request.emotion is not None and request.audio_prompt_path is None and audio_prompt_path is not None
    )
    if used_db_prompt_flag:
        try:
            info = await audio_db.get_audio_prompt_info(request.actor_name, request.emotion)
            if info:
                db_lang = info.get("language_code")
                language_id_to_use = (db_lang.strip() if isinstance(db_lang, str) and db_lang.strip() else "en")
                language_source = "db"
        except Exception as _e:
            # Fall back to request/default
            language_id_to_use = request.language_id or "en"
            language_source = "request"

    temp_file_to_cleanup = None
    if audio_prompt_path != request.audio_prompt_path:
        # This is a temporary file from database, mark for cleanup
        temp_file_to_cleanup = audio_prompt_path
    
    try:
        if audio_prompt_path:
            logger.info(f"Using audio prompt: {audio_prompt_path}")
        else:
            logger.info("No audio prompt specified, using default voice")
        
        # Generate audio using the non-streaming method
        start_time = asyncio.get_event_loop().time()
        
        # Run generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        audio_tensor = await loop.run_in_executor(
            None,
            lambda: model.generate(
                text=request.text,
                audio_prompt_path=audio_prompt_path,
                language_id=language_id_to_use,
                exaggeration=request.exaggeration,
                cfg_weight=request.cfg_weight,
                temperature=request.temperature
            )
        )
        
        generation_time = asyncio.get_event_loop().time() - start_time
        
        # Convert to WAV bytes
        audio_bytes = tensor_to_wav_bytes(audio_tensor, model.sr)
        
        # Calculate audio duration
        audio_duration = audio_tensor.shape[-1] / model.sr
        
        logger.info(f"Generated audio: {audio_duration:.2f}s in {generation_time:.2f}s (RTF: {generation_time/audio_duration:.3f})")
        
        # Prepare response headers
        headers = {
            "Content-Disposition": "attachment; filename=generated_audio.wav",
            "X-Audio-Duration": str(audio_duration),
            "X-Sample-Rate": str(model.sr),
            "X-Generation-Time": str(generation_time),
            "X-RTF": str(generation_time/audio_duration)
        }
        
        # Add audio prompt info to headers if used
        if request.actor_name:
            headers["X-Actor-Name"] = request.actor_name
        if request.emotion:
            headers["X-Emotion"] = request.emotion
        if audio_prompt_path:
            headers["X-Used-Audio-Prompt"] = "true"
        if language_id_to_use:
            headers["X-Language-ID"] = language_id_to_use
            headers["X-Language-Source"] = language_source
        
        # Return WAV file
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers=headers
        )
        
    except Exception as e:
        logger.error(f"Error during TTS generation: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")
    finally:
        # Clean up temporary file if created
        if temp_file_to_cleanup:
            await cleanup_temp_file(temp_file_to_cleanup)

@app.post("/generate/json")
async def generate_tts_json(request: TTSRequest):
    """
    Generate TTS audio and return as JSON with base64-encoded audio
    
    Alternative endpoint that returns JSON instead of binary WAV file.
    Useful for web applications that need structured responses.
    
    - **text**: Text to synthesize (required)
    - **audio_prompt_path**: Path to reference audio for voice cloning (optional)
    - **actor_name**: Actor name for database audio prompt lookup (optional)
    - **emotion**: Emotion for database audio prompt lookup (optional)
    - **language_id**: Two-letter language code (e.g. "en", "fr", "zh")
    - **exaggeration**: Emotion intensity control (0.0-1.0+, default: 0.5)
    - **cfg_weight**: Classifier-free guidance weight (0.0-1.0, default: 0.5)
    - **temperature**: Sampling randomness (0.1-1.0, default: 0.7)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    logger.info(f"Processing JSON TTS request: '{request.text[:50]}...'")
    
    # Get audio prompt path (from database or direct path)
    audio_prompt_path = await get_audio_prompt_path(
        request.actor_name, 
        request.emotion, 
        request.audio_prompt_path
    )

    # Require an audio prompt if no default voice (conds) is available
    if audio_prompt_path is None and getattr(model, "conds", None) is None:
        raise HTTPException(status_code=400, detail="No default voice configured. Provide audio_prompt_path or configure a default voice.")
    
    # Determine language: prefer DB language when using DB prompt; default to 'en' if missing
    language_source = "request"
    language_id_to_use = request.language_id or "en"
    used_db_prompt_flag = (
        request.actor_name is not None and request.emotion is not None and request.audio_prompt_path is None and audio_prompt_path is not None
    )
    if used_db_prompt_flag:
        try:
            info = await audio_db.get_audio_prompt_info(request.actor_name, request.emotion)
            if info:
                db_lang = info.get("language_code")
                language_id_to_use = (db_lang.strip() if isinstance(db_lang, str) and db_lang.strip() else "en")
                language_source = "db"
        except Exception as _e:
            language_id_to_use = request.language_id or "en"
            language_source = "request"

    temp_file_to_cleanup = None
    if audio_prompt_path != request.audio_prompt_path:
        temp_file_to_cleanup = audio_prompt_path
    
    try:
        if audio_prompt_path:
            logger.info(f"Using audio prompt: {audio_prompt_path}")
        else:
            logger.info("No audio prompt specified, using default voice")
        
        # Generate audio using the non-streaming method
        start_time = asyncio.get_event_loop().time()
        
        # Run generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        audio_tensor = await loop.run_in_executor(
            None,
            lambda: model.generate(
                text=request.text,
                audio_prompt_path=audio_prompt_path,
                language_id=language_id_to_use,
                exaggeration=request.exaggeration,
                cfg_weight=request.cfg_weight,
                temperature=request.temperature
            )
        )
        
        generation_time = asyncio.get_event_loop().time() - start_time
        
        # Convert to WAV bytes and encode as base64
        audio_bytes = tensor_to_wav_bytes(audio_tensor, model.sr)
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Calculate audio duration
        audio_duration = audio_tensor.shape[-1] / model.sr
        
        logger.info(f"Generated JSON audio: {audio_duration:.2f}s in {generation_time:.2f}s (RTF: {generation_time/audio_duration:.3f})")
        
        response_data = {
            "message": "TTS generation completed",
            "audio": audio_base64,
            "audio_duration": audio_duration,
            "sample_rate": model.sr,
            "generation_time": generation_time,
            "rtf": generation_time / audio_duration,
            "format": "wav"
        }
        
        # Add audio prompt info if used
        if request.actor_name:
            response_data["actor_name"] = request.actor_name
        if request.emotion:
            response_data["emotion"] = request.emotion
        if audio_prompt_path:
            response_data["used_audio_prompt"] = True
        if language_id_to_use:
            response_data["language_id"] = language_id_to_use
            response_data["language_source"] = language_source
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error during JSON TTS generation: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")
    finally:
        # Clean up temporary file if created
        if temp_file_to_cleanup:
            await cleanup_temp_file(temp_file_to_cleanup)


def tensor_to_wav_bytes(audio_tensor: torch.Tensor, sample_rate: int) -> bytes:
    """Convert audio tensor to WAV bytes"""
    import io
    import numpy as np

    # Ensure tensor is on CPU and correct shape
    audio_cpu = audio_tensor.detach().cpu()
    # Collapse single-channel shape (1, T) to (T)
    if audio_cpu.dim() == 2 and audio_cpu.size(0) == 1:
        audio_cpu = audio_cpu.squeeze(0)

    # Convert to numpy in (T,) or (T, C) format for soundfile
    if audio_cpu.dim() == 2:
        # torch shape is (C, T) -> transpose to (T, C)
        audio_np = audio_cpu.transpose(0, 1).numpy()
    else:
        audio_np = audio_cpu.numpy()

    buffer = io.BytesIO()
    sf.write(buffer, audio_np, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
