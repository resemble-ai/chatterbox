"""
FastAPI REST API server for Chatterbox TTS.
"""

import os
import io
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
import torch
import torchaudio

# Try to import pydub for mp3 support
try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
from chatterbox.tts_turbo import ChatterboxTurboTTS
from chatterbox.vc import ChatterboxVC
from chatterbox.voice_bank import VoiceBank
from chatterbox.batch_processor import BatchProcessor

app = FastAPI(title="Chatterbox API Server", version="1.0.0")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals for lazy loading
models = {}
device = "cpu"
voice_bank = None
batch_processor = None

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_models():
    global models, device, voice_bank, batch_processor
    if not models:
        device = get_device()
        print(f"Loading models on {device}...")
        try:
            models['standard'] = ChatterboxTTS().to(device)
            models['multilingual'] = ChatterboxMultilingualTTS().to(device)
            models['turbo'] = ChatterboxTurboTTS().to(device)
            models['vc'] = ChatterboxVC().to(device)
            print("Models loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load some models: {e}")
            
    if not voice_bank:
        voice_bank = VoiceBank()
        
    if not batch_processor:
        batch_processor = BatchProcessor(models=models)

@app.on_event("startup")
async def startup_event():
    global device
    device = get_device()
    print(f"Starting server... Device detected: {device}")
    print("Models will be lazily loaded on first request.")

class TTSRequest(BaseModel):
    text: str
    lang: str = "en"
    model_type: str = "multilingual"
    emotion: Optional[str] = None
    exaggeration: float = 0.5
    temperature: float = 0.8
    output_format: str = "wav"

@app.post("/v1/tts")
async def generate_tts(request: TTSRequest, voice_ref: Optional[UploadFile] = File(None)):
    load_models()
    
    if request.model_type not in models:
        raise HTTPException(status_code=400, detail=f"Invalid model_type. Available: {list(models.keys())}")
        
    model = models[request.model_type]
    
    audio_prompt_path = None
    if voice_ref:
        temp_path = f"temp_{voice_ref.filename}"
        with open(temp_path, "wb") as f:
            f.write(await voice_ref.read())
        audio_prompt_path = temp_path
        
    try:
        if request.model_type == 'multilingual':
            audio = model.generate(request.text, audio_prompt_path=audio_prompt_path, lang=request.lang)
        elif request.model_type == 'standard':
            audio = model.generate(request.text, audio_prompt_path=audio_prompt_path, 
                                   exaggeration=request.exaggeration, temperature=request.temperature)
        else: # turbo
            audio = model.generate(request.text, audio_prompt_path=audio_prompt_path)
            
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio, model.sr, format="wav")
        audio_bytes = buffer.getvalue()
        
        if request.output_format.lower() == "mp3":
            if not HAS_PYDUB:
                raise HTTPException(status_code=400, detail="MP3 output requested but pydub/ffmpeg not available.")
            
            wav_io = io.BytesIO(audio_bytes)
            audio_seg = AudioSegment.from_wav(wav_io)
            mp3_io = io.BytesIO()
            audio_seg.export(mp3_io, format="mp3")
            audio_bytes = mp3_io.getvalue()
            media_type = "audio/mpeg"
        else:
            media_type = "audio/wav"
            
        return Response(content=audio_bytes, media_type=media_type)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if audio_prompt_path and os.path.exists(audio_prompt_path):
            os.remove(audio_prompt_path)

@app.post("/v1/tts/batch")
async def submit_batch_tts(request: TTSRequest):
    load_models()
    global batch_processor
    if not batch_processor.worker_task:
        await batch_processor.start_worker()
        
    job_id = await batch_processor.submit(
        text=request.text,
        lang=request.lang,
        model_type=request.model_type,
        emotion=request.emotion
    )
    return {"job_id": job_id}

@app.get("/v1/tts/batch/{job_id}")
async def get_batch_status(job_id: str):
    load_models()
    try:
        job = await batch_processor.get_status(job_id)
        return {
            "id": str(job.id),
            "status": job.status,
            "error": job.error,
            "created_at": job.created_at,
            "completed_at": job.completed_at
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")

@app.get("/v1/tts/batch/{job_id}/audio")
async def get_batch_audio(job_id: str):
    load_models()
    try:
        audio_bytes = await batch_processor.get_result(job_id)
        return Response(content=audio_bytes, media_type="audio/wav")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")

@app.get("/v1/languages")
async def list_languages():
    return {"languages": SUPPORTED_LANGUAGES}

@app.get("/v1/models")
async def list_models():
    return {"models": ["standard", "multilingual", "turbo", "vc"]}

@app.get("/v1/emotions")
async def list_emotions():
    return {"emotions": ["happy", "sad", "angry", "calm", "excited"]}

@app.get("/v1/voices")
async def list_voices():
    load_models()
    return {"voices": voice_bank.list_voices()}

@app.post("/v1/voices")
async def register_voice(
    name: str = Form(...),
    lang: str = Form("en"),
    gender: str = Form("neutral"),
    audio_files: List[UploadFile] = File(...)
):
    load_models()
    temp_paths = []
    for f in audio_files:
        path = f"temp_vb_{f.filename}"
        with open(path, "wb") as out:
            out.write(await f.read())
        temp_paths.append(path)
        
    try:
        # Assuming voice bank accepts list of files
        voice_bank.register_voice(name, temp_paths, lang=lang, gender=gender)
        return {"status": "success", "voice": name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in temp_paths:
            if os.path.exists(p):
                os.remove(p)

@app.delete("/v1/voices/{name}")
async def delete_voice(name: str):
    load_models()
    try:
        voice_bank.delete_voice(name)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok", "device": device}
