import asyncio
import os
import re
import tempfile
import threading
from pathlib import Path

import soundfile as sf
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS
from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from chunking import narration_chunks


API_KEY = os.environ["TTS_API_KEY"]
VOICE_DIR = Path("/voices")
MODEL_KIND = os.environ.get("CHATTERBOX_MODEL", "turbo")
model = ChatterboxTurboTTS.from_pretrained(device="cuda", nano=MODEL_KIND == "nano")
model_lock = threading.Lock()

app = FastAPI()


class SpeechRequest(BaseModel):
    input: str
    model: str = "chatterbox-turbo"
    voice: str | None = None
    response_format: str = "wav"


def reference_voice(name: str | None) -> str | None:
    if not name:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise HTTPException(400, "Invalid voice name")
    path = VOICE_DIR / f"{name}.wav"
    if not path.is_file():
        raise HTTPException(404, "Voice not found")
    return str(path)


def pcm_s16le(audio: torch.Tensor) -> bytes:
    samples = audio.squeeze().float().cpu().clamp(-1, 1)
    return (samples * 32767).to(torch.int16).numpy().tobytes()


def generate_audio(text: str, voice: str | None) -> torch.Tensor:
    """Run the process-global model one request at a time."""
    with model_lock:
        return model.generate(text, audio_prompt_path=voice, norm_loudness=False)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_KIND, "device": torch.cuda.get_device_name(0)}


@app.post("/v1/audio/speech")
def speech(request: SpeechRequest, authorization: str | None = Header(default=None)):
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(401, "Invalid API key")
    if request.response_format != "wav":
        raise HTTPException(400, "Only wav output is supported")
    if not request.input.strip():
        raise HTTPException(400, "Input cannot be empty")

    voice = reference_voice(request.voice)
    audio = generate_audio(request.input, voice)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output:
        sf.write(output.name, audio.squeeze().float().cpu().numpy(), model.sr)
        path = output.name
    return FileResponse(
        path,
        media_type="audio/wav",
        filename="speech.wav",
        background=BackgroundTask(os.unlink, path),
    )


@app.websocket("/v1/audio/speech/stream")
async def speech_stream(websocket: WebSocket):
    """Emit one binary PCM frame per synthesized narration chunk."""
    authorization = websocket.headers.get("authorization")
    supplied_key = websocket.query_params.get("api_key")
    if authorization != f"Bearer {API_KEY}" and supplied_key != API_KEY:
        await websocket.close(code=1008, reason="Invalid API key")
        return

    await websocket.accept()
    try:
        request = SpeechRequest.model_validate(await websocket.receive_json())
        if request.response_format != "pcm_s16le":
            await websocket.send_json({"type": "error", "message": "Use response_format=pcm_s16le"})
            await websocket.close(code=1003)
            return
        if not request.input.strip():
            await websocket.send_json({"type": "error", "message": "Input cannot be empty"})
            await websocket.close(code=1003)
            return

        voice = reference_voice(request.voice)
        chunks = narration_chunks(request.input)
        await websocket.send_json(
            {
                "type": "start",
                "format": "pcm_s16le",
                "sample_rate": model.sr,
                "channels": 1,
                "chunks": len(chunks),
            }
        )
        for index, chunk in enumerate(chunks):
            audio = await asyncio.to_thread(generate_audio, chunk.text, voice)
            await websocket.send_json(
                {"type": "chunk", "index": index, "text": chunk.text, "boundary": chunk.boundary}
            )
            await websocket.send_bytes(pcm_s16le(audio))
        await websocket.send_json({"type": "end"})
        await websocket.close(code=1000, reason="complete")
    except WebSocketDisconnect:
        return
    except HTTPException as error:
        await websocket.send_json({"type": "error", "message": error.detail})
        await websocket.close(code=1003)
    except Exception:
        await websocket.send_json({"type": "error", "message": "Speech generation failed"})
        await websocket.close(code=1011)
