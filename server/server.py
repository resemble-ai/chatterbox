"""
Chatterbox streaming TTS server.

Streams audio chunks as float32 PCM over HTTP chunked transfer encoding.
Frame format: [4-byte little-endian int32 = sample count] [float32 samples...]
Special frames:
  sample count = 0  → end-of-stream sentinel
  sample count = -1 → error frame: [4-byte msg length][UTF-8 message]

Endpoints:
  POST /generate          Stream TTS audio (JSON body, see GenerateRequest)
  GET  /speakers          List uploaded reference voices
  POST /upload-audio      Upload a reference audio file for voice cloning
  DELETE /speakers/{id}   Remove an uploaded reference voice
  GET  /models            List available model names
  GET  /languages         List supported language codes (multilingual model)
  GET  /health            Server health + loaded models
  GET  /                  HTML test client

Usage:
    cd /path/to/chatterbox
    python server/server.py [--model base|multilingual|turbo] [--device cuda|cpu|mps] [--port 8000]
"""

import argparse
import io
import os
import pathlib
import struct
import sys
import tempfile
import uuid
import wave

# Make sure the chatterbox package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(
    title="Chatterbox Streaming TTS",
    description="Streaming text-to-speech server with voice cloning support.",
    version="0.1.0",
)

# Allow the HTML client to call the API from any origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "DELETE"],
    allow_headers=["*"],
)

# Serve the HTML client at /
app.mount("/static", StaticFiles(directory=os.path.dirname(__file__)), name="static")


# ─── Model registry ──────────────────────────────────────────────────────────

_loaded: dict = {}  # { model_key: model_instance }
_cond_cache: dict = {}  # { (model_key, audio_id, exaggeration): Conditionals }
_default_dtype = None  # set at startup from --dtype; used when requests omit dtype


def get_device(device_str: str) -> str:
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_str


_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def parse_dtype(dtype_str: str):
    """Convert a dtype string ('fp16', 'bf16', 'fp32') to a torch.dtype, or None."""
    effective = dtype_str if dtype_str is not None else _default_dtype
    if effective is None:
        return None
    dtype = _DTYPE_MAP.get(effective.lower())
    if dtype is None:
        raise ValueError(f"Unknown dtype '{effective}'. Choose from: fp16, bf16, fp32.")
    return dtype


def load_model(model_name: str, device: str, dtype=None):
    key = f"{model_name}:{device}:{dtype}"
    if key in _loaded:
        return _loaded[key]

    # Evict any previously loaded model to free VRAM before loading the new one
    for old_key in list(_loaded.keys()):
        print(f"Unloading '{old_key}' to free VRAM…")
        del _loaded[old_key]
        # Drop cached conditionals tied to the evicted model
        for ck in [k for k in _cond_cache if k[0] == old_key]:
            del _cond_cache[ck]
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    dtype_label = dtype if dtype is not None else "default"
    print(f"Loading model '{model_name}' on {device} (dtype={dtype_label})…")

    if model_name == "base":
        from chatterbox.tts import ChatterboxTTS
        model = ChatterboxTTS.from_pretrained(device=device, dtype=dtype)
    elif model_name == "multilingual":
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        model = ChatterboxMultilingualTTS.from_pretrained(device=device, dtype=dtype)
    elif model_name == "turbo":
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        model = ChatterboxTurboTTS.from_pretrained(device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown model '{model_name}'")

    _loaded[key] = model
    print(f"Model '{model_name}' ready.")
    return model


# ─── Speaker registry ────────────────────────────────────────────────────────

_speakers: dict = {}  # { audio_id: {"name": str, "path": str} }

# ─── Debug flag ───────────────────────────────────────────────────────────────

DEBUG = False  # set to True via --debug CLI flag


# ─── Request / Response schema ────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize.")
    model: str = Field("multilingual", description="Model to use: base | multilingual | turbo.")
    device: str = Field("auto", description="Compute device: auto | cuda | cpu | mps.")
    dtype: Optional[str] = Field(None, description="Model precision: fp16 | bf16 | fp32 | null (server default). fp16/bf16 halve VRAM and can significantly improve throughput on CUDA GPUs.")
    language_id: str = Field("en", description="BCP-47 language code (multilingual model only).")
    audio_prompt_path: Optional[str] = Field(None, description="Server-side path to a reference audio file for voice cloning (API use).")
    audio_id: Optional[str] = Field(None, description="ID of a previously uploaded reference audio file (returned by POST /upload-audio).")
    exaggeration: float = Field(0.5, description="Emotion exaggeration factor (base/multilingual only).")
    cfg_weight: float = Field(0.5, description="Classifier-free guidance weight (base/multilingual only).")
    temperature: float = Field(0.8, description="Sampling temperature.")
    chunk_size: int = Field(25, description="Token chunk size for streaming. 25 tokens ≈ 1 second of audio.")
    # base / multilingual only
    repetition_penalty: float = Field(1.2, description="Repetition penalty (base/multilingual only).")
    min_p: float = Field(0.05, description="Min-p sampling threshold (base/multilingual only).")
    top_p: float = Field(1.0, description="Top-p (nucleus) sampling threshold.")
    cfm_steps: int = Field(10, description="CFM flow-matching steps for S3Gen vocoder (default 10). Lower = faster but may reduce quality. Try 4-6 for low-latency streaming.")
    # turbo only
    top_k: int = Field(1000, description="Top-k sampling (turbo only).")
    # sentence splitting
    sentence_split: bool = Field(False, description="Split text into sentences and stream each sequentially.")
    min_chars: int = Field(100, description="Minimum characters per sentence segment (sentence_split only).")
    max_chars: int = Field(250, description="Maximum characters per sentence segment (sentence_split only).")
    first_sentence_half: bool = Field(True, description="Use half of min/max limits for the first sentence to reduce time-to-first-audio (sentence_split only).")


# ─── Sentence splitter ────────────────────────────────────────────────────────

import re as _re

def split_sentences(
    text: str,
    min_chars: int,
    max_chars: int,
    first_min: int = None,
    first_max: int = None,
) -> list:
    """
    Split text into segments breaking at sentence-ending punctuation.

    Thresholds:
      - Segments below their min threshold are merged with the next piece.
      - Segments above their max threshold are force-split at a comma or space.
      - first_min / first_max apply only to the first output segment;
        all subsequent segments use min_chars / max_chars.
        If not provided, the same thresholds apply to all segments.

    When DEBUG is True, prints intermediate steps to the console.
    """
    eff_first_min = first_min if first_min is not None else min_chars
    eff_first_max = first_max if first_max is not None else max_chars

    # Step 1: split on sentence-ending punctuation
    raw = _re.split(r'(?<=[.!?。？！])\s+', text.strip())
    raw = [s.strip() for s in raw if s.strip()]

    if DEBUG:
        print(f"    punctuation split → {len(raw)} piece(s)")
        print(f"    thresholds: first=[{eff_first_min}, {eff_first_max}]  rest=[{min_chars}, {max_chars}]")
        for i, p in enumerate(raw):
            print(f"      [{i+1}] ({len(p)} chars) {p!r}")

    # Step 2: merge pieces that are below their min threshold into the next piece.
    # A piece already at or above its min threshold is flushed as its own segment.
    # TODO: if a piece is below min but merging with the next would exceed max,
    #       it still gets merged and force-split in step 3 (see TODO.md).
    segments = []
    current = ""
    for part in raw:
        cur_min = eff_first_min if len(segments) == 0 else min_chars
        if not current:
            current = part
        elif len(current) < cur_min:
            current = current + " " + part  # too short — merge
        else:
            segments.append(current)        # in range — flush
            current = part
    if current:
        segments.append(current)

    if DEBUG and len(segments) != len(raw):
        print(f"    after merge → {len(segments)} segment(s):")
        for i, s in enumerate(segments):
            print(f"      [{i+1}] ({len(s)} chars) {s!r}")

    # Step 3: force-split any segment exceeding its max threshold.
    # First segment uses eff_first_max; all others use max_chars.
    # Prefer splitting at a comma, fall back to last space before the limit.
    result = []
    for seg_idx, seg in enumerate(segments):
        cur_max = eff_first_max if seg_idx == 0 else max_chars
        first_piece = True
        while len(seg) > cur_max:
            split_at = seg.rfind(',', 0, cur_max)
            if split_at > 0:
                split_at += 1  # keep comma with left part
            else:
                split_at = seg.rfind(' ', 0, cur_max)
                if split_at <= 0:
                    split_at = cur_max
            result.append(seg[:split_at].strip())
            seg = seg[split_at:].strip()
            # After the first piece is split off, remaining pieces of this
            # segment are no longer "first" and use the normal max.
            if first_piece and seg_idx == 0:
                first_piece = False
                cur_max = max_chars
        if seg:
            result.append(seg)

    if DEBUG and len(result) != len(segments):
        print(f"    after force-split → {len(result)} segment(s):")
        for i, s in enumerate(result):
            print(f"      [{i+1}] ({len(s)} chars) {s!r}")

    return result


# ─── Per-sentence generator helper ───────────────────────────────────────────

def _make_gen(model, req: "GenerateRequest", text: str, audio_path):
    """Return a generate_stream generator for the given model/text/audio_path."""
    if req.model == "base":
        return model.generate_stream(
            text=text,
            audio_prompt_path=audio_path,
            exaggeration=req.exaggeration,
            cfg_weight=req.cfg_weight,
            temperature=req.temperature,
            repetition_penalty=req.repetition_penalty,
            min_p=req.min_p,
            top_p=req.top_p,
            chunk_size=req.chunk_size,
        )
    elif req.model == "multilingual":
        return model.generate_stream(
            text=text,
            language_id=req.language_id,
            audio_prompt_path=audio_path,
            exaggeration=req.exaggeration,
            cfg_weight=req.cfg_weight,
            temperature=req.temperature,
            repetition_penalty=req.repetition_penalty,
            min_p=req.min_p,
            top_p=req.top_p,
            chunk_size=req.chunk_size,
            cfm_steps=req.cfm_steps,
        )
    elif req.model == "turbo":
        return model.generate_stream(
            text=text,
            audio_prompt_path=audio_path,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            chunk_size=req.chunk_size,
        )
    else:
        raise ValueError(f"Unknown model '{req.model}'")


# ─── Streaming generator ─────────────────────────────────────────────────────

def audio_stream(req: "GenerateRequest", resolved_audio_path):
    """
    Generator that yields binary frames: 4-byte sample count + float32 PCM.
    Sends a final zero-length frame to signal end-of-stream.
    resolved_audio_path is the already-validated audio file path (or None).
    """
    import time

    device = get_device(req.device)
    dtype = parse_dtype(req.dtype)
    model = load_model(req.model, device, dtype=dtype)
    model_key = f"{req.model}:{device}:{req.dtype}"

    # Pre-compute and cache conditionals (mel extraction + speaker embedding + S3 tokenization)
    # so the same reference audio is never processed more than once per model/exaggeration combo.
    if resolved_audio_path and req.audio_id:
        cond_key = (model_key, req.audio_id, req.exaggeration)
        if cond_key not in _cond_cache:
            print(f"  Computing conditionals for speaker '{req.audio_id}' (exaggeration={req.exaggeration})…")
            model.prepare_conditionals(resolved_audio_path, exaggeration=req.exaggeration)
            _cond_cache[cond_key] = model.conds
        else:
            model.conds = _cond_cache[cond_key]
        resolved_audio_path = None  # conds already set; skip re-preparation inside generate_stream

    def make_frame(audio_tensor) -> bytes:
        samples = audio_tensor.squeeze().cpu().numpy().astype("float32")
        header = struct.pack("<i", len(samples))
        return header + samples.tobytes()

    # Build sentence list
    if req.sentence_split:
        first_min = req.min_chars // 2 if req.first_sentence_half else None
        first_max = req.max_chars // 2 if req.first_sentence_half else None
        if DEBUG:
            print(f"  Sentence split (min={req.min_chars}, max={req.max_chars}, first_half={req.first_sentence_half}):")
        sentences = split_sentences(
            req.text, req.min_chars, req.max_chars,
            first_min=first_min, first_max=first_max,
        )
        print(f"  Sentence split → {len(sentences)} segment(s):")
        for i, s in enumerate(sentences):
            preview = s[:60] + "…" if len(s) > 60 else s
            print(f"    [{i+1}] ({len(s)} chars) {preview!r}")
    else:
        sentences = [req.text]

    gen_start = time.time()
    total_chunks = 0
    total_audio_s = 0.0
    first_ttfa_logged = False

    try:
        for i, sentence in enumerate(sentences):
            # Pass audio path only for the first sentence; model reuses conds after
            sentence_audio_path = resolved_audio_path if i == 0 else None
            gen = _make_gen(model, req, sentence, sentence_audio_path)

            for audio_chunk, metrics in gen:
                yield make_frame(audio_chunk)
                total_chunks += 1
                samples_np = audio_chunk.squeeze().cpu().numpy()
                total_audio_s += len(samples_np) / 24000

                if not first_ttfa_logged and metrics.latency_to_first_chunk:
                    first_ttfa_logged = True
                    print(f"  ⚡ First chunk latency: {metrics.latency_to_first_chunk:.3f}s")

        # End-of-stream sentinel (zero samples)
        yield struct.pack("<i", 0)

        gen_time = time.time() - gen_start
        if total_audio_s > 0:
            rtf = gen_time / total_audio_s
            print(f"  ✅ Done. RTF={rtf:.3f}  chunks={total_chunks}  "
                  f"total={gen_time:.2f}s  audio={total_audio_s:.2f}s")

    except Exception as e:
        import traceback
        traceback.print_exc()
        # Send an error frame: -1 sample count + UTF-8 error message
        msg = str(e).encode()
        yield struct.pack("<i", -1) + struct.pack("<i", len(msg)) + msg


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def index():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(os.path.dirname(__file__), "client.html"))


@app.post("/generate", summary="Stream TTS audio")
def generate(req: GenerateRequest):
    """
    Generate speech from text and stream it as binary audio frames.

    Response body is a sequence of length-prefixed float32 PCM frames at 24000 Hz:
    - Normal frame: `[int32 sample_count][float32 × sample_count]`
    - End-of-stream: `[int32 = 0]`
    - Error frame: `[int32 = -1][int32 msg_len][UTF-8 message]`

    The `X-Sample-Rate` response header carries the sample rate (always 24000).
    """
    # Resolve audio prompt path before starting the generator
    resolved_audio_path = req.audio_prompt_path
    if req.audio_id:
        if req.audio_id not in _speakers:
            raise HTTPException(status_code=404, detail=f"Speaker '{req.audio_id}' not found. Upload it first via POST /upload-audio.")
        resolved_audio_path = _speakers[req.audio_id]["path"]

    return StreamingResponse(
        audio_stream(req, resolved_audio_path),
        media_type="application/octet-stream",
        headers={"X-Sample-Rate": "24000"},
    )


@app.post("/generate/wav", summary="Generate TTS audio as a WAV file")
def generate_wav(req: GenerateRequest):
    """
    Generate speech and return it as a standard WAV file (16-bit PCM, 24000 Hz, mono).

    Accepts the same request body as `POST /generate` but waits for the full audio
    to be generated before responding. Use this for simple integrations that don't
    need low-latency streaming. For streaming, use `POST /generate` instead.

    Example:
        curl -X POST http://localhost:8000/generate/wav \\
             -H "Content-Type: application/json" \\
             -d '{"text": "Hello world", "model": "multilingual", "language_id": "en"}' \\
             --output speech.wav
    """
    from fastapi.responses import Response

    resolved_audio_path = req.audio_prompt_path
    if req.audio_id:
        if req.audio_id not in _speakers:
            raise HTTPException(status_code=404, detail=f"Speaker '{req.audio_id}' not found.")
        resolved_audio_path = _speakers[req.audio_id]["path"]

    # Drain the streaming generator and collect all float32 samples
    all_samples = []
    for frame in audio_stream(req, resolved_audio_path):
        if len(frame) < 4:
            continue
        sample_count = struct.unpack_from("<i", frame, 0)[0]
        if sample_count == 0:
            break  # end-of-stream sentinel
        if sample_count == -1:
            msg_len = struct.unpack_from("<i", frame, 4)[0]
            msg = frame[8:8 + msg_len].decode("utf-8", errors="replace")
            raise HTTPException(status_code=500, detail=msg)
        samples = np.frombuffer(frame[4:], dtype="float32")
        all_samples.append(samples)

    if not all_samples:
        raise HTTPException(status_code=500, detail="No audio was generated.")

    # Encode as 16-bit PCM WAV
    pcm16 = (np.clip(np.concatenate(all_samples), -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(pcm16.tobytes())

    return Response(
        content=buf.getvalue(),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
    )


@app.get("/speakers", summary="List uploaded reference voices")
def list_speakers():
    """Return all uploaded reference audio files available for voice cloning."""
    return [
        {"audio_id": aid, "name": info["name"]}
        for aid, info in _speakers.items()
    ]


@app.post("/upload-audio", summary="Upload a reference voice for cloning")
async def upload_audio(file: UploadFile = File(...)):
    """
    Upload an audio file (WAV/MP3/FLAC) to use as a reference voice for cloning.
    The file must be at least 5 seconds long.
    Returns an `audio_id` that can be passed to `POST /generate`.
    """
    original_name = pathlib.Path(file.filename).stem if file.filename else "speaker"
    suffix = pathlib.Path(file.filename).suffix if file.filename else ".wav"

    # Save to a temp file that persists for the server lifetime
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
    finally:
        tmp.close()

    audio_id = uuid.uuid4().hex
    _speakers[audio_id] = {"name": original_name, "path": tmp.name}
    print(f"Uploaded speaker '{original_name}' → {tmp.name} (id={audio_id})")
    return {"audio_id": audio_id, "name": original_name}


@app.delete("/speakers/{audio_id}", status_code=204, summary="Remove an uploaded reference voice")
def delete_speaker(audio_id: str):
    """Delete an uploaded reference voice and its temp file."""
    if audio_id not in _speakers:
        raise HTTPException(status_code=404, detail=f"Speaker '{audio_id}' not found.")
    info = _speakers.pop(audio_id)
    try:
        os.unlink(info["path"])
    except OSError:
        pass
    # Drop any cached conditionals for this speaker
    for ck in [k for k in _cond_cache if k[1] == audio_id]:
        del _cond_cache[ck]
    print(f"Deleted speaker '{info['name']}' (id={audio_id})")


@app.get("/models", summary="List available models")
def list_models():
    """Return the list of supported model names."""
    return {"models": ["base", "multilingual", "turbo"]}


@app.get("/languages", summary="List supported languages")
def list_languages():
    """Return a dict of BCP-47 code → language name supported by the multilingual model."""
    from chatterbox.mtl_tts import SUPPORTED_LANGUAGES
    return SUPPORTED_LANGUAGES


@app.get("/health", summary="Server health check")
def health():
    """Return server status and currently loaded models."""
    return {"status": "ok", "loaded_models": list(_loaded.keys())}


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="multilingual",
                        choices=["base", "multilingual", "turbo"],
                        help="Model to pre-load on startup (default: multilingual)")
    parser.add_argument("--device", default="auto",
                        help="Device: auto|cuda|cpu|mps (default: auto)")
    parser.add_argument("--dtype", default=None, choices=["fp16", "bf16", "fp32"],
                        help="Model precision: fp16 | bf16 | fp32 (default: keep weights as-is). "
                             "fp16/bf16 halve VRAM usage and can significantly improve throughput on CUDA GPUs.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose debug logging (sentence split steps, etc.)")
    parser.add_argument("--warmup", action="store_true",
                        help="Run a short dummy synthesis after loading to trigger CUDA kernel JIT compilation")
    args = parser.parse_args()

    DEBUG = args.debug
    if DEBUG:
        print("🐛 Debug logging enabled")

    _default_dtype = args.dtype  # requests that omit dtype will use this

    # Enable TF32 tensor cores for ~2x faster float32 matmuls on Ampere+ GPUs
    torch.set_float32_matmul_precision('high')

    device = get_device(args.device)
    dtype = parse_dtype(args.dtype)
    print(f"Pre-loading '{args.model}' on {device}…")
    load_model(args.model, device, dtype=dtype)

    if args.warmup:
        print("Warming up (triggering CUDA kernel JIT compilation)…")
        _warmup_req = GenerateRequest(text="ok", model=args.model, device=args.device, dtype=args.dtype)
        for _ in audio_stream(_warmup_req, None):
            pass
        print("Warmup complete.")

    print(f"\n🎙  Chatterbox streaming server running at http://{args.host}:{args.port}")
    print(f"   Open http://{args.host}:{args.port} in your browser")
    print(f"   API docs at http://{args.host}:{args.port}/docs\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
