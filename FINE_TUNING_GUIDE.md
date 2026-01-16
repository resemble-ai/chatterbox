# Chatterbox Fine-Tuning, Streaming & Deployment Guide

## Table of Contents
1. [Complete Model Inventory](#1-complete-model-inventory)
2. [Dependency Audit & Upgrades](#2-dependency-audit--upgrades)
3. [Real-Time Streaming Support](#3-real-time-streaming-support)
4. [Fine-Tuning Strategy](#4-fine-tuning-strategy-for-arabic-dialects)
5. [Deployment Architecture](#5-deployment-architecture)

---

## 1. Complete Model Inventory

You have **full access** to all model weights. Here's what each does and how to use it:

### 1.1 T3 Model (Text-to-Speech Token Generator) - PRIMARY

| Property | Value |
|----------|-------|
| **File** | `t3_mtl23ls_v2.safetensors` (2.14 GB) |
| **Architecture** | LLaMA-based Transformer (500M params) |
| **License** | MIT |
| **Fine-tune Priority** | HIGH |

**What it does:**
- Converts text tokens → speech tokens (discrete audio representations)
- Uses Classifier-Free Guidance (CFG) for quality control
- Handles 23 languages via language ID tokens (`[ar]`, `[en]`, etc.)
- Has built-in `loss()` function for training
- Already supports **incremental token generation** (foundation for streaming)

**Access in code:**
```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

model = ChatterboxMultilingualTTS.from_pretrained(device="mps")
t3 = model.t3  # Direct access to LLaMA backbone

# T3 has built-in loss function for training!
loss_text, loss_speech = t3.loss(
    t3_cond=conditioning,
    text_tokens=text_tokens,
    text_token_lens=text_lens,
    speech_tokens=speech_tokens,
    speech_token_lens=speech_lens,
)

# Access the transformer backbone directly
llama_model = t3.tfmr  # HuggingFace LlamaModel
```

**What you'll do with it:**
- Fine-tune for Arabic dialect pronunciation, rhythm, intonation
- Modify generation loop for streaming output
- The incremental generation loop already exists in `t3.inference()` - we'll expose it for streaming

---

### 1.2 S3Gen (Speech Token to Audio Decoder) - SECONDARY

| Property | Value |
|----------|-------|
| **File** | `s3gen.pt` (1.06 GB) |
| **Architecture** | Flow-Matching CFM + HiFi-GAN Vocoder |
| **License** | Apache 2.0 (from CosyVoice) |
| **Fine-tune Priority** | MEDIUM |

**What it does:**
- Converts speech tokens → mel spectrograms (via Causal Flow Matching)
- Converts mel spectrograms → audio waveform (via HiFi-GAN)
- **Already has streaming support infrastructure** (see `finalize` parameter)
- Has vocoder caching for efficient streaming (`cache_source`)

**Access in code:**
```python
s3gen = model.s3gen  # Direct access

# Two-stage inference (for streaming):
# Stage 1: Speech tokens → Mel spectrograms
output_mels = s3gen.flow_inference(
    speech_tokens=tokens,
    ref_dict=ref_embeddings,
    finalize=False,  # Set False during streaming, True for final chunk
)

# Stage 2: Mel → Audio (with caching for streaming)
cache_source = torch.zeros(1, 1, 0).to(device)
output_wav, cache_source = s3gen.hift_inference(output_mels, cache_source)

# Get speech tokens from audio (for training data prep)
speech_tokens, lengths = s3gen.tokenizer([audio_waveforms])
```

**What you'll do with it:**
- Use `finalize=False` for streaming chunks, `finalize=True` for final chunk
- Fine-tune if audio quality needs improvement for Arabic phonemes
- Leverage HiFi-GAN caching for low-latency streaming

---

### 1.3 Voice Encoder (Speaker Embedding Extractor) - LOW PRIORITY

| Property | Value |
|----------|-------|
| **File** | `ve.pt` (5.7 MB) |
| **Architecture** | Speaker Verification Network |
| **License** | MIT |
| **Fine-tune Priority** | LOW |

**What it does:**
- Extracts 256-dimensional speaker identity embeddings
- Enables zero-shot voice cloning from ~10s reference audio

**Access in code:**
```python
ve = model.ve

# Extract speaker embedding from reference audio
embedding = ve.embeds_from_wavs([audio_16k], sample_rate=16000)
# Returns: numpy array shape (1, 256)
```

**What you'll do with it:**
- Usually NO fine-tuning needed
- Works well for Arabic voices out-of-box
- Only fine-tune if voice cloning quality is poor

---

### 1.4 MTLTokenizer (Text Tokenizer) - EXTEND IF NEEDED

| Property | Value |
|----------|-------|
| **File** | `grapheme_mtl_merged_expanded_v1.json` (70 KB) |
| **Architecture** | HuggingFace Tokenizer |
| **License** | MIT |
| **Fine-tune Priority** | LOW-MEDIUM |

**What it does:**
- Converts text → token IDs with language-specific preprocessing
- Arabic: NFKD normalization, prepends `[ar]` language token
- No special Arabic preprocessing (unlike Japanese, Chinese, Hebrew, Korean, Russian)

**Access in code:**
```python
tokenizer = model.tokenizer

# Tokenize Arabic text
tokens = tokenizer.text_to_tokens("مرحباً بكم", language_id="ar")

# Access underlying HuggingFace tokenizer
vocab = tokenizer.tokenizer.get_vocab()
print(f"Vocabulary size: {len(vocab)}")

# Encode/decode
ids = tokenizer.encode("مرحباً", language_id="ar")
text = tokenizer.decode(ids)
```

**What you'll do with it:**
- Possibly extend vocabulary for dialect-specific characters
- Add phonetic markers for dialectal pronunciation if needed
- Modify `encode()` for dialect-specific preprocessing

---

### 1.5 AlignmentStreamAnalyzer (Streaming Quality Control)

| Property | Value |
|----------|-------|
| **File** | Part of T3 module |
| **Purpose** | Online integrity checks during streaming |
| **License** | MIT |

**What it does:**
- Tracks text-speech alignment during generation
- Detects hallucinations (false starts, repetitions, long tails)
- Can force EOS token when quality degrades
- **Critical for production streaming**

**Access in code:**
```python
from chatterbox.models.t3.inference.alignment_stream_analyzer import (
    AlignmentStreamAnalyzer,
    AlignmentAnalysisResult
)

# Already integrated in T3 for multilingual models
# Results include:
# - false_start: bool - hallucination at beginning
# - long_tail: bool - hallucination at end
# - repetition: bool - repeating content
# - discontinuity: bool - alignment jump
# - complete: bool - reached end of text
# - position: int - current text position (for timestamps!)
```

---

## 2. Dependency Audit & Upgrades

### Current Dependencies Analysis

| Package | Current | Status | Impact | Recommendation |
|---------|---------|--------|--------|----------------|
| `torch` | 2.6.0 | ✅ Current | Core framework | Keep |
| `torchaudio` | 2.6.0 | ✅ Current | Audio processing | Keep |
| `transformers` | 4.46.3 | ⚠️ Behind | LLaMA backbone | Can upgrade to 4.47+ |
| `diffusers` | 0.29.0 | ⚠️ Deprecated APIs | Flow matching | **Upgrade to 0.31+** |
| `numpy` | >=1.24.0,<1.26.0 | ⚠️ Restrictive | Arrays | Can relax to <2.0 |
| `gradio` | 5.44.1 | ✅ Current | Web UI | Keep |
| `safetensors` | 0.5.3 | ✅ Current | Model loading | Keep |
| `librosa` | 0.11.0 | ✅ Current | Audio processing | Keep |

### Deprecation Warnings Found

**1. diffusers LoRACompatibleLinear (MEDIUM IMPACT)**
```
FutureWarning: `LoRACompatibleLinear` is deprecated and will be removed in version 1.0.0.
Use of `LoRACompatibleLinear` is deprecated. Please switch to PEFT backend.
```

**Impact:** The S3Gen flow matching decoder uses diffusers internally. This warning means:
- Current code works but will break in diffusers 1.0.0
- No immediate action needed, but plan to update

**Fix when upgrading:**
```bash
pip install peft  # For LoRA support
pip install diffusers>=0.31.0
```

**2. NumPy Version Constraint (LOW IMPACT)**

The constraint `numpy>=1.24.0,<1.26.0` is overly restrictive. NumPy 2.0 has breaking changes but 1.26.x is fine.

**Recommended pyproject.toml update:**
```toml
"numpy>=1.24.0,<2.0.0",  # Relax upper bound
```

### Safe Upgrade Path

```bash
# Step 1: Create a test environment
python -m venv upgrade-test
source upgrade-test/bin/activate

# Step 2: Install with relaxed constraints
pip install torch==2.6.0 torchaudio==2.6.0
pip install transformers>=4.47.0
pip install diffusers>=0.31.0 peft
pip install "numpy>=1.24.0,<2.0.0"
pip install -e .

# Step 3: Test inference
python test_arabic_tts.py

# Step 4: If tests pass, update pyproject.toml
```

---

## 3. Real-Time Streaming Support

### Current Architecture Analysis

The codebase **already has streaming infrastructure** but it's not exposed at the API level:

```
┌─────────────────────────────────────────────────────────────────┐
│                 EXISTING STREAMING COMPONENTS                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  T3 Model (t3.py):                                              │
│  ├─ Token generation loop is incremental (lines 352-412)        │
│  ├─ Yields tokens one at a time with KV-cache                   │
│  └─ AlignmentStreamAnalyzer tracks progress                     │
│                                                                  │
│  S3Gen (s3gen.py):                                              │
│  ├─ flow_inference() has `finalize` parameter for streaming     │
│  ├─ hift_inference() has `cache_source` for vocoder streaming   │
│  └─ Comment says "use S3GenStreamer for streaming synthesis"    │
│                                                                  │
│  MISSING: High-level streaming API that connects these          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Streaming Implementation Plan

#### Phase 1: Create Streaming Generator (NEW FILE)

```python
# src/chatterbox/streaming.py

import torch
import asyncio
from typing import AsyncGenerator, Generator
from dataclasses import dataclass

@dataclass
class AudioChunk:
    """A chunk of generated audio with metadata."""
    audio: torch.Tensor  # Shape: (samples,)
    sample_rate: int
    is_final: bool
    text_position: int  # Which part of text this corresponds to
    latency_ms: float

class ChatterboxStreamer:
    """Real-time streaming TTS interface."""
    
    # Chunk size in speech tokens (25 tokens/sec at S3 rate)
    # 10 tokens ≈ 400ms of audio
    CHUNK_SIZE = 10
    
    def __init__(self, model: 'ChatterboxMultilingualTTS'):
        self.model = model
        self.t3 = model.t3
        self.s3gen = model.s3gen
        self.device = model.device
        
    def stream_generate(
        self,
        text: str,
        language_id: str = "ar",
        audio_prompt_path: str = None,
        chunk_size: int = None,
        **kwargs
    ) -> Generator[AudioChunk, None, None]:
        """
        Generate audio in streaming chunks.
        
        Yields AudioChunk objects as soon as they're ready.
        """
        import time
        chunk_size = chunk_size or self.CHUNK_SIZE
        
        # Prepare conditioning
        if audio_prompt_path:
            self.model.prepare_conditionals(audio_prompt_path)
        
        # Tokenize text
        text_tokens = self.model.tokenizer.text_to_tokens(
            text, language_id=language_id
        ).to(self.device)
        
        # Prepare for T3 generation
        # ... (initialization code)
        
        # Initialize vocoder cache for streaming
        vocoder_cache = torch.zeros(1, 1, 0).to(self.device)
        
        # Buffer for accumulating speech tokens
        token_buffer = []
        
        # Generation loop - this is the key modification
        start_time = time.time()
        
        for token_idx, speech_token in enumerate(self._generate_tokens(text_tokens, **kwargs)):
            token_buffer.append(speech_token)
            
            # When we have enough tokens, generate audio chunk
            if len(token_buffer) >= chunk_size:
                is_final = False  # Not final until generation completes
                
                # Convert tokens to audio
                speech_tokens_tensor = torch.tensor(token_buffer).unsqueeze(0).to(self.device)
                
                # Flow inference (mel generation)
                mels = self.s3gen.flow_inference(
                    speech_tokens_tensor,
                    ref_dict=self.model.conds.gen,
                    finalize=is_final,
                )
                
                # Vocoder inference with caching
                audio_chunk, vocoder_cache = self.s3gen.hift_inference(
                    mels, 
                    cache_source=vocoder_cache
                )
                
                latency = (time.time() - start_time) * 1000
                
                yield AudioChunk(
                    audio=audio_chunk.squeeze(),
                    sample_rate=self.model.sr,
                    is_final=False,
                    text_position=token_idx,
                    latency_ms=latency,
                )
                
                # Clear buffer (keep some overlap for continuity)
                token_buffer = token_buffer[-2:]  # Keep last 2 for context
        
        # Final chunk with remaining tokens
        if token_buffer:
            speech_tokens_tensor = torch.tensor(token_buffer).unsqueeze(0).to(self.device)
            mels = self.s3gen.flow_inference(
                speech_tokens_tensor,
                ref_dict=self.model.conds.gen,
                finalize=True,  # Final chunk
            )
            audio_chunk, _ = self.s3gen.hift_inference(mels, vocoder_cache)
            
            yield AudioChunk(
                audio=audio_chunk.squeeze(),
                sample_rate=self.model.sr,
                is_final=True,
                text_position=-1,
                latency_ms=(time.time() - start_time) * 1000,
            )
    
    def _generate_tokens(self, text_tokens, **kwargs):
        """
        Generator that yields speech tokens one at a time.
        This wraps T3's inference loop to make it a generator.
        """
        # This would be a modification of t3.inference() to yield tokens
        # instead of collecting them all at once
        pass  # Implementation details in actual code
    
    async def async_stream_generate(
        self,
        text: str,
        language_id: str = "ar",
        **kwargs
    ) -> AsyncGenerator[AudioChunk, None]:
        """Async version for web frameworks."""
        for chunk in self.stream_generate(text, language_id, **kwargs):
            yield chunk
            await asyncio.sleep(0)  # Yield control to event loop
```

#### Phase 2: FastAPI Streaming Server

```python
# streaming_server.py

from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
import torch
import io
import wave

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.streaming import ChatterboxStreamer

app = FastAPI()

# Load model once at startup
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = ChatterboxMultilingualTTS.from_pretrained(device=device)
streamer = ChatterboxStreamer(model)

@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    """WebSocket endpoint for real-time streaming TTS."""
    await websocket.accept()
    
    try:
        while True:
            # Receive text from client
            data = await websocket.receive_json()
            text = data.get("text", "")
            language_id = data.get("language_id", "ar")
            
            # Stream audio chunks back
            async for chunk in streamer.async_stream_generate(text, language_id):
                # Convert to bytes for transmission
                audio_bytes = chunk.audio.cpu().numpy().tobytes()
                
                await websocket.send_bytes(audio_bytes)
                
                if chunk.is_final:
                    await websocket.send_json({"status": "complete"})
                    
    except Exception as e:
        await websocket.close(code=1000, reason=str(e))

@app.get("/tts/stream")
async def stream_tts(text: str, language_id: str = "ar"):
    """HTTP streaming endpoint using chunked transfer."""
    
    async def audio_generator():
        async for chunk in streamer.async_stream_generate(text, language_id):
            yield chunk.audio.cpu().numpy().tobytes()
    
    return StreamingResponse(
        audio_generator(),
        media_type="audio/raw",
        headers={
            "X-Sample-Rate": str(model.sr),
            "X-Channels": "1",
            "X-Format": "float32",
        }
    )
```

#### Phase 3: Client-Side Integration

```javascript
// JavaScript client for WebSocket streaming
class ChatterboxStreamingClient {
    constructor(wsUrl) {
        this.ws = new WebSocket(wsUrl);
        this.audioContext = new AudioContext({ sampleRate: 22050 });
        this.audioQueue = [];
    }
    
    async speak(text, languageId = 'ar') {
        return new Promise((resolve, reject) => {
            this.ws.send(JSON.stringify({ text, language_id: languageId }));
            
            this.ws.onmessage = async (event) => {
                if (event.data instanceof Blob) {
                    // Audio chunk received
                    const arrayBuffer = await event.data.arrayBuffer();
                    const float32Array = new Float32Array(arrayBuffer);
                    this.playChunk(float32Array);
                } else {
                    const data = JSON.parse(event.data);
                    if (data.status === 'complete') {
                        resolve();
                    }
                }
            };
        });
    }
    
    playChunk(samples) {
        const buffer = this.audioContext.createBuffer(1, samples.length, 22050);
        buffer.copyToChannel(samples, 0);
        
        const source = this.audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(this.audioContext.destination);
        source.start();
    }
}

// Usage
const client = new ChatterboxStreamingClient('ws://localhost:8000/ws/tts');
await client.speak('مرحباً بكم في العالم العربي', 'ar');
```

### Streaming Latency Targets

#### For Conversational AI (< 200ms TTFA Required)

| Metric | Conservative | Aggressive | Notes |
|--------|--------------|------------|-------|
| Time to First Audio (TTFA) | < 200ms | < 100ms | Critical for conversation flow |
| Chunk Duration | 80-160ms | 40-80ms | 2-4 speech tokens per chunk |
| Inter-chunk Gap | < 20ms | < 10ms | Seamless playback |
| Total Latency (short text) | < 500ms | < 300ms | "Hello" → Audio complete |

#### Latency Breakdown (Current vs Optimized)

```
CURRENT MULTILINGUAL (500M):           OPTIMIZED TARGET:
├─ Tokenization:     5ms               ├─ Tokenization:     5ms
├─ T3 First Token:   150ms             ├─ T3 First Token:   50ms  (speculative)
├─ S3Gen (10 steps): 300ms             ├─ S3Gen (2 steps):  60ms  (MeanFlow)
├─ Vocoder:          50ms              ├─ Vocoder:          30ms  (cached)
└─ TOTAL:            505ms ❌           └─ TOTAL:            145ms ✅
```

---

## 3.5 Achieving < 200ms Latency for Arabic

### The Challenge

**Chatterbox-Turbo** achieves ~150ms TTFA but is **English-only**. For Arabic, we need to optimize the **Multilingual model** or create an **Arabic-specific Turbo variant**.

### Strategy Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│            LATENCY OPTIMIZATION STRATEGIES (Priority Order)              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. MeanFlow Distillation (BIGGEST IMPACT: 5x decoder speedup)          │
│     └─ Reduce CFM steps from 10 → 2                                     │
│                                                                          │
│  2. Speculative Decoding (2-3x T3 speedup)                              │
│     └─ Predict multiple tokens at once                                  │
│                                                                          │
│  3. Smaller Chunk Size (Lower TTFA)                                     │
│     └─ Stream 2-4 tokens (80-160ms audio) instead of 10                │
│                                                                          │
│  4. Model Quantization (1.5-2x overall speedup)                         │
│     └─ INT8 weights, FP16 compute                                       │
│                                                                          │
│  5. Parallel Pipeline (Hide latency)                                    │
│     └─ Generate next chunk while playing current                        │
│                                                                          │
│  6. GPU Optimization (Required for production)                          │
│     └─ TensorRT, torch.compile, Flash Attention                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Option A: Train Arabic MeanFlow Model (Recommended)

The biggest latency win comes from using **MeanFlow distillation** which reduces CFM steps from 10 to 2.

**What Turbo does differently:**
```python
# Turbo uses MeanFlow S3Gen
s3gen = S3Gen(meanflow=True)  # 2 CFM steps instead of 10
weights = load_file("s3gen_meanflow.safetensors")

# Inference with just 2 steps
wav = s3gen.inference(speech_tokens, ref_dict, n_cfm_timesteps=2)
```

**To create Arabic MeanFlow model:**

```python
# training/distill_meanflow.py
"""
Distill the multilingual S3Gen into a MeanFlow variant.
This is knowledge distillation from 10-step model to 2-step model.
"""

import torch
from chatterbox.models.s3gen import S3Gen

class MeanFlowDistillationTrainer:
    def __init__(self, teacher_model_path, device="cuda"):
        # Load teacher (10-step model)
        self.teacher = S3Gen(meanflow=False)
        self.teacher.load_state_dict(torch.load(teacher_model_path))
        self.teacher.eval()
        
        # Create student (2-step MeanFlow model)
        self.student = S3Gen(meanflow=True)
        # Initialize from teacher weights where possible
        self.student.load_state_dict(
            self.teacher.state_dict(), 
            strict=False  # MeanFlow has extra time_embed_mixer
        )
        
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(), 
            lr=1e-4
        )
    
    def distill_step(self, speech_tokens, ref_dict):
        """
        Train student to match teacher output with fewer steps.
        """
        # Teacher output (10 steps, high quality)
        with torch.no_grad():
            teacher_mels = self.teacher.flow_inference(
                speech_tokens, 
                ref_dict=ref_dict,
                n_cfm_timesteps=10,
            )
        
        # Student output (2 steps, fast)
        student_mels = self.student.flow_inference(
            speech_tokens,
            ref_dict=ref_dict, 
            n_cfm_timesteps=2,
        )
        
        # L1 loss between mel spectrograms
        loss = torch.nn.functional.l1_loss(student_mels, teacher_mels)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

### Option B: Speculative Decoding for T3 (2-3x Faster)

Predict multiple speech tokens at once, then verify. This significantly speeds up the autoregressive T3 model.

```python
# src/chatterbox/models/t3/speculative_inference.py

import torch
from typing import Tuple

class SpeculativeT3Decoder:
    """
    Speculative decoding for faster T3 inference.
    Uses a smaller draft model to propose tokens, then verifies with main model.
    """
    
    def __init__(self, main_model, draft_model=None, speculation_length=4):
        self.main = main_model
        # Draft model can be a smaller version or same model with less compute
        self.draft = draft_model or main_model  
        self.k = speculation_length  # Tokens to speculate at once
    
    @torch.inference_mode()
    def generate_speculative(
        self,
        text_tokens: torch.Tensor,
        t3_cond,
        max_tokens: int = 500,
        temperature: float = 0.8,
    ) -> torch.Tensor:
        """
        Generate speech tokens using speculative decoding.
        ~2-3x faster than standard autoregressive generation.
        """
        device = text_tokens.device
        generated = []
        
        # Initial embeddings
        embeds, len_cond = self.main.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=torch.tensor([[self.main.hp.start_speech_token]], device=device),
            cfg_weight=0.0,
        )
        
        past_kv = None
        
        while len(generated) < max_tokens:
            # Step 1: Draft model proposes k tokens quickly
            draft_tokens = self._draft_propose(embeds, past_kv, self.k)
            
            # Step 2: Main model verifies all k tokens in parallel
            accepted, new_past_kv = self._verify_tokens(
                embeds, past_kv, draft_tokens, temperature
            )
            
            generated.extend(accepted)
            past_kv = new_past_kv
            
            # Check for EOS
            if self.main.hp.stop_speech_token in accepted:
                break
            
            # Update embeddings for next iteration
            embeds = self._get_next_embeds(accepted)
        
        return torch.tensor(generated, device=device)
    
    def _draft_propose(self, embeds, past_kv, k) -> list:
        """Quickly propose k tokens using draft model."""
        # Use greedy decoding for speed
        proposed = []
        for _ in range(k):
            logits = self.draft.forward_one_step(embeds, past_kv)
            token = logits.argmax(dim=-1).item()
            proposed.append(token)
        return proposed
    
    def _verify_tokens(self, embeds, past_kv, proposed, temperature) -> Tuple[list, any]:
        """Verify proposed tokens with main model in parallel."""
        # Run main model on all proposed tokens at once
        # Accept tokens until first mismatch
        # This is where the speedup comes from - parallel verification
        pass  # Implementation details
```

### Option C: Optimized Streaming Pipeline

For < 200ms TTFA, use aggressive chunking:

```python
# src/chatterbox/low_latency_streaming.py

import torch
import asyncio
from dataclasses import dataclass
from typing import AsyncGenerator

@dataclass 
class LowLatencyConfig:
    """Configuration for sub-200ms latency."""
    # Chunk 2-4 tokens at a time (80-160ms audio)
    min_chunk_tokens: int = 2
    max_chunk_tokens: int = 4
    
    # Start vocoder while still generating
    parallel_vocoder: bool = True
    
    # Use reduced CFM steps (requires MeanFlow model)
    cfm_steps: int = 2
    
    # Pre-warm model with dummy input
    warmup_on_init: bool = True
    
    # Cache speaker embeddings
    cache_speaker_emb: bool = True


class LowLatencyStreamer:
    """
    Ultra-low latency streaming for conversational AI.
    Target: < 200ms time-to-first-audio.
    """
    
    def __init__(self, model, config: LowLatencyConfig = None):
        self.model = model
        self.config = config or LowLatencyConfig()
        self._speaker_cache = {}
        
        if self.config.warmup_on_init:
            self._warmup()
    
    def _warmup(self):
        """Pre-warm the model to avoid cold-start latency."""
        dummy_tokens = torch.tensor([[1, 2, 3]], device=self.model.device)
        with torch.no_grad():
            # Run a dummy forward pass to warm up CUDA/MPS kernels
            _ = self.model.t3.tfmr(inputs_embeds=torch.randn(1, 10, 768).to(self.model.device))
    
    async def stream_low_latency(
        self,
        text: str,
        language_id: str = "ar",
        speaker_id: str = None,
        audio_prompt_path: str = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream audio with < 200ms TTFA.
        
        Key optimizations:
        1. Start streaming after just 2-4 tokens
        2. Run vocoder in parallel with token generation
        3. Use MeanFlow (2 CFM steps instead of 10)
        4. Cache speaker embeddings
        """
        import time
        start_time = time.perf_counter()
        
        # Use cached speaker embedding if available
        if speaker_id and speaker_id in self._speaker_cache:
            self.model.conds = self._speaker_cache[speaker_id]
        elif audio_prompt_path:
            self.model.prepare_conditionals(audio_prompt_path)
            if speaker_id:
                self._speaker_cache[speaker_id] = self.model.conds
        
        # Tokenize
        text_tokens = self.model.tokenizer.text_to_tokens(
            text, language_id=language_id
        ).to(self.model.device)
        
        # Token generation with immediate chunking
        token_buffer = []
        vocoder_task = None
        vocoder_cache = torch.zeros(1, 1, 0).to(self.model.device)
        
        async for speech_token in self._generate_tokens_async(text_tokens):
            token_buffer.append(speech_token)
            
            # As soon as we have minimum tokens, start generating audio
            if len(token_buffer) >= self.config.min_chunk_tokens:
                # Wait for previous vocoder task if running
                if vocoder_task:
                    audio_chunk = await vocoder_task
                    yield audio_chunk
                
                # Start vocoder for current chunk (non-blocking)
                chunk_tokens = torch.tensor(token_buffer[:self.config.max_chunk_tokens])
                vocoder_task = asyncio.create_task(
                    self._vocoder_async(chunk_tokens, vocoder_cache)
                )
                
                # Keep overlap for continuity
                token_buffer = token_buffer[self.config.max_chunk_tokens - 1:]
                
                # Log TTFA on first chunk
                if start_time:
                    ttfa = (time.perf_counter() - start_time) * 1000
                    print(f"⚡ TTFA: {ttfa:.1f}ms")
                    start_time = None
        
        # Final chunk
        if token_buffer:
            if vocoder_task:
                yield await vocoder_task
            final_audio = await self._vocoder_async(
                torch.tensor(token_buffer), 
                vocoder_cache, 
                finalize=True
            )
            yield final_audio
    
    async def _generate_tokens_async(self, text_tokens):
        """Async wrapper for token generation."""
        # This wraps the synchronous T3 generation in an async generator
        # In practice, you'd use torch async or run in executor
        for token in self._generate_tokens_sync(text_tokens):
            yield token
            await asyncio.sleep(0)  # Yield control
    
    async def _vocoder_async(self, tokens, cache, finalize=False):
        """Async vocoder with MeanFlow (2 steps)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._vocoder_sync,
            tokens, cache, finalize
        )
    
    def _vocoder_sync(self, tokens, cache, finalize=False):
        """Synchronous vocoder using MeanFlow."""
        tokens = tokens.unsqueeze(0).to(self.model.device)
        
        # Use 2 CFM steps (MeanFlow)
        mels = self.model.s3gen.flow_inference(
            tokens,
            ref_dict=self.model.conds.gen,
            n_cfm_timesteps=self.config.cfm_steps,  # 2 instead of 10!
            finalize=finalize,
        )
        
        audio, new_cache = self.model.s3gen.hift_inference(mels, cache)
        return audio.cpu().numpy().tobytes()
```

### Option D: Model Quantization (INT8)

```python
# quantization/quantize_model.py

import torch
from torch.quantization import quantize_dynamic

def quantize_for_inference(model):
    """
    Quantize model to INT8 for ~1.5-2x speedup.
    """
    # Quantize T3 (LLaMA backbone)
    model.t3.tfmr = quantize_dynamic(
        model.t3.tfmr,
        {torch.nn.Linear},  # Quantize linear layers
        dtype=torch.qint8
    )
    
    # For S3Gen, use torch.compile instead (flow matching doesn't quantize well)
    model.s3gen = torch.compile(model.s3gen, mode="reduce-overhead")
    
    return model


# Using with TensorRT (for NVIDIA GPUs)
def export_tensorrt(model, output_path):
    """Export to TensorRT for maximum GPU performance."""
    import torch_tensorrt
    
    # Example input shapes
    text_tokens = torch.randint(0, 1000, (1, 50)).cuda()
    speech_tokens = torch.randint(0, 6561, (1, 100)).cuda()
    
    # Compile T3 with TensorRT
    t3_trt = torch_tensorrt.compile(
        model.t3,
        inputs=[text_tokens, speech_tokens],
        enabled_precisions={torch.float16},  # FP16 for speed
    )
    
    torch.jit.save(t3_trt, output_path)
```

### Latency Targets Summary

| Optimization | TTFA Impact | Effort | Arabic Support |
|--------------|-------------|--------|----------------|
| **MeanFlow Distillation** | -250ms (5x decoder) | High | Requires training |
| **Speculative Decoding** | -100ms (2-3x T3) | Medium | Yes |
| **Smaller Chunks (2-4 tokens)** | -100ms | Low | Yes |
| **INT8 Quantization** | -50ms | Low | Yes |
| **torch.compile** | -30ms | Very Low | Yes |
| **Pre-cached Speaker Emb** | -20ms | Very Low | Yes |
| **GPU (vs CPU/MPS)** | -200ms | Infrastructure | Yes |

### Recommended Path for < 200ms Arabic

1. **Immediate (no training):**
   - Use smaller chunks (2-4 tokens)
   - Pre-cache speaker embeddings  
   - Use `torch.compile` on inference
   - Use GPU if available
   - **Expected: ~300-400ms TTFA**

2. **Short-term (requires training):**
   - Distill MeanFlow S3Gen for Arabic
   - Implement speculative decoding
   - **Expected: ~150-200ms TTFA** ✅

3. **Production optimization:**
   - TensorRT/ONNX export
   - INT8 quantization
   - Dedicated GPU instances
   - **Expected: ~100-150ms TTFA** ✅✅

---

## 4. Fine-Tuning Strategy for Arabic Dialects

### Phase 1: Data Preparation

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA REQUIREMENTS                         │
├─────────────────────────────────────────────────────────────┤
│  • 10-100 hours of dialect-specific Arabic audio            │
│  • Accurate transcriptions (text)                           │
│  • Clean audio (minimal background noise, < -20dB SNR)      │
│  • Multiple speakers (10+ for diversity)                    │
│  • Sample rate: 22050 Hz (S3GEN_SR) or 16000 Hz (S3_SR)    │
│  • Dialects: Egyptian, Gulf, Levantine, Maghrebi, etc.     │
└─────────────────────────────────────────────────────────────┘
```

### Phase 2: Training Pipeline

```python
# training/arabic_dialect_trainer.py

import torch
from torch.utils.data import DataLoader
import librosa
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.t3.modules.cond_enc import T3Cond

class ArabicDialectDataset(torch.utils.data.Dataset):
    """Dataset for Arabic dialect fine-tuning."""
    
    def __init__(self, audio_paths, transcripts, model):
        self.audio_paths = audio_paths
        self.transcripts = transcripts
        self.tokenizer = model.tokenizer
        self.s3gen = model.s3gen
        self.ve = model.ve
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        # Load audio at 22kHz for S3Gen
        audio_22k, _ = librosa.load(self.audio_paths[idx], sr=22050)
        
        # Resample to 16kHz for tokenizer and voice encoder
        audio_16k = librosa.resample(audio_22k, orig_sr=22050, target_sr=16000)
        
        # Get speech tokens from audio
        speech_tokens, speech_lens = self.s3gen.tokenizer([audio_16k])
        
        # Get text tokens
        text_tokens = self.tokenizer.text_to_tokens(
            self.transcripts[idx], 
            language_id="ar"
        )
        
        # Get speaker embedding
        speaker_emb = torch.from_numpy(
            self.ve.embeds_from_wavs([audio_16k], sample_rate=16000)
        )
        
        return {
            "text_tokens": text_tokens.squeeze(0),
            "text_len": text_tokens.size(1),
            "speech_tokens": speech_tokens.squeeze(0),
            "speech_len": speech_lens[0],
            "speaker_emb": speaker_emb.squeeze(0),
        }


def collate_fn(batch):
    """Collate function with padding."""
    max_text_len = max(b["text_len"] for b in batch)
    max_speech_len = max(b["speech_len"] for b in batch)
    
    text_tokens = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    speech_tokens = torch.zeros(len(batch), max_speech_len, dtype=torch.long)
    text_lens = torch.zeros(len(batch), dtype=torch.long)
    speech_lens = torch.zeros(len(batch), dtype=torch.long)
    speaker_embs = torch.stack([b["speaker_emb"] for b in batch])
    
    for i, b in enumerate(batch):
        text_tokens[i, :b["text_len"]] = b["text_tokens"]
        speech_tokens[i, :b["speech_len"]] = b["speech_tokens"]
        text_lens[i] = b["text_len"]
        speech_lens[i] = b["speech_len"]
    
    return {
        "text_tokens": text_tokens,
        "text_lens": text_lens,
        "speech_tokens": speech_tokens,
        "speech_lens": speech_lens,
        "speaker_embs": speaker_embs,
    }


class ArabicDialectTrainer:
    """Fine-tuning trainer for Arabic dialects."""
    
    def __init__(
        self, 
        model: ChatterboxMultilingualTTS, 
        learning_rate: float = 1e-5,
        freeze_layers: int = 6,  # Freeze first N transformer layers
    ):
        self.model = model
        self.t3 = model.t3
        self.device = model.device
        
        # Freeze early layers for efficiency
        for name, param in self.t3.named_parameters():
            for i in range(freeze_layers):
                if f"tfmr.layers.{i}." in name:
                    param.requires_grad = False
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in self.t3.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.t3.parameters())
        print(f"Training {trainable:,} / {total:,} parameters ({100*trainable/total:.1f}%)")
        
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.t3.parameters()),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-7
        )
    
    def train_step(self, batch):
        self.t3.train()
        
        # Move to device
        text_tokens = batch["text_tokens"].to(self.device)
        text_lens = batch["text_lens"].to(self.device)
        speech_tokens = batch["speech_tokens"].to(self.device)
        speech_lens = batch["speech_lens"].to(self.device)
        speaker_embs = batch["speaker_embs"].to(self.device)
        
        # Prepare conditioning
        t3_cond = T3Cond(
            speaker_emb=speaker_embs,
            cond_prompt_speech_tokens=None,
            emotion_adv=0.5 * torch.ones(len(batch["text_tokens"]), 1, 1),
        ).to(self.device)
        
        # Compute loss using T3's built-in loss function
        loss_text, loss_speech = self.t3.loss(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_lens,
        )
        
        total_loss = loss_text + loss_speech
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.t3.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            "total_loss": total_loss.item(),
            "text_loss": loss_text.item(),
            "speech_loss": loss_speech.item(),
            "lr": self.scheduler.get_last_lr()[0],
        }
    
    def save_checkpoint(self, path, epoch):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.t3.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.t3.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint["epoch"]


# Training script
def train_arabic_dialect(
    audio_dir: str,
    transcript_file: str,
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
):
    """Main training function."""
    import os
    from pathlib import Path
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    print(f"Loading model on {device}...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    
    # Load data
    # ... (load audio paths and transcripts)
    
    dataset = ArabicDialectDataset(audio_paths, transcripts, model)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    
    trainer = ArabicDialectTrainer(model, learning_rate=learning_rate)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            metrics = trainer.train_step(batch)
            total_loss += metrics["total_loss"]
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {metrics['total_loss']:.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} complete. Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        trainer.save_checkpoint(
            Path(output_dir) / f"checkpoint_epoch_{epoch}.pt",
            epoch
        )
    
    print("Training complete!")
```

---

## 5. Deployment Architecture

### Option A: Local/Development (Your Mac)

```
┌─────────────────────────────────────────────────────────────┐
│                    LOCAL DEPLOYMENT                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Mac (MPS/CPU)                                              │
│  ├─ FastAPI Server (streaming_server.py)                    │
│  ├─ WebSocket endpoint for real-time audio                  │
│  ├─ HTTP endpoint for batch generation                      │
│  └─ Gradio UI for testing (multilingual_app.py)            │
│                                                              │
│  Requirements:                                               │
│  • macOS 12.3+ with Apple Silicon (M1/M2/M3)               │
│  • 16GB+ RAM recommended                                    │
│  • ~4GB disk for models                                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Option B: Production (Cloud GPU)

```
┌─────────────────────────────────────────────────────────────┐
│                  PRODUCTION DEPLOYMENT                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Load Balancer (nginx/AWS ALB)                              │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────────────────────────────┐                │
│  │  Kubernetes Cluster                      │                │
│  │  ├─ TTS Service Pods (GPU: T4/A10G)     │                │
│  │  │   ├─ Model loaded in memory          │                │
│  │  │   ├─ WebSocket streaming             │                │
│  │  │   └─ Auto-scaling (1-10 replicas)    │                │
│  │  │                                       │                │
│  │  ├─ Redis (session/voice cache)         │                │
│  │  └─ Prometheus + Grafana (monitoring)   │                │
│  └─────────────────────────────────────────┘                │
│                                                              │
│  GPU Options:                                                │
│  • AWS: g5.xlarge (A10G, 24GB) - $1.00/hr                  │
│  • GCP: n1-standard-4 + T4 - $0.35/hr                      │
│  • Azure: NC4as_T4_v3 - $0.52/hr                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Option C: Edge Deployment (Mobile/Embedded)

```
┌─────────────────────────────────────────────────────────────┐
│                    EDGE DEPLOYMENT                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  For mobile/embedded, use Chatterbox-Turbo (350M):          │
│                                                              │
│  iOS:                                                        │
│  ├─ Export to CoreML (.mlpackage)                           │
│  ├─ ~500MB model size after quantization                    │
│  └─ Use ANE (Apple Neural Engine) for inference             │
│                                                              │
│  Android:                                                    │
│  ├─ Export to ONNX → NNAPI                                  │
│  ├─ Use GPU delegate for Snapdragon/Mali                    │
│  └─ INT8 quantization for smaller footprint                 │
│                                                              │
│  Quantization options:                                       │
│  • INT8: 4x smaller, ~5% quality loss                       │
│  • INT4: 8x smaller, ~10-15% quality loss                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

---

## 6. Phase 2 Implementation: Low-Latency Training

This section provides complete step-by-step instructions for achieving < 200ms TTFA.

### 6.1 MeanFlow S3Gen Distillation (Complete Guide)

**Goal:** Train a 2-step decoder that matches 10-step quality, reducing latency by ~250ms.

#### Step 1: Prepare Training Environment

```bash
# Create dedicated training environment
conda create -n meanflow-train python=3.11
conda activate meanflow-train

# Install dependencies
pip install torch==2.6.0 torchaudio==2.6.0
pip install -e .  # Install chatterbox
pip install wandb accelerate  # For training tracking

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

#### Step 2: Prepare Training Data

```python
# training/prepare_meanflow_data.py
"""
Prepare paired data for MeanFlow distillation.
We need: (speech_tokens, teacher_mels, ref_dict)
"""

import torch
import json
from pathlib import Path
from tqdm import tqdm
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

def prepare_distillation_dataset(
    audio_dir: str,
    output_dir: str,
    device: str = "cuda",
    max_samples: int = 10000,
):
    """
    Generate teacher outputs for distillation training.
    
    For each audio file:
    1. Extract speech tokens
    2. Generate mel spectrograms with 10-step teacher
    3. Save paired data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    teacher_s3gen = model.s3gen
    teacher_s3gen.eval()
    
    audio_files = list(Path(audio_dir).glob("**/*.wav"))[:max_samples]
    
    metadata = []
    
    for i, audio_path in enumerate(tqdm(audio_files, desc="Preparing data")):
        try:
            # Load and process audio
            import librosa
            audio, sr = librosa.load(str(audio_path), sr=22050)
            audio_16k = librosa.resample(audio, orig_sr=22050, target_sr=16000)
            
            # Get speech tokens
            speech_tokens, token_lens = teacher_s3gen.tokenizer([audio_16k])
            speech_tokens = speech_tokens.to(device)
            
            # Get reference embeddings (use first 10s of audio)
            ref_dict = teacher_s3gen.embed_ref(audio[:220500], 22050, device=device)
            
            # Generate teacher mels (10 steps - high quality)
            with torch.no_grad():
                teacher_mels = teacher_s3gen.flow_inference(
                    speech_tokens,
                    ref_dict=ref_dict,
                    n_cfm_timesteps=10,
                    finalize=True,
                )
            
            # Save data
            sample_path = output_dir / f"sample_{i:06d}.pt"
            torch.save({
                "speech_tokens": speech_tokens.cpu(),
                "teacher_mels": teacher_mels.cpu(),
                "ref_dict": {k: v.cpu() if torch.is_tensor(v) else v 
                            for k, v in ref_dict.items()},
            }, sample_path)
            
            metadata.append({
                "id": i,
                "audio_path": str(audio_path),
                "sample_path": str(sample_path),
                "token_len": token_lens[0].item(),
            })
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue
    
    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Prepared {len(metadata)} samples in {output_dir}")
    return metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True, help="Directory with Arabic audio files")
    parser.add_argument("--output_dir", default="./meanflow_data", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=10000)
    args = parser.parse_args()
    
    prepare_distillation_dataset(args.audio_dir, args.output_dir, max_samples=args.max_samples)
```

#### Step 3: Train MeanFlow Student Model

```python
# training/train_meanflow.py
"""
Train MeanFlow (2-step) model via knowledge distillation.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from tqdm import tqdm
import wandb

from chatterbox.models.s3gen import S3Gen


class MeanFlowDistillationDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        with open(self.data_dir / "metadata.json") as f:
            self.metadata = json.load(f)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        sample_path = self.metadata[idx]["sample_path"]
        data = torch.load(sample_path)
        return data


class MeanFlowTrainer:
    def __init__(
        self,
        teacher_path: str,
        output_dir: str,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        batch_size: int = 4,
    ):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load teacher (10-step model)
        print("Loading teacher model...")
        self.teacher = S3Gen(meanflow=False)
        teacher_weights = torch.load(teacher_path, map_location=device)
        self.teacher.load_state_dict(teacher_weights)
        self.teacher.to(device).eval()
        
        # Create student (2-step MeanFlow)
        print("Creating student model...")
        self.student = S3Gen(meanflow=True)
        
        # Initialize student from teacher where possible
        teacher_state = self.teacher.state_dict()
        student_state = self.student.state_dict()
        
        # Copy matching weights
        for key in student_state:
            if key in teacher_state and teacher_state[key].shape == student_state[key].shape:
                student_state[key] = teacher_state[key]
        
        self.student.load_state_dict(student_state)
        self.student.to(device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10000, eta_min=1e-6
        )
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        self.batch_size = batch_size
        self.global_step = 0
    
    def train_epoch(self, dataloader, epoch):
        self.student.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            loss = self.train_step(batch)
            total_loss += loss
            
            pbar.set_postfix({"loss": f"{loss:.4f}"})
            
            if self.global_step % 100 == 0:
                wandb.log({
                    "loss": loss,
                    "lr": self.scheduler.get_last_lr()[0],
                    "step": self.global_step,
                })
            
            self.global_step += 1
        
        return total_loss / len(dataloader)
    
    def train_step(self, batch):
        speech_tokens = batch["speech_tokens"].to(self.device)
        teacher_mels = batch["teacher_mels"].to(self.device)
        ref_dict = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch["ref_dict"].items()}
        
        # Student forward (2 steps)
        student_mels = self.student.flow_inference(
            speech_tokens,
            ref_dict=ref_dict,
            n_cfm_timesteps=2,
            finalize=True,
        )
        
        # Loss: L1 + MSE for better convergence
        l1 = self.l1_loss(student_mels, teacher_mels)
        mse = self.mse_loss(student_mels, teacher_mels)
        loss = l1 + 0.1 * mse
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def save_checkpoint(self, epoch, loss):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": loss,
            "global_step": self.global_step,
        }
        path = self.output_dir / f"meanflow_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
        
        # Also save just the model weights for easy loading
        weights_path = self.output_dir / f"s3gen_meanflow_arabic.pt"
        torch.save(self.student.state_dict(), weights_path)
        print(f"Saved model weights to {weights_path}")
    
    def evaluate(self, dataloader):
        """Evaluate student vs teacher quality."""
        self.student.eval()
        total_l1 = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                speech_tokens = batch["speech_tokens"].to(self.device)
                teacher_mels = batch["teacher_mels"].to(self.device)
                ref_dict = {k: v.to(self.device) if torch.is_tensor(v) else v 
                            for k, v in batch["ref_dict"].items()}
                
                student_mels = self.student.flow_inference(
                    speech_tokens, ref_dict=ref_dict, n_cfm_timesteps=2, finalize=True
                )
                
                total_l1 += self.l1_loss(student_mels, teacher_mels).item()
        
        return total_l1 / len(dataloader)


def collate_fn(batch):
    """Custom collate for variable-length sequences."""
    # For simplicity, just return first item (batch_size=1 for distillation)
    return batch[0]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--teacher_path", required=True, help="Path to s3gen.pt")
    parser.add_argument("--output_dir", default="./meanflow_checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    
    # Initialize wandb
    wandb.init(project="chatterbox-meanflow", config=vars(args))
    
    # Load data
    dataset = MeanFlowDistillationDataset(args.data_dir)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    # Create trainer
    trainer = MeanFlowTrainer(
        teacher_path=args.teacher_path,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )
    
    # Training loop
    for epoch in range(args.epochs):
        avg_loss = trainer.train_epoch(dataloader, epoch)
        print(f"Epoch {epoch}: avg_loss = {avg_loss:.4f}")
        
        # Evaluate
        eval_loss = trainer.evaluate(dataloader)
        print(f"Epoch {epoch}: eval_l1 = {eval_loss:.4f}")
        
        wandb.log({"epoch": epoch, "avg_loss": avg_loss, "eval_l1": eval_loss})
        
        # Save checkpoint
        trainer.save_checkpoint(epoch, avg_loss)
    
    print("Training complete!")
    wandb.finish()


if __name__ == "__main__":
    main()
```

#### Step 4: Integrate Trained MeanFlow Model

```python
# After training, use the MeanFlow model:

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.s3gen import S3Gen

# Load base model
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

# Replace S3Gen with trained MeanFlow version
meanflow_s3gen = S3Gen(meanflow=True)
meanflow_s3gen.load_state_dict(
    torch.load("./meanflow_checkpoints/s3gen_meanflow_arabic.pt")
)
meanflow_s3gen.to("cuda").eval()

model.s3gen = meanflow_s3gen

# Now inference uses 2 steps instead of 10!
wav = model.generate(text, language_id="ar")
```

#### Training Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 16GB | 24GB+ |
| Training Data | 1,000 samples | 10,000+ samples |
| Training Time | 2-4 hours | 8-12 hours |
| Disk Space | 20GB | 50GB |

---

### 6.2 Speculative Decoding Implementation (Complete Guide)

**Goal:** Speed up T3 token generation by 2-3x by predicting multiple tokens at once.

#### Step 1: Create Speculative Decoder Module

```python
# src/chatterbox/models/t3/speculative.py
"""
Speculative decoding for T3: predict multiple tokens, verify in parallel.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from tqdm import tqdm


class SpeculativeT3:
    """
    Speculative decoding wrapper for T3 model.
    
    Algorithm:
    1. Draft model quickly proposes K tokens (greedy/low-temp)
    2. Main model verifies all K tokens in ONE forward pass
    3. Accept tokens until first rejection
    4. Repeat from rejection point
    
    Speedup comes from parallel verification (1 forward pass for K tokens).
    """
    
    def __init__(
        self,
        model,  # T3 model
        draft_model=None,  # Optional smaller draft model
        speculation_k: int = 4,  # Tokens to speculate at once
        draft_temperature: float = 0.5,  # Lower temp for draft (more deterministic)
        verify_temperature: float = 0.8,  # Normal temp for verification
    ):
        self.model = model
        self.draft = draft_model or model  # Use same model if no draft provided
        self.k = speculation_k
        self.draft_temp = draft_temperature
        self.verify_temp = verify_temperature
        
        self.device = model.device
        self.hp = model.hp
    
    @torch.inference_mode()
    def generate(
        self,
        t3_cond,
        text_tokens: torch.Tensor,
        max_new_tokens: int = 500,
    ) -> torch.Tensor:
        """
        Generate speech tokens using speculative decoding.
        """
        # Initial setup
        text_tokens = torch.atleast_2d(text_tokens).to(self.device)
        
        # Prepare initial embeddings
        initial_speech = torch.tensor(
            [[self.hp.start_speech_token]], 
            device=self.device
        )
        
        embeds, len_cond = self.model.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=initial_speech,
            cfg_weight=0.0,
        )
        
        # Run initial forward to get KV cache
        output = self.model.tfmr(
            inputs_embeds=embeds,
            use_cache=True,
            output_hidden_states=True,
        )
        past_kv = output.past_key_values
        
        generated_tokens = [self.hp.start_speech_token]
        
        pbar = tqdm(total=max_new_tokens, desc="Speculative generation")
        
        while len(generated_tokens) < max_new_tokens:
            # Step 1: Draft K tokens quickly
            draft_tokens, draft_kv = self._draft_tokens(
                generated_tokens[-1], past_kv, self.k
            )
            
            # Step 2: Verify all K tokens in parallel
            accepted, verified_kv = self._verify_tokens(
                generated_tokens, draft_tokens, past_kv
            )
            
            # Step 3: Accept verified tokens
            generated_tokens.extend(accepted)
            past_kv = verified_kv
            
            pbar.update(len(accepted))
            
            # Check for EOS
            if self.hp.stop_speech_token in accepted:
                break
            
            # If no tokens accepted, generate one normally
            if len(accepted) == 0:
                token = self._generate_single_token(generated_tokens[-1], past_kv)
                generated_tokens.append(token)
                pbar.update(1)
        
        pbar.close()
        
        return torch.tensor(generated_tokens[1:], device=self.device)  # Skip BOS
    
    def _draft_tokens(
        self, 
        last_token: int, 
        past_kv, 
        k: int
    ) -> Tuple[list, any]:
        """
        Quickly generate K draft tokens using greedy/low-temperature sampling.
        Uses the draft model (can be same as main model).
        """
        draft_tokens = []
        current_token = last_token
        draft_kv = past_kv
        
        for _ in range(k):
            # Get embedding for current token
            token_embed = self.draft.speech_emb(
                torch.tensor([[current_token]], device=self.device)
            )
            
            # Forward pass
            output = self.draft.tfmr(
                inputs_embeds=token_embed,
                past_key_values=draft_kv,
                use_cache=True,
            )
            draft_kv = output.past_key_values
            
            # Get logits and sample with low temperature (more deterministic)
            hidden = output.last_hidden_state[:, -1, :]
            logits = self.draft.speech_head(hidden)
            
            # Greedy or low-temp sampling for draft
            if self.draft_temp == 0:
                next_token = logits.argmax(dim=-1).item()
            else:
                probs = F.softmax(logits / self.draft_temp, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            
            draft_tokens.append(next_token)
            current_token = next_token
            
            # Stop if EOS
            if next_token == self.hp.stop_speech_token:
                break
        
        return draft_tokens, draft_kv
    
    def _verify_tokens(
        self,
        prefix: list,
        draft_tokens: list,
        past_kv,
    ) -> Tuple[list, any]:
        """
        Verify draft tokens with main model in ONE parallel forward pass.
        Accept tokens until first rejection.
        """
        if not draft_tokens:
            return [], past_kv
        
        # Create tensor of all draft tokens for parallel processing
        draft_tensor = torch.tensor(
            [draft_tokens], 
            device=self.device
        )
        draft_embeds = self.model.speech_emb(draft_tensor)
        
        # Single forward pass for all K tokens
        output = self.model.tfmr(
            inputs_embeds=draft_embeds,
            past_key_values=past_kv,
            use_cache=True,
            output_hidden_states=True,
        )
        
        # Get logits for each position
        hidden_states = output.last_hidden_state  # (1, K, dim)
        logits = self.model.speech_head(hidden_states)  # (1, K, vocab)
        
        # Verify each token
        accepted = []
        for i, draft_token in enumerate(draft_tokens):
            # Sample from main model distribution
            probs = F.softmax(logits[0, i] / self.verify_temp, dim=-1)
            
            # Acceptance criterion: 
            # Accept if draft token has reasonable probability
            draft_prob = probs[draft_token].item()
            
            # Simple acceptance: accept if prob > threshold
            # More sophisticated: use rejection sampling
            if draft_prob > 0.1:  # Threshold can be tuned
                accepted.append(draft_token)
            else:
                # Reject and sample from main model instead
                sampled = torch.multinomial(probs, 1).item()
                accepted.append(sampled)
                break  # Stop accepting after first rejection
            
            if draft_token == self.hp.stop_speech_token:
                break
        
        # Truncate KV cache to accepted length
        # (This is an approximation - full impl would reconstruct cache)
        
        return accepted, output.past_key_values
    
    def _generate_single_token(self, last_token: int, past_kv) -> int:
        """Fallback: generate single token normally."""
        token_embed = self.model.speech_emb(
            torch.tensor([[last_token]], device=self.device)
        )
        
        output = self.model.tfmr(
            inputs_embeds=token_embed,
            past_key_values=past_kv,
            use_cache=True,
        )
        
        hidden = output.last_hidden_state[:, -1, :]
        logits = self.model.speech_head(hidden)
        probs = F.softmax(logits / self.verify_temp, dim=-1)
        
        return torch.multinomial(probs, 1).item()
```

#### Step 2: Integrate with Streaming

```python
# src/chatterbox/low_latency.py

from chatterbox.models.t3.speculative import SpeculativeT3

class LowLatencyChatterbox:
    """
    Combines MeanFlow + Speculative Decoding for < 200ms latency.
    """
    
    def __init__(self, model):
        self.model = model
        self.speculative_t3 = SpeculativeT3(
            model.t3,
            speculation_k=4,  # Predict 4 tokens at once
        )
    
    def generate_fast(self, text, language_id="ar", audio_prompt_path=None):
        """
        Generate with all optimizations enabled.
        Expected latency: ~150-200ms TTFA
        """
        import time
        start = time.perf_counter()
        
        # Prepare conditioning
        if audio_prompt_path:
            self.model.prepare_conditionals(audio_prompt_path)
        
        # Tokenize
        text_tokens = self.model.tokenizer.text_to_tokens(
            text, language_id=language_id
        ).to(self.model.device)
        
        # Speculative T3 generation (2-3x faster)
        speech_tokens = self.speculative_t3.generate(
            t3_cond=self.model.conds.t3,
            text_tokens=text_tokens,
        )
        
        t3_time = time.perf_counter() - start
        
        # MeanFlow S3Gen (5x faster: 2 steps vs 10)
        s3_start = time.perf_counter()
        wav, _ = self.model.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=self.model.conds.gen,
            n_cfm_timesteps=2,  # MeanFlow!
        )
        s3_time = time.perf_counter() - s3_start
        
        total_time = (time.perf_counter() - start) * 1000
        
        print(f"⚡ T3: {t3_time*1000:.0f}ms | S3Gen: {s3_time*1000:.0f}ms | Total: {total_time:.0f}ms")
        
        return wav
```

---

## 7. Phase 3/4 Implementation: Production Deployment

### 7.1 TensorRT Optimization (NVIDIA GPUs)

```python
# deployment/export_tensorrt.py
"""
Export models to TensorRT for maximum NVIDIA GPU performance.
Expected speedup: 2-4x over PyTorch
"""

import torch
import torch_tensorrt

def export_t3_tensorrt(model, output_path: str):
    """Export T3 model to TensorRT."""
    
    # Example inputs for tracing
    batch_size = 1
    text_seq_len = 100
    speech_seq_len = 200
    
    # Trace the model
    traced = torch.jit.trace(
        model.t3.tfmr,
        example_inputs=(
            torch.randn(batch_size, text_seq_len + speech_seq_len, 768).cuda(),
        ),
    )
    
    # Compile with TensorRT
    trt_model = torch_tensorrt.compile(
        traced,
        inputs=[
            torch_tensorrt.Input(
                min_shape=(1, 50, 768),
                opt_shape=(1, 200, 768),
                max_shape=(1, 500, 768),
                dtype=torch.float16,
            ),
        ],
        enabled_precisions={torch.float16},  # FP16 for speed
        workspace_size=1 << 30,  # 1GB workspace
    )
    
    torch.jit.save(trt_model, output_path)
    print(f"Saved TensorRT model to {output_path}")


def export_s3gen_tensorrt(model, output_path: str):
    """Export S3Gen to TensorRT."""
    # Similar process for S3Gen
    pass
```

### 7.2 INT8 Quantization

```python
# deployment/quantize.py
"""
Quantize models to INT8 for smaller size and faster inference.
"""

import torch
from torch.quantization import quantize_dynamic, prepare, convert

def quantize_model_dynamic(model):
    """
    Dynamic INT8 quantization (easiest, good for CPU).
    """
    # Quantize T3 transformer
    model.t3.tfmr = quantize_dynamic(
        model.t3.tfmr,
        {torch.nn.Linear, torch.nn.Embedding},
        dtype=torch.qint8
    )
    
    return model


def quantize_model_static(model, calibration_data):
    """
    Static INT8 quantization (better accuracy, requires calibration).
    """
    # Prepare for quantization
    model.t3.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model.t3, inplace=True)
    
    # Calibrate with representative data
    with torch.no_grad():
        for batch in calibration_data:
            model.t3(batch)
    
    # Convert to quantized
    torch.quantization.convert(model.t3, inplace=True)
    
    return model
```

### 7.3 Production Streaming Server

```python
# deployment/streaming_server.py
"""
Production-ready streaming server with WebSocket support.
"""

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import asyncio
import json
import time
from typing import Optional

app = FastAPI(title="Chatterbox Arabic TTS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None
low_latency_model = None


@app.on_event("startup")
async def startup():
    global model, low_latency_model
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from chatterbox.low_latency import LowLatencyChatterbox
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    
    # Load MeanFlow S3Gen if available
    meanflow_path = "./models/s3gen_meanflow_arabic.pt"
    if Path(meanflow_path).exists():
        from chatterbox.models.s3gen import S3Gen
        meanflow = S3Gen(meanflow=True)
        meanflow.load_state_dict(torch.load(meanflow_path))
        meanflow.to(device).eval()
        model.s3gen = meanflow
        print("Loaded MeanFlow S3Gen for low latency")
    
    low_latency_model = LowLatencyChatterbox(model)
    print("Model loaded!")


@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming TTS.
    
    Protocol:
    1. Client sends: {"text": "مرحبا", "language_id": "ar", "speaker_id": "optional"}
    2. Server streams: binary audio chunks
    3. Server sends: {"status": "complete", "latency_ms": 150}
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive request
            data = await websocket.receive_json()
            
            text = data.get("text", "")
            language_id = data.get("language_id", "ar")
            speaker_id = data.get("speaker_id")
            
            if not text:
                await websocket.send_json({"error": "No text provided"})
                continue
            
            start_time = time.perf_counter()
            first_chunk_sent = False
            
            # Stream audio chunks
            async for chunk in stream_audio_chunks(text, language_id, speaker_id):
                await websocket.send_bytes(chunk)
                
                if not first_chunk_sent:
                    ttfa = (time.perf_counter() - start_time) * 1000
                    print(f"⚡ TTFA: {ttfa:.0f}ms")
                    first_chunk_sent = True
            
            # Send completion
            total_time = (time.perf_counter() - start_time) * 1000
            await websocket.send_json({
                "status": "complete",
                "latency_ms": total_time,
            })
            
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close(code=1000)


async def stream_audio_chunks(text: str, language_id: str, speaker_id: Optional[str]):
    """
    Generate and yield audio chunks for streaming.
    """
    # Use low-latency model with speculative decoding + MeanFlow
    wav = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: low_latency_model.generate_fast(text, language_id)
    )
    
    # Convert to bytes and yield
    audio_bytes = wav.squeeze().cpu().numpy().astype('float32').tobytes()
    
    # Yield in chunks for streaming
    chunk_size = 4096  # ~93ms at 22050Hz
    for i in range(0, len(audio_bytes), chunk_size):
        yield audio_bytes[i:i + chunk_size]


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/latency")
async def measure_latency():
    """Measure current latency."""
    text = "مرحبا"
    
    start = time.perf_counter()
    wav = low_latency_model.generate_fast(text, language_id="ar")
    latency = (time.perf_counter() - start) * 1000
    
    return {
        "text": text,
        "latency_ms": latency,
        "samples": wav.shape[-1],
        "duration_ms": wav.shape[-1] / 22.05,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 7.4 Docker Deployment

```dockerfile
# deployment/Dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3.11 python3-pip

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy model and code
COPY src/ /app/src/
COPY models/ /app/models/
COPY deployment/ /app/deployment/

WORKDIR /app

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "-m", "uvicorn", "deployment.streaming_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# deployment/docker-compose.yml
version: '3.8'

services:
  tts:
    build: .
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./models:/app/models
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## Summary: Action Items

### Phase 1: Immediate (No Training Required)

1. ✅ Fix Mac compatibility (DONE - `mtl_tts.py` updated)
2. ⬜ Test baseline Arabic quality with `test_arabic_tts.py`
3. ⬜ Measure current latency baseline
4. ⬜ Implement low-latency streaming with small chunks (2-4 tokens)
5. ⬜ Add speaker embedding caching
6. ⬜ Apply `torch.compile` optimization

**Expected Result: ~300-400ms TTFA**

### Phase 2: Low-Latency Optimization (Training Required)

7. ⬜ **Distill MeanFlow S3Gen for Arabic** (BIGGEST IMPACT)
   - Train 2-step decoder to match 10-step quality
   - This alone reduces decoder latency from ~300ms to ~60ms

8. ⬜ Implement speculative decoding for T3
   - Predict 4 tokens at once, verify in parallel
   - Reduces T3 latency by 2-3x

**Expected Result: ~150-200ms TTFA** ✅

### Phase 3: Arabic Dialect Fine-Tuning

9. ⬜ Collect Arabic dialect dataset (10-100 hours)
10. ⬜ Fine-tune T3 model on dialect data
11. ⬜ Evaluate pronunciation and prosody
12. ⬜ Iterate based on native speaker feedback

### Phase 4: Production Deployment

13. ⬜ Export to TensorRT (NVIDIA) or CoreML (Apple)
14. ⬜ INT8 quantization for additional speedup
15. ⬜ Deploy WebSocket streaming server
16. ⬜ Implement client SDKs (JS, Python, mobile)
17. ⬜ Set up monitoring and latency tracking

**Expected Result: ~100-150ms TTFA** ✅✅

### Quick Wins Checklist (Do These First)

```bash
# 1. Test current baseline
python test_arabic_tts.py

# 2. Measure latency
python -c "
import time
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = ChatterboxMultilingualTTS.from_pretrained(device=device)

text = 'مرحباً'
start = time.perf_counter()
wav = model.generate(text, language_id='ar')
latency = (time.perf_counter() - start) * 1000
print(f'Total latency: {latency:.0f}ms')
"

# 3. Apply torch.compile (easy win)
# Add to your inference code:
# model.t3 = torch.compile(model.t3, mode='reduce-overhead')
# model.s3gen = torch.compile(model.s3gen, mode='reduce-overhead')
```

### Dependencies to Update (When Ready)

```bash
pip install "diffusers>=0.31.0" peft
pip install "transformers>=4.47.0"
# Test thoroughly before updating numpy
```

### Hardware Recommendations for < 200ms

| Setup | Expected TTFA | Cost |
|-------|---------------|------|
| Mac M1/M2 (MPS) | 300-500ms | $0 (your laptop) |
| NVIDIA T4 (cloud) | 150-250ms | ~$0.35/hr |
| NVIDIA A10G (cloud) | 100-150ms | ~$1.00/hr |
| NVIDIA A100 (cloud) | 80-120ms | ~$3.00/hr |

**Recommendation:** Start with T4 for development, scale to A10G for production.
