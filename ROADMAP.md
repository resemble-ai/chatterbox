# Chatterbox Low-Latency Arabic TTS Roadmap

> Target: < 200ms Time-to-First-Audio (TTFA) for Conversational AI Agents

## 🚨 Status Update (2026-01-16)

| Metric | Baseline (MPS) | Target | Gap Factor |
|--------|----------------|--------|------------|
| **Latency** | ~20,000ms | < 200ms | **100x** |
| **T3 Speed** | 2-14 it/s | ~500 it/s | 35-250x |
| **Real-time Factor** | 4.43x | < 0.2x | 22x |

**Blocker:** CUDA environment required. MPS/CPU cannot achieve target.  
**Next Step:** Complete Task 0.1 (CUDA Setup) before continuing.

---

## Overview

| Phase | Focus | Target TTFA | Status |
|-------|-------|-------------|--------|
| **Phase 0** | **CUDA Environment Setup** | **Prerequisite** | **Not Started** |
| Phase 1 | Quick Wins (No Training) | ~300-400ms | Blocked by Phase 0 |
| Phase 2 | Low-Latency Training | ~150-200ms | Not Started |
| Phase 3 | Arabic Dialect Fine-Tuning | ~150-200ms | Not Started |
| Phase 4 | Production Deployment | ~100-150ms | Not Started |

### Critical Finding from Baseline (2026-01-16)

```
⚠️  MPS (Apple Silicon) is NOT viable for development:
    - Current latency: ~20,000ms (20 seconds!)
    - Target latency: < 200ms
    - MPS is 100x slower than target
    
✅  CUDA GPU is REQUIRED for:
    - Meaningful latency measurements
    - Phase 1 quick wins to be measurable
    - Phase 2 training
    - Production deployment
```

---

## Phase 0: CUDA Environment Setup (PREREQUISITE)

### Task 0.1: Set Up CUDA Development Environment

| Field | Value |
|-------|-------|
| **Priority** | CRITICAL |
| **Effort** | Low-Medium (1-4 hours) |
| **Status** | [ ] Not Started |
| **Assignee** | |
| **Due Date** | |

**Description:**
Set up a CUDA-enabled environment for development and benchmarking. This is a **prerequisite** for all other tasks - MPS/CPU are too slow for meaningful iteration.

**Options:**

#### Option A: Local NVIDIA GPU (Recommended if available)
- RTX 3060+ / RTX 4060+ (8GB+ VRAM)
- Install CUDA toolkit + cuDNN
- Expected latency: ~1-2 seconds baseline

#### Option B: Cloud GPU Instance
| Provider | Instance | GPU | Cost | Setup Time |
|----------|----------|-----|------|------------|
| AWS | g5.xlarge | A10G (24GB) | ~$1.00/hr | 30 min |
| GCP | n1-standard-4 + T4 | T4 (16GB) | ~$0.35/hr | 30 min |
| Lambda Labs | gpu_1x_a10 | A10 (24GB) | ~$0.60/hr | 15 min |
| RunPod | RTX 4090 | RTX 4090 (24GB) | ~$0.44/hr | 10 min |

#### Option C: Google Colab Pro (For testing)
- Free tier: T4 GPU (limited hours)
- Pro: A100 access (~$10/month)
- Good for initial testing, not for extended development

**Setup Commands (Cloud/Local):**
```bash
# 1. Verify CUDA
nvidia-smi

# 2. Create environment
conda create -n chatterbox python=3.11
conda activate chatterbox

# 3. Install PyTorch with CUDA
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121

# 4. Clone and install chatterbox
git clone https://github.com/resemble-ai/chatterbox.git
cd chatterbox
pip install -e .

# 5. Run benchmark
python benchmark_latency.py
```

**Acceptance Criteria:**
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] Benchmark runs successfully on CUDA
- [ ] Baseline latency < 3,000ms (vs 20,000ms on MPS)
- [ ] Results saved to `benchmark_results.json`

**Expected Results on CUDA:**
```
Device: NVIDIA T4/A10G
Expected Latency: 1,000-2,000ms (10-20x faster than MPS)
This gives us room to optimize toward 200ms target
```

---

### Task 0.2: Sync Codebase to CUDA Machine

| Field | Value |
|-------|-------|
| **Priority** | High |
| **Effort** | Very Low (15 min) |
| **Status** | [ ] Not Started |
| **Depends On** | Task 0.1 |

**Description:**
Transfer the codebase (with our modifications) to the CUDA machine.

**Options:**
1. **Git**: Push changes to a branch, clone on CUDA machine
2. **rsync/scp**: Direct file transfer
3. **Cloud sync**: Dropbox/Google Drive

**Files to Transfer:**
```
chatterbox/
├── src/chatterbox/mtl_tts.py      # Mac compatibility fix
├── benchmark_latency.py            # Benchmark script
├── benchmark_results.json          # History (optional)
├── test_arabic_tts.py             # Test script
├── ROADMAP.md                      # Task tracking
└── FINE_TUNING_GUIDE.md           # Implementation guide
```

**Acceptance Criteria:**
- [ ] All modified files present on CUDA machine
- [ ] `pip install -e .` succeeds
- [ ] `python benchmark_latency.py` runs and shows CUDA device

---

## Phase 1: Quick Wins (No Training Required)

> ⚠️ **Note:** Phase 1 tasks require CUDA to measure improvements. On MPS, the ~30-100ms savings are lost in 20,000ms noise.

### Task 1.1: Baseline Latency Measurement ✅ COMPLETE

| Field | Value |
|-------|-------|
| **Priority** | High |
| **Effort** | Low (1-2 hours) |
| **Status** | [x] **COMPLETE** (2026-01-16) |
| **Assignee** | |
| **Due Date** | |

**Description:**
Run `benchmark_latency.py` to establish current TTFA and chunk duration on existing hardware.

**RESULTS (MPS - Apple Silicon):**
```
Device: mps (Apple Silicon M-series)
PyTorch: 2.6.0

Trial 1: 21,272ms (audio: 2,830ms)
Trial 2: 17,458ms (audio: 4,093ms)
Trial 3: 21,490ms (audio: 4,528ms)

Average Latency: ~20,000ms
Best Latency: 17,458ms
Real-time Factor: 4.43x (slower than real-time!)

BOTTLENECK: T3 token generation on MPS (12-18 seconds)
OBSERVATION: AlignmentStreamAnalyzer detecting hallucinations (working as expected)
```

**Task Details:**
1. Activate the virtual environment: `source tts-venv/bin/activate`
2. Run the test script: `python test_arabic_tts.py`
3. Measure and document:
   - Total generation time
   - T3 token generation time
   - S3Gen decoder time
   - HiFi-GAN vocoder time
4. Identify the primary bottleneck (T3 vs S3Gen)

**Commands:**
```bash
# Measure baseline latency
python -c "
import time
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Device: {device}')

model = ChatterboxMultilingualTTS.from_pretrained(device=device)

text = 'مرحباً، كيف حالك اليوم؟'
print(f'Text: {text}')

# Warm-up run
_ = model.generate(text, language_id='ar')

# Measured run
start = time.perf_counter()
wav = model.generate(text, language_id='ar')
latency = (time.perf_counter() - start) * 1000

print(f'Total latency: {latency:.0f}ms')
print(f'Audio samples: {wav.shape[-1]}')
print(f'Audio duration: {wav.shape[-1] / 22050 * 1000:.0f}ms')
"
```

**Reusable Benchmark Script:** `benchmark_latency.py`
```bash
# Run benchmark (saves to benchmark_results.json)
python benchmark_latency.py

# Run with more trials
python benchmark_latency.py --trials 5

# View progress history
python benchmark_latency.py --history

# Test custom text
python benchmark_latency.py --text "نص مخصص"
```

**Acceptance Criteria:**
- [x] Baseline metrics documented in this file
- [x] Current bottleneck identified (T3 vs S3Gen)
- [x] Hardware specs recorded (CPU/GPU, RAM)

**Results:** ✅ COMPLETED 2026-01-16
```
Device: Apple Silicon (MPS) - MacBook
PyTorch: 2.6.0

Trial 1: 21,272ms (audio: 2,830ms)
Trial 2: 17,458ms (audio: 4,093ms)  
Trial 3: 21,490ms (audio: 4,528ms)

Average Latency: 20,073ms (~20 seconds!)
Best Latency: 17,458ms
Audio Duration: ~3-4.5 seconds
Real-time Factor: 4.43x (slower than real-time)

BOTTLENECK ANALYSIS:
├─ T3 Token Generation: ~12-18 seconds (5-14 tokens/sec on MPS, highly variable)
├─ S3Gen Decoder: ~2-4 seconds (10 CFM steps)
└─ PRIMARY BOTTLENECK: T3 on MPS is extremely slow

OBSERVATIONS:
1. MPS backend not well-optimized for LLaMA attention
2. Falls back to manual attention (sdp_kernel deprecated warning)
3. AlignmentStreamAnalyzer detecting "long_tail" hallucinations
4. High variance in generation speed (2-14 it/s)

CRITICAL: Need CUDA GPU for acceptable performance!
- Expected CUDA speedup: 10-20x faster
- Target on CUDA: ~1-2 seconds total latency
```

---

### Task 1.2: Small-Chunk Implementation

| Field | Value |
|-------|-------|
| **Priority** | High |
| **Effort** | Low (2-4 hours) |
| **Status** | [ ] Not Started |
| **Assignee** | |
| **Due Date** | |
| **Depends On** | Task 1.1 |

**Description:**
Modify the generation loop to produce aggressive chunks of 2-4 speech tokens instead of waiting for full generation.

**Task Details:**
1. Create `src/chatterbox/streaming.py` with `ChatterboxStreamer` class
2. Implement token buffering with configurable chunk size (default: 2-4 tokens)
3. Add overlap between chunks for audio continuity
4. Yield audio chunks as soon as minimum tokens are available

**Files to Create/Modify:**
- [ ] `src/chatterbox/streaming.py` (new)
- [ ] `src/chatterbox/__init__.py` (add export)

**Key Code Changes:**
```python
# Chunk configuration
CHUNK_SIZE = 3  # 3 tokens ≈ 120ms audio
OVERLAP = 1     # Keep 1 token overlap for continuity
```

**Acceptance Criteria:**
- [ ] TTFA reduced by ~100ms compared to baseline
- [ ] Inter-chunk gap remains < 20ms
- [ ] Audio playback is seamless (no audible gaps)
- [ ] Unit tests pass

---

### Task 1.3: Speaker Embedding Caching

| Field | Value |
|-------|-------|
| **Priority** | High |
| **Effort** | Very Low (1-2 hours) |
| **Status** | [ ] Not Started |
| **Assignee** | |
| **Due Date** | |

**Description:**
Pre-extract and cache the 256-dimensional speaker identity embedding from the reference audio to avoid redundant extraction during inference.

**Task Details:**
1. Add `_speaker_cache` dictionary to the model/streamer
2. Compute speaker embedding once per unique speaker ID
3. Reuse cached embedding for subsequent requests
4. Add cache invalidation mechanism

**Files to Modify:**
- [ ] `src/chatterbox/streaming.py`
- [ ] `src/chatterbox/mtl_tts.py` (optional: add caching to base class)

**Key Code:**
```python
class ChatterboxStreamer:
    def __init__(self, model):
        self.model = model
        self._speaker_cache = {}  # speaker_id -> Conditionals
    
    def get_or_compute_conds(self, speaker_id: str, audio_path: str):
        if speaker_id not in self._speaker_cache:
            self.model.prepare_conditionals(audio_path)
            self._speaker_cache[speaker_id] = self.model.conds
        return self._speaker_cache[speaker_id]
```

**Acceptance Criteria:**
- [ ] Saves ~20ms per request after first request for same speaker
- [ ] Cache correctly invalidated when reference audio changes
- [ ] Memory usage bounded (LRU eviction for large caches)

---

### Task 1.4: Inference Optimization (torch.compile)

| Field | Value |
|-------|-------|
| **Priority** | Medium |
| **Effort** | Very Low (1 hour) |
| **Status** | [ ] Not Started |
| **Assignee** | |
| **Due Date** | |

**Description:**
Apply `torch.compile` to the T3 and S3Gen inference scripts for graph-level optimization.

**Task Details:**
1. Add `torch.compile` with `mode="reduce-overhead"` to T3 and S3Gen
2. Test on both CUDA and MPS backends
3. Measure speedup vs baseline
4. Handle compilation warm-up (first inference slower)

**Files to Modify:**
- [ ] `src/chatterbox/mtl_tts.py` (add compile option)

**Key Code:**
```python
# In ChatterboxMultilingualTTS.__init__ or from_pretrained:
if torch.__version__ >= "2.0" and compile_model:
    self.t3 = torch.compile(self.t3, mode="reduce-overhead")
    self.s3gen = torch.compile(self.s3gen, mode="reduce-overhead")
```

**Acceptance Criteria:**
- [ ] Consistent ~30ms reduction in generation latency
- [ ] No regression in output quality
- [ ] Graceful fallback if compilation fails

---

## Phase 2: Low-Latency Training

### Task 2.1: MeanFlow S3Gen Distillation

| Field | Value |
|-------|-------|
| **Priority** | Critical |
| **Effort** | High (1-2 weeks) |
| **Status** | [ ] Not Started |
| **Assignee** | |
| **Due Date** | |
| **Depends On** | Phase 1 complete |

**Description:**
Train a "Student" MeanFlow model to reduce CFM steps from 10 to 2 using 1k-10k Arabic audio samples. This is the **single biggest latency improvement** (~250ms saved).

**Task Details:**

#### Step 1: Data Preparation (2-3 days)
1. Collect 1,000-10,000 Arabic audio samples
2. Run `training/prepare_meanflow_data.py` to generate teacher outputs
3. Verify data quality and format

```bash
python training/prepare_meanflow_data.py \
    --audio_dir ./arabic_audio \
    --output_dir ./meanflow_data \
    --max_samples 10000
```

#### Step 2: Training (3-5 days)
1. Set up training environment with GPU (24GB+ VRAM recommended)
2. Run `training/train_meanflow.py`
3. Monitor loss convergence on Wandb
4. Save checkpoints every epoch

```bash
python training/train_meanflow.py \
    --data_dir ./meanflow_data \
    --teacher_path ~/.cache/huggingface/hub/models--ResembleAI--chatterbox/s3gen.pt \
    --output_dir ./meanflow_checkpoints \
    --epochs 10 \
    --lr 1e-4
```

#### Step 3: Evaluation (1-2 days)
1. Compare student vs teacher mel spectrogram quality (L1 loss)
2. Listen to generated samples for quality regression
3. Measure latency improvement

**Files to Create:**
- [ ] `training/prepare_meanflow_data.py`
- [ ] `training/train_meanflow.py`
- [ ] `training/evaluate_meanflow.py`

**Hardware Requirements:**
| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 16GB | 24GB+ |
| Training Data | 1,000 samples | 10,000 samples |
| Training Time | 8 hours | 24 hours |
| Disk Space | 20GB | 50GB |

**Acceptance Criteria:**
- [ ] S3Gen latency drops from ~300ms to ~60ms
- [ ] MOS (Mean Opinion Score) >= 4.0 (or within 0.2 of teacher)
- [ ] L1 loss between student and teacher mels < 0.1
- [ ] Model weights saved and loadable

**Results:**
```
Teacher (10 steps) Latency: [TO BE FILLED] ms
Student (2 steps) Latency: [TO BE FILLED] ms
Speedup: [TO BE FILLED]x
L1 Loss: [TO BE FILLED]
```

---

### Task 2.2: Speculative Decoding for T3

| Field | Value |
|-------|-------|
| **Priority** | High |
| **Effort** | Medium (3-5 days) |
| **Status** | [ ] Not Started |
| **Assignee** | |
| **Due Date** | |
| **Depends On** | Task 1.1 |

**Description:**
Create a module to predict 4 tokens simultaneously, verified in parallel by the primary T3 model. This speeds up the autoregressive T3 by 2-3x.

**Task Details:**

#### Step 1: Implement SpeculativeT3 (2-3 days)
1. Create `src/chatterbox/models/t3/speculative.py`
2. Implement draft token generation (low temperature)
3. Implement parallel verification
4. Add fallback to single-token generation

#### Step 2: Integration (1 day)
1. Create `src/chatterbox/low_latency.py` wrapper
2. Integrate with streaming pipeline
3. Add configuration options (speculation_k, temperatures)

#### Step 3: Testing & Tuning (1-2 days)
1. Compare output quality vs standard autoregressive
2. Tune acceptance threshold
3. Measure speedup on various text lengths

**Files to Create:**
- [ ] `src/chatterbox/models/t3/speculative.py`
- [ ] `src/chatterbox/low_latency.py`

**Key Parameters:**
```python
speculation_k = 4        # Tokens to speculate at once
draft_temperature = 0.5  # Lower temp for draft (more deterministic)
verify_temperature = 0.8 # Normal temp for verification
acceptance_threshold = 0.1  # Min probability to accept
```

**Acceptance Criteria:**
- [ ] T3 generation speed increases by 2-3x
- [ ] Output quality matches autoregressive mode (human evaluation)
- [ ] Works correctly with streaming pipeline
- [ ] Handles edge cases (short text, EOS detection)

**Results:**
```
Autoregressive T3 Latency: [TO BE FILLED] ms
Speculative T3 Latency: [TO BE FILLED] ms
Speedup: [TO BE FILLED]x
Acceptance Rate: [TO BE FILLED]%
```

---

## Phase 3: Arabic Dialect Fine-Tuning

### Task 3.1: Dialect Dataset Collection

| Field | Value |
|-------|-------|
| **Priority** | Medium |
| **Effort** | Medium (2-4 weeks) |
| **Status** | [ ] Not Started |
| **Assignee** | |
| **Due Date** | |

**Description:**
Gather 10-100 hours of Arabic dialect audio covering Gulf, Levantine, Egyptian, and other target dialects.

**Task Details:**

#### Data Requirements:
| Dialect | Target Hours | Speakers (min) |
|---------|--------------|----------------|
| Gulf (خليجي) | 20-30 hrs | 10+ |
| Egyptian (مصري) | 20-30 hrs | 10+ |
| Levantine (شامي) | 20-30 hrs | 10+ |
| MSA (فصحى) | 10-20 hrs | 5+ |

#### Data Quality Standards:
- Sample rate: 22050 Hz or 16000 Hz
- Format: WAV or FLAC
- SNR: > 20dB (clean audio, minimal background noise)
- Accurate transcriptions aligned with audio
- Balanced gender distribution

#### Potential Sources:
1. Common Voice Arabic (open source)
2. MGB-2/MGB-3 Arabic Broadcast (academic)
3. Custom recording sessions
4. Licensed datasets (Resemble AI, etc.)

**Files to Create:**
- [ ] `data/README.md` (data documentation)
- [ ] `scripts/validate_dataset.py` (quality checks)
- [ ] `scripts/prepare_dialect_data.py` (preprocessing)

**Acceptance Criteria:**
- [ ] 10-100 hours of clean, transcribed audio collected
- [ ] Dataset balanced across target dialects
- [ ] Gender balance within each dialect
- [ ] Data properly formatted for T3 training
- [ ] Validation script confirms quality standards

---

### Task 3.2: T3 Dialect Fine-Tuning

| Field | Value |
|-------|-------|
| **Priority** | Medium |
| **Effort** | High (2-3 weeks) |
| **Status** | [ ] Not Started |
| **Assignee** | |
| **Due Date** | |
| **Depends On** | Task 3.1 |

**Description:**
Fine-tune the primary T3 model on the collected dialect data to improve rhythm, prosody, and pronunciation for Arabic dialects.

**Task Details:**

#### Step 1: Data Preparation (2-3 days)
1. Preprocess audio (resampling, normalization)
2. Tokenize text with `MTLTokenizer`
3. Extract speech tokens with S3Tokenizer
4. Create train/validation splits

#### Step 2: Training Configuration (1 day)
1. Freeze first 6 transformer layers (preserve general knowledge)
2. Fine-tune remaining layers on dialect data
3. Use T3's built-in `loss()` function

```python
# Training hyperparameters
learning_rate = 1e-5
batch_size = 4
epochs = 10
warmup_steps = 500
freeze_layers = 6
```

#### Step 3: Training (1-2 weeks)
1. Run training with monitoring (Wandb)
2. Save checkpoints every epoch
3. Evaluate on held-out validation set

#### Step 4: Evaluation (3-5 days)
1. Generate samples for each dialect
2. Native speaker evaluation (pronunciation, naturalness)
3. A/B testing vs baseline model

**Files to Create/Reference:**
- [ ] `training/prepare_dialect_data.py`
- [ ] `training/train_dialect.py` (see FINE_TUNING_GUIDE.md)
- [ ] `training/evaluate_dialect.py`

**Acceptance Criteria:**
- [ ] Native speaker evaluation confirms improved pronunciation
- [ ] Dialect-specific prosody and rhythm captured
- [ ] No regression on MSA quality
- [ ] Model generalizes to unseen speakers

---

## Phase 4: Production Deployment

### Task 4.1: Quantization & Hardware Export

| Field | Value |
|-------|-------|
| **Priority** | Medium |
| **Effort** | Low (2-3 days) |
| **Status** | [ ] Not Started |
| **Assignee** | |
| **Due Date** | |
| **Depends On** | Phase 2 complete |

**Description:**
Apply INT8 quantization and export the model to TensorRT (NVIDIA) or CoreML (Apple) for maximum inference speed.

**Task Details:**

#### For NVIDIA GPUs (TensorRT):
1. Export T3 and S3Gen to ONNX
2. Convert ONNX to TensorRT with FP16 precision
3. Test inference latency on T4/A10G/A100

```bash
# TensorRT export
python deployment/export_tensorrt.py \
    --model_path ./checkpoints/final \
    --output_path ./deployment/models/trt
```

#### For Apple Silicon (CoreML):
1. Export to CoreML format
2. Enable ANE (Apple Neural Engine) acceleration
3. Test on M1/M2/M3 devices

#### INT8 Quantization:
1. Apply dynamic quantization to linear layers
2. Calibrate with representative data
3. Measure accuracy vs latency tradeoff

**Files to Create:**
- [ ] `deployment/export_tensorrt.py`
- [ ] `deployment/export_coreml.py`
- [ ] `deployment/quantize.py`

**Acceptance Criteria:**
- [ ] TTFA target of 100-150ms reached on NVIDIA A10G
- [ ] INT8 model within 5% quality of FP32
- [ ] Export scripts documented and reproducible

**Results:**
| Platform | Precision | TTFA |
|----------|-----------|------|
| A10G FP16 | [TO BE FILLED] ms |
| A10G INT8 | [TO BE FILLED] ms |
| T4 FP16 | [TO BE FILLED] ms |
| M2 CoreML | [TO BE FILLED] ms |

---

### Task 4.2: WebSocket Server Deployment

| Field | Value |
|-------|-------|
| **Priority** | High |
| **Effort** | Medium (3-5 days) |
| **Status** | [ ] Not Started |
| **Assignee** | |
| **Due Date** | |
| **Depends On** | Tasks 2.1, 2.2 |

**Description:**
Build a FastAPI/WebSocket server to handle persistent client connections and real-time audio streaming.

**Task Details:**

#### Step 1: Server Implementation (2-3 days)
1. Create FastAPI application with WebSocket support
2. Implement `/ws/tts` endpoint for streaming
3. Add health check and latency measurement endpoints
4. Handle concurrent connections

#### Step 2: Client SDK (1-2 days)
1. JavaScript/TypeScript client for web
2. Python client for backend integration
3. Example usage documentation

#### Step 3: Docker & Deployment (1 day)
1. Create Dockerfile with CUDA support
2. Set up docker-compose for local development
3. Configure for Kubernetes deployment

**Files to Create:**
- [ ] `deployment/streaming_server.py`
- [ ] `deployment/Dockerfile`
- [ ] `deployment/docker-compose.yml`
- [ ] `clients/js/chatterbox-client.js`
- [ ] `clients/python/chatterbox_client.py`

**API Specification:**
```
WebSocket /ws/tts
  <- {"text": "مرحبا", "language_id": "ar", "speaker_id": "user123"}
  -> [binary audio chunk]
  -> [binary audio chunk]
  -> ...
  -> {"status": "complete", "latency_ms": 150}
```

**Acceptance Criteria:**
- [ ] Server supports 10+ concurrent connections
- [ ] Audio playback is seamless with no audible gaps
- [ ] TTFA < 200ms on GPU server
- [ ] Graceful error handling and reconnection

---

### Task 4.3: AlignmentStreamAnalyzer Tuning

| Field | Value |
|-------|-------|
| **Priority** | Medium |
| **Effort** | Low (1-2 days) |
| **Status** | [ ] Partially Working |
| **Assignee** | |
| **Due Date** | |
| **Depends On** | Task 4.2 |

**Description:**
The `AlignmentStreamAnalyzer` is **already integrated and working** (observed in baseline test). This task is to **tune thresholds** for Arabic and optimize for streaming.

**Baseline Observation (2026-01-16):**
```
✅ AlignmentStreamAnalyzer is ACTIVE and detecting issues:
   - "forcing EOS token, long_tail=True" (detected 3 times)
   - "Detected 2x repetition of token 6486" (detected 1 time)
   
⚠️ May need tuning:
   - Triggered on most generations (possibly too sensitive?)
   - Need to verify audio quality when EOS is forced
```

**Task Details:**

1. Enable `AlignmentStreamAnalyzer` in streaming pipeline
2. Configure detection thresholds for:
   - False starts (hallucinations at beginning)
   - Long tails (hallucinations at end)
   - Token repetitions (3+ same token)
   - Alignment discontinuities
3. Force EOS when quality issues detected
4. Log quality metrics for monitoring

**Files to Modify:**
- [ ] `src/chatterbox/streaming.py`
- [ ] `deployment/streaming_server.py`

**Key Configuration:**
```python
# AlignmentStreamAnalyzer thresholds
false_start_threshold = 0.1
long_tail_threshold = 5  # frames
repetition_threshold = 3  # consecutive tokens
discontinuity_threshold = 7  # position jump
```

**Acceptance Criteria:**
- [ ] System automatically detects hallucinations
- [ ] EOS forced when quality drops below threshold
- [ ] Quality metrics logged for monitoring
- [ ] False positive rate < 5%

---

## Progress Tracking

### Milestones

| Milestone | Target Date | Status | Actual Date |
|-----------|-------------|--------|-------------|
| Phase 1 Complete (Quick Wins) | | [ ] | |
| MeanFlow Training Complete | | [ ] | |
| Speculative Decoding Complete | | [ ] | |
| Phase 2 Complete (< 200ms TTFA) | | [ ] | |
| Dialect Dataset Ready | | [ ] | |
| Dialect Fine-Tuning Complete | | [ ] | |
| Production Server Deployed | | [ ] | |
| Full System Live | | [ ] | |

### Weekly Check-ins

#### Week 1
- [ ] Task 1.1: Baseline measurement
- [ ] Task 1.2: Small-chunk implementation (start)

#### Week 2
- [ ] Task 1.2: Small-chunk implementation (complete)
- [ ] Task 1.3: Speaker embedding caching
- [ ] Task 1.4: torch.compile optimization

#### Week 3-4
- [ ] Task 2.1: MeanFlow distillation (data prep + training)

#### Week 5
- [ ] Task 2.1: MeanFlow evaluation
- [ ] Task 2.2: Speculative decoding

#### Week 6+
- [ ] Phase 3 & 4 tasks (as resources allow)

---

## Notes & Decisions

### Technical Decisions
| Date | Decision | Rationale |
|------|----------|-----------|
| | | |

### Blockers & Risks
| Date | Blocker/Risk | Mitigation |
|------|--------------|------------|
| | | |

### Learnings
| Date | Learning |
|------|----------|
| | |

---

## References

- [FINE_TUNING_GUIDE.md](./FINE_TUNING_GUIDE.md) - Detailed implementation code
- [test_arabic_tts.py](./test_arabic_tts.py) - Baseline test script
- [Chatterbox GitHub](https://github.com/resemble-ai/chatterbox) - Original repository

