# Chatterbox Streaming TTS Server

A FastAPI server that exposes Chatterbox TTS models over HTTP, streaming audio as it is generated so playback can start before synthesis completes.

Audio is streamed as length-prefixed **float32 PCM frames** at 24 000 Hz:

```
[int32 sample_count][float32 × sample_count]   ← normal audio frame
[int32 = 0]                                     ← end-of-stream sentinel
[int32 = -1][int32 msg_len][UTF-8 message]      ← error frame
```

All three models are supported: `base`, `multilingual`, and `turbo`.

## Running

```bash
cd /path/to/chatterbox
python server/server.py [--model base|multilingual|turbo] [--device auto|cuda|cpu|mps] [--dtype fp16|bf16|fp32] [--port 8000] [--host 127.0.0.1] [--debug]
```

The server pre-loads the requested model on startup.  Only one model is kept in memory at a time; switching models evicts the previous one to free VRAM.

**Note on `--dtype`:** The current model weights were trained in fp32. In testing, `bf16` and `fp16` produce noticeably degraded or unintelligible audio and are **not recommended** for production use. They are exposed as an experimental option for future fine-tuned checkpoints that may tolerate reduced precision. Omitting `--dtype` (the default) keeps weights in fp32 and gives the best audio quality. The dtype chosen at startup is used as the server default; per-request `dtype` fields in the API can override it but carry the same quality caveats.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | HTML test client (see [Frontend](#frontend)) |
| `POST` | `/generate` | Stream TTS audio as binary frames |
| `POST` | `/generate/wav` | Generate and return a complete WAV file |
| `GET` | `/speakers` | List uploaded reference voices |
| `POST` | `/upload-audio` | Upload a reference audio file for voice cloning |
| `DELETE` | `/speakers/{id}` | Remove an uploaded reference voice |
| `GET` | `/models` | List available model names |
| `GET` | `/languages` | List supported language codes (multilingual model) |
| `GET` | `/health` | Server health + currently loaded models |
| `GET` | `/docs` | Auto-generated OpenAPI / Swagger UI |

### POST /generate

Accepts a JSON body (`GenerateRequest`) and streams binary audio frames back via chunked transfer encoding.  Key fields:

| Field | Default | Description |
|-------|---------|-------------|
| `text` | — | Text to synthesize |
| `model` | `multilingual` | `base` \| `multilingual` \| `turbo` |
| `device` | `auto` | `auto` \| `cuda` \| `cpu` \| `mps` |
| `dtype` | `null` | `fp16` \| `bf16` \| `fp32` \| `null` (server default). fp16/bf16 halve VRAM and improve throughput on CUDA GPUs |
| `language_id` | `en` | BCP-47 code (multilingual only) |
| `audio_id` | `null` | ID returned by `POST /upload-audio` |
| `exaggeration` | `0.5` | Emotion exaggeration (base/multilingual) |
| `cfg_weight` | `0.5` | Classifier-free guidance weight (base/multilingual) |
| `temperature` | `0.8` | Sampling temperature |
| `chunk_size` | `25` | Tokens per streaming chunk (~1 s of audio) |
| `cfm_steps` | `10` | CFM flow-matching steps for the S3Gen vocoder (1–20). Lower values speed up decoding at some quality cost. `5`–`6` is a good trade-off for low-latency streaming; `4` is aggressive. |
| `sentence_split` | `false` | Split text into sentences before synthesis |
| `min_chars` | `100` | Minimum chars per sentence segment |
| `max_chars` | `250` | Maximum chars per sentence segment |
| `first_sentence_half` | `true` | Halve limits for the first sentence (lower time-to-first-audio) |

The response header `X-Sample-Rate: 24000` carries the sample rate.

### POST /generate/wav

Same request body as `/generate`.  Waits for the full audio and returns a standard 16-bit PCM WAV file.  Useful for simple integrations that do not need low-latency streaming.

### POST /upload-audio

Upload a WAV/MP3/FLAC reference audio file (≥ 5 s recommended).  Returns `{"audio_id": "...", "name": "..."}`.  Pass `audio_id` in subsequent `/generate` calls to clone that voice.  Uploaded files are stored in OS temp directories and lost on server restart.

## Frontend

`client.html` is a single-page test client served at `/`.  It connects to the streaming `/generate` endpoint, schedules audio chunks through the Web Audio API as they arrive, and assembles the full recording for download once generation is complete.

Features: model/language selection, voice upload and management, generation parameter sliders, sentence-splitting controls, real-time metrics (time-to-first-audio, RTF, chunk count, audio duration), and chunk-arrival visualisation.

A deeper look at the full API schema is available at `/docs` once the server is running.
