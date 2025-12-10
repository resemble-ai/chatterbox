# Chatterbox TTS Service

This document describes how to set up and run the Chatterbox TTS service as a FastAPI server for production deployment.

## Overview

The Chatterbox TTS service provides a RESTful API for text-to-speech synthesis using the ChatterboxMultilingualTTS model. The service supports:

- **23 languages** with multilingual zero-shot voice cloning
- **Zero-shot voice cloning** from reference audio prompts
- **PCM format WAV output** (16-bit signed integer) for maximum compatibility
- **FastAPI-based REST API** with automatic API documentation
- **CORS support** for cross-origin requests
- **Thread-safe model inference** with voice prompt caching

### Known Issues and Limitations

> **⚠️ Important Notice:** ChatterboxMultilingualTTS has known quality issues with non-English languages, particularly Chinese.
>
> **Chinese Language Issues:**
> - Audio outputs frequently contain unexpected artifacts at the end, such as:
>   - Extended breathing sounds beyond expected length
>   - Low-volume speech or whispering
>   - Other audio artifacts
>
> **Other Languages:**
> - Similar issues have been reported by other users for various non-English languages in the [GitHub issues](https://github.com/resemble-ai/chatterbox/issues/)
>
> **Recommendation:**
> - **Currently, Chatterbox is only recommended for English language use.**

### API Endpoints

- `GET /api/v1/list_voice_names` - List all available voice configurations
- `POST /api/v1/generate_audio` - Generate audio from text using specified voice
- `GET /health` - Health check endpoint
- `GET /` - Redirects to API documentation at `/docs`

## Audio Prompts Preparation

Audio prompts are reference audio files used for zero-shot voice cloning. Each prompt file defines a voice that can be used for synthesis.

### File Naming Convention

Audio prompt files must follow this naming pattern:
```
{voice_key}_{language_id}.wav
```

Where:
- `voice_key`: A unique identifier for the voice (e.g., "刻晴", "keqing")
- `language_id`: Two-letter language code (e.g., "zh", "en", "fr")

**Example:**
```
刻晴_zh.wav
keqing_en.wav
```

### File Requirements

- **Format**: WAV files
- **Sample Rate**: 24000 Hz (recommended, will be resampled if needed)
- **Channels**: Mono (single channel)
- **Duration**: At least 6-10 seconds for best results

### Directory Structure

Place all audio prompt files in a directory (default: `data/`):

```
data/
├── 刻晴_zh.wav
├── keqing_en.wav
├── voice1_fr.wav
└── voice2_es.wav
```

The service will automatically scan this directory on startup and register all valid audio prompts.

### Pre-configured Audio Prompts

This repository provides pre-configured audio prompt files for DLP3D project, including reference audio files for some characters in both Chinese and English. You can download them from:

1. **Baidu Netdisk**: [https://pan.baidu.com/s/18Syh-_uwEoN-jVSDc--zBQ?pwd=r8ev](https://pan.baidu.com/s/18Syh-_uwEoN-jVSDc--zBQ?pwd=r8ev)
   - Enter extract password: `r8ev`
   - Download the `voices.zip` file and extract it to the `data/` directory

2. **GitHub Releases**: [https://github.com/LazyBusyYang/chatterbox/releases/download/voices/voices.zip](https://github.com/LazyBusyYang/chatterbox/releases/download/voices/voices.zip)
   - Direct download link for `voices.zip`
   - Extract the contents to the `data/` directory

After downloading and extracting, the audio prompt files will be ready to use. Restart the service to register them.

### Adding New Voices

To add a new voice, you need to convert your audio file to the required WAV format. If you have an audio file in MP3 or other formats, use `ffmpeg` to convert it:

```bash
ffmpeg \
    -i input_audio.mp3 \
    -ar 24000 \
    -ac 1 \
    -acodec pcm_s16le \
    data/{voice_key}_{language_id}.wav
```

**Parameters:**
- `-i input_audio.mp3`: Input audio file (can be MP3, WAV, or other formats)
- `-ar 24000`: Set sample rate to 24000 Hz
- `-ac 1`: Convert to mono (single channel)
- `-acodec pcm_s16le`: Use PCM 16-bit little-endian encoding
- `data/{voice_key}_{language_id}.wav`: Output file path following the naming convention

**Example:**
```bash
# Convert MP3 to WAV for a new voice
ffmpeg \
    -i my_voice.mp3 \
    -ar 24000 \
    -ac 1 \
    -acodec pcm_s16le \
    data/myvoice_en.wav
```

After adding the new voice file to the `data/` directory, restart the service to register it.

## Checkpoint Preparation

The service can load the TTS model from either:
1. Local checkpoint directory (recommended for production)
2. HuggingFace pretrained model (automatic download)

### Option 1: Local Checkpoint Directory

**Download Options:**

You can download the model checkpoint files manually from either of the following sources:

1. **HuggingFace**: [https://huggingface.co/ResembleAI/chatterbox/tree/main](https://huggingface.co/ResembleAI/chatterbox/tree/main)
   - Navigate to the repository and download the required files listed below

2. **Baidu Netdisk**: [https://pan.baidu.com/s/1ivYmAmZS4t1ec-edwtJ9dA?pwd=3nuq](https://pan.baidu.com/s/1ivYmAmZS4t1ec-edwtJ9dA?pwd=3nuq)
   - Enter extract password: `3nuq`
   - Download the checkpoint files from the shared folder

After downloading, organize the files as follows:

```
weights/
├── ve.pt
├── t3_mtl23ls_v2.safetensors
├── s3gen.pt
├── grapheme_mtl_merged_expanded_v1.json
├── Cangjie5_TC.json
└── conds.pt
```

**Required files:**
- `ve.pt` - Voice encoder model
- `t3_mtl23ls_v2.safetensors` - T3 multilingual model
- `s3gen.pt` - S3Gen audio generation model
- `grapheme_mtl_merged_expanded_v1.json` - Multilingual tokenizer vocabulary
- `Cangjie5_TC.json` - Chinese Cangjie conversion mapping (for Chinese support)
- `conds.pt` - Built-in voice conditionals

**Important Notes:**
- Ensure HuggingFace network connectivity is available. Even if local files exist, the service may still download additional small files not listed here from HuggingFace during initialization.

### Option 2: HuggingFace Pretrained Model

If no local checkpoint is provided, the service will automatically download the pretrained model from HuggingFace on first startup. This requires internet connectivity and may take some time.

## Docker Deployment

### Building the Docker Image

**Note:** For amd64 platforms, you can directly use the pre-built image from Docker Hub without building locally:

```bash
# Option 1: Use pre-built image (recommended for amd64)
docker pull dockersenseyang/service_chatterbox:latest
```

If you need to build the image yourself (e.g., for other platforms or custom modifications), the service includes a Dockerfile for containerized deployment with CUDA support:

```bash
# Option 2: Build from source
docker build -f service/Dockerfile -t dockersenseyang/service_chatterbox:latest .
```

### Running with Docker

Assuming you have already prepared the `data/` and `weights/` directories as described in the previous sections, run the container:

```bash
docker run -d \
  --gpus all \
  -p 18085:18085 \
  -v $(pwd)/data:/workspace/chatterbox/data \
  -v $(pwd)/weights:/workspace/chatterbox/weights \
  -v $(pwd)/logs:/workspace/chatterbox/logs \
  dockersenseyang/service_chatterbox:latest
```

## Local Development Setup

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended) or CPU
- CUDA 12.4+ (for GPU support)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/LazyBusyYang/chatterbox
   cd chatterbox
   ```

2. **Create a conda environment:**
   ```bash
   conda create -n chatterbox python=3.10 -y
   conda activate chatterbox
   ```

3. **Install PyTorch (with CUDA support):**
   ```bash
   pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
     --index-url https://download.pytorch.org/whl/cu124
   ```

4. **Install dependencies:**
   ```bash
   pip install -e .
   ```

### Configuration

Edit `service/config.py` to customize server settings:

```python
type = 'FastAPIServer'
checkpoint_dir = "weights"          # Path to model checkpoints
audio_prompts_dir = "data"          # Path to audio prompt files
host = '0.0.0.0'                    # Server host
port = 18085                        # Server port
logger_cfg = __logger_cfg__         # Logger configuration
```

### Running the Service

1. **Prepare directories:**
   ```bash
   mkdir -p data weights logs
   ```

2. **Place audio prompts and checkpoints:**
   - Copy audio prompt files to `data/`
   - Copy model checkpoint files to `weights/` (or leave empty to use HuggingFace)

3. **Start the server:**
   ```bash
   python service/main.py --config_path service/config.py
   ```

   Or with custom config:
   ```bash
   python service/main.py --config_path path/to/your/config.py
   ```

4. **Access the API:**
   - API Documentation: http://localhost:18085/docs
   - Health Check: http://localhost:18085/health
   - List Voices: http://localhost:18085/api/v1/list_voice_names

## API Usage Examples

### List Available Voices

```bash
curl http://localhost:18085/api/v1/list_voice_names
```

Response:
```json
{
  "voice_names": {
    "刻晴_zh": "刻晴",
    "keqing_en": "keqing"
  }
}
```

### Generate Audio

```bash
curl -X POST http://localhost:18085/api/v1/generate_audio \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test.",
    "voice_key": "keqing_en"
  }' \
  --output output.wav
```

### Python Client Example

```python
import requests

# List voices
response = requests.get("http://localhost:18085/api/v1/list_voice_names")
voices = response.json()["voice_names"]
print(f"Available voices: {list(voices.keys())}")

# Generate audio
response = requests.post(
    "http://localhost:18085/api/v1/generate_audio",
    json={
        "text": "Hello, this is a test.",
        "voice_key": "keqing_en"
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

## License

This repository is a fork of [https://github.com/resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) and follows the original repository's MIT License. See the LICENSE file for details.

