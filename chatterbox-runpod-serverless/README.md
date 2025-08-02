# ChatterboxTTS Serverless Worker

A RunPod serverless worker for ChatterboxTTS text-to-speech generation. This worker processes text input and generates high-quality speech audio using the ChatterboxTTS model with support for voice cloning and long text processing.

## Features

- **Text-to-Speech Generation**: Convert text to natural-sounding speech
- **Voice Cloning**: Use audio prompts for voice cloning/style transfer
- **Long Text Support**: Automatic chunking for long text processing
- **Configurable Parameters**: Control voice characteristics with exaggeration, cfg_weight, and temperature
- **Robust Error Handling**: Comprehensive error handling and validation
- **GPU Acceleration**: Optimized for CUDA-enabled RunPod workers
- **Multiple Output Formats**: Support for WAV, MP3, and other audio formats

## Best Parameters (for most prompts)
- **temperature**: 0.8
- **cfg_weight**: 0.5
- **exaggeration**: 0.5
- **max_chars_per_chunk**: 550
- **inter_chunk_silence_ms**: 100


## API Parameters

The serverless worker accepts the following parameters:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | string | Yes | - | Text to convert to speech |
| `audio_prompt_base64` | string | No | null | Base64-encoded audio file for voice cloning |
| `temperature` | float | No | 0.8 | Controls randomness in generation (0.0-1.0) |
| `cfg_weight` | float | No | 0.5 | Classifier-free guidance weight |
| `exaggeration` | float | No | 0.5 | Controls expression emphasis |
| `max_chars_per_chunk` | integer | No | 300 | Maximum characters per audio chunk |
| `inter_chunk_silence_ms` | integer | No | 350 | Silence between chunks in milliseconds |

## API Usage Examples

### Basic Text-to-Speech

```json
{
    "input": {
        "text": "Hello world! This is a test of the ChatterboxTTS system.",
        "temperature": 0.8,
        "cfg_weight": 0.5,
        "exaggeration": 0.5
    }
}
```

### Voice Cloning with Audio Prompt

```json
{
    "input": {
        "text": "Clone this voice and speak this text with the same characteristics.",
        "audio_prompt_base64": "UklGRi4EAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQgEAAAA...",
        "temperature": 0.7,
        "cfg_weight": 0.6,
        "exaggeration": 0.4
    }
}
```

### Long Text with Custom Chunking

```json
{
    "input": {
        "text": "This is a very long text that will be automatically split into smaller chunks for processing. Each chunk will be processed separately and then combined into a single audio file.",
        "max_chars_per_chunk": 150,
        "inter_chunk_silence_ms": 500,
        "temperature": 0.9
    }
}
```

### Custom Audio Format

```json
{
    "input": {
        "text": "Generate audio in a specific format.",
        "output_format": "mp3",
        "temperature": 0.8
    }
}
```

## Response Format

### Success Response

```json
{
    "success": true,
    "audio_base64": "UklGRi4EAABXQVZFZm10...",
    "metadata": {
        "duration_seconds": 5.2,
        "sample_rate": 44100,
        "format": "wav",
        "chunks_generated": 3,
        "total_characters": 156,
        "voice_cloning_used": false
    },
    "processing_time": 2.34
}
```

### Error Response

```json
{
    "success": false,
    "error": "Error message describing what went wrong",
    "error_type": "validation_error|processing_error|system_error",
    "processing_time": 0.12
}
```

## Audio Prompt Guidelines

When using voice cloning with `audio_prompt_base64`:

1. **Audio Format**: WAV format recommended for best results
2. **Duration**: 5-10 seconds of clear speech
3. **Quality**: High-quality recording with minimal background noise
4. **Encoding**: Use base64 encoding of the raw audio file
5. **Sample Rate**: 44.1kHz or 22kHz recommended

**Example of preparing audio prompt:**

```python
import base64

# Read your audio file
with open("voice_sample.wav", "rb") as audio_file:
    audio_data = audio_file.read()
    audio_base64 = base64.b64encode(audio_data).decode('utf-8')

# Use in API request
request_data = {
    "input": {
        "text": "Your text here",
        "audio_prompt_base64": audio_base64
    }
}
```

## Local Development

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- Docker (for containerization)

### Setup

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install ChatterboxTTS (replace with actual installation method):
   ```bash
   # Example - adjust based on actual ChatterboxTTS installation
   pip install chatterbox-tts
   # OR
   git clone https://github.com/your-repo/chatterbox-tts.git
   cd chatterbox-tts
   pip install -e .
   ```


## Deployment

### Option 1: Deploy via GitHub Integration (Recommended)

1. Fork this repository
2. Connect your GitHub account to RunPod Console
3. Create a new Serverless Endpoint
4. Select "GitHub Repo" as source
5. Choose this repository, branch and Dockerfile
6. Configure compute resources (GPU recommended)
7. Deploy


## Configuration

### GPU Requirements

- **Minimum**: 16GB VRAM (RTX 4080/A4000)             # works well for non concurent on single worker
- **Recommended**: 24GB+ VRAM (RTX 4090/A5000/A6000)  # for concurent on single worker


## API Usage

### cURL Example

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "text": "Hello world! This is a test of the ChatterboxTTS system.",
      "temperature": 0.8,
      "exaggeration": 0.5
    }
  '
```

### Python SDK Example

```python
import runpod

runpod.api_key = "YOUR_API_KEY"

endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

# Basic usage
result = endpoint.run_sync({
    "input": {
        "text": "Welcome to ChatterboxTTS!",
        "temperature": 0.7,
        "cfg_weight": 0.6
    }
})

print(result)

# Voice cloning example
import base64

with open("voice_sample.wav", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode('utf-8')

result = endpoint.run_sync({
    "input": {
        "text": "This will sound like the voice sample.",
        "audio_prompt_base64": audio_base64,
        "temperature": 0.8
    }
})
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');

const runPodRequest = async () => {
    try {
        const response = await axios.post(
            'https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync',
            {
                input: {
                    text: "Hello from JavaScript!",
                    temperature: 0.8,
                    cfg_weight: 0.5
                }
            },
            {
                headers: {
                    'Authorization': 'Bearer YOUR_API_KEY',
                    'Content-Type': 'application/json'
                }
            }
        );
        
        console.log(response.data);
    } catch (error) {
        console.error('Error:', error.response.data);
    }
};

runPodRequest();
```

## Monitoring and Troubleshooting

### Logs

Monitor worker logs in the RunPod Console under Endpoint Details > Logs tab.

### Common Issues

1. **Out of Memory**: 
   - Reduce `max_chars_per_chunk` parameter
   - Use larger GPU configuration
   - Check for memory leaks in model loading

2. **Model Loading Timeout**: 
   - Increase worker timeout settings
   - Ensure model files are properly cached
   - Check internet connectivity for model downloads

3. **Audio Quality Issues**: 
   - Adjust `temperature` (lower = more consistent)
   - Tune `cfg_weight` and `exaggeration` parameters
   - Ensure audio prompts are high quality

4. **Base64 Encoding Issues**:
   - Verify audio file format is supported
   - Check base64 encoding is properly formatted
   - Ensure audio file size is reasonable
5. **Cuda not avaiable error on Runpod**: (Already fixed in Dockerfile)
   - when installing chatterbox-tts, pip installs chatterbox-tts including its specified torch==2.6.0 and torchaudio==2.6.0 (which will be CPU-only from PyPI if not constrained).
   - **Solution**: Running `pip install --force-reinstall torch==2.6.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124` after installing chatterbox-tts overwrites the previously installed torch and torchaudio with the CUDA-enabled versions from the PyTorch wheel index. The versions (2.6.0) match, but the build (+cu124) is different.

### Performance Optimization

- Use GPU workers for faster inference
- Optimize chunk size based on text length and GPU memory
- Enable model caching between requests
- Use appropriate timeout settings
- Preload models during worker initialization

## File Structure

```
chatterbox-tts-serverless/
├── rp_handler.py              # Main handler function
├── requirements.txt           # Python dependencies
├── test.py             # experiment script
├── test_input.json           # Basic test input
├── test_input_long.json      # Long text test input
├── test_input_voice_clone.json # Voice cloning test input
└── README.md                 # This file
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Check the [RunPod Documentation](https://docs.runpod.io/)
- Open an issue in this repository
- Contact RunPod support for infrastructure issues
- Review ChatterboxTTS documentation for model-specific questions

# Future Plans
- Concurrent requests on Single Worker
- Streaming Responses