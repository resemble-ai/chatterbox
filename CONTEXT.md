# Chatterbox TTS — Domain Context

## Project
Chatterbox is an open-source TTS (text-to-speech) and voice cloning system by Resemble AI. It uses a three-stage architecture: a voice encoder (speaker embedding), a T3 autoregressive transformer (text→speech tokens), and an S3Gen decoder (speech tokens→mel→audio).

## Key Components

| Term | Description |
|---|---|
| **T3** | Autoregressive transformer (520M LLaMA-3-style) that generates speech tokens from text + conditioning |
| **S3Gen** | Flow-matching decoder that converts speech tokens to audio waveform (24kHz output) |
| **VoiceEncoder** | Extracts speaker embeddings from reference audio (256-dim vectors) |
| **S3Tokenizer** | Converts audio ↔ speech tokens at 25 tokens/sec |
| **Perceiver** | Cross-attention resampler that compresses 150 reference speech tokens into 32 |

## Generation Pipeline
1. `prepare_conditionals`: load reference audio → extract speaker embedding + speech prompt tokens
2. `prepare_conditioning`: project speaker emb + compress prompt via Perceiver
3. `forward`: concat [cond_emb, text_emb, speech_emb] → LLaMA backbone → logits
4. Autoregressive loop: sample one speech token at a time, feed back as next input
5. `S3Gen.inference`: decode speech tokens → waveform

## MLX Port (feat/mlx-port)
The T3 backbone (steps 3-4) is ported to MLX for Metal-native inference on Apple Silicon. Conditioning (steps 1-2) and audio decoding (step 5) remain in PyTorch.
