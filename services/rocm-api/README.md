# ROCm API service

This optional service packages this fork for an AMD ROCm host and exposes an
OpenAI-compatible WAV endpoint plus progressive WebSocket narration.

The stream sends a `start` JSON frame, then one `chunk` metadata frame and one
24 kHz mono signed-16-bit PCM frame per phrase, followed by `end`. Chunk metadata
includes `boundary` (`clause`, `sentence`, `paragraph`, or `final`) so clients can
apply natural pauses while caching the combined audio.

## Run

From this directory, copy `.env.example` to `.env`, provide the host's Tailscale
IP and a random API key, create `models/` and `voices/`, then run:

```sh
docker compose up -d --build
```

Voice references are read from `/voices/<name>.wav`. The service serializes GPU
generation because the ROCm kernels and process-global model are not safe to
enter concurrently.

The repository remains covered by Chatterbox's MIT license. This service is a
deployment integration and is not an official Resemble AI API.
