FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Set non-interactive timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies including Python and FFmpeg
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

WORKDIR /app
COPY . /app/

# Install the base chatterbox package
RUN pip install --no-cache-dir -e .

# Install API requirements
RUN pip install --no-cache-dir -r requirements-api.txt

# Install specific torch version with CUDA 12.4 support
RUN pip install --no-cache-dir torch==2.6.0 torchaudio --index-url https://download.pytorch.org/whl/cu124

# Create voices directory
RUN mkdir -p /app/voices

EXPOSE 8000 7860

# Default command to run the API server
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
