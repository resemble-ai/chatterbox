# Multi-stage Dockerfile for better optimization
# This is an alternative version that uses multi-stage builds

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && python -m pip install --upgrade pip

RUN python -m pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
WORKDIR /app


# Copy requirements and install Python dependencies
COPY chatterbox-runpod-serverless/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt




# Create necessary directories
RUN mkdir -p /tmp /app/outputs

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_VISIBLE_DEVICES=all

# Copy application code (done last for best cache utilization)
COPY chatterbox-runpod-serverless/rp_handler.py /app/

# Expose port (optional, for debugging)
EXPOSE 8000

# Add a health check (optional - can be removed if not needed)
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
  CMD python -c "import torch; print('CUDA available:', torch.cuda.is_available()); exit(0 if torch.cuda.is_available() else 1)" || exit 1

# Start the RunPod handler
CMD ["python", "-u", "rp_handler.py"]