# Chatterbox TTS Streaming Container with NVIDIA CUDA 12.8.1 support
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ARG RUNTIME=nvidia

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/app/hf_cache
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Alternative approach: Use different mirrors and bypass GPG temporarily
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirror.math.princeton.edu/pub/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|http://mirror.math.princeton.edu/pub/ubuntu|g' /etc/apt/sources.list && \
    apt-get update -o Acquire::AllowInsecureRepositories=true -o Acquire::AllowDowngradeToInsecureRepositories=true && \
    apt-get install -y --no-install-recommends --allow-unauthenticated \
        ca-certificates \
        gnupg \
        wget \
        curl \
        build-essential \
        libsndfile1 \
        libsndfile1-dev \
        ffmpeg \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        git \
        vim \
        net-tools \
        htop \
        procps && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a symlink for python3 to be python for convenience
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set up working directory
WORKDIR /app

# Upgrade pip first
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements file
COPY requirements.txt ./

# Pre-install numpy to satisfy pkuseg (chatterbox-tts dep) metadata import
RUN pip3 install --no-cache-dir "numpy>=1.24.0,<1.26.0"

# Install base requirements first
RUN pip3 install --no-cache-dir -r requirements.txt

# (Optional) Previously, NVIDIA-specific requirements were installed from a separate file.
# This image now relies on a single requirements.txt with any necessary extra-index URLs.

# Copy the rest of the application code
COPY . .

# Install the local chatterbox package from the current repository (src/ layout)
RUN pip3 install --no-cache-dir -e .

# Create cache directories
RUN mkdir -p $HF_HOME && chmod 755 $HF_HOME

# Expose the port for FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Make start script executable and set as entrypoint
RUN chmod +x start.sh

# Run the service using our startup script
ENTRYPOINT ["./start.sh"]
