FROM python:3.11-slim-bullseye
WORKDIR /app

# Install system dependencies required for Python packages and Chatterbox
RUN apt-get update && apt-get install -y \
    git \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .
# RUN git clone https://github.com/resemble-ai/chatterbox.git . && \
# Install numpy explicitly to avoid dependency issues with pkuseg
RUN pip install --no-cache-dir "numpy>=1.24.0,<1.26.0" "torchaudio==2.6.0" "librosa==0.11.0" gradio[mcp]&& \
pip install --no-cache-dir -e .

EXPOSE 7860

ENV PYTHONUNBUFFERED=1

CMD ["bash"]