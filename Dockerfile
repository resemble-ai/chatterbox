FROM python:3.11-slim-bullseye
WORKDIR /app

# Install system dependencies required for Python packages and Chatterbox
RUN apt-get update && apt-get install -y \
    git \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .
# Inject custom parameters for docker to communicate to host
RUN sed -i 's/\.launch(/.launch(server_name="0.0.0.0", server_port=7860,/' gradio_tts_app.py && \
sed -i 's/\.launch(/.launch(server_name="0.0.0.0", server_port=7860,/' gradio_vc_app.py && \
sed -i 's/\.launch(/.launch(server_name="0.0.0.0", server_port=7860,/' multilingual_app.py
# Install numpy explicitly to avoid dependency issues with pkuseg
RUN pip install --no-cache-dir "numpy>=1.24.0,<1.26.0" "torchaudio==2.6.0" "librosa==0.11.0" gradio[mcp] && \
pip install --no-cache-dir -e .

EXPOSE 7860

ENV PYTHONUNBUFFERED=1

CMD ["bash"]