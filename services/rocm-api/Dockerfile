FROM rocm/pytorch:latest

WORKDIR /src/chatterbox
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" soundfile && \
    pip install --no-cache-dir librosa s3tokenizer transformers==5.2.0 \
      diffusers==0.29.0 conformer==0.3.2 safetensors==0.5.3 \
      spacy-pkuseg pykakasi==2.3.0 pyloudnorm omegaconf \
      git+https://github.com/resemble-ai/Perth.git@master

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

WORKDIR /app
COPY services/rocm-api/app.py services/rocm-api/chunking.py /app/

EXPOSE 8880
# Authentication for the built-in Node WebSocket client uses a query parameter.
# Access logging is disabled so Uvicorn never persists that credential.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8880", "--no-access-log"]
