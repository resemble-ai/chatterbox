#!/bin/bash

# Chatterbox TTS Service Startup Script

set -e  # Exit on any error

echo "🚀 Starting Chatterbox TTS Service..."

# Print system information
echo "📊 System Information:"
echo "  - Python version: $(python --version)"
echo "  - Working directory: $(pwd)"
echo "  - Available GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo 'No GPU detected')"

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "  - CUDA available: Yes"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
else
    echo "  - CUDA available: No (CPU mode)"
fi

# Set default values for environment variables
export PYTHONPATH="${PYTHONPATH}:/app"
export PYTHONUNBUFFERED=1

# Configuration from environment variables
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"
LOG_LEVEL="${LOG_LEVEL:-info}"

echo "🔧 Service Configuration:"
echo "  - Host: $HOST"
echo "  - Port: $PORT"
echo "  - Workers: $WORKERS"
echo "  - Log Level: $LOG_LEVEL"

# Pre-flight checks
echo "🔍 Running pre-flight checks..."

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found!"
    exit 1
fi

# Check if required packages are installed
python -c "import fastapi, uvicorn, torch, torchaudio" 2>/dev/null || {
    echo "❌ Error: Required packages not installed!"
    exit 1
}

echo "✅ Pre-flight checks passed"

# Handle graceful shutdown
cleanup() {
    echo "🛑 Received shutdown signal, stopping service gracefully..."
    kill -TERM "$child" 2>/dev/null || true
    wait "$child"
    echo "👋 Service stopped"
    exit 0
}

trap cleanup SIGTERM SIGINT

echo "🎵 Starting Chatterbox TTS service on $HOST:$PORT..."

# Start the service
uvicorn main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level "$LOG_LEVEL" \
    --access-log \
    --loop uvloop \
    --http httptools \
    --ws-ping-interval 20 \
    --ws-ping-timeout 300 \
    --timeout-keep-alive 300 &

child=$!

# Wait for the service to start
sleep 2

# Health check
echo "🏥 Performing initial health check..."
for i in {1..300}; do
    if curl -f -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "✅ Service is healthy and ready to accept connections!"
        break
    fi
    
    if [ $i -eq 300 ]; then
        echo "❌ Health check failed after 300 attempts"
        exit 1
    fi
    
    echo "⏳ Waiting for service to be ready... (attempt $i/300)"
    sleep 1
done

echo "🎉 Chatterbox TTS Service is running!"
echo "📚 API documentation: http://$HOST:$PORT/docs"
echo "🏥 Health check: http://$HOST:$PORT/health"

# Keep the script running and wait for the service
wait "$child"
