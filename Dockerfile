# Start from a Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install system dependencies that might be needed by torch, soundfile, etc.
# For example, libsndfile1 for soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Consider using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
# Assuming your main.py and any related modules (like a local chatterbox src) are in the current directory
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable if necessary (e.g., for model paths, though chatterbox downloads its own)
# ENV MODEL_PATH=/app/models 

# Command to run the application using LitServe
# Ensure main.py has `api = ChatterboxLitAPI()` at the global scope or similar for LitServe to find.
CMD ["litestar", "run", "main:api", "--host", "0.0.0.0", "--port", "8000"]
