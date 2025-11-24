# metrics imports
from dataclasses import dataclass
import time
from typing import Optional

# streamer imports
import queue
import threading
import multiprocessing


# audio imports
import torch
import numpy as np
from chatterbox import ChatterboxTTS
from chatterbox.tts import EOS, END_OF_REQUEST
from pathlib import Path
import soundfile as sf

# server imports
import struct
import socket

SAMPLE_RATE = 24000

FADE_DURATION = 0.005
CONTEXT_WINDOW = 250

EXAGGERATION = 0.5
CFG_WEIGHT = 0.5
TEMPERATURE = 0.8
CHUNK_SIZE = 15

prompt_file_name = "1_0.5_1.0_0.5_True_0.9.wav"
AUDIO_PROMPT_PATH = Path(__file__).resolve().parents[4] / f"inputs/audio_prompts/{prompt_file_name}"

HOST = "0.0.0.0"
PORT = 9000

@dataclass
class Metrics:
    """Metrics for streaming TTS generation"""
    start_time: Optional[float] = None
    first_chunk_time: Optional[float] = None
    playback_start_time: Optional[float] = None
    generation_end_time: Optional[float] = None
    audio_duration: Optional[float] = None

def load_model():
    model = ChatterboxTTS.from_pretrained(device = "cuda")
    model.setup_stream(audio_prompt_path=AUDIO_PROMPT_PATH, exaggeration=EXAGGERATION, fade_duration=FADE_DURATION)
    return model

def generate_chunks(
    request_queue: queue.Queue, 
    chunk_queue: queue.Queue
):
    # Create model instance
    model = load_model()

    while True:
        # BLOCKS until request is available
        text = request_queue.get()

        # Checks for shutdown signal
        if text is None:
            break

        # Perform tts generation
        for chunk in model.generate_stream(
            text=text,
            cfg_weight=CFG_WEIGHT,
            temperature=TEMPERATURE,
            chunk_size = CHUNK_SIZE,
        ):
            # Blocks if audio queue is full
            chunk_queue.put(chunk)

def process_chunks(
    model: ChatterboxTTS,
    chunk_queue: torch.multiprocessing.Queue,
    conn: socket.socket,
    context_window: int = 25,
    metrics = None
):
    prev_tail = None # Keeps track of the previous chunk tail for cross fading
    all_tokens_processed = [] # Stores previous tokens to fill context window

    while True:
        # Wait on generated tokens
        token_chunk = chunk_queue.get()

        # Check for end of sentence signal
        if isinstance(token_chunk, type(EOS)) and token_chunk.name == "EOS":
            all_tokens_processed = []
            prev_tail = None
            continue
            
        # Stream termiate signal
        if isinstance(token_chunk, type(END_OF_REQUEST)) and token_chunk.name == "END_OF_REQUEST":
            break

        # Extract only the conditional batch
        token_chunk = token_chunk[0]

        # Process chunk TODO-> consider only using the previous chunk for each context window cut down on np cat operations
        audio_chunk, new_tail, success = model._process_token_buffer(
            token_buffer=token_chunk,
            all_tokens_so_far=all_tokens_processed,
            context_window=context_window,
            prev_tail=prev_tail
        )

        if success:
            if metrics.first_chunk_time is None:
                metrics.first_chunk_time = time.time()
            metrics.audio_duration += len(audio_chunk) / SAMPLE_RATE

            # Send audio over TCP
            data = audio_chunk.tobytes()
            conn.sendall(data)

            # Store new tail
            prev_tail = new_tail

        # Store new tokens
        if len(all_tokens_processed) == 0:
            all_tokens_processed = token_chunk
        else:
            all_tokens_processed = torch.cat([all_tokens_processed, token_chunk], dim=-1)

    metrics.generation_end_time = time.time()

def main():
    # Initialize metrics
    metrics = Metrics()
    metrics.audio_duration = 0.0

    # Initialize shared multiprocessing queues
    request_queue = torch.multiprocessing.Queue()
    chunk_queue = torch.multiprocessing.Queue()

    # Start generation subprocess
    generation = torch.multiprocessing.Process(target=generate_chunks, args=(request_queue, chunk_queue))
    generation.start()

    # Create processing model instance
    proc_model = load_model()

    # wait for models to finish loading
    time.sleep(15.0)

    # Create TCP conneciton with client
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print("waiting for connection...")
    conn, addr = server.accept()
    print(f"client {addr} connected to port {PORT}\n")

    # Send warmup request
    request = "Warming up the model."
    request_queue.put(request)

    # Process generated chunks
    process_chunks(model=proc_model, chunk_queue=chunk_queue, conn=conn, context_window=CONTEXT_WINDOW, metrics=metrics)

    # reset metrics
    metrics.audio_duration = 0.0
    metrics.first_chunk_time = None
    time.sleep(5.0)

    # Send request
    request = "Active-duty U S military personnel get special baggage allowances with Delta. When traveling on orders or for personal travel, you’ll receive baggage fee exceptions and extra checked bag benefits. These allowances apply to all branches, including the Marine Corps, Army, Air Force, Space Force, Navy, and Coast Guard. There may be some regional weight or embargo restrictions. Would you like me to text you a link with the full details for military baggage policies?" 
    request_queue.put(request)
    metrics.start_time = time.time()

    # Process generated chunks
    process_chunks(model=proc_model, chunk_queue=chunk_queue, conn=conn, context_window=CONTEXT_WINDOW, metrics=metrics)


    # Send termination signal
    request_queue.put(None)
    generation.join()

    # Close TCP connection
    conn.close()

    print(f"Latency to first chunk: {metrics.first_chunk_time - metrics.start_time}")
    print(f"Total generation time: {metrics.generation_end_time - metrics.start_time}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
