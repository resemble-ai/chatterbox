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

FADE_DURATION = 0.001
CONTEXT_WINDOW = 50

EXAGGERATION = 0.5
CFG_WEIGHT = 0.5
TEMPERATURE = 0.8
CHUNK_SIZE = 50

prompt_file_name = "1_0.5_1.0_0.5_True_0.9.wav"
AUDIO_PROMPT_PATH = Path(__file__).resolve().parents[4] / f"inputs/audio_prompts/{prompt_file_name}"

HOST = "0.0.0.0"
PORT = 9000

# @dataclass
# class Metrics:
#     """Metrics for streaming TTS generation"""
#     networking_cost: Optional[float] = None
#     text_processing_cost: Optional[float] = None
#     generation_cost: Optional[float] = None
#     speech_processing_cost: Optional[float] = None
#     total_time: Optional[float] = None
#     total_audio_duration: Optional[float] = None
#     rtf: Optional[float] = None
#     total_chunks = 0

# class AudioBuffer:
#     def __init__(
#         self, 
#         fade_duration: float, 
#         sample_rate: int,
#         dtype=np.float32
#     ):  
#         # Define buffer
#         self.dtype = dtype
#         self.buffer = np.zeros(0, dtype=dtype)

#         # Cross fading variables
#         self.sample_rate = sample_rate
#         self.fade_samples = int(round(fade_duration * self.sample_rate))
#         self.fade_out = np.linspace(1.0, 0, self.fade_samples, endpoint=True, dtype=self.dtype)
#         self.fade_in = 1.0 - self.fade_out

#         self._lock = threading.Lock()
#         self._running = True
    
#     def add_chunk(self, chunk: np.ndarray):
#         """
#         Adds processed chunks to buffer
#         """
#         with self._lock:
#             # No previous audio in buffer
#             if self.buffer.size == 0:
#                 self.buffer = chunk.copy()
#                 return

#             # Get available fade samples
#             fade_len = min(self.fade_samples, self.buffer.size, chunk.size)

#             # Not enough samples to apply cross fading
#             if fade_len == 0:
#                 self.buffer = np.concatenate([self.buffer, chunk])

#             # Get prev tail and new head
#             tail = self.buffer[-fade_len:].copy()
#             head = chunk[:fade_len].copy()

#             if fade_len != self.fade_samples:
#                 # Buffer or chunk is smaller than fade_samples
#                 fade_out = np.linspace(1.0, 0, fade_len, endpoint=True, dtype=self.dtype)
#                 fade_in = 1.0 - fade_out
#                 fade_chunk = tail * fade_out + head * fade_in
#             else:
#                 # fade_samples is smaller than buffer and chunk size
#                 fade_chunk = tail * self.fade_out + head * self.fade_in

#             self.buffer = np.concatenate([self.buffer[:-fade_len], fade_chunk, chunk[fade_len:]]) # Rebuilds buffer with new chunk and crossfade
    
#     def get_samples(self, num_samples: int) -> np.ndarray:
#         """
#         Pop 'num_samples' samples from the front of the buffer.

#         Returns:
#             Exactly num_saples, padded with zeros if needed.
#         """
#         if num_samples <= 0:
#             raise ValueError("num_samples must be > 0")

#         with self._lock:
#             # get number of samples available in buffer
#             available_samples = self.buffer.size

#             # buffer is empty
#             if available_samples == 0:
#                 return np.zeros(num_samples, dtype=self.dtype)

#             # buffer has has more samples than requested
#             if available_samples >= num_samples:
#                 out = self.buffer[:num_samples].copy()
#                 self.buffer = self.buffer[num_samples:]
#                 return out
                
#             # samples requested is greater than buffer
#             print(f"WARNING: {num_samples - available_samples} more samples requested than were available")
#             out = self.buffer.copy()
#             self.buffer = np.zeros(0, dtype=self.dtype)
#             pad = np.zeros(num_samples - available_samples, dtype=self.dtype)
#             out = np.concatenate([out, pad])
#             return out

#     def available_samples(self) -> int:
#         with self._lock:
#             return self.buffer.size

#     def terminate(self):
#         with self._lock:
#             self._running = False

#     def is_running(self):
#         with self._lock:
#             return self._running


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
    context_window: int = 50
):
    prev_tail = None # Keeps track of the previous chunk tail for cross fading
    all_tokens_processed = [] # Stores previous tokens to fill context window
    all_audio_processed = np.zeros(0, dtype=np.float32) # Stores all audio chunks produced

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
            token_buffer=[token_chunk],
            all_tokens_so_far=all_tokens_processed,
            context_window=context_window,
            prev_tail=prev_tail
        )

        if success:
            # Send audio over TCP
            data = audio_chunk.tobytes()
            conn.sendall(data)

            # Store new tail
            prev_tail = new_tail

            # Store new audio
            all_audio_processed = np.concatenate([all_audio_processed, audio_chunk], dtype=np.float32)
        
        # Store new tokens
        if len(all_tokens_processed) == 0:
            all_tokens_processed = token_chunk
        else:
            all_tokens_processed = torch.cat([all_tokens_processed, token_chunk], dim=-1)

    # Reconstruct stored audio
    sf.write("stream_snapshot.wav", audio, SAMPLE_RATE)

def main():
    # Initialize shared multiprocessing queues
    request_queue = torch.multiprocessing.Queue()
    chunk_queue = torch.multiprocessing.Queue()

    # Start generation subprocess
    generation = torch.multiprocessing.Process(target=generate_chunks, args=(request_queue, chunk_queue))
    generation.start()

    # Create processing model instance
    proc_model = load_model()
    time.sleep(10.0)

    # Create TCP conneciton with client
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print("waiting for connection...")
    conn, addr = server.accept()
    print(f"client {addr} connected to port {PORT}\n")

    # Send request
    request = "Active-duty U S military personnel get special baggage allowances with Delta. When traveling on orders or for personal travel, you’ll receive baggage fee exceptions and extra checked bag benefits. These allowances apply to all branches, including the Marine Corps, Army, Air Force, Space Force, Navy, and Coast Guard. There may be some regional weight or embargo restrictions. Would you like me to text you a link with the full details for military baggage policies?" 
    request_queue.put(request)

    # Process generated chunks
    process_chunks(model=proc_model, chunk_queue=chunk_queue, conn=conn, context_window=CONTEXT_WINDOW)

    # Send termination signal
    request_queue.put(None)
    generation.join()

    # Close TCP connection
    conn.close()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
