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

HOST = "0.0.0.0"
PORT = 9000

@dataclass
class Metrics:
    """Metrics for streaming TTS generation"""
    networking_cost: Optional[float] = None
    text_processing_cost: Optional[float] = None
    generation_cost: Optional[float] = None
    speech_processing_cost: Optional[float] = None
    total_time: Optional[float] = None
    total_audio_duration: Optional[float] = None
    rtf: Optional[float] = None
    total_chunks = 0

class AudioBuffer:
    def __init__(
        self, 
        fade_duration: float, 
        sample_rate: int,
        dtype=np.float32
    ):  
        # Define buffer
        self.dtype = dtype
        self.buffer = np.zeros(0, dtype=dtype)

        # Cross fading variables
        self.sample_rate = sample_rate
        self.fade_samples = int(round(fade_duration * self.sample_rate))
        self.fade_out = np.linspace(1.0, 0, self.fade_samples, endpoint=True, dtype=self.dtype)
        self.fade_in = 1.0 - self.fade_out

        self.lock = threading.Lock()
        self.terminate = False
    
    def add_chunk(self, chunk: np.ndarray):
        """
        Adds processed chunks to buffer
        """
        with self.lock:
            # No previous audio in buffer
            if self.buffer.size == 0:
                self.buffer = chunk.copy()
                return

            # Get available fade samples
            fade_len = min(self.fade_samples, self.buffer.size, chunk.size)

            # Not enough samples to apply cross fading
            if fade_len == 0:
                self.buffer = np.concatenate([self.buffer, chunk])

            # Get prev tail and new head
            tail = self.buffer[-fade_len:].copy()
            head = chunk[:fade_len].copy()

            if fade_len != self.fade_samples:
                # Buffer or chunk is smaller than fade_samples
                fade_out = np.linspace(1.0, 0, fade_len, endpoint=True, dtype=self.dtype)
                fade_in = 1.0 - fade_out
                fade_chunk = tail * fade_out + head * fade_in
            else:
                # fade_samples is smaller than buffer and chunk size
                fade_chunk = tail * self.fade_out + head * self.fade_in

            self.buffer = np.concatenate([self.buffer[:-fade_len], fade_chunk, chunk[fade_len:]]) # Rebuilds buffer with new chunk and crossfade
    
    def get_samples(self, num_samples: int) -> np.ndarray:
        """
        Pop 'num_samples' samples from the front of the buffer.

        Returns:
            Exactly num_saples, padded with zeros if needed.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")

        with self.lock:
            # get number of samples available in buffer
            available_samples = self.buffer.size

            # buffer is empty
            if available_samples == 0:
                return np.zeros(num_samples, dtype=self.dtype)

            # buffer has has more samples than requested
            if available_samples >= num_samples:
                out = self.buffer[:num_samples].copy()
                self.buffer = self.buffer[num_samples:]
                return out
                
            # samples requested is greater than buffer
            print(f"WARNING: {num_samples - available_samples} more samples requested than were available")
            out = self.buffer.copy()
            self.buffer = np.zeros(0, dtype=self.dtype)
            pad = np.zeros(num_samples - available_samples, dtype=self.dtype)
            out = np.concatenate([out, pad])
            return out

    def available_samples(self) -> int:
        with self.lock:
            return self.buffer.size

    def flush(self):
        with self.lock:
            self.flush = True

    def get_flush(self):
        with self.lock:
            return self.flush


def generate_chunks(
    request_queue: queue.Queue, 
    chunk_queue: queue.Queue
):
    # Load chatterbox model and perform setup
    model = ChatterboxTTS.from_pretrained(device = "cuda")
    prompt_file_name = "1_0.5_1.0_0.5_True_0.9.wav"
    prompt_path = Path(__file__).resolve().parents[4] / f"inputs/audio_prompts/{prompt_file_name}"
    model.setup_stream(audio_prompt_path=prompt_path, exaggeration=EXAGGERATION, fade_duration=FADE_DURATION)

    while True:
        # BLOCKS until request is available
        request = request_queue.get()

        start_time = time.time()

        # Checks for shutdown signal
        if request is None:
            break

        # Perform tts generation
        for chunk in model.generate_stream(
            text=request,
            cfg_weight=CFG_WEIGHT,
            temperature=TEMPERATURE,
            chunk_size = CHUNK_SIZE,
        ):
            # Blocks if audio queue is full
            chunk_queue.put(chunk)
        
        # Puts end-of-request signal onto chunk
        print(f"Generation Time Cost: {time.time() - start_time}")
        chunk_queue.put(None)

def process_chunks(chunk_queue, audio_buffer):

    model = ChatterboxTTS.from_pretrained(device = "cuda")
    # prompt_file_name = "1_0.5_1.0_0.5_True_0.9.wav"
    # prompt_path = Path(__file__).resolve().parents[4] / f"inputs/audio_prompts/{prompt_file_name}"
    model.setup_stream(audio_prompt_path=None, exaggeration=EXAGGERATION, fade_duration=FADE_DURATION)

    # Process and send tokens as they're generated
    all_tokens_processed = []
    audio = np.zeros(0, dtype=np.float32)

    while True:
        # TODO -> if processing time is slow enough maybe we could process multiple chunks together if there are multiple in the queue
        token_chunk = chunk_queue.get()

        # Check for end-of-request signal
        if token_chunk is None:
            # TODO end-of-request logic
            all_tokens_passed = []
            break

        # Extract only the conditional batch
        token_chunk = token_chunk[0]

        # Process each chunk immediately
        # TODO switch to saving only the previous chunk, to cut down on torch cat operations
        audio_chunk, success = model._process_token_buffer(
            token_buffer=[token_chunk],
            all_tokens_so_far=all_tokens_processed, 
            context_window=CONTEXT_WINDOW,
        )

        if success:
            audio = np.concatenate([audio, audio_chunk], dtype=np.float32)
            audio_buffer.add_chunk(audio_chunk)

        # Update all_tokens_processed with the new tokens
        # TODO switch to saving only the previous chunk, to cut down on torch cat operations 
        # TODO also check that that this shouldn't only be applied when processing is a sucess
        if len(all_tokens_processed) == 0:
            all_tokens_processed = token_chunk
        else:
            all_tokens_processed = torch.cat([all_tokens_processed, token_chunk], dim=-1)

    sf.write("stream_snapshot.wav", audio, SAMPLE_RATE)

def send_audio(audio_buffer):
    # Setup Server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print("waiting for connection...")
    client, addr = server.accept()
    print(f"client {addr} connected to port {PORT}\n")

    while not audio_buffer.get_flush() and audio_buffer.available_samples() != 0:
        audio_chunk = audio_buffer.get_samples(1024)
        raw_data = audio_chunk.tobytes()
        sock.sendall(raw_data)

    sock.close()



def main():
    # Initialize shared multiprocessing queues
    request_queue = torch.multiprocessing.Queue()
    chunk_queue = torch.multiprocessing.Queue()
    audio_buffer = AudioBuffer(FADE_DURATION, SAMPLE_RATE)

    # Create generation subprocess
    generation = torch.multiprocessing.Process(target=generate_chunks, args=(request_queue, chunk_queue))
    generation.start()

    # Create processing thread
    processing_thread = threading.Thread(target=process_chunks, args=(chunk_queue, audio_buffer,))
    processing_thread.start()

    # Create networking thread with connected client
    networking_thread = threading.Thread(target=process_chunks, args=(audio_buffer,))
    networking_thread.start()

    time.sleep(10.0)

    # Populate request queue with test request
    request = "Active-duty U S military personnel get special baggage allowances with Delta. When traveling on orders or for personal travel, you’ll receive baggage fee exceptions and extra checked bag benefits. These allowances apply to all branches, including the Marine Corps, Army, Air Force, Space Force, Navy, and Coast Guard. There may be some regional weight or embargo restrictions. Would you like me to text you a link with the full details for military baggage policies?" 
    request_queue.put(request)

    # Send generation termination signal
    request_queue.put(None)
    generation.join() # Wait for generation to finish

    processing_thread.join() # Wait for processing to finish
    audio_buffer.flush() # Flush audio buffer
    networking_thread.join() # Wait for audio buffer to be cleared

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
