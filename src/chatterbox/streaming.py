# metrics imports
from dataclasses import dataclass
import time
from typing import Optional

# streamer imports
import queue
import threading

# tts generation imports
import torch
import numpy as np
from chatterbox import ChatterboxTTS

# server imports
import struct
import socket

@dataclass
class StreamingMetrics:
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
        self.fade_duration = fade_duration
        self.sample_rate = sample_rate
        self.fade_samples = int(round(self.fade_duration * self.sample_rate))
        if self.fade_samples == 0:
            raise ValueError("fade_samples must be > 0")
        self.dtype = dtype
        self.buffer = np.zeros(0, dtype=dtype)

    
    def add_chunk(self, chunk: np.ndarray):
        # No previous audio in buffer
        if self.buffer.size == 0:
            self.buffer = chunk.copy()
            return

        fade_len = min(self.fade_samples, self.buffer.size, chunk.size)
        if fade_len == 0:
            self.buffer = np.concatenate([self.buffer, chunk])

        # Apply equal-part linear crossfade
        tail = self.buffer[-fade_len:]
        head = chunk[:fade_len]

        fade_out = np.linspace(1.0, 0, fade_len, endpoint=True, dtype=self.dtype)
        fade_in = 1.0 - fade_out

        fade_chunk = tail * fade_out + head * fade_in

        self.buffer = np.concatenate([self.buffer[:-fade_len], fade_chunk, chunk[fade_len:]]) # Rebuilds buffer with new chunk and crossfade
    
    def get_samples(self, num_samples: int) -> np.ndarray:
        """
        Pop 'num_samples' samples from the front of the buffer.

        Returns:
            Exactly num_saples, padded with zeros if needed.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")

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
            
        # num_samples is greater than the number of samples in buffer
        print(f"WARNING: {num_samples - available_samples} more samples requested than were available")
        out = self.buffer.copy()
        self.buffer = np.zeros(0, dtype=self.dtype)
        pad = np.zeros(num_samples - available_samples, dtype=self.dtype)
        out = np.concatenate([out, pad])
        return out

    def flush(self) -> np.ndarray:
        if self.buffer.size == 0:
            return np.zeros(0, dtype=self.dtype)
        
        out = self.buffer.copy
        self.buffer = np.zeros(0, dtype=self.dtype)
        return out

    def available_samples(self) -> int:
        return self.buffer.size


class ChatterboxStreamer:

    # sentinel for "end of this utterance"
    _EOS = object()

    def __init__(
        self, 
        sample_rate: int = 24000,
        context_window: int = 50,
        dtype = np.float32,
        metrics = None
    ):
        self._model = self.load_model()
        self.sr = sample_rate
        self.dtype = np.float32
        self.context_window = context_window

        self.metrics = metrics

        self.request_queue = queue.Queue()
        self.audio_queue = queue.Queue(maxsize=10)

        # Creates tts generator thread
        self._running = False
        self._tts_thread = threading.Thread(
            target=self._tts_loop, daemon=True
        )

    def start(self):
        """
        Starts TTS generation thread
        """
        self._running = True
        self._tts_thread.start()
        self._model.setup_stream()

    def _tts_loop(self):
        while self._running:
            # Block for next request instead of polling
            text = self.request_queue.get()

            # check for shutdown signal
            if text is None:
                break
        
            # Perform tts generation
            for chunk in self._model.generate_stream(text=text, metrics=self.metrics):
                # Terminates if thread is stopped
                if not self._running:
                    break

                # Blocks if audio queue is full
                self.audio_queue.put(chunk)
            
            # singal end-of-request
            self.audio_queue.put(self._EOS)
    
    def make_request(self, request):
        """
        Makes requests for tts audio generation.
        """
        self.request_queue.put(request)
    
    def get_frame(self) -> np.ndarray:
        """
        Called from audio streaming side.
        """
        # Grabs chunk from queue (blocks until data arrives)
        chunk = self.audio_queue.get()

        if chunk is self._EOS:
            return None, True

        return chunk, False

    def stop(self):
        """
        Terminates tts generation thread
        """
        self._running = False
        self.request_queue.put(None)
        self._tts_thread.join(timeout=1.0)

    def load_model(self):
        """Loads chatterbox model for tts generation"""
        model = None

        model = ChatterboxTTS.from_pretrained(device="cuda")

        if model is None:
            raise Exception("Failed to load chatterbox model.")
        
        return model


def main():
    request = "Active-duty U S military personnel get special baggage allowances with Delta. When traveling on orders or for personal travel, you’ll receive baggage fee exceptions and extra checked bag benefits. These allowances apply to all branches, including the Marine Corps, Army, Air Force, Space Force, Navy, and Coast Guard. There may be some regional weight or embargo restrictions. Would you like me to text you a link with the full details for military baggage policies?" 

    # stream configs
    sample_rate = 24000
    frame_size = 210
    frame_duration = frame_size / sample_rate
    dtype = np.float32

    # Setup Server
    HOST = "0.0.0.0"
    PORT = 9000
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print("waiting for connection...")
    client, addr = server.accept()
    print(f"client {addr} connected to port {PORT}\n")

    with client:
        # Setup metrics
        metrics = StreamingMetrics()

        # initializing and starting chatterbox stream
        stream = ChatterboxStreamer(
            sample_rate=sample_rate,
            context_window=50,
            dtype=dtype,
            metrics=metrics
        )
        stream.start()

        # Setup main thread metrics
        metrics.networking_cost = 0.0
        metrics.total_audio_duration = 0.0
        ref_time = time.time()

        # make stream request
        client.send("START".encode())
        start_time = time.time()
        stream.make_request(request)
        
        while True:
            # update networking cost metrics
            metrics.networking_cost += time.time() - ref_time

            # BLOCKS when no available chunks
            chunk, request_complete = stream.get_frame()
            
            # Signal to terminate thread and close socket
            if chunk is None:
                time.sleep(5.0) # wait for audio to finish streaming
                break

            # set ref time for networking cost calculation
            ref_time = time.time()

            # Update total audio duration with duration of curr chunk
            metrics.total_audio_duration += len(chunk) / sample_rate
            
            # Package and send all data
            payload = chunk.tobytes(order="C")
            header = struct.pack("!I", len(payload))
            client.sendall(header + payload)

        
        print("\nPROCESS SPECIFIC TIME COSTS")
        print(f"Networking: {metrics.networking_cost}")
        print(f"Text Processing: {metrics.text_processing_cost}")
        print(f"Token Generation: {metrics.generation_cost}")
        print(f"Speech Processing: {metrics.speech_processing_cost}")

        print("\nPER CHUNK TIME COSTS")
        print(f"Generation Cost Per Chunk: {metrics.generation_cost / metrics.total_chunks}")
        print(f"Processing Cost Per Chunk: {metrics.speech_processing_cost / metrics.total_chunks}")
        
        print("\nTOTAL TIME COSTS")
        total_time = metrics.networking_cost + metrics.text_processing_cost + metrics.generation_cost + metrics.speech_processing_cost
        print(f"Calculated Total Time: {total_time}")
        print(f"Actual Total Time: {time.time() - start_time - 5.0}")
        print(f"Total Audio Duration: {metrics.total_audio_duration}")
        print(f"RTF: {total_time / metrics.total_audio_duration}")
    stream.stop()
    client.close()

if __name__ == "__main__":
    main()
