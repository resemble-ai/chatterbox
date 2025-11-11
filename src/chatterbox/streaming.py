# streamer imports
import queue
import threading
import soundfile as sf

# tts generation imports
import time
import torch
import numpy as np
from chatterbox import ChatterboxTTS

# server imports
import struct
import socket

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
        dtype = np.float32
    ):
        self._model = self.load_model()
        self.sr = sample_rate
        self.dtype = np.float32

        self.context_window = context_window

        self.request_queue = queue.Queue()
        self.audio_queue = queue.Queue(maxsize=10)
        self.all_tokens_processed = []

        # Creates tts generation thread and starts
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
            # retrieve request
            text = self.request_queue.get()

            # check for shutdown signal
            if text is None:
                break
        
            # Perform tts generation
            for chunk in self._model.generate_stream(text=text):
                # Terminates if thread is stopped
                if not self._running:
                    break

                # Blocks if audio queue is full
                self.audio_queue.put(chunk)
            
            self.audio_queue.put(self._EOS)
    
    def make_request(self, request):
        """
        Makes requests for tts audio generation.
        """
        self.request_queue.put(request)

    def process_queue(self):
        """
        Drain queue and process all chunks.
        """

        request_complete = False

        speech_tokens: List[torch.Tensor] = []

        while self.audio_queue.empty():
            time.sleep(0.1)

        while True:
            try:
                token = self.audio_queue.get_nowait() # chunk: List[torch.Tensor]
            except queue.Empty:
                break
                
            if token is self._EOS:
                # TODO Add end of utterance logic here
                request_complete = True
                break
            
            speech_tokens.append(token)
        
        
        with torch.no_grad():
            speech_tokens = torch.cat(speech_tokens, dim=1)    
        
        # Extract only the conditional batch
        speech_tokens = speech_tokens[0]

        # Process each chunk immediately
        audio, success = self._model._process_token_buffer(
            [speech_tokens], self.all_tokens_processed, self.context_window
        )

        # Reset all_tokens_processed if request complete, otherwise update with new tokens
        if request_complete == True:
            self.all_tokens_processed = []
        elif len(self.all_tokens_processed) == 0:
            self.all_tokens_processed = speech_tokens
        else:
            self.all_tokens_processed = torch.cat([self.all_tokens_processed, speech_tokens], dim=-1)

        if success == False:
            raise ValueError("Failed chunk generation.")
        return audio, request_complete

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

    # server configs
    HOST = "0.0.0.0"
    PORT = 9000

    # Setup Server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)
    print("waiting for connection...")
    client, addr = server.accept()
    print(f"client {addr} connected to port {PORT}\n")

    with client:
        # initializing and starting chatterbox stream
        stream = ChatterboxStreamer(
            sample_rate=sample_rate,
            context_window=50,
            dtype=dtype
        )
        stream.start()
        
        # making test request
        start_time = time.time()
        latency_to_first_chunk = None
        stream.make_request(request)
        
        while True:
            chunk, request_complete = stream.process_queue()

            if latency_to_first_chunk is None:
                latency_to_first_chunk = time.time() - start_time
                print(f"Latency to first chunk: {latency_to_first_chunk}")
            
            # Turn the array into raw bytes
            payload = chunk.tobytes(order="C")

            # 4 byte big-endian length header
            header = struct.pack("!I", len(payload))

            # loops internally until all data is sent
            client.sendall(header + payload)

            if request_complete:
                print(f"Toal audio playtime: {len(audio) / sample_rate}")
                print(f"Total generation time: {time.time() - start_time}")
                break  
    
    stream.stop()
    client.close()
        
    sf.write("stream_snapshot.wav", audio, sample_rate)

if __name__ == "__main__":
    main()
