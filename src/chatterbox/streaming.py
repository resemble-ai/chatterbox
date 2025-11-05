import numpy as np
import queue
import threading
from chatterbox import ChatterboxTTS
import time
import soundfile as sf

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
        fade_duration: float = 0.02,
        dtype = np.float32
    ):
        self._model = self.load_model()
        self.sr = sample_rate
        self.dtype = np.float32

        self.request_queue = queue.Queue()
        self.audio_queue = queue.Queue(maxsize=10)

        self.buffer = AudioBuffer(
            fade_duration=fade_duration,
            sample_rate=sample_rate,
            dtype=dtype
        )
        self.target_buffer_samples = int(1.0 * sample_rate)

        # Creates tts generation thread and starts
        self._running = True
        self._tts_thread = threading.Thread(
            target=self._tts_loop, daemon=True
        )
        self._tts_thread.start()

    def _tts_loop(self):
        while self._running:
            text, start_time = self.request_queue.get()
            latency_to_first_chunk = None

            if text is None:
                # Shutdown signal
                break
        
            for chunk, metrics in self._model.generate_stream(text=text):
                # Terminates tts generation if thread is stopped
                if not self._running:
                    break
            
                if latency_to_first_chunk is None:
                    latency_to_first_chunk = time.time() - start_time
                    print(f"Latency to first chunk: {latency_to_first_chunk}")

                # blocks if audio queue is full
                self.audio_queue.put(chunk)
            
            self.audio_queue.put(self._EOS)
    
    def make_request(self, request):
        """
        Makes requests for tts audio generation.
        """
        self.request_queue.put(request)

    def get_frame(self, num_samples: int) -> np.ndarray:
        """
        Called from audio streaming side.
        """
        req_finished = False

        while True:
            try:
                chunk = self.audio_queue.get_nowait()
            except queue.Empty:
                break
                
            if chunk is self._EOS:
                # TODO Add end of utterance logic here
                req_finished = True
                continue
            
            self.buffer.add_chunk(chunk)
        
        # Return audio frame
        return self.buffer.get_samples(num_samples), req_finished

    def stop(self):
        """
        Terminates tts generation thread
        """
        self._running = False
        self.request_queue.put(None)
        self._tts_thread.join(timeout=1.0)

    def available_samples(self) -> int:
        return self.buffer.available_samples()

    def load_model(self):
        """Loads chatterbox model for tts generation"""
        model = None

        model = ChatterboxTTS.from_pretrained(device="cuda")

        if model is None:
            raise Exception("Failed to load chatterbox model.")
        
        return model


def main():
    sample_rate = 24000
    frame_size = 210
    frame_duration = frame_size / sample_rate
    dtype = np.float32

    streamer = ChatterboxStreamer(
        sample_rate = sample_rate,
        fade_duration = 0.02,
        dtype=dtype
    )

    # text = [
    #     "Active-duty U.S. military personnel get special baggage allowances with Delta. When traveling on orders or for personal travel, you’ll receive baggage fee exceptions and extra checked bag benefits. These allowances apply to all branches, including the Marine Corps, Army, Air Force, Space Force, Navy, and Coast Guard. There may be some regional weight or embargo restrictions. Would you like me to text you a link with the full details for military baggage policies?", 
    #     "Yes, there are specific restrictions for minors and unaccompanied minors traveling internationally with Delta Air Lines. For international travel, Delta requires that all passengers under the age of fifteen use the Unaccompanied Minor Service. This service provides supervision from boarding until the child is met at their destination."
    # ]

    request = "Hi, I'm Delta's AI assistant! How can I help you today?"

    audio = np.zeros(0, dtype=dtype)
    start_time = time.time()
    streamer.make_request((request, start_time))
    terminate = False

    while True:
        frame, request_finished = streamer.get_frame(frame_size)
        audio = np.concatenate([audio, frame])
        time.sleep(frame_duration)

        if request_finished: 
            print(f"Total generation time: {time.time() - start_time}")
            terminate = True
        
        if streamer.available_samples() == 0 and terminate:
            break
        
    
    print(f"Total audio play time: {time.time() - start_time}")
    sf.write("stream_snapshot.wav", audio, sample_rate)

if __name__ == "__main__":
    main()
