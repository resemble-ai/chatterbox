import numpy as np
import queue
import threading
from chatterbox import ChatterboxTTS
import time

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
        self.dtype = dtype
        self.buffer = np.zeros(0, dtype=dtype)

    
    def add_chunk(self, chunk: np.ndarray):
        # ensures chunk is a ndarray and is 1 dimensional
        chunk = np.asarray(chunk, self.dtype).ravel()

        # No audio in provided chunk
        if chunk.size == 0:
            return
        
        # No previous audio in buffer
        if self.buffer.size == 0:
            self.buffer = chunk.copy()
            return
        
        # No crossfading
        if self.fade_samples == 0:
            self.buffer = np.concatenate([self.buffer, chunk])
            return

        # Add chunk and apply equal-part linear crossfade
        n = min(self.fade_samples, self.buffer.size, chunk.size)
        tail = self.buffer[-n:]
        head = chunk[:n]

        fade_out = np.linspace(1.0, 0, n, endpoint=True, dtype=self.dtype)
        fade_in = 1.0 - fade_out

        fade_chunk = tail * fade_out + head * fade_in

        self.buffer = np.concatenate([self.buffer[:-n], fade_chunk, chunk[n:]]) # Rebuilds buffer with new chunk and crossfade
    
    def get_samples(self, num_samples: int) -> np.ndarray:
        """
        Pop 'num_samples' samples from the fornt of the buffer.

        Returns:
            Exactly num_saples, padded with zeros if needed.
        """
        if self.buffer.size == 0 or num_samples == 0:
            return np.zeros(num_samples, dtype=self.dtype)

        if self.buffer.size >= num_samples:
            out = self.buffer[:num_samples].copy()
            self.buffer = self.buffer[num_samples:]
            return out
            
        # num_samples is greater than the number of samples in buffer
        out = self.buffer.copy()
        out = np.pad(out, (0, num_samples - self.buffer.size), mode='constant', constant_values=0)
        return out
    
    def flush(self) -> np.ndarray:
        """
        Drains remaining samples.

        Returns
            Remaining samples in buffer as a np.ndarray
        """
        if self.buffer.size == 0:
            return np.zeros(0, dtype=self.dtype)
        
        out = self.buffer.copy()
        self.buffer = np.zeros(0, dtype=self.dtype)
        return out

    def available_samples(self) -> int:
        return self.buffer.size


class ThreadedStreamer:
    def __init__(
        self, 
        sample_rate: int = 24000,
        fade_duration: float = 0.02,
        dtype = np.float32
    ):
        self.sr = sample_rate
        self.dtype = np.float32
        self.queue = queue.Queue(maxsize=10)
        self.buffer = AudioBuffer(
            fade_duration=fade_duration,
            sample_rate=sample_rate,
            dtype=dtype
        )
        self.target_buffer_samples = int(60 * sample_rate)
        self._model = self.load_model()
        self._producer_thread = threading.Thread(
            target=self._generation_loop, daemon=True
        )
        self._finished = False # generation finished?
        self._started = False # generation started?
    
    def start(self):
        """
        Starts the producer thread. This method should be called once before streaming.
        """
        if not self._started:
            self._started = True
            self._producer_thread.start()
    
    def _generation_loop(self):
        text = "Hi, I'm delta's AI assistant! How can I help you today?"
        exaggeration = 0.5
        cfg_weight = 0.5
        chunk_size = 60
        temperature = 0.8
        context_window = 150

        for chunk, metrics in self._model.generate_stream(
            text=text,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            chunk_size=chunk_size,
            context_window=context_window,
            temperature=temperature,
        ):
            # blocks if queue is full
            self.queue.put(chunk)
            
        # Signal end of stream
        self._finished = True

        # Adds sentinel value to signal end of stream to consumer
        # if the queue is full then the consumer will catch that the producer is terminated with _finished.
        self.queue.put_nowait(None)
    
    def _push_to_buffer(self):
        """Push chunks from the queue onto the audio buffer."""
        
        # Prevents unbounded memory growth. `target_buffer_samples` is set to 60 seconds of audio so it shouldn't be reached.
        while self.buffer.available_samples() < self.target_buffer_samples:
            try:
                chunk = self.queue.get_nowait()
            except queue.Empty:
                break

            # Sentinel: End of streaming
            if chunk is None:
                self._finished = True
                break
            
            self.buffer.add_chunk(chunk)

    def get_frame(self, num_samples: int) -> np.ndarray:
        # Top up buffer from queue (wont't block)
        print(self.queue.qsize())
        self._push_to_buffer()

        # Get a frame
        return self.buffer.get_samples(num_samples)

    def finished(self) -> bool:
        """
        True if the processor has finished and there are no samples left
        """
        return self._finished and self.buffer.available_samples() == 0

    def load_model(self):
        """Loads chatterbox model for tts generation"""
        model = None

        model = ChatterboxTTS.from_pretrained(device="cuda")

        if model is None:
            raise Exception("Failed to load chatterbox model.")
        
        return model


def main():


    streamer = ThreadedStreamer(
        sample_rate = 24000,
        fade_duration = 0.02,
    )

    streamer.start()
    frames = []
    frame_size = 240

    while not streamer.finished():
        frame = streamer.get_frame(frame_size)
        frames.append(frame)
    
    audio = np.concatenate(frames)
    sf.write("stream_snapshot.wav", audio, sample_rate)

if __name__ == "__main__":
    main()
