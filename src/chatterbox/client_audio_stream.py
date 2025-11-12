import pyaudio
import time

# client imports
import socket
import struct

# audio buffer/processing
import numpy as np

# threading
import threading

# metrics
from dataclasses import dataclass
from typing import Optional

HOST = "195.26.233.44"
PORT = 34521
SAMPLE_RATE = 24000

@dataclass
class Metrics:
    chunk_processing_cost: Optional[float] = None
    networking_cost: Optional[float] = None


class AudioBuffer:
    def __init__(
        self,
        fade_duration: float, 
        sample_rate: int,
        dtype=np.float32
    ):  
        self.buffer = np.zeros(0, dtype=dtype)
        self.sample_rate = sample_rate
        self.fade_samples = int(round(fade_duration * self.sample_rate))
        self.dtype = dtype
        self.lock = threading.Lock()

    
    def add_chunk(self, chunk: np.ndarray):
        """
        Adds chunks to audio buffer after applying cross-fading

        Args:
            chunk (np.ndarray): Chunk of processed audio recieved from TCP connection
        """
        with self.lock:
            # No previous audio in buffer
            if self.buffer.size == 0:
                self.buffer = chunk.copy()
                return
            
            # Fade duration is 
            if self.fade_samples == 0:
                self.buffer = np.concatenate([self.buffer, chunk])
                return

            # Apply equal-part linear crossfade
            tail = self.buffer[-self.fade_samples:]
            head = chunk[:self.fade_samples]

            fade_out = np.linspace(1.0, 0, self.fade_samples, endpoint=True, dtype=self.dtype)
            fade_in = 1.0 - fade_out

            fade_chunk = tail * fade_out + head * fade_in

            self.buffer = np.concatenate([self.buffer[:-self.fade_samples], fade_chunk, chunk[self.fade_samples:]]) # Rebuilds buffer with new chunk and crossfade
        
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
            #print(f"Available Samples: {available_samples}")

            # buffer is empty
            if available_samples == 0:
                print("WARNING: no available audio")
                return np.zeros(0, dtype=self.dtype)

            # buffer has has more samples than requested
            if available_samples >= num_samples:
                out = self.buffer[:num_samples].copy()
                self.buffer = self.buffer[num_samples:]
                return out
                
            # num_samples is greater than the number of samples in buffer
            print(f"WARNING: {num_samples - available_samples} more samples requested than available.")
            out = self.buffer.copy()
            self.buffer = np.zeros(0, dtype=self.dtype)
            return out

    def available_samples(self) -> int:
        with self.lock:
            return self.buffer.size

def recv_exact(client: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    total = 0
    while total < n:
        recieved = client.recv_into(view[total:])
        if recieved == 0:
            raise EOFError("Socket closed while recieving data")
        total += recieved
    return bytes(buf)

def network_thread(client: socket.socket, audio_buffer: AudioBuffer, stop_event: threading.Event):
    try:
        # Make recv interruptible so we can shut down promptly
        client.settimeout(1.0)
        while not stop_event.is_set():
            try:
                header = recv_exact(client, 4)
            except socket.timeout:
                continue

            (nbytes,) = struct.unpack("!I", header)

            if nbytes == 0:
                print("Network thread: EOF received")
                break

            # Receive payload; allow timeouts to be retried
            try:
                data = recv_exact(client, nbytes)
            except socket.timeout:
                # partial read timed out; loop back to check stop_event
                continue

            # TODO -> Bytes are being converted to numpy array and then back to bytes, this is costly
            samples = np.frombuffer(data, dtype=np.float32)

            audio_buffer.add_chunk(samples)
    except EOFError:
        print("Network thread: server closed connection")
    finally:
        try:
            client.close()
        except Exception:
            pass
        print("Network thread exiting")

def playback_thread(audio_buffer: AudioBuffer, stop_event: threading.Event, start_time):
    latency_to_first_chunk = None
    p = pyaudio.PyAudio()
    frames_per_buffer = 1024
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=SAMPLE_RATE,
        output=True,
        frames_per_buffer=frames_per_buffer
    )
    try:
        while not stop_event.is_set():
            if audio_buffer.available_samples() >= frames_per_buffer:
                chunk = audio_buffer.get_samples(frames_per_buffer)
                if chunk.size == 0:
                    # If nothing available you can stream silence or sleep briefly
                    time.sleep(0.1)
                    continue

                if latency_to_first_chunk is None:
                    latency_to_first_chunk = time.time() - start_time
                    print(f"Latency_to_first_chunk: {latency_to_first_chunk}")

                stream.write(chunk.tobytes())
            else: 
                # Not enough data, sleep to avoid busy-waiting

                print(f"WARNING: {frames_per_buffer - audio_buffer.available_samples()} more samples requested than available.")
                time.sleep(0.5)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Playback thread exiting")

    
def main():
    buffer = AudioBuffer(
        fade_duration=0.001,
        sample_rate=SAMPLE_RATE,
        dtype=np.float32
    )

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((HOST, PORT))
    print("connected to server")

    start_time = None
    start = client.recv(1024).decode()
    if start == "START":
        start_time = time.time()

    # Use an Event to request a clean shutdown. Don't make threads daemons so
    # we can join them and allow PyAudio/threads to clean up properly.
    stop_event = threading.Event()

    net_thread = threading.Thread(target=network_thread, args=(client, buffer, stop_event), daemon=False)
    play_thread = threading.Thread(target=playback_thread, args=(buffer, stop_event, start_time), daemon=False)

    print("Starting threads")
    net_thread.start()
    play_thread.start()

    try:
        # Keep main alive until user interrupts or threads finish
        while net_thread.is_alive() and play_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: stopping threads")
        stop_event.set()
        try:
            client.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
    finally:
        # Ensure stop flag is set and join threads
        stop_event.set()
        net_thread.join(timeout=5.0)
        play_thread.join(timeout=5.0)
        print("Stopping streaming")

if __name__ == "__main__":
    main()


