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

def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    total = 0
    while total < n:
        recieved = sock.recv_into(view[total:])
        if recieved == 0:
            raise EOFError("Socket closed while recieving data")
        total += recieved
    return bytes(buf)
    
def main():
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=SAMPLE_RATE,
        output=True,
        frames_per_buffer=1024
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    print("connected to server")

    try:
        while True:
            # header = recv_exact(sock, 4)
            # (chunk_size,) = struct.unpack("!I", header)
            # data = recv_exact(sock, chunk_size)
            data = sock.recv(1024)
            stream.write(data)
    except EOFError:
        print("Network thread: server closed connection")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate
        try:
            sock.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()


