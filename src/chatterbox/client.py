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
PORT = 57049
SAMPLE_RATE = 24000
    
def main():

    # setup pyaudio stream
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=SAMPLE_RATE,
        output=True,
        frames_per_buffer=1024
    )

    # connect to server
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn.connect((HOST, PORT))
    conn.settimeout(10.0)
    print("connected to server")

    try:
        while True:
            # receive data from connection
            data = conn.recv(1024)

            # If server closes connection, recv returns b''
            if not data:
                print("Server closed connection")
                break

            # write data to pyaudio stream
            stream.write(data)
    except socket.timeout:
        print("Network thread: socket timeout, no data received")
    except EOFError:
        print("Network thread: server closed connection")
    finally:
        # cleanup pyaudio
        stream.stop_stream()
        stream.close()
        p.terminate()

        # close connection       
        try:
            conn.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()


