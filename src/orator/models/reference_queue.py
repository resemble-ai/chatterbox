# Copyright (c) 2025 Resemble AI
# Author: John Meade
# MIT License
from collections import deque
from queue import Empty, Full
from threading import Lock
from time import sleep
from tqdm import tqdm
import torch


class Closed(Exception):
    pass


class Paused(Exception):
    pass


class ReferenceQueue:
    """
    Thread-safe Queue for object references with built-in stop signal.
    For eg, adding CUDA tensors to a queue.Queue() can easily result in problems.

    Usage is similar to queue.Queue(), but with a "close" and "pause" mechanism:
        rq = ReferenceQueue(1)
        rq.put(1)
        rq.get() # => 1
        rq.get_nowait() # => raises `Empty` exception
        rq.put(2)
        rq.put(3) # => raises `Full` exception
        rq.close()
        rq.get_nowait() # => 2
        rq.get_nowait() # => raises `Closed` exception
        rq.get() # => raises `Closed` exception

    Pausing is a temporary closure for `put` operations (not get), for eg:
        rq.put(4)
        rq.paused = True
        rq.put() # => raises `Paused` exception
        rq.get() # => 4
        rq.get() # => raises `Paused` exception
        rq.paused = False

    Example async usage:
        def my_stream_func():
            while True:
                try:
                    yield rq.get()
                except Closed:
                    return
    """

    def __init__(self, maxlen=99, dt=1/10_000):
        self.lock = Lock()
        self.buffer = deque()
        self.maxlen = maxlen
        self.dt = dt
        self._closed = False
        self.paused = False
        # Note: add separate pausing for `get` and `put` if needed.

    def close(self):
        "No more items can be added after calling this, and `Closed` exceptions will be raised when the queue is empty."
        self._closed = True

    def _get(self):
        empty = len(self.buffer) == 0
        if self._closed and empty:
            raise Closed("can't `get`, queue is closed and empty.")
        if self.paused and empty:
            raise Paused("can't `get`, queue is paused and empty.")
        if not empty:
            return self.buffer.popleft()
        raise Empty("can't `get`, queue is empty")

    def _put(self, obj):
        if self._closed:
            raise Closed("can't `put`, queue is closed.")
        if self.paused:
            raise Paused("can't `put`, queue is paused.")
        if len(self.buffer) < self.maxlen:
            self.buffer.append(obj)
            return
        raise Full("can't `put`, queue is full.")

    def get(self):
        while True:
            try:
                with self.lock:
                    return self._get()
            except (Empty, Paused):
                sleep(self.dt)

    def get_nowait(self):
        with self.lock:
            return self._get()

    def get_iter(self, print_progress=False):
        pbar = tqdm() if print_progress else None
        while True:
            try:
                yield self.get_nowait()
                if print_progress:
                    pbar.update()
            except Closed:
                return
            except Empty:
                sleep(self.dt)

    def get_all(self, print_progress=False):
        return list(self.get_iter(print_progress=print_progress))

    def put(self, tensor):
        while True:
            try:
                with self.lock:
                    return self._put(tensor)
            except (Full, Paused):
                sleep(self.dt)

    def put_nowait(self, tensor):
        with self.lock:
            return self._put(tensor)


def test_ref_queue(device='cuda'):
    shape, n = (2,3,4), 5
    rq = ReferenceQueue(n)

    try:
        rq.get_nowait()
        raise RuntimeError()
    except Empty:
        pass

    for i in range(n):
        rq.put_nowait(i * torch.ones(*shape, device=device))

    try:
        rq.put_nowait(torch.rand(*shape, device=device))
        raise RuntimeError()
    except Full:
        pass

    for i in range(n // 2):
        t = rq.get_nowait()
        assert i == int(t[0, 0, 0])

    for i in range(n - n // 2):
        rq.put_nowait(torch.randn(*shape, device=device))
        t = rq.get_nowait()
        assert n // 2 + i == int(t[0, 0, 0])

    return True


if __name__ == '__main__':
    test_ref_queue()
    print("tests OK")
