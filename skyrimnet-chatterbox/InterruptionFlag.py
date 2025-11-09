import asyncio
import functools


class InterruptionFlag:
    def __init__(self):
        self._interrupted = False
        self._ack_event = asyncio.Event()

    def interrupt(self):
        self._interrupted = True
        # Do NOT set ack_event here — only acknowledge() should

    def reset(self):
        self._interrupted = False
        self._ack_event.clear()

    def is_interrupted(self):
        return self._interrupted

    def acknowledge(self):
        self._ack_event.set()

    async def join(self, timeout=None):
        """Wait until acknowledge() is called after interrupt()."""
        try:
            await asyncio.wait_for(self._ack_event.wait(), timeout)
        except asyncio.TimeoutError:
            raise RuntimeError("Timeout waiting for interruption to be acknowledged.")


def interruptible(gen_func):
    @functools.wraps(gen_func)
    def wrapper(*args, interrupt_flag: InterruptionFlag = None, **kwargs):
        interrupt_flag.reset()
        gen = gen_func(*args, **kwargs)
        try:
            for item in gen:
                yield item
                if interrupt_flag and interrupt_flag.is_interrupted():
                    print("Interrupted.")
                    break
        finally:
            interrupt_flag.acknowledge()
            if hasattr(gen, "close"):
                gen.close()

    return wrapper