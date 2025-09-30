from __future__ import annotations

import ctypes
import logging
import platform
import subprocess
from typing import Optional


logger = logging.getLogger(__name__)


class AntiSleepGuard:
    """Best-effort guard that keeps the host machine awake."""

    def __init__(self) -> None:
        self._platform = platform.system()
        self._active = False
        self._blocker_proc: Optional[subprocess.Popen] = None

    def enable(self) -> None:
        if self._active:
            return

        supported = False
        try:
            if self._platform == "Windows":
                self._enable_windows()
                supported = True
            elif self._platform == "Darwin":
                self._enable_macos()
                supported = True
            elif self._platform == "Linux":
                supported = self._enable_linux()
            else:
                logger.warning("Anti-sleep guard is not implemented for %s", self._platform)
        except Exception:  # pragma: no cover - platform specific
            logger.exception("Failed to enable anti-sleep guard")
            return

        if supported:
            self._active = True
            logger.info("Anti-sleep guard enabled")
        else:
            logger.info("Anti-sleep guard could not be enabled on this platform")

    def disable(self) -> None:
        if not self._active:
            return

        try:
            if self._platform == "Windows":
                self._disable_windows()
            elif self._platform == "Darwin":
                self._disable_macos()
            elif self._platform == "Linux":
                self._disable_linux()
        except Exception:  # pragma: no cover - platform specific
            logger.exception("Failed to disable anti-sleep guard")
        finally:
            self._active = False
            logger.info("Anti-sleep guard disabled")

    def __enter__(self) -> "AntiSleepGuard":
        self.enable()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disable()

    def _enable_windows(self) -> None:
        es_continuous = 0x80000000
        es_system_required = 0x00000001
        es_display_required = 0x00000002
        flags = es_continuous | es_system_required | es_display_required
        result = ctypes.windll.kernel32.SetThreadExecutionState(flags)
        if result == 0:
            raise OSError("SetThreadExecutionState failed")

    def _disable_windows(self) -> None:
        es_continuous = 0x80000000
        if ctypes.windll.kernel32.SetThreadExecutionState(es_continuous) == 0:
            raise OSError("SetThreadExecutionState reset failed")

    def _enable_macos(self) -> None:
        self._blocker_proc = subprocess.Popen(["caffeinate"])

    def _disable_macos(self) -> None:
        if self._blocker_proc:
            self._terminate_blocker()

    def _enable_linux(self) -> bool:
        try:
            self._blocker_proc = subprocess.Popen(
                [
                    "systemd-inhibit",
                    "--what=idle:sleep",
                    "--mode=block",
                    "--who=Chattractive",
                    "--why=Prevent system sleep while Chattractive runs",
                    "sleep",
                    "infinity",
                ]
            )
        except FileNotFoundError:
            logger.warning("systemd-inhibit is not available; anti-sleep guard cannot run")
            return False

        if self._blocker_proc.poll() is not None:
            raise RuntimeError("systemd-inhibit exited unexpectedly")
        return True

    def _disable_linux(self) -> None:
        if self._blocker_proc:
            self._terminate_blocker()

    def _terminate_blocker(self) -> None:
        if not self._blocker_proc:
            return
        self._blocker_proc.terminate()
        try:
            self._blocker_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._blocker_proc.kill()
        finally:
            self._blocker_proc = None
