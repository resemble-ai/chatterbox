import torch
import logging
import os
import warnings
import linecache
import traceback
from contextlib import contextmanager

# 1. Setup Persistent Logger (Module Level)
# Unique name per PID to ensure safety in Ray workers
LOGGER_NAME = f"sync_hunter_{os.getpid()}"
hunter_logger = logging.getLogger(LOGGER_NAME)
hunter_logger.propagate = False 
hunter_logger.setLevel(logging.WARNING)

# Track initialization state to avoid handler churn
_INITIALIZED_FILES = set()

# Configuration
LOG_FULL_STACK_TRACE = True

def _ensure_handler_for_file(log_file):
    """
    Ensures the logger has a handler for the specific file, initializing it if needed.
    """
    if log_file in _INITIALIZED_FILES:
        return

    # If this is the first time we see this file in this process, wipe it (fresh run)
    try:
        # Ensure dir exists
        dir_name = os.path.dirname(os.path.abspath(log_file))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        # Check if we already have a handler for this file (redundancy check)
        for h in hunter_logger.handlers:
            if isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file):
                return 

        # Create new handler
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(message)s'))
        hunter_logger.addHandler(fh)
        
        _INITIALIZED_FILES.add(log_file)
        
        # Log header
        hunter_logger.warning(f">>> SYNC HUNTER INITIALIZED. PID: {os.getpid()}")

    except Exception as e:
        print(f"SyncHunter failed to init log file {log_file}: {e}")

@contextmanager
def hunt_syncs(enabled=True, log_file=None):
    """
    Context manager to detect and log CUDA-CPU synchronizations.
    """
    if not enabled:
        yield
        return

    if log_file is None:
        log_file = f"runs/sync_report_pid_{os.getpid()}.txt"

    # 1. Setup Logger (Idempotent)
    _ensure_handler_for_file(log_file)

    # 2. Capture Original Warning Handler
    original_showwarning = warnings.showwarning
    cwd = os.getcwd()

    def custom_sync_handler(message, category, filename, lineno, file=None, line=None):
        """
        Intercepts warnings to check for CUDA synchronization triggers.
        """
        msg_str = str(message)
        
        # Filter unrelated warnings
        if "prototype feature" in msg_str: 
            return

        # Check for PyTorch Sync Warning (UserWarning class + specific text)
        is_sync_warning = issubclass(category, UserWarning) and ("synchroniz" in msg_str.lower())
        
        if not is_sync_warning:
            # Pass through non-sync warnings to the original handler
            if original_showwarning:
                original_showwarning(message, category, filename, lineno, file, line)
            return
        
        # --- LOGGING LOGIC ---
        try:
            clean_path = os.path.relpath(filename, cwd)
        except ValueError:
            clean_path = filename

        try:
            code_line = linecache.getline(filename, lineno).strip()
        except Exception:
            code_line = "<could not read source>"
            
        if LOG_FULL_STACK_TRACE:
            # [SAFETY FIX] Use extract_stack with limit to avoid deep object inspection overhead
            # and prevent potential interference with Autograd graph retention.
            try:
                # Limit depth to 15 frames to reduce overhead
                stack_summary = traceback.extract_stack(limit=15)
                # Filter out the sync_hunter frames themselves
                stack_summary = stack_summary[:-2] 
                formatted_stack = "".join(traceback.format_list(stack_summary))
            except Exception:
                formatted_stack = "<stack trace failed>"

            log_msg = (
                f"SYNC DETECTED\n"
                f"Trigger: {clean_path}:{lineno}\n"
                f"Code: {code_line}\n"
                f"Message: {msg_str}\n"
                f"Stack Trace:\n{formatted_stack}\n"
                f"--\n"
            )
        else:
            log_msg = f"{clean_path}:{lineno} :: {code_line}\n"

        hunter_logger.warning(log_msg)

        # Force flush
        for h in hunter_logger.handlers: 
            if hasattr(h, 'flush'): h.flush()

    # 3. Enable Debug Mode
    try:
        warnings.showwarning = custom_sync_handler
        
        # Only set mode if CUDA is actually available
        if torch.cuda.is_available():
            torch.cuda.set_sync_debug_mode("warn")
            
        yield

    finally:
        # 4. Restore State (Critical)
        # [IMPROVEMENT] Check availability directly to avoid UnboundLocalError
        if torch.cuda.is_available():
            torch.cuda.set_sync_debug_mode("default")
            
        warnings.showwarning = original_showwarning