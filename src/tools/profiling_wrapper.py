# tools/profiling_wrapper.py
# Adapted for Chatterbox TTS profiling
import time
import os
import sys
import logging
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from tools.profiler_python_utils import process_profile_json, process_torch_trace_json

logger = logging.getLogger(__name__)

class AuraProfiler:
    def __init__(self, log_dir: str, actor_name: str, enabled: bool = True, 
                 enable_python: bool = True, enable_torch: bool = True, 
                 schedule_config: dict = None, 
                 delay_sec: float = 0.0, max_duration_sec: float = 0.0, max_events: int = 0):
        
        self.enabled = enabled
        self.enable_python = enable_python and enabled
        self.enable_torch = enable_torch and enabled
        self.delay_sec = delay_sec
        self.max_duration_sec = max_duration_sec
        self.max_events = max_events
        
        if actor_name:
            self.log_dir = os.path.join(log_dir, "profiling", actor_name)
        else:
            self.log_dir = log_dir

        self.actor_name = actor_name or "Profiler"
        self.py_profiler = None
        self.torch_profiler = None
        self._is_legacy_torch = False 
        
        # Internal State Machine
        self._state = "IDLE"  # IDLE -> WAITING -> RECORDING -> DONE
        self._global_start_time = 0.0
        self._recording_start_time = 0.0
        self._event_count = 0

        if not self.enabled: return

        os.makedirs(self.log_dir, exist_ok=True)
            
        if self.enable_python:
            try:
                from pyinstrument import Profiler
                self.py_profiler = Profiler(interval=0.001)
            except ImportError:
                logger.warning("pyinstrument not installed. Python profiling disabled. Install with: pip install pyinstrument")

        if self.enable_torch:
            try:
                if not schedule_config:
                    self.enable_torch = False
                elif schedule_config.get('repeat', 1) == 0:
                    self.torch_profiler = torch.autograd.profiler.profile(
                        use_device='cuda' if torch.cuda.is_available() else 'cpu',
                        record_shapes=True,
                        profile_memory=False, 
                        with_stack=True, 
                        use_kineto=False 
                    )
                    self._is_legacy_torch = True
                else:
                    # Kineto (Distributed Mode) - not used for local generation
                    pass 
            except Exception as e:
                logger.error(f"Failed to init Torch Profiler: {e}")
                self.torch_profiler = None

    def _profiler_hook(self, frame, event, arg):
        """State Machine running on every Python function call."""
        if event != 'call': return

        current_time = time.time()

        # State 1: WAITING for Delay
        if self._state == "WAITING":
            if (current_time - self._global_start_time) >= self.delay_sec:
                logger.info(f"[{self.actor_name}] Delay passed. STARTING capture.")
                self._state = "RECORDING"
                self._recording_start_time = current_time
                self._event_count = 0
                
                if self.torch_profiler and self._is_legacy_torch:
                    self.torch_profiler.__enter__()

        # State 2: RECORDING
        elif self._state == "RECORDING":
            self._event_count += 1
            
            stop_by_time = (self.max_duration_sec > 0 and (current_time - self._recording_start_time) >= self.max_duration_sec)
            stop_by_events = (self.max_events > 0 and self._event_count >= self.max_events)
            
            if stop_by_time or stop_by_events:
                reason = "Time" if stop_by_time else "Events"
                logger.info(f"[{self.actor_name}] Limit reached ({reason}). STOPPING capture and flushing.")
                
                self._state = "DONE"
                
                if self.torch_profiler and self._is_legacy_torch:
                    self.torch_profiler.__exit__(None, None, None)
                    self._save_torch_trace(self.torch_profiler, "auto_capture")
                    self.torch_profiler = None
                
                sys.setprofile(None)

    def start(self):
        if not self.enabled: return
        
        if self.py_profiler and not self.py_profiler.is_running:
            self.py_profiler.start()
            
        if self.torch_profiler:
            if self._is_legacy_torch:
                if self.delay_sec > 0 or self.max_duration_sec > 0 or self.max_events > 0:
                    self._global_start_time = time.time()
                    self._state = "WAITING"
                    sys.setprofile(self._profiler_hook)
                else:
                    self.torch_profiler.__enter__()
                    self._state = "RECORDING"
            else:
                self.torch_profiler.start()

    def step(self):
        if self.enabled and self.torch_profiler and not self._is_legacy_torch:
            self.torch_profiler.step()

    def _save_torch_trace(self, p, tag):
        try:
            trace_path = os.path.join(self.log_dir, f"trace_{tag}.json")
            if self._is_legacy_torch:
                logger.info(f"[{self.actor_name}] Exporting legacy trace...")

            try: p.export_chrome_trace(trace_path)
            except (RuntimeError, AssertionError):
                return

            logger.info(f"[{self.actor_name}] Torch trace saved: {trace_path}")
            summary_path = os.path.join(self.log_dir, f"summary_trace_{tag}.txt")
            process_torch_trace_json(trace_path, summary_path)
            if os.path.exists(summary_path): os.remove(trace_path)
        except BaseException as e:
            logger.error(f"[{self.actor_name}] Error processing/saving trace: {repr(e)}")

    def stop_and_save(self, tag: str):
        if not self.enabled: return

        if sys.getprofile() == self._profiler_hook:
            sys.setprofile(None)

        if self.torch_profiler:
            try: 
                if self._is_legacy_torch:
                    if self._state == "RECORDING":
                        self.torch_profiler.__exit__(None, None, None)
                else:
                    self.torch_profiler.stop()
            except BaseException: pass 

            if self.torch_profiler:
                 self._save_torch_trace(self.torch_profiler, tag)
            self.torch_profiler = None 

        if self.py_profiler and self.py_profiler.is_running:
            try:
                self.py_profiler.stop()
                html_path = os.path.join(self.log_dir, f"py_profile_{tag}.html")
                with open(html_path, "w", encoding='utf-8') as f:
                    f.write(self.py_profiler.output_html())
                
                json_path = os.path.join(self.log_dir, f"py_profile_{tag}.json")
                from pyinstrument.renderers import JSONRenderer
                with open(json_path, "w", encoding='utf-8') as f:
                    f.write(self.py_profiler.output(renderer=JSONRenderer()))
                    
                summary_path = os.path.join(self.log_dir, f"summary_{tag}.txt")
                process_profile_json(json_path, summary_path)
                
                if os.path.exists(summary_path):
                    os.remove(json_path)
                    os.remove(html_path)
                self.py_profiler.reset()
            except BaseException as e:
                logger.error(f"[{self.actor_name}] Error saving python profile: {e}")