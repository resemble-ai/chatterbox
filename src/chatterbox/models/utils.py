from os import environ
from torch import bfloat16, float16, float32, cuda, backends, mps
from psutil import virtual_memory

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def is_debug():
    """Return True when `CHATTERBOX_DEBUG` is set to the string '1'.

    Use this helper when you need the current environment value at runtime.
    """
    return environ.get("CHATTERBOX_DEBUG", "0") == "1"

# Module-level constant preserved for import-time compatibility.
DEBUG_LOGGING = is_debug()

# Enable MPS allocator debug logging when CHATTERBOX_DEBUG is enabled
# See: https://docs.pytorch.org/docs/stable/mps_environment_variables.html
if DEBUG_LOGGING:
    environ["PYTORCH_DEBUG_MPS_ALLOCATOR"] = "1"

def get_optimal_dtype(device=None):
    """
    Returns the optimal dtype for the given device.
    - CUDA: bfloat16 (best performance)
    - MPS: float16 (bfloat16 has limited support, many ops fall back to CPU)
    - CPU: float32 (most compatible)
    """
    if device is None:
        if cuda.is_available():
            return bfloat16
        elif hasattr(backends, 'mps') and backends.mps.is_available():
            return float16
        else:
            return float32
    
    device_str = str(device).lower()
    if 'cuda' in device_str:
        return bfloat16
    elif 'mps' in device_str:
        return float16
    else:
        return float32


def get_optimal_dtype_str(device=None):
    """Returns the optimal dtype as a string for config."""
    dtype = get_optimal_dtype(device)
    if dtype == bfloat16:
        return "bfloat16"
    elif dtype == float16:
        return "float16"
    else:
        return "float32"

def get_memory_info():
    """Get comprehensive memory info matching Activity Monitor on Mac."""
    vm = virtual_memory()
    
    info = {
        'sys_used_gb': vm.used / 1024**3,
        'sys_available_gb': vm.available / 1024**3,
        'sys_percent': vm.percent,
    }
    
    # macOS specific: get wired and app memory via vm_stat
    try:
        import subprocess
        result = subprocess.run(['vm_stat'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        page_size = 16384  # Default for Apple Silicon
        
        stats = {}
        for line in lines:
            if ':' in line:
                key, val = line.split(':')
                try:
                    stats[key.strip()] = int(val.strip().rstrip('.'))
                except:
                    pass
        
        if 'Pages wired down' in stats:
            info['wired_gb'] = (stats['Pages wired down'] * page_size) / 1024**3
        if 'Pages occupied by compressor' in stats:
            info['compressed_gb'] = (stats['Pages occupied by compressor'] * page_size) / 1024**3
        if 'Pages active' in stats:
            info['active_gb'] = (stats['Pages active'] * page_size) / 1024**3
    except:
        pass
    
    # MPS memory
    if hasattr(backends, 'mps') and backends.mps.is_available():
        try:
            mps.synchronize()
            info['mps_allocated_mb'] = mps.current_allocated_memory() / 1024**2
            if hasattr(mps, 'driver_allocated_memory'):
                info['mps_driver_mb'] = mps.driver_allocated_memory() / 1024**2
        except:
            pass
    
    return info


def log_memory(step, label=""):
    """Log comprehensive memory usage at a specific step."""
    if not is_debug():
        return
    
    info = get_memory_info()
    
    parts = [f"[T3 MEM] Step {step:4d} {label}:"]
    parts.append(f"Sys={info['sys_used_gb']:.1f}GB ({info['sys_percent']:.0f}%)")
    
    if 'wired_gb' in info:
        parts.append(f"Wired={info['wired_gb']:.1f}GB")
    if 'compressed_gb' in info:
        parts.append(f"Compressed={info['compressed_gb']:.1f}GB")
    if 'active_gb' in info:
        parts.append(f"Active={info['active_gb']:.1f}GB")
    if 'mps_allocated_mb' in info:
        parts.append(f"MPS={info['mps_allocated_mb']:.0f}MB")
    if 'mps_driver_mb' in info:
        parts.append(f"MPSDriver={info['mps_driver_mb']:.0f}MB")
    
    print(" | ".join(parts))