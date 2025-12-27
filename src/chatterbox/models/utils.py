from os import environ
import gc
import logging
import torch
from torch import bfloat16, float16, float32, cuda, backends, mps
from psutil import virtual_memory

logger = logging.getLogger(__name__)


# =============================================================================
# Tensor Contiguity Management for MPS
# =============================================================================
#
# Problem: MPS Metal kernels require contiguous tensors. Non-contiguous
# operations (transpose, permute, slicing) create views with changed strides.
# Many Metal kernels fall back to slow "Gather-Scatter" patterns or fail
# silently when given non-contiguous tensors.
#
# Critical Issue: Operations like addcmul_ and addcdiv_ (used in optimizers
# and flow matching solvers) fail to update non-contiguous tensors correctly
# on MPS, leading to static or garbage output.
#
# Solution: Ensure contiguity before intensive MPS operations (matmul,
# attention, convolutions). The copy cost is negligible vs performance penalty.
# =============================================================================


def ensure_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor is contiguous for MPS kernel compatibility.

    MPS Metal kernels are optimized for contiguous memory layouts. When
    non-contiguous tensors (from transpose, permute, slicing) are passed
    to optimized kernels, the backend falls back to slow paths or fails.

    Args:
        tensor: Input tensor that may be non-contiguous

    Returns:
        Contiguous tensor (original if already contiguous, copy otherwise)

    Memory Impact: 5-10% overhead for non-contiguous tensors
    Quality Impact: None (prevents silent failures)
    Speed Impact: 5-15% speedup from kernel optimization
    """
    if tensor is None:
        return tensor
    if not tensor.is_contiguous():
        return tensor.contiguous()
    return tensor


def ensure_contiguous_pair(a: torch.Tensor, b: torch.Tensor) -> tuple:
    """
    Ensure both tensors in a pair are contiguous for binary operations.

    Use before matmul, addcmul_, addcdiv_, and other binary MPS operations.

    Args:
        a: First tensor
        b: Second tensor

    Returns:
        Tuple of contiguous tensors
    """
    return ensure_contiguous(a), ensure_contiguous(b)


def safe_mps_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Safe matrix multiplication that ensures contiguity for MPS.

    Prevents silent failures from non-contiguous tensors in attention
    score computation and other matmul-heavy operations.

    Args:
        a: First tensor for matmul
        b: Second tensor for matmul

    Returns:
        Result of torch.matmul with contiguous inputs
    """
    a, b = ensure_contiguous_pair(a, b)
    return torch.matmul(a, b)


def contiguous_transpose(tensor: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    """
    Transpose followed by contiguous() in a single allocation.

    Preferred pattern for MPS: tensor.transpose(...).contiguous()
    This creates a single new tensor instead of a view + copy.

    Args:
        tensor: Input tensor
        dim0: First dimension to swap
        dim1: Second dimension to swap

    Returns:
        Contiguous transposed tensor
    """
    return tensor.transpose(dim0, dim1).contiguous()


def contiguous_permute(tensor: torch.Tensor, *dims: int) -> torch.Tensor:
    """
    Permute followed by contiguous() in a single allocation.

    Args:
        tensor: Input tensor
        *dims: New dimension ordering

    Returns:
        Contiguous permuted tensor
    """
    return tensor.permute(*dims).contiguous()


def contiguous_view(tensor: torch.Tensor, *shape: int) -> torch.Tensor:
    """
    Safe view that ensures contiguity first.

    View operations require contiguous tensors. This helper prevents
    runtime errors from non-contiguous view attempts.

    Args:
        tensor: Input tensor
        *shape: Target shape

    Returns:
        View of contiguous tensor
    """
    return ensure_contiguous(tensor).view(*shape)


def clear_device_memory():
    """Clear GPU memory for both CUDA and MPS devices."""
    gc.collect()
    if cuda.is_available():
        cuda.empty_cache()
    elif hasattr(backends, "mps") and backends.mps.is_available():
        mps.empty_cache()
        mps.synchronize()


# =============================================================================
# MLX Memory Management
# =============================================================================


def _get_mlx():
    """Safely import MLX if available."""
    try:
        import mlx.core as mx

        return mx
    except ImportError:
        return None


def set_mlx_cache_limit(limit_gb: float = 4.0):
    """
    Set MLX cache memory limit to prevent unbounded growth.

    The cache holds reusable memory allocations. Setting a limit prevents
    MLX from holding onto too much cached memory between operations.

    Args:
        limit_gb: Maximum cache size in GB (default: 4GB)

    Returns:
        Previous cache limit in GB, or None if MLX unavailable
    """
    mx = _get_mlx()
    if mx is None:
        return None
    try:
        limit_bytes = int(limit_gb * 1024 * 1024 * 1024)
        old_limit = mx.set_cache_limit(limit_bytes)
        old_limit_gb = old_limit / (1024 * 1024 * 1024)
        logger.info(
            f"[MLX MEMORY] Set cache limit to {limit_gb}GB (was {old_limit_gb:.1f}GB)"
        )
        return old_limit_gb
    except Exception as e:
        logger.warning(f"Could not set MLX cache limit: {e}")
        return None


def set_mlx_memory_limit(limit_gb: float = 8.0):
    """
    Set MLX total memory limit to prevent system instability.

    This controls the maximum total memory MLX can allocate. Setting this
    prevents MLX from consuming all available system memory.

    Args:
        limit_gb: Maximum memory in GB (default: 8GB)

    Returns:
        Previous memory limit in GB, or None if MLX unavailable
    """
    mx = _get_mlx()
    if mx is None:
        return None
    try:
        limit_bytes = int(limit_gb * 1024 * 1024 * 1024)
        old_limit = mx.set_memory_limit(limit_bytes)
        old_limit_gb = old_limit / (1024 * 1024 * 1024)
        logger.info(
            f"[MLX MEMORY] Set memory limit to {limit_gb}GB (was {old_limit_gb:.1f}GB)"
        )
        return old_limit_gb
    except Exception as e:
        logger.warning(f"Could not set MLX memory limit: {e}")
        return None


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
        elif hasattr(backends, "mps") and backends.mps.is_available():
            return float16
        else:
            return float32

    device_str = str(device).lower()
    if "cuda" in device_str:
        return bfloat16
    elif "mps" in device_str:
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
        "sys_used_gb": vm.used / 1024**3,
        "sys_available_gb": vm.available / 1024**3,
        "sys_percent": vm.percent,
    }

    # macOS specific: get wired and app memory via vm_stat
    try:
        import subprocess

        result = subprocess.run(["vm_stat"], capture_output=True, text=True)
        lines = result.stdout.split("\n")
        page_size = 16384  # Default for Apple Silicon

        stats = {}
        for line in lines:
            if ":" in line:
                key, val = line.split(":")
                try:
                    stats[key.strip()] = int(val.strip().rstrip("."))
                except Exception:
                    pass

        if "Pages wired down" in stats:
            info["wired_gb"] = (stats["Pages wired down"] * page_size) / 1024**3
        if "Pages occupied by compressor" in stats:
            info["compressed_gb"] = (
                stats["Pages occupied by compressor"] * page_size
            ) / 1024**3
        if "Pages active" in stats:
            info["active_gb"] = (stats["Pages active"] * page_size) / 1024**3
    except Exception:
        pass

    # MPS memory
    if hasattr(backends, "mps") and backends.mps.is_available():
        try:
            mps.synchronize()
            info["mps_allocated_mb"] = mps.current_allocated_memory() / 1024**2
            if hasattr(mps, "driver_allocated_memory"):
                info["mps_driver_mb"] = mps.driver_allocated_memory() / 1024**2
        except Exception:
            pass

    return info


def log_memory(step, label=""):
    """Log comprehensive memory usage at a specific step."""
    if not is_debug():
        return

    info = get_memory_info()

    parts = [f"[T3 MEM] Step {step:4d} {label}:"]
    parts.append(f"Sys={info['sys_used_gb']:.1f}GB ({info['sys_percent']:.0f}%)")

    if "wired_gb" in info:
        parts.append(f"Wired={info['wired_gb']:.1f}GB")
    if "compressed_gb" in info:
        parts.append(f"Compressed={info['compressed_gb']:.1f}GB")
    if "active_gb" in info:
        parts.append(f"Active={info['active_gb']:.1f}GB")
    if "mps_allocated_mb" in info:
        parts.append(f"MPS={info['mps_allocated_mb']:.0f}MB")
    if "mps_driver_mb" in info:
        parts.append(f"MPSDriver={info['mps_driver_mb']:.0f}MB")

    print(" | ".join(parts))
