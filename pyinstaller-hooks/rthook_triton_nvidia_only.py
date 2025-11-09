"""
PyInstaller runtime hook to enable Triton JIT compilation in frozen environment.

This sets up the necessary environment for Triton to compile CUDA kernels
at runtime, including Python headers and compilation paths.
"""
import os
import sys
from pathlib import Path

# Set up Python compilation environment for Triton JIT
if hasattr(sys, '_MEIPASS'):
    meipass = Path(sys._MEIPASS)
    
    # Set Python include path for headers (Python.h)
    python_include = meipass / 'include'
    if python_include.exists():
        os.environ['PYTHON_INCLUDE'] = str(python_include)
    
    # Set Python libs path for linking
    python_libs = meipass / 'libs'
    if python_libs.exists():
        os.environ['PYTHON_LIBS'] = str(python_libs)
    
    # Set TRITON_CACHE_DIR to writable temp directory
    import tempfile
    triton_cache = Path(tempfile.gettempdir()) / 'triton_cache'
    triton_cache.mkdir(exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = str(triton_cache)
    
    # Enable Triton JIT compilation (remove interpret mode)
    # Remove TRITON_INTERPRET if it was set to disable JIT
    os.environ.pop('TRITON_INTERPRET', None)
    
    print(f"Triton JIT enabled with cache: {triton_cache}")
    if python_include.exists():
        print(f"Python headers available: {python_include}")
    if python_libs.exists():
        print(f"Python libs available: {python_libs}")

# Restrict Triton to NVIDIA backend only (keep this part)
os.environ['TRITON_BACKENDS'] = 'nvidia'

def _patch_triton_backends():
    """Patch Triton to only discover NVIDIA backends in frozen app."""
    try:
        import triton.backends
        
        original_discover = triton.backends._discover_backends
        
        def _discover_nvidia_only():
            """Only discover NVIDIA backend, ignore AMD/others."""
            backends = original_discover()
            # Filter to only nvidia backend
            return {k: v for k, v in backends.items() if k == 'nvidia'}
        
        triton.backends._discover_backends = _discover_nvidia_only
    except (ImportError, AttributeError):
        # Triton not available or API changed, ignore
        pass

# Apply patch when this hook runs
_patch_triton_backends()
