"""
Runtime hook to ensure CUDA libraries are available in system PATH.
This hook adds the CUDA installation directory to the system PATH so that
CUDA DLLs can be found at runtime.

CRITICAL: Also patches DeepSpeed's CUDA detection to avoid subprocess handle
issues in PyInstaller frozen executables on Windows.
"""
import os
import sys

def patch_deepspeed_cuda_detection():
    """
    Monkeypatch DeepSpeed's subprocess-based CUDA detection to avoid handle issues
    in PyInstaller frozen executables. This patches at import-time using sys.meta_path.
    """
    print("=== Patching DeepSpeed CUDA Detection ===")
    
    import subprocess
    import importlib.abc
    import importlib.machinery
    
    # Store original subprocess.check_output
    _original_check_output = subprocess.check_output
    
    def patched_check_output(cmd, *args, **kwargs):
        """
        Patched check_output that intercepts nvcc calls and returns fake version
        to avoid subprocess handle issues in PyInstaller frozen executables.
        """
        # Check if this is an nvcc version check call from DeepSpeed
        if isinstance(cmd, list) and len(cmd) >= 2 and 'nvcc' in cmd[0] and '-V' in cmd:
            print(f"🔧 Intercepting nvcc subprocess call: {cmd}")
            
            # Extract CUDA version from CUDA_HOME/CUDA_PATH
            cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
            if cuda_home:
                import re
                match = re.search(r'v(\d+)\.(\d+)', cuda_home)
                if match:
                    major, minor = int(match.group(1)), int(match.group(2))
                    print(f"✅ Returning fake nvcc output for CUDA {major}.{minor}")
                    # Return fake nvcc output that DeepSpeed expects
                    return f"nvcc: NVIDIA (R) Cuda compiler driver\nCopyright (c) 2005-2024 NVIDIA Corporation\nBuilt on Wed_Oct_30_01:18:48_Pacific_Daylight_Time_2024\nCuda compilation tools, release {major}.{minor}, V{major}.{minor}.0\nBuild cuda_{major}.{minor}.r{major}.{minor}/compiler.12345678_0"
            
            # Fallback: return CUDA 12.0 format
            print("⚠️ Could not detect CUDA version, returning fake CUDA 12.0")
            return "nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 12.0, V12.0.0"
        
        # For all other subprocess calls, use original
        try:
            return _original_check_output(cmd, *args, **kwargs)
        except OSError as e:
            # If we still get handle errors, at least provide a useful message
            if '[WinError 6]' in str(e):
                print(f"⚠️ Subprocess handle error for command: {cmd}")
                print("⚠️ This is a known PyInstaller limitation on Windows")
                raise
            raise
    
    # Replace subprocess.check_output globally
    subprocess.check_output = patched_check_output
    
    print("✅ subprocess.check_output patched for DeepSpeed compatibility")
    print("=== End DeepSpeed Patch ===\n")

def setup_cuda_path():
    """Add CUDA installation to PATH if available."""
    print("=== CUDA Path Setup Hook ===")
    
    cuda_path = os.environ.get('CUDA_PATH')
    print(f"CUDA_PATH environment variable: {cuda_path}")
    
    if cuda_path:
        if 'CUDA_PATH' in os.environ and 'CUDA_HOME' not in os.environ:
            os.environ['CUDA_HOME'] = os.environ['CUDA_PATH']
            print(f"✅ Set CUDA_HOME to: {os.environ['CUDA_HOME']}")
        cuda_bin = os.path.join(cuda_path, 'bin')
        print(f"Expected CUDA bin directory: {cuda_bin}")
        
        if os.path.exists(cuda_bin):
            # Add CUDA bin directory to PATH if not already present
            current_path = os.environ.get('PATH', '')
            if cuda_bin not in current_path:
                os.environ['PATH'] = cuda_bin + os.pathsep + current_path
                print(f"✅ Added CUDA bin directory to PATH: {cuda_bin}")
            else:
                print(f"✅ CUDA bin directory already in PATH: {cuda_bin}")
                
            # List some key CUDA DLLs that should be available
            key_dlls = ['cudart64_12.dll', 'cublas64_12.dll', 'nvrtc64_120_0.dll']
            for dll in key_dlls:
                dll_path = os.path.join(cuda_bin, dll)
                if os.path.exists(dll_path):
                    print(f"✅ Found CUDA DLL: {dll}")
                else:
                    print(f"❌ Missing CUDA DLL: {dll}")
        else:
            print(f"❌ CUDA bin directory not found: {cuda_bin}")
    else:
        print("❌ CUDA_PATH environment variable not set.")
        print("Please ensure NVIDIA CUDA toolkit is installed and CUDA_PATH is set.")
        print("Expected: CUDA_PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9")
    
    # Also add common CUDA locations to PATH as fallback
    fallback_paths = [
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\bin",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\bin",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.7\\bin",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1\\bin",
    ]
    
    current_path = os.environ.get('PATH', '')
    for fallback_path in fallback_paths:
        if os.path.exists(fallback_path) and fallback_path not in current_path:
            os.environ['PATH'] = fallback_path + os.pathsep + current_path
            print(f"✅ Added fallback CUDA path: {fallback_path}")
            current_path = os.environ['PATH']
            break
    
    print("=== End CUDA Path Setup ===\n")

# Call setup functions when this hook is loaded
# CRITICAL: Patch DeepSpeed BEFORE setting up paths to ensure the patch
# is in place before any deepspeed imports happen
patch_deepspeed_cuda_detection()
setup_cuda_path()