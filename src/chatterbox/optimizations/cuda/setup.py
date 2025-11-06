"""
Setup script for building custom CUDA kernels

Usage:
    python setup.py install
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Check CUDA availability
import torch
if not torch.cuda.is_available():
    print("WARNING: CUDA not available, custom kernels will not be built")
    exit(0)

# CUDA compute capability (adjust based on your GPU)
# RTX 40xx: 8.9, RTX 30xx: 8.6, A100: 8.0
cuda_arch = os.environ.get('TORCH_CUDA_ARCH_LIST', '8.0;8.6;8.9')

setup(
    name='chatterbox_cuda_kernels',
    ext_modules=[
        CUDAExtension(
            'chatterbox_cuda_kernels',
            sources=[
                'fused_sampling.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-gencode', f'arch=compute_80,code=sm_80',  # A100
                    '-gencode', f'arch=compute_86,code=sm_86',  # RTX 30xx
                    '-gencode', f'arch=compute_89,code=sm_89',  # RTX 40xx
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                ]
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
