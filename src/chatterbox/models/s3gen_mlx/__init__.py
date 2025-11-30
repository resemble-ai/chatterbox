# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
MLX implementation of S3Gen (Semantic to Mel-Spectrogram Generator).
Port of PyTorch S3Gen to MLX for Apple Silicon optimization.

This package provides a complete MLX port of the S3Gen speech synthesis pipeline,
including:
- Conformer encoder with upsampling
- Conditional Flow Matching (CFM) decoder
- HiFiGAN vocoder (HiFTNet)
- Speaker encoder (CAMPPlus)
- F0 predictor
"""

# Main model classes
from .s3gen_mlx import (
    S3Token2MelMLX,
    S3Token2WavMLX,
    load_s3gen_mlx,
    convert_s3gen_weights,
)

# Flow module
from .flow_mlx import (
    MaskedDiffWithXvecMLX,
    CausalMaskedDiffWithXvecMLX,
)

# Flow matching
from .flow_matching_mlx import (
    ConditionalCFMMLX,
    CausalConditionalCFMMLX,
)

# Decoder
from .decoder_mlx import ConditionalDecoderMLX

# Vocoder
from .hifigan_mlx import HiFTGeneratorMLX

# Speaker encoder
from .xvector_mlx import CAMPPlusMLX

# F0 predictor
from .f0_predictor_mlx import ConvRNNF0PredictorMLX

# Encoder
from .transformer.upsample_encoder_mlx import UpsampleConformerEncoderMLX

__all__ = [
    # Main models
    "S3Token2MelMLX",
    "S3Token2WavMLX",
    "load_s3gen_mlx",
    "convert_s3gen_weights",
    # Flow
    "MaskedDiffWithXvecMLX",
    "CausalMaskedDiffWithXvecMLX",
    # Flow matching
    "ConditionalCFMMLX",
    "CausalConditionalCFMMLX",
    # Decoder
    "ConditionalDecoderMLX",
    # Vocoder
    "HiFTGeneratorMLX",
    # Speaker encoder
    "CAMPPlusMLX",
    # F0 predictor
    "ConvRNNF0PredictorMLX",
    # Encoder
    "UpsampleConformerEncoderMLX",
]
