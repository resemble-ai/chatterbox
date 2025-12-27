# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
MLX Transformer modules for S3Gen.
"""

from .attention_mlx import MultiHeadedAttentionMLX, RelPositionMultiHeadedAttentionMLX
from .convolution_mlx import ConvolutionModuleMLX, glu
from .encoder_layer_mlx import TransformerEncoderLayerMLX, ConformerEncoderLayerMLX
from .feed_forward_mlx import PositionwiseFeedForwardMLX
from .embedding_mlx import (
    PositionalEncodingMLX,
    RelPositionalEncodingMLX,
    EspnetRelPositionalEncodingMLX,
    NoPositionalEncodingMLX,
)
from .subsampling_mlx import (
    LinearNoSubsamplingMLX,
    EmbeddingNoSubsamplingMLX,
    Conv1dSubsampling2MLX,
)
from .upsample_encoder_mlx import UpsampleConformerEncoderMLX, Upsample1DMLX

__all__ = [
    # Attention
    "MultiHeadedAttentionMLX",
    "RelPositionMultiHeadedAttentionMLX",
    # Convolution
    "ConvolutionModuleMLX",
    "glu",
    # Encoder layers
    "TransformerEncoderLayerMLX",
    "ConformerEncoderLayerMLX",
    # Feed-forward
    "PositionwiseFeedForwardMLX",
    # Embeddings
    "PositionalEncodingMLX",
    "RelPositionalEncodingMLX",
    "EspnetRelPositionalEncodingMLX",
    "NoPositionalEncodingMLX",
    # Subsampling
    "LinearNoSubsamplingMLX",
    "EmbeddingNoSubsamplingMLX",
    "Conv1dSubsampling2MLX",
    # Encoder
    "UpsampleConformerEncoderMLX",
    "Upsample1DMLX",
]
