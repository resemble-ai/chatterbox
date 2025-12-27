# Modified from CosyVoice https://github.com/FunAudioLLM/CosyVoice
# MLX port for Apple Silicon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
S3Gen MLX Implementation

Main speech synthesis pipeline that converts S3 speech tokens to audio.
Combines token-to-mel (CFM) and mel-to-waveform (HiFiGAN) stages.
"""

import logging
import os
from typing import Optional, Dict, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Import MLX components
from .flow_mlx import CausalMaskedDiffWithXvecMLX
from .xvector_mlx import CAMPPlusMLX
from .f0_predictor_mlx import ConvRNNF0PredictorMLX
from .hifigan_mlx import HiFTGeneratorMLX
from .transformer.upsample_encoder_mlx import UpsampleConformerEncoderMLX
from .flow_matching_mlx import CausalConditionalCFMMLX
from .decoder_mlx import ConditionalDecoderMLX
from ..utils import get_memory_info, is_debug

logger = logging.getLogger(__name__)

# Constants from original implementation
S3GEN_SR = 24000  # 24kHz output sample rate
S3_SR = 16000  # 16kHz for S3 tokenizer


def _log_s3gen_memory(label: str):
    """Log memory for S3Gen MLX debugging. Enabled via DEBUG_MEMORY=1."""
    if not is_debug() and os.environ.get("DEBUG_MEMORY", "0") != "1":
        return
    info = get_memory_info()
    parts = [f"[S3GEN_MLX] {label}:", f"Sys={info['sys_used_gb']:.2f}GB"]
    if "mps_allocated_mb" in info:
        parts.append(f"MPS={info['mps_allocated_mb']:.0f}MB")
    mx.eval(mx.array([0]))  # Force MLX sync
    print(" | ".join(parts))


class CFMParams:
    """CFM parameters configuration."""

    sigma_min: float = 1e-06
    solver: str = "euler"
    t_scheduler: str = "cosine"
    training_cfg_rate: float = 0.2
    inference_cfg_rate: float = 0.7
    reg_loss_type: str = "l1"


class S3Token2MelMLX(nn.Module):
    """
    S3 Token to Mel-spectrogram converter using Conditional Flow Matching.

    This module:
    1. Takes S3 speech tokens as input
    2. Extracts speaker embedding from reference audio
    3. Uses CFM decoder to generate mel-spectrograms
    """

    def __init__(
        self,
        # Encoder config
        encoder_output_size: int = 512,
        encoder_attention_heads: int = 8,
        encoder_linear_units: int = 2048,
        encoder_num_blocks: int = 6,
        encoder_dropout_rate: float = 0.1,
        # Decoder config
        decoder_in_channels: int = 320,
        decoder_out_channels: int = 80,
        decoder_channels: list = [256],
        decoder_num_heads: int = 8,
        decoder_num_mid_blocks: int = 12,
        decoder_n_blocks: int = 4,
        # Flow config
        vocab_size: int = 6561,
        input_frame_rate: int = 25,
        spk_embed_dim: int = 192,
        output_size: int = 80,
    ):
        super().__init__()

        logger.info("[MLX] Initializing S3Token2MelMLX")

        # Speaker encoder (CAMPPlus)
        self.speaker_encoder = CAMPPlusMLX(feat_dim=80, embedding_size=spk_embed_dim)

        # Upsample Conformer Encoder (created locally, stored only in flow)
        encoder = UpsampleConformerEncoderMLX(
            output_size=encoder_output_size,
            attention_heads=encoder_attention_heads,
            linear_units=encoder_linear_units,
            num_blocks=encoder_num_blocks,
            dropout_rate=encoder_dropout_rate,
            positional_dropout_rate=encoder_dropout_rate,
            attention_dropout_rate=encoder_dropout_rate,
            normalize_before=True,
            input_layer="linear",
            pos_enc_layer_type="rel_pos_espnet",
            selfattention_layer_type="rel_selfattn",
            input_size=encoder_output_size,
            use_cnn_module=False,
            macaron_style=False,
        )

        # Conditional Decoder (U-Net style)
        estimator = ConditionalDecoderMLX(
            in_channels=decoder_in_channels,
            out_channels=decoder_out_channels,
            causal=True,
            channels=decoder_channels,
            dropout=0.0,
            attention_head_dim=64,
            n_blocks=decoder_n_blocks,
            num_mid_blocks=decoder_num_mid_blocks,
            num_heads=decoder_num_heads,
            act_fn="gelu",
        )

        # CFM decoder (created locally, stored only in flow)
        cfm_params = CFMParams()
        decoder = CausalConditionalCFMMLX(
            spk_emb_dim=output_size,
            cfm_params=cfm_params,
            estimator=estimator,
        )

        # Flow module (contains encoder and decoder)
        self.flow = CausalMaskedDiffWithXvecMLX(
            input_size=encoder_output_size,
            output_size=output_size,
            spk_embed_dim=spk_embed_dim,
            vocab_size=vocab_size,
            input_frame_rate=input_frame_rate,
            encoder=encoder,
            decoder=decoder,
        )

        logger.info("[MLX] S3Token2MelMLX initialization complete")

    def embed_ref(
        self,
        ref_wav: mx.array,
        ref_sr: int,
        ref_mels: Optional[mx.array] = None,
        ref_tokens: Optional[mx.array] = None,
        ref_token_lens: Optional[mx.array] = None,
    ) -> Dict[str, mx.array]:
        """
        Compute reference embedding from audio.

        Note: For full functionality, audio resampling and mel extraction
        should be done in PyTorch/NumPy and passed to this function.

        Args:
            ref_wav: Reference waveform (for speaker embedding)
            ref_sr: Sample rate
            ref_mels: Pre-computed mel spectrogram [B, T, 80]
            ref_tokens: Pre-computed speech tokens [B, T]
            ref_token_lens: Token lengths

        Returns:
            Dictionary with prompt_token, prompt_feat, embedding
        """
        logger.info("[MLX] Computing reference embedding")

        # Compute speaker embedding
        # Note: The original uses Kaldi fbank features - we assume pre-computed
        # For inference, ref_wav should be fbank features [B, T, 80]
        ref_x_vector = self.speaker_encoder(ref_wav)

        return dict(
            prompt_token=ref_tokens,
            prompt_token_len=(
                int(ref_token_lens[0])
                if ref_token_lens is not None
                else (ref_tokens.shape[1] if ref_tokens is not None else 0)
            ),
            prompt_feat=ref_mels,
            prompt_feat_len=ref_mels.shape[1] if ref_mels is not None else None,
            embedding=ref_x_vector,
        )

    def __call__(
        self,
        speech_tokens: mx.array,
        ref_dict: Dict[str, mx.array],
        finalize: bool = False,
    ) -> mx.array:
        """
        Generate mel spectrogram from speech tokens.

        Args:
            speech_tokens: S3 speech tokens [B, T] or [T]
            ref_dict: Reference embedding dictionary
            finalize: Whether this is the final chunk

        Returns:
            Generated mel spectrogram [B, C, T]
        """
        logger.debug("[MLX] Running S3Token2Mel forward")

        if len(speech_tokens.shape) == 1:
            speech_tokens = mx.expand_dims(speech_tokens, axis=0)

        speech_token_lens = mx.array([speech_tokens.shape[1]])

        # Run flow inference
        output_mels, _ = self.flow(
            token=speech_tokens,
            token_len=speech_token_lens,
            prompt_token=ref_dict["prompt_token"],
            prompt_token_len=ref_dict["prompt_token_len"],
            prompt_feat=ref_dict["prompt_feat"],
            prompt_feat_len=ref_dict.get("prompt_feat_len"),
            embedding=ref_dict["embedding"],
            finalize=finalize,
        )

        return output_mels


class S3Token2WavMLX(S3Token2MelMLX):
    """
    S3 Token to Waveform converter.

    Extends S3Token2MelMLX with HiFiGAN vocoder for waveform synthesis.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        logger.info("[MLX] Initializing HiFiGAN vocoder")

        # F0 predictor
        f0_predictor = ConvRNNF0PredictorMLX()

        # HiFiGAN vocoder
        self.mel2wav = HiFTGeneratorMLX(
            sampling_rate=S3GEN_SR,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            f0_predictor=f0_predictor,
        )

        # Trim fade buffer for artifact reduction
        n_trim = S3GEN_SR // 50  # 20ms
        # Cosine fade-in: first n_trim samples are zeros, then fade-in
        zeros_part = mx.zeros(n_trim)
        fade_in = (mx.cos(mx.linspace(mx.array(np.pi), mx.array(0.0), n_trim)) + 1) / 2
        trim_fade = mx.concatenate([zeros_part, fade_in])
        self.trim_fade = trim_fade

        logger.info("[MLX] S3Token2WavMLX initialization complete")

    def __call__(
        self,
        speech_tokens: mx.array,
        ref_dict: Dict[str, mx.array],
        finalize: bool = False,
    ) -> mx.array:
        """
        Generate waveform from speech tokens.

        Args:
            speech_tokens: S3 speech tokens [B, T]
            ref_dict: Reference embedding dictionary
            finalize: Whether this is the final chunk

        Returns:
            Generated waveform [B, T_audio]
        """
        # Get mel spectrogram
        output_mels = super().__call__(speech_tokens, ref_dict, finalize)

        # Convert to waveform
        output_wavs, _ = self.mel2wav.inference(
            speech_feat=output_mels, cache_source=mx.zeros((1, 1, 0))
        )

        # Apply trim fade
        fade_len = self.trim_fade.shape[0]
        if output_wavs.shape[-1] >= fade_len:
            output_wavs = output_wavs.at[:, :fade_len].multiply(self.trim_fade)

        return output_wavs

    def flow_inference(
        self,
        speech_tokens: mx.array,
        ref_dict: Dict[str, mx.array],
        finalize: bool = False,
    ) -> mx.array:
        """
        Run only the flow (token-to-mel) inference.

        Args:
            speech_tokens: S3 speech tokens
            ref_dict: Reference embedding dictionary
            finalize: Whether this is the final chunk

        Returns:
            Generated mel spectrogram
        """
        result = S3Token2MelMLX.__call__(self, speech_tokens, ref_dict, finalize)
        # Force evaluation to prevent lazy computation graph buildup
        mx.eval(result)
        return result

    def hift_inference(
        self, speech_feat: mx.array, cache_source: Optional[mx.array] = None
    ) -> Tuple[mx.array, mx.array]:
        """
        Run only the HiFiGAN (mel-to-waveform) inference.

        Args:
            speech_feat: Mel spectrogram [B, C, T]
            cache_source: Optional cache for streaming

        Returns:
            Generated waveform, source signal
        """
        if cache_source is None:
            cache_source = mx.zeros((1, 1, 0))

        _log_s3gen_memory("hift_inference_start")
        result = self.mel2wav.inference(
            speech_feat=speech_feat, cache_source=cache_source
        )
        # Force evaluation to prevent lazy computation graph buildup
        mx.eval(result[0])
        _log_s3gen_memory("hift_inference_end")
        return result

    def inference(
        self,
        speech_tokens: mx.array,
        ref_dict: Dict[str, mx.array],
        finalize: bool = True,
    ) -> Tuple[mx.array, mx.array]:
        """
        Full inference pipeline.

        Args:
            speech_tokens: S3 speech tokens
            ref_dict: Reference embedding dictionary
            finalize: Whether this is the final chunk

        Returns:
            Generated waveform, source signal
        """
        logger.debug("[MLX] Running full inference pipeline")
        _log_s3gen_memory("s3gen_inference_start")

        # Flow inference (token -> mel)
        _log_s3gen_memory("before_flow_inference")
        output_mels = self.flow_inference(speech_tokens, ref_dict, finalize)
        _log_s3gen_memory("after_flow_inference")

        # HiFiGAN inference (mel -> waveform)
        output_wavs, output_sources = self.hift_inference(output_mels)

        # Apply trim fade
        fade_len = self.trim_fade.shape[0]
        if output_wavs.shape[-1] >= fade_len:
            output_wavs = output_wavs.at[:, :fade_len].multiply(self.trim_fade)

        _log_s3gen_memory("s3gen_inference_end")
        return output_wavs, output_sources


def convert_s3gen_weights(pytorch_state_dict: dict) -> dict:
    """
    Convert PyTorch S3Gen weights to MLX format.

    This is the main weight conversion function that handles
    the entire S3Gen model hierarchy. It maps PyTorch weight names
    to MLX weight names and converts tensor formats appropriately.

    Args:
        pytorch_state_dict: Full PyTorch state dict

    Returns:
        Dictionary with MLX-compatible weights
    """
    import numpy as np
    import torch

    mlx_weights = {}

    def to_numpy(v):
        """Convert a tensor to numpy array."""
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().numpy()
        elif hasattr(v, "numpy"):
            return v.numpy()
        return v

    def convert_conv1d_weight(weight):
        """Convert Conv1d: PyTorch [out, in, k] -> MLX [out, k, in]"""
        return np.transpose(weight, (0, 2, 1))

    def convert_conv2d_weight(weight):
        """Convert Conv2d: PyTorch [out, in, h, w] -> MLX [out, h, w, in]"""
        return np.transpose(weight, (0, 2, 3, 1))

    # Key mapping patterns for MLX model structure
    # The MLX model uses different naming conventions

    for key, value in pytorch_state_dict.items():
        np_value = to_numpy(value)

        # Skip buffers and non-weight parameters
        if "num_batches_tracked" in key:
            continue
        if "running_mean" in key or "running_var" in key:
            # These are batch norm running statistics - keep them
            pass

        # Determine if this is a weight that needs format conversion
        new_key = key
        new_value = np_value

        # Handle Conv1d weights (3D tensors)
        if len(np_value.shape) == 3:
            # Check if it's a weight (not bias)
            if "weight" in key and "weight_g" not in key and "weight_v" not in key:
                new_value = convert_conv1d_weight(np_value)

        # Handle Conv2d weights (4D tensors)
        elif len(np_value.shape) == 4:
            if "weight" in key:
                new_value = convert_conv2d_weight(np_value)

        # Handle weight_norm by combining weight_g and weight_v
        if "weight_v" in key:
            base_key = key.replace(".weight_v", "")
            g_key = key.replace(".weight_v", ".weight_g")
            if g_key in pytorch_state_dict:
                weight_g = to_numpy(pytorch_state_dict[g_key])
                weight_v = np_value

                # Compute normalized weight: w = g * (v / ||v||)
                if len(weight_v.shape) == 3:  # Conv1d
                    norm = np.linalg.norm(
                        weight_v.reshape(weight_v.shape[0], -1), axis=1, keepdims=True
                    )
                    norm = norm.reshape(weight_v.shape[0], 1, 1)
                else:
                    norm = np.linalg.norm(
                        weight_v.reshape(weight_v.shape[0], -1), axis=1, keepdims=True
                    )

                combined = weight_g.reshape(-1, 1, 1) * (weight_v / (norm + 1e-12))

                # Convert to MLX format
                if len(combined.shape) == 3:
                    combined = convert_conv1d_weight(combined)

                new_key = base_key + ".weight"
                new_value = combined
            else:
                continue  # Skip weight_v if no corresponding weight_g
        elif "weight_g" in key:
            continue  # Skip weight_g, already handled with weight_v

        mlx_weights[new_key] = mx.array(new_value)

    logger.info(f"[MLX] Converted {len(mlx_weights)} weights")
    return mlx_weights


def load_s3gen_mlx(
    pytorch_checkpoint_path: str, model_class: str = "S3Token2WavMLX"
) -> nn.Module:
    """
    Load S3Gen MLX model from PyTorch checkpoint.

    Args:
        pytorch_checkpoint_path: Path to PyTorch checkpoint
        model_class: Which model class to instantiate

    Returns:
        Loaded MLX model
    """
    import torch

    logger.info(f"[MLX] Loading checkpoint from {pytorch_checkpoint_path}")

    # Load PyTorch weights
    checkpoint = torch.load(pytorch_checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Create model
    if model_class == "S3Token2WavMLX":
        model = S3Token2WavMLX()
    else:
        model = S3Token2MelMLX()

    # Convert and load weights
    mlx_weights = convert_s3gen_weights(state_dict)
    model.load_weights(list(mlx_weights.items()))

    logger.info("[MLX] Model loaded successfully")

    return model
