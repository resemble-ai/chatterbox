# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
MLX implementation of Conditional Flow Matching (CFM).
Port of PyTorch implementation from s3gen/flow_matching.py
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@dataclass
class CFMParamsMLX:
    """Configuration for CFM."""

    sigma_min: float = 1e-4
    t_scheduler: str = "cosine"
    training_cfg_rate: float = 0.0
    inference_cfg_rate: float = 0.7


class ConditionalCFMMLX(nn.Module):
    """Conditional Flow Matching decoder for MLX.

    Uses midpoint (2nd order) ODE solver for flow matching inference.
    """

    def __init__(
        self,
        in_channels: int,
        cfm_params: CFMParamsMLX,
        n_spks: int = 1,
        spk_emb_dim: int = 64,
        estimator: Optional[nn.Module] = None,
    ):
        """Initialize ConditionalCFMMLX.

        Args:
            in_channels: Number of mel channels.
            cfm_params: CFM configuration.
            n_spks: Number of speakers (for conditioning).
            spk_emb_dim: Speaker embedding dimension.
            estimator: Neural network estimator (U-Net decoder).
        """
        super().__init__()
        self.n_feats = in_channels
        self.sigma_min = cfm_params.sigma_min
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        self.estimator = estimator

    def __call__(
        self,
        mu: mx.array,
        mask: mx.array,
        n_timesteps: int,
        temperature: float = 1.0,
        spks: Optional[mx.array] = None,
        cond: Optional[mx.array] = None,
        prompt_len: int = 0,
        flow_cache: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """Forward diffusion (inference).

        Args:
            mu: Encoder output (batch, n_feats, mel_timesteps).
            mask: Output mask (batch, 1, mel_timesteps).
            n_timesteps: Number of ODE steps.
            temperature: Temperature for noise scaling.
            spks: Speaker embedding (batch, spk_emb_dim).
            cond: Additional conditioning.
            prompt_len: Length of prompt for caching.
            flow_cache: Cache for streaming (not fully implemented).

        Returns:
            Generated mel-spectrogram (batch, n_feats, mel_timesteps).
            Updated flow cache.
        """
        # Generate initial noise
        z = mx.random.normal(mu.shape) * temperature

        # Handle cache for streaming (basic implementation)
        cache_size = 0 if flow_cache is None else flow_cache.shape[2]
        if cache_size != 0:
            z = mx.concatenate([flow_cache[:, :, :, 0], z[:, :, cache_size:]], axis=2)
            mu = mx.concatenate([flow_cache[:, :, :, 1], mu[:, :, cache_size:]], axis=2)

        # Build new cache
        if prompt_len > 0:
            z_cache = mx.concatenate([z[:, :, :prompt_len], z[:, :, -34:]], axis=2)
            mu_cache = mx.concatenate([mu[:, :, :prompt_len], mu[:, :, -34:]], axis=2)
            new_flow_cache = mx.stack([z_cache, mu_cache], axis=-1)
        else:
            new_flow_cache = None

        # Time span
        t_span = mx.linspace(0, 1, n_timesteps + 1)
        if self.t_scheduler == "cosine":
            t_span = 1 - mx.cos(t_span * 0.5 * mx.array(3.14159265))

        # Solve ODE with midpoint method
        result = self.solve_midpoint(z, t_span, mu, mask, spks, cond)

        return result.astype(mx.float32), new_flow_cache

    def solve_midpoint(
        self,
        x: mx.array,
        t_span: mx.array,
        mu: mx.array,
        mask: mx.array,
        spks: Optional[mx.array],
        cond: Optional[mx.array],
    ) -> mx.array:
        """Midpoint (2nd order) ODE solver.

        Achieves same quality as Euler with fewer steps.

        Args:
            x: Initial noise.
            t_span: Time steps.
            mu: Encoder output.
            mask: Output mask.
            spks: Speaker embedding.
            cond: Additional conditioning.

        Returns:
            Solved trajectory at t=1.
        """
        t = t_span[0]
        dt = t_span[1] - t_span[0]

        # Pre-allocate CFG buffers (batch size 2 for conditional/unconditional)
        mel_len = x.shape[2]
        x_in = mx.zeros((2, 80, mel_len))
        mask_in = mx.zeros((2, 1, mel_len))
        mu_in = mx.zeros((2, 80, mel_len))
        t_in = mx.zeros((2,))
        spks_in = mx.zeros((2, 80)) if spks is not None else None
        cond_in = mx.zeros((2, 80, mel_len)) if cond is not None else None

        for step in range(1, len(t_span)):
            # Half step: evaluate derivative at current position
            x_in = mx.broadcast_to(x, (2,) + x.shape[1:])
            mask_in = mx.broadcast_to(mask, (2,) + mask.shape[1:])
            mu_in = mx.concatenate([mu, mx.zeros_like(mu)], axis=0)
            t_in = mx.broadcast_to(mx.expand_dims(t, axis=0), (2,))

            if spks is not None:
                spks_in = mx.concatenate([spks, mx.zeros_like(spks)], axis=0)
            if cond is not None:
                cond_in = mx.concatenate([cond, mx.zeros_like(cond)], axis=0)

            k1 = self.estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)

            # Apply CFG to k1
            k1_cond = k1[:1]
            k1_uncond = k1[1:]
            k1 = (
                1.0 + self.inference_cfg_rate
            ) * k1_cond - self.inference_cfg_rate * k1_uncond

            # Compute midpoint
            x_mid = x + (dt / 2) * k1
            t_mid = t + dt / 2

            # Full step: evaluate derivative at midpoint
            x_in = mx.broadcast_to(x_mid, (2,) + x_mid.shape[1:])
            t_in = mx.broadcast_to(mx.expand_dims(t_mid, axis=0), (2,))

            k2 = self.estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)

            # Apply CFG to k2
            k2_cond = k2[:1]
            k2_uncond = k2[1:]
            k2 = (
                1.0 + self.inference_cfg_rate
            ) * k2_cond - self.inference_cfg_rate * k2_uncond

            # Update x using midpoint derivative
            x = x + dt * k2
            t = t + dt

            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        # Evaluate only once at the end to allow command buffer fusion
        mx.eval(x)

        return x


class CausalConditionalCFMMLX(ConditionalCFMMLX):
    """Causal version of Conditional CFM for streaming."""

    def __init__(
        self,
        in_channels: int = 240,
        cfm_params: CFMParamsMLX = None,
        n_spks: int = 1,
        spk_emb_dim: int = 80,
        estimator: Optional[nn.Module] = None,
    ):
        if cfm_params is None:
            cfm_params = CFMParamsMLX()
        super().__init__(in_channels, cfm_params, n_spks, spk_emb_dim, estimator)

    def __call__(
        self,
        mu: mx.array,
        mask: mx.array,
        n_timesteps: int,
        temperature: float = 1.0,
        spks: Optional[mx.array] = None,
        cond: Optional[mx.array] = None,
    ) -> Tuple[mx.array, None]:
        """Forward diffusion for causal mode."""
        # Generate fresh random noise each time (do NOT cache - causes hiss)
        z = mx.random.normal(mu.shape) * temperature

        t_span = mx.linspace(0, 1, n_timesteps + 1)
        if self.t_scheduler == "cosine":
            t_span = 1 - mx.cos(t_span * 0.5 * mx.array(3.14159265))

        result = self.solve_midpoint(z, t_span, mu, mask, spks, cond)
        return result, None
