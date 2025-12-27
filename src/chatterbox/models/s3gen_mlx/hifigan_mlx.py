# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
MLX implementation of HiFiGAN vocoder (HiFTNet).
Port of PyTorch implementation from s3gen/hifigan.py

CRITICAL: This module MUST use FP32 for numerical stability.
The Snake activation is unstable in FP16.
"""

from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class SnakeMLX(nn.Module):
    """Snake activation function for MLX.

    Snake(x) = x + (1/alpha) * sin^2(alpha * x)

    MUST use FP32 - numerically unstable in FP16!
    """

    def __init__(
        self, in_features: int, alpha: float = 1.0, alpha_trainable: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        # Initialize alpha as ones
        self.alpha = mx.ones((in_features,)) * alpha
        self.no_div_by_zero = 1e-9

    def __call__(self, x: mx.array) -> mx.array:
        """Apply Snake activation (MUST be FP32).

        Args:
            x: Input tensor [B, T, C] (MLX channels-last format)
        """
        # Force FP32 for numerical stability
        x = x.astype(mx.float32)

        # Reshape alpha for broadcasting: (C,) -> (1, 1, C) for [B, T, C] format
        alpha = mx.expand_dims(mx.expand_dims(self.alpha, axis=0), axis=0)

        # Snake: x + (1/alpha) * sin^2(alpha * x)
        x = x + (1.0 / (alpha + self.no_div_by_zero)) * mx.power(mx.sin(x * alpha), 2)

        return x


class ResBlockMLX(nn.Module):
    """Residual block for HiFiGAN.

    Uses Snake activation (requires FP32).
    """

    def __init__(
        self,
        channels: int = 512,
        kernel_size: int = 3,
        dilations: List[int] = [1, 3, 5],
    ):
        super().__init__()

        self.num_layers = len(dilations)

        for i, dilation in enumerate(dilations):
            padding = (kernel_size * dilation - dilation) // 2
            conv1 = nn.Conv1d(
                channels, channels, kernel_size, padding=padding, dilation=dilation
            )
            conv2 = nn.Conv1d(
                channels, channels, kernel_size, padding=(kernel_size - 1) // 2
            )
            act1 = SnakeMLX(channels)
            act2 = SnakeMLX(channels)
            # Use explicit attribute names for MLX parameter tracking
            setattr(self, f"convs1_{i}", conv1)
            setattr(self, f"convs2_{i}", conv2)
            setattr(self, f"activations1_{i}", act1)
            setattr(self, f"activations2_{i}", act2)

    def __call__(self, x: mx.array) -> mx.array:
        for i in range(self.num_layers):
            act1 = getattr(self, f"activations1_{i}")
            conv1 = getattr(self, f"convs1_{i}")
            act2 = getattr(self, f"activations2_{i}")
            conv2 = getattr(self, f"convs2_{i}")
            xt = act1(x)
            xt = conv1(xt)
            xt = act2(xt)
            xt = conv2(xt)
            x = xt + x
        return x


class SineGenMLX(nn.Module):
    """Sine wave generator for source-filter model.

    Generates harmonic sine waves from F0.

    Note: Uses deterministic noise for consistent audio quality.
    """

    def __init__(
        self,
        samp_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0: mx.array) -> mx.array:
        """Generate UV (unvoiced) signal."""
        return (f0 > self.voiced_threshold).astype(mx.float32)

    def __call__(self, f0: mx.array) -> tuple:
        """Generate sine waves from F0.

        Args:
            f0: F0 tensor (batch, 1, sample_len) in Hz.

        Returns:
            sine_waves: Harmonic sine waves.
            uv: Unvoiced mask.
            noise: Noise component.
        """
        batch_size = f0.shape[0]

        # NOTE: Do NOT use fixed random seed here - it causes repeating noise
        # pattern that manifests as constant background "hiss" in the audio.

        # Compute frequency matrix for each harmonic using concatenation
        harmonics = []
        for i in range(self.harmonic_num + 1):
            harmonics.append(f0 * (i + 1) / self.sampling_rate)
        F_mat = mx.concatenate(harmonics, axis=1)  # [B, harmonic_num+1, T]

        # Cumulative phase
        theta_mat = 2 * np.pi * (mx.cumsum(F_mat, axis=-1) % 1)

        # Random initial phase (except fundamental)
        phase_vec = mx.random.uniform(
            low=-np.pi, high=np.pi, shape=(batch_size, self.harmonic_num + 1, 1)
        )
        # Set fundamental to 0 using concatenation
        phase_vec_fundamental = mx.zeros((batch_size, 1, 1))
        phase_vec = mx.concatenate([phase_vec_fundamental, phase_vec[:, 1:, :]], axis=1)

        # Generate sine waves
        sine_waves = self.sine_amp * mx.sin(theta_mat + phase_vec)

        # UV signal
        uv = self._f02uv(f0)

        # Noise for unvoiced regions
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * mx.random.normal(sine_waves.shape)

        # Apply UV mask
        sine_waves = sine_waves * uv + noise

        return sine_waves, uv, noise


class SourceModuleHnNSFMLX(nn.Module):
    """Neural Source Filter source module for MLX."""

    def __init__(
        self,
        sampling_rate: int,
        upsample_scale: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        self.l_sin_gen = SineGenMLX(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold
        )

        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def __call__(self, x: mx.array) -> tuple:
        """Generate source excitation.

        Args:
            x: F0 tensor (batch, time, 1).

        Returns:
            sine_merge: Merged sine source.
            noise: Noise source.
            uv: Unvoiced mask.
        """
        # Transpose for SineGen: (batch, time, 1) -> (batch, 1, time)
        sine_wavs, uv, _ = self.l_sin_gen(mx.transpose(x, (0, 2, 1)))

        # Transpose back: (batch, harmonics, time) -> (batch, time, harmonics)
        sine_wavs = mx.transpose(sine_wavs, (0, 2, 1))
        uv = mx.transpose(uv, (0, 2, 1))

        # Merge harmonics
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        # Noise source
        noise = mx.random.normal(uv.shape) * self.sine_amp / 3

        return sine_merge, noise, uv


class HiFTGeneratorMLX(nn.Module):
    """HiFTNet Generator for MLX.

    Neural Source Filter + ISTFT architecture.

    CRITICAL: Must use FP32 throughout due to Snake activation instability.
    """

    def __init__(
        self,
        in_channels: int = 80,
        base_channels: int = 512,
        nb_harmonics: int = 8,
        sampling_rate: int = 22050,
        nsf_alpha: float = 0.1,
        nsf_sigma: float = 0.003,
        nsf_voiced_threshold: float = 10,
        upsample_rates: List[int] = [8, 8],
        upsample_kernel_sizes: List[int] = [16, 16],
        istft_params: Dict[str, int] = {"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes: List[int] = [7, 11],
        source_resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5]],
        lrelu_slope: float = 0.1,
        audio_limit: float = 0.99,
        f0_predictor: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.out_channels = 1
        self.nb_harmonics = nb_harmonics
        self.sampling_rate = sampling_rate
        self.istft_params = istft_params
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        # Source module
        upsample_scale = int(np.prod(upsample_rates) * istft_params["hop_len"])
        self.m_source = SourceModuleHnNSFMLX(
            sampling_rate=sampling_rate,
            upsample_scale=upsample_scale,
            harmonic_num=nb_harmonics,
            sine_amp=nsf_alpha,
            add_noise_std=nsf_sigma,
            voiced_threshold=nsf_voiced_threshold,
        )

        # F0 upsampling will be handled via interpolation
        self.f0_upsample_scale = upsample_scale

        # Pre-conv
        self.conv_pre = nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3)

        # Upsampling layers - use explicit attributes
        self.num_ups = len(upsample_rates)
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            up = nn.ConvTranspose1d(
                base_channels // (2**i),
                base_channels // (2 ** (i + 1)),
                kernel_size=k,
                stride=u,
                padding=(k - u) // 2,
            )
            setattr(self, f"ups_{i}", up)

        # Source processing (downsampling path)
        downsample_rates = [1] + upsample_rates[::-1][:-1]
        downsample_cum_rates = np.cumprod(downsample_rates)

        self.num_source_blocks = len(source_resblock_kernel_sizes)
        for i, (u, k, d) in enumerate(
            zip(
                downsample_cum_rates[::-1],
                source_resblock_kernel_sizes,
                source_resblock_dilation_sizes,
            )
        ):
            in_ch = istft_params["n_fft"] + 2
            out_ch = base_channels // (2 ** (i + 1))

            if u == 1:
                down = nn.Conv1d(in_ch, out_ch, kernel_size=1)
            else:
                down = nn.Conv1d(
                    in_ch,
                    out_ch,
                    kernel_size=int(u * 2),
                    stride=int(u),
                    padding=int(u // 2),
                )

            setattr(self, f"source_downs_{i}", down)
            setattr(self, f"source_resblocks_{i}", ResBlockMLX(out_ch, k, d))

        # Main resblocks - use explicit attributes
        resblock_idx = 0
        for i in range(len(upsample_rates)):
            ch = base_channels // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                setattr(self, f"resblocks_{resblock_idx}", ResBlockMLX(ch, k, d))
                resblock_idx += 1
        self.num_resblocks = resblock_idx

        # Post-conv
        ch = base_channels // (2 ** len(upsample_rates))
        self.conv_post = nn.Conv1d(
            ch, istft_params["n_fft"] + 2, kernel_size=7, padding=3
        )

        # STFT window
        from scipy.signal import get_window

        window_np = get_window("hann", istft_params["n_fft"], fftbins=True).astype(
            np.float32
        )
        self.stft_window = mx.array(window_np)

        # F0 predictor (kept as placeholder - will use PyTorch version)
        self.f0_predictor = f0_predictor

    def _upsample_f0(self, f0: mx.array) -> mx.array:
        """Upsample F0 using nearest neighbor."""
        return mx.repeat(f0, self.f0_upsample_scale, axis=-1)

    def decode(self, x: mx.array, s: mx.array) -> mx.array:
        """Decode mel to audio.

        Args:
            x: Mel spectrogram (batch, time, mel_channels) - MLX NLC format.
            s: Source signal (batch, 1, sample_len) - NCL format.

        Returns:
            Audio waveform (batch, sample_len).
        """
        # Ensure FP32
        x = x.astype(mx.float32)
        s = s.astype(mx.float32)

        # x is in MLX format [B, T, C] - use directly for conv_pre

        # STFT on source
        # Note: MLX doesn't have native STFT, we'll need to use FFT
        s_squeezed = mx.squeeze(s, axis=1)
        s_stft_real, s_stft_imag = self._stft(s_squeezed)
        s_stft = mx.concatenate([s_stft_real, s_stft_imag], axis=1)  # [B, 2*freq, T]

        # Convert s_stft to MLX format [B, T, C]
        s_stft = s_stft.transpose(0, 2, 1)  # [B, C, T] -> [B, T, C]

        # Pre-conv (x is already in MLX [B, T, C] format)
        x = self.conv_pre(x)  # [B, T, C_out]

        # Note: Removed mx.eval() here - let MLX fuse operations

        # Upsample and fuse with source
        for i in range(self.num_ups):
            x = nn.leaky_relu(x, self.lrelu_slope)
            up = getattr(self, f"ups_{i}")
            x = up(x)  # ConvTranspose1d in MLX format

            if i == self.num_ups - 1:
                # Reflection padding on time dimension (axis 1)
                x = mx.pad(x, [(0, 0), (1, 0), (0, 0)], mode="edge")

            # Fuse with source
            source_down = getattr(self, f"source_downs_{i}")
            source_resblock = getattr(self, f"source_resblocks_{i}")
            si = source_down(s_stft)  # MLX format [B, T, C]
            si = source_resblock(si)
            x = x + si

            # ResBlocks
            xs = None
            for j in range(self.num_kernels):
                resblock = getattr(self, f"resblocks_{i * self.num_kernels + j}")
                if xs is None:
                    xs = resblock(x)
                else:
                    xs = xs + resblock(x)
            x = xs / self.num_kernels

            # Note: Removed mx.eval() here - let MLX fuse across upsample blocks

        x = nn.leaky_relu(x, self.lrelu_slope)
        x = self.conv_post(x)  # [B, T, n_fft+2]

        # Convert to channel-first for magnitude/phase split
        x = x.transpose(0, 2, 1)  # [B, T, C] -> [B, C, T]

        # Magnitude and phase
        n_fft_half = self.istft_params["n_fft"] // 2 + 1
        magnitude = mx.exp(x[:, :n_fft_half, :])
        phase = mx.sin(x[:, n_fft_half:, :])

        # Note: Removed mx.eval() here - let ISTFT fuse with magnitude/phase

        # ISTFT
        result = self._istft(magnitude, phase)
        result = mx.clip(result, -self.audio_limit, self.audio_limit)

        # Single mx.eval() at the very end - allows MLX to fuse all operations
        mx.eval(result)

        return result

    def _reflect_pad_1d(self, x: mx.array, pad_left: int, pad_right: int) -> mx.array:
        """Apply reflect padding to 1D signal.

        Args:
            x: Input tensor [batch, time]
            pad_left: Left padding amount
            pad_right: Right padding amount

        Returns:
            Padded tensor [batch, time + pad_left + pad_right]
        """
        if pad_left > 0:
            # Reflect left: indices pad_left, pad_left-1, ..., 1
            left_pad = x[:, pad_left:0:-1]
            x = mx.concatenate([left_pad, x], axis=1)
        if pad_right > 0:
            # Reflect right: indices -2, -3, ..., -(pad_right+1)
            right_pad = x[:, -2 : -(pad_right + 2) : -1]
            x = mx.concatenate([x, right_pad], axis=1)
        return x

    def _stft(self, x: mx.array) -> tuple:
        """Compute STFT with reflect padding and windowing.

        Args:
            x: Input signal [batch, time]

        Returns:
            real: Real part of STFT [batch, freq_bins, n_frames]
            imag: Imaginary part of STFT [batch, freq_bins, n_frames]
        """
        n_fft = self.istft_params["n_fft"]
        hop_len = self.istft_params["hop_len"]
        window = self.stft_window

        if x.ndim == 1:
            x = x[None, :]
        batch_size, signal_len = x.shape

        # Center padding (like PyTorch center=True)
        pad_len = n_fft // 2
        x = self._reflect_pad_1d(x, pad_len, pad_len)
        signal_len = x.shape[1]

        # Compute number of frames
        n_frames = (signal_len - n_fft) // hop_len + 1

        # Extract frames using indexing
        frame_starts = mx.arange(0, n_frames * hop_len, hop_len)
        frame_indices = frame_starts[:, None] + mx.arange(n_fft)[None, :]

        # Index and reshape: [batch, n_frames, n_fft]
        frames = x[:, frame_indices.flatten()].reshape(batch_size, n_frames, n_fft)

        # Apply window
        windowed = frames * window[None, None, :]

        # FFT
        spectrum = mx.fft.rfft(windowed, axis=-1)  # [batch, n_frames, freq_bins]

        # Transpose to [batch, freq_bins, n_frames]
        spectrum = spectrum.transpose(0, 2, 1)

        return mx.real(spectrum), mx.imag(spectrum)

    def _istft(self, magnitude: mx.array, phase: mx.array) -> mx.array:
        """Compute inverse STFT with fully vectorized overlap-add.

        OPTIMIZED: Uses vectorized scatter-add instead of frame-by-frame loop.
        This allows MLX to fuse operations and prevents graph explosion.

        Args:
            magnitude: Magnitude spectrogram [batch, freq_bins, n_frames]
            phase: Phase spectrogram [batch, freq_bins, n_frames]

        Returns:
            Reconstructed signal [batch, time]
        """
        # Clip magnitude to prevent numerical issues (match PyTorch: only clip max)
        # MLX requires both a_min and a_max, so use None for no lower bound
        magnitude = mx.clip(magnitude, a_min=None, a_max=1e2)
        n_fft = self.istft_params["n_fft"]
        hop_len = self.istft_params["hop_len"]
        window = self.stft_window

        # Build complex spectrum from magnitude and phase
        # Use float32 for better precision in trig operations
        magnitude = magnitude.astype(mx.float32)
        phase = phase.astype(mx.float32)
        real = magnitude * mx.cos(phase)
        imag = magnitude * mx.sin(phase)
        spectrum = real + 1j * imag  # [batch, freq_bins, n_frames]

        # Transpose to [batch, n_frames, freq_bins]
        spectrum = spectrum.transpose(0, 2, 1)
        batch_size, n_frames, freq_bins = spectrum.shape

        # IRFFT to get time-domain frames
        frames = mx.fft.irfft(spectrum, n=n_fft, axis=-1)  # [batch, n_frames, n_fft]

        # Apply window to all frames at once (vectorized)
        windowed_frames = frames * window[None, None, :]  # [batch, n_frames, n_fft]

        # Compute output length
        output_len = (n_frames - 1) * hop_len + n_fft

        # === Vectorized overlap-add using scatter indices ===
        # Create index tensor for scatter-add: shape [n_frames, n_fft]
        frame_starts = mx.arange(n_frames) * hop_len  # [n_frames]
        sample_offsets = mx.arange(n_fft)  # [n_fft]
        indices = frame_starts[:, None] + sample_offsets[None, :]  # [n_frames, n_fft]

        # Flatten for scatter
        flat_indices = indices.flatten()  # [n_frames * n_fft]

        # Process batch dimension - typically batch_size=1 for inference
        # Stack results for each batch element
        outputs = []
        for b in range(batch_size):
            flat_values = windowed_frames[b].flatten()  # [n_frames * n_fft]
            # Use at-indexing with accumulated addition
            output_b = mx.zeros((output_len,))
            output_b = output_b.at[flat_indices].add(flat_values)
            outputs.append(output_b)

        # Stack batch results
        output = mx.stack(outputs, axis=0)  # [batch, output_len]

        # Compute window normalization (same for all batches)
        # Note: window applied in both STFT analysis and ISTFT synthesis,
        # so we normalize by window**2 for proper reconstruction
        window_sq = window**2
        window_norm = mx.zeros((output_len,))
        window_sq_repeated = mx.broadcast_to(
            window_sq[None, :], (n_frames, n_fft)
        ).flatten()
        window_norm = window_norm.at[flat_indices].add(window_sq_repeated)

        # Normalize by window sum
        # Use small epsilon to avoid division by zero but don't add noise floor
        window_norm = mx.maximum(window_norm, 1e-8)
        output = output / window_norm[None, :]

        # Remove center padding
        pad_len = n_fft // 2
        output = output[:, pad_len:-pad_len]

        return output

    def inference(
        self, speech_feat: mx.array, cache_source: Optional[mx.array] = None
    ) -> tuple:
        """Run inference.

        Args:
            speech_feat: Mel spectrogram (batch, mel_channels, time) - PyTorch NCL format.
            cache_source: Cached source for streaming.

        Returns:
            Audio waveform and source signal.
        """
        batch_size = speech_feat.shape[0]
        mel_len = speech_feat.shape[-1]

        # Predict F0 from mel spectrogram using the neural F0 predictor
        if self.f0_predictor is not None:
            # f0_predictor expects [B, C, T] and returns [B, T]
            f0 = self.f0_predictor(speech_feat)  # [B, T]
            f0 = mx.expand_dims(f0, axis=1)  # [B, 1, T]
        else:
            # Fallback to dummy F0 if predictor not available
            f0 = mx.ones((batch_size, 1, mel_len)) * 200  # 200 Hz placeholder

        # Upsample F0 using nearest neighbor interpolation
        f0_upsampled = self._upsample_f0(f0)  # [B, 1, T*upsample_scale]
        f0_upsampled = mx.transpose(f0_upsampled, (0, 2, 1))  # (batch, time, 1)

        # Generate source
        s, _, _ = self.m_source(f0_upsampled)
        s = mx.transpose(s, (0, 2, 1))  # (batch, 1, time)

        # Use cache if provided
        if cache_source is not None and cache_source.shape[2] > 0:
            s = mx.concatenate([cache_source, s[:, :, cache_source.shape[2] :]], axis=2)

        # Convert speech_feat from NCL to NLC format for decode
        speech_feat_nlc = mx.transpose(speech_feat, (0, 2, 1))  # [B, C, T] -> [B, T, C]

        # Decode
        generated_speech = self.decode(speech_feat_nlc, s)

        return generated_speech, s
