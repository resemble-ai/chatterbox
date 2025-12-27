# Modified from CosyVoice https://github.com/FunAudioLLM/CosyVoice
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

import logging
import numpy as np
import torch
import torchaudio as ta
from functools import lru_cache
from typing import Optional
import psutil
import os
import gc
from ..s3tokenizer import S3_SR, SPEECH_VOCAB_SIZE, S3Tokenizer
from .const import S3GEN_SR
from .flow import CausalMaskedDiffWithXvec
from .xvector import CAMPPlus
from .utils.mel import mel_spectrogram
from .f0_predictor import ConvRNNF0Predictor
from .hifigan import HiFTGenerator
from .transformer.upsample_encoder import UpsampleConformerEncoder
from .flow_matching import CausalConditionalCFM
from .decoder import ConditionalDecoder
from .configs import CFM_PARAMS
from chatterbox.models.utils import is_debug, contiguous_transpose


def log_message(msg, level="info"):
    """Log a message only when `DEBUG_LOGGING` is enabled.

    This reduces stdout I/O during production inference. When disabled,
    this function is a no-op.
    """
    if level == "error":
        logging.error(msg)
    elif not is_debug():
        return
    if level == "debug":
        logging.debug(msg)
    elif level == "warning":
        logging.warning(msg)
    else:
        logging.info(msg)


def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def get_gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def log_memory(stage, extra_info=""):
    cpu_mem = get_memory_mb()
    gpu_mem = get_gpu_memory_mb()
    log_message(
        f"[MEMORY] {stage}: CPU={cpu_mem:.1f}MB, GPU={gpu_mem:.1f}MB {extra_info}",
        "debug",
    )


def log_tensor_info(tensor, name):
    if torch.is_tensor(tensor):
        log_message(
            f"[TENSOR] {name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, mem={tensor.numel() * tensor.element_size() / 1024 / 1024:.2f}MB",
            "debug",
        )
    else:
        log_message(f"[TENSOR] {name}: not a tensor, type={type(tensor)}", "debug")


def drop_invalid_tokens(x):
    assert (
        len(x.shape) <= 2 and x.shape[0] == 1
    ), "only batch size of one allowed for now"
    return x[x < SPEECH_VOCAB_SIZE]


# TODO: global resampler cache
@lru_cache(100)
def get_resampler(src_sr, dst_sr, device):
    log_message(f"[CACHE] Creating resampler {src_sr}->{dst_sr} on {device}", "debug")
    log_memory("before_resampler_creation")
    resampler = ta.transforms.Resample(src_sr, dst_sr).to(device)
    log_memory("after_resampler_creation")
    return resampler


class S3Token2Mel(torch.nn.Module):
    """
    CosyVoice2's CFM decoder maps S3 speech tokens to mel-spectrograms.
    TODO: make these modules configurable?
    """

    def __init__(self):
        log_message("[INIT] Starting S3Token2Mel initialization")
        log_memory("start_s3token2mel_init")

        super().__init__()

        log_message("[INIT] Creating tokenizer...")
        log_memory("before_tokenizer")
        self.tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz")
        log_memory("after_tokenizer")

        log_message("[INIT] Setting up mel extractor...")
        self.mel_extractor = mel_spectrogram  # TODO: make it a torch module?

        log_message("[INIT] Creating speaker encoder...")
        log_memory("before_speaker_encoder")
        self.speaker_encoder = CAMPPlus()  # use default args
        log_memory("after_speaker_encoder")

        log_message("[INIT] Creating encoder...")
        log_memory("before_encoder")
        encoder = UpsampleConformerEncoder(
            output_size=512,
            attention_heads=8,
            linear_units=2048,
            num_blocks=6,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            normalize_before=True,
            input_layer="linear",
            pos_enc_layer_type="rel_pos_espnet",
            selfattention_layer_type="rel_selfattn",
            input_size=512,
            use_cnn_module=False,
            macaron_style=False,
        )
        log_memory("after_encoder")

        log_message("[INIT] Creating estimator...")
        log_memory("before_estimator")
        estimator = ConditionalDecoder(
            in_channels=320,
            out_channels=80,
            causal=True,
            channels=[256],
            dropout=0.0,
            attention_head_dim=64,
            n_blocks=4,
            num_mid_blocks=12,
            num_heads=8,
            act_fn="gelu",
        )
        log_memory("after_estimator")

        log_message("[INIT] Creating CFM decoder...")
        log_memory("before_cfm_decoder")
        cfm_params = CFM_PARAMS
        decoder = CausalConditionalCFM(
            spk_emb_dim=80,
            cfm_params=cfm_params,
            estimator=estimator,
        )
        log_memory("after_cfm_decoder")

        log_message("[INIT] Creating flow...")
        log_memory("before_flow")
        self.flow = CausalMaskedDiffWithXvec(encoder=encoder, decoder=decoder)
        log_memory("after_flow")

        self.resamplers = {}
        log_memory("end_s3token2mel_init")
        log_message("[INIT] S3Token2Mel initialization complete")

    @property
    def device(self):
        params = self.tokenizer.parameters()
        return next(params).device

    def embed_ref(
        self,
        ref_wav: torch.Tensor,
        ref_sr: int,
        device="auto",
        ref_fade_out=True,
    ):
        log_message(f"[EMBED_REF] Starting reference embedding, ref_sr={ref_sr}")
        log_memory("start_embed_ref")
        log_tensor_info(ref_wav, "input_ref_wav")

        device = self.device if device == "auto" else device

        if isinstance(ref_wav, np.ndarray):
            log_message("[EMBED_REF] Converting numpy array to tensor")
            ref_wav = torch.from_numpy(ref_wav).float()
            log_tensor_info(ref_wav, "ref_wav_after_numpy_conversion")

        if ref_wav.device != device:
            log_message(f"[EMBED_REF] Moving ref_wav to {device}")
            ref_wav = ref_wav.to(device)
            log_memory("after_ref_wav_device_move")

        if len(ref_wav.shape) == 1:
            ref_wav = ref_wav.unsqueeze(0)  # (B, L)
            log_tensor_info(ref_wav, "ref_wav_after_unsqueeze")

        if ref_wav.size(1) > 10 * ref_sr:
            log_message("WARNING: cosydec received ref longer than 10s", "warning")

        log_message("[EMBED_REF] Resampling to 24kHz...")
        log_memory("before_24khz_resample")
        ref_wav_24 = ref_wav
        if ref_sr != S3GEN_SR:
            ref_wav_24 = get_resampler(ref_sr, S3GEN_SR, device)(ref_wav)
            log_tensor_info(ref_wav_24, "ref_wav_24_after_resample")
        log_memory("after_24khz_resample")

        log_message("[EMBED_REF] Extracting mel spectrogram...")
        log_memory("before_mel_extraction")
        # transpose creates non-contiguous view - make contiguous for MPS kernels
        ref_mels_24 = contiguous_transpose(self.mel_extractor(ref_wav_24), 1, 2).to(
            device
        )
        log_tensor_info(ref_mels_24, "ref_mels_24")
        log_memory("after_mel_extraction")
        ref_mels_24_len = None

        log_message("[EMBED_REF] Resampling to 16kHz...")
        log_memory("before_16khz_resample")
        ref_wav_16 = get_resampler(ref_sr, S3_SR, device)(ref_wav).to(device)
        log_tensor_info(ref_wav_16, "ref_wav_16")
        log_memory("after_16khz_resample")

        log_message("[EMBED_REF] Computing speaker embedding...")
        log_memory("before_speaker_embedding")
        ref_x_vector = self.speaker_encoder.inference(ref_wav_16)
        log_tensor_info(ref_x_vector, "ref_x_vector")
        log_memory("after_speaker_embedding")

        log_message("[EMBED_REF] Tokenizing reference...")
        log_memory("before_tokenization")
        ref_speech_tokens, ref_speech_token_lens = self.tokenizer(ref_wav_16)
        log_tensor_info(ref_speech_tokens, "ref_speech_tokens")
        log_memory("after_tokenization")

        # Make sure mel_len = 2 * stoken_len (happens when the input is not padded to multiple of 40ms)
        if ref_mels_24.shape[1] != 2 * ref_speech_tokens.shape[1]:
            logging.warning(
                "Reference mel length is not equal to 2 * reference token length.\n"
            )
            log_message(
                f"[EMBED_REF] Adjusting token length: {ref_speech_tokens.shape[1]} -> {ref_mels_24.shape[1] // 2}"
            )
            ref_speech_tokens = ref_speech_tokens[:, : ref_mels_24.shape[1] // 2]
            ref_speech_token_lens[0] = ref_speech_tokens.shape[1]
            log_tensor_info(ref_speech_tokens, "ref_speech_tokens_adjusted")

        # Clean up intermediate tensors
        del ref_wav_24, ref_wav_16
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_memory("end_embed_ref_after_cleanup")

        result = dict(
            prompt_token=ref_speech_tokens.to(device),
            prompt_token_len=ref_speech_token_lens,
            prompt_feat=ref_mels_24,
            prompt_feat_len=ref_mels_24_len,
            embedding=ref_x_vector,
        )

        log_message("[EMBED_REF] Reference embedding complete")
        return result

    def forward(
        self,
        speech_tokens: torch.LongTensor,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor],
        ref_sr: Optional[int],
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        finalize: bool = False,
    ):
        """
        Generate waveforms from S3 speech tokens and a reference waveform, which the speaker timbre is inferred from.
        """
        log_message(f"[FORWARD] Starting S3Token2Mel forward pass, finalize={finalize}")
        log_memory("start_s3token2mel_forward")
        log_tensor_info(speech_tokens, "input_speech_tokens")

        assert (ref_wav is None) ^ (
            ref_dict is None
        ), f"Must provide exactly one of ref_wav or ref_dict (got {ref_wav} and {ref_dict})"

        if ref_dict is None:
            log_message("[FORWARD] Computing reference embedding...")
            ref_dict = self.embed_ref(ref_wav, ref_sr)
        else:
            log_message("[FORWARD] Using pre-computed reference embedding...")
            log_memory("before_ref_dict_processing")
            # type/device casting (all values will be numpy if it's from a prod API call)
            for rk in list(ref_dict):
                if isinstance(ref_dict[rk], np.ndarray):
                    log_message(f"[FORWARD] Converting {rk} from numpy to tensor")
                    ref_dict[rk] = torch.from_numpy(ref_dict[rk])
                if torch.is_tensor(ref_dict[rk]):
                    ref_dict[rk] = ref_dict[rk].to(self.device)
                    log_tensor_info(ref_dict[rk], f"ref_dict[{rk}]")
            log_memory("after_ref_dict_processing")

        if len(speech_tokens.shape) == 1:
            speech_tokens = speech_tokens.unsqueeze(0)
            log_tensor_info(speech_tokens, "speech_tokens_after_unsqueeze")

        speech_token_lens = torch.LongTensor([speech_tokens.size(1)]).to(self.device)
        log_tensor_info(speech_token_lens, "speech_token_lens")

        log_message("[FORWARD] Running flow inference...")
        log_memory("before_flow_inference")
        output_mels, flow_cache = self.flow.inference(
            token=speech_tokens,
            token_len=speech_token_lens,
            finalize=finalize,
            **ref_dict,
        )
        # Explicitly delete flow_cache to prevent memory leak
        if flow_cache is not None:
            del flow_cache
        log_tensor_info(output_mels, "output_mels")
        log_memory("after_flow_inference")

        log_message("[FORWARD] S3Token2Mel forward pass complete")
        return output_mels


class S3Token2Wav(S3Token2Mel):
    """
    The decoder of CosyVoice2 is a concat of token-to-mel (CFM) and a mel-to-waveform (HiFiGAN) modules.
    TODO: make these modules configurable?
    """

    def __init__(self):
        log_message("[INIT] Starting S3Token2Wav initialization")
        log_memory("start_s3token2wav_init")

        super().__init__()

        log_message("[INIT] Creating F0 predictor...")
        log_memory("before_f0_predictor")
        f0_predictor = ConvRNNF0Predictor()
        log_memory("after_f0_predictor")

        log_message("[INIT] Creating HiFiGAN...")
        log_memory("before_mel2wav")
        self.mel2wav = HiFTGenerator(
            sampling_rate=S3GEN_SR,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            f0_predictor=f0_predictor,
        )
        log_memory("after_mel2wav")

        log_message("[INIT] Creating trim fade buffer...")
        # silence out a few ms and fade audio in to reduce artifacts
        n_trim = S3GEN_SR // 50  # 20ms = half of a frame
        trim_fade = torch.zeros(2 * n_trim)
        trim_fade[n_trim:] = (torch.cos(torch.linspace(torch.pi, 0, n_trim)) + 1) / 2
        self.register_buffer(
            "trim_fade", trim_fade, persistent=False
        )  # (buffers get automatic device casting)
        log_tensor_info(self.trim_fade, "trim_fade_buffer")

        log_memory("end_s3token2wav_init")
        log_message("[INIT] S3Token2Wav initialization complete")

    def forward(
        self,
        speech_tokens,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor],
        ref_sr: Optional[int],
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        finalize: bool = False,
    ):
        log_message(f"[FORWARD] Starting S3Token2Wav forward pass, finalize={finalize}")
        log_memory("start_s3token2wav_forward")

        log_message("[FORWARD] Getting mel spectrograms...")
        log_memory("before_parent_forward")
        output_mels = super().forward(
            speech_tokens,
            ref_wav=ref_wav,
            ref_sr=ref_sr,
            ref_dict=ref_dict,
            finalize=finalize,
        )
        log_memory("after_parent_forward")

        log_message("[FORWARD] Converting mels to waveform...")
        log_memory("before_mel2wav")
        # TODO jrm: ignoring the speed control (mel interpolation) and the HiFTGAN caching mechanisms for now.
        hift_cache_source = torch.zeros(1, 1, 0).to(self.device)
        log_tensor_info(hift_cache_source, "hift_cache_source")

        output_wavs, *_ = self.mel2wav.inference(
            speech_feat=output_mels, cache_source=hift_cache_source
        )
        log_tensor_info(output_wavs, "output_wavs")
        log_memory("after_mel2wav")

        if not self.training:
            log_message("[FORWARD] Applying trim fade...")
            log_memory("before_trim_fade")
            # NOTE: ad-hoc method to reduce "spillover" from the reference clip.
            output_wavs[:, : len(self.trim_fade)] *= self.trim_fade
            log_memory("after_trim_fade")

        # Clean up intermediate tensors
        del output_mels, hift_cache_source
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_memory("end_s3token2wav_forward_after_cleanup")

        log_message("[FORWARD] S3Token2Wav forward pass complete")
        return output_wavs

    @torch.inference_mode()
    def flow_inference(
        self,
        speech_tokens,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor] = None,
        ref_sr: Optional[int] = None,
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        finalize: bool = False,
    ):
        log_message(f"[FLOW_INFERENCE] Starting flow inference, finalize={finalize}")
        log_memory("start_flow_inference")
        log_tensor_info(speech_tokens, "flow_input_speech_tokens")

        result = super().forward(
            speech_tokens,
            ref_wav=ref_wav,
            ref_sr=ref_sr,
            ref_dict=ref_dict,
            finalize=finalize,
        )

        log_memory("end_flow_inference")
        log_tensor_info(result, "flow_output_mels")
        log_message("[FLOW_INFERENCE] Flow inference complete")
        return result

    @torch.inference_mode()
    def hift_inference(self, speech_feat, cache_source: torch.Tensor = None):
        log_message("[HIFT_INFERENCE] Starting HiFiGAN inference")
        log_memory("start_hift_inference")
        log_tensor_info(speech_feat, "hift_input_speech_feat")

        # Create local cache source to avoid potential retention
        local_cache_source = torch.zeros(1, 1, 0).to(self.device)
        log_tensor_info(local_cache_source, "hift_local_cache_source")

        log_memory("before_mel2wav_inference")

        # CRITICAL: Force cleanup before HiFiGAN inference
        gc.collect()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        result = self.mel2wav.inference(
            speech_feat=speech_feat, cache_source=local_cache_source
        )
        log_memory("after_mel2wav_inference")

        output_wavs, output_sources = result
        log_tensor_info(output_wavs, "hift_output_wavs")
        if output_sources is not None:
            log_tensor_info(output_sources, "hift_output_sources")

        # Aggressive cleanup of HiFiGAN internals
        log_message("[HIFT_INFERENCE] Performing aggressive cleanup...")
        del local_cache_source

        # Try to clear any potential HiFiGAN internal caches
        if hasattr(self.mel2wav, "clear_cache"):
            self.mel2wav.clear_cache()

        # Clear any computational graph references
        if hasattr(output_wavs, "detach"):
            output_wavs = output_wavs.detach()
        if hasattr(output_sources, "detach"):
            output_sources = output_sources.detach()

        gc.collect()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        log_memory("end_hift_inference_after_aggressive_cleanup")
        log_message("[HIFT_INFERENCE] HiFiGAN inference complete")
        return output_wavs, output_sources

    @torch.inference_mode()
    def inference(
        self,
        speech_tokens,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor] = None,
        ref_sr: Optional[int] = None,
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        cache_source: torch.Tensor = None,  # NOTE: this arg is for streaming, it can probably be removed here
        finalize: bool = True,
    ):
        log_message(
            f"[INFERENCE] Starting full inference pipeline, finalize={finalize}"
        )
        log_memory("start_full_inference")
        log_tensor_info(speech_tokens, "inference_input_speech_tokens")

        # Force aggressive garbage collection before starting
        log_message("[INFERENCE] Pre-inference cleanup...")
        gc.collect()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_memory("after_pre_inference_cleanup")

        before_flow_inference = get_memory_mb()
        log_message(f"before_flow_inference: {before_flow_inference:.1f}MB", "debug")

        log_message("[INFERENCE] Running flow inference...")
        output_mels = self.flow_inference(
            speech_tokens,
            ref_wav=ref_wav,
            ref_sr=ref_sr,
            ref_dict=ref_dict,
            finalize=finalize,
        )

        after_flow_inference = get_memory_mb()
        log_message(f"after_flow_inference: {after_flow_inference:.1f}MB", "debug")
        log_tensor_info(output_mels, "inference_output_mels")

        # Clean up immediately after flow inference
        log_message("[INFERENCE] Mid-inference cleanup...")
        gc.collect()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_memory("after_mid_inference_cleanup")

        log_message("[INFERENCE] Running HiFiGAN inference...")
        log_memory("before_hift_in_inference")
        output_wavs, output_sources = self.hift_inference(output_mels, cache_source)
        log_memory("after_hift_in_inference")

        log_message("[INFERENCE] Applying final trim fade...")
        log_memory("before_final_trim_fade")
        # NOTE: ad-hoc method to reduce "spillover" from the reference clip.
        output_wavs[:, : len(self.trim_fade)] *= self.trim_fade
        log_memory("after_final_trim_fade")

        log_message("[INFERENCE] Final aggressive cleanup...")
        log_memory("before_final_cleanup")

        # Delete intermediate tensors explicitly
        del output_mels
        if cache_source is not None:
            del cache_source

        # Force multiple rounds of garbage collection
        for _ in range(3):
            gc.collect()

        # Clear device cache multiple times
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()  # Ensure all operations complete
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        log_memory("after_final_cleanup")

        final_memory = get_memory_mb()
        log_message(
            f"[INFERENCE] Full inference complete. Final memory: {final_memory:.1f}MB",
            "info",
        )

        return output_wavs, output_sources
