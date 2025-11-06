"""
Optimized TTS with significant performance improvements for production use
"""
from pathlib import Path
import torch
import torch.nn.functional as F
import torchaudio
import perth
from typing import Optional

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond
from .tts import Conditionals, punc_norm, REPO_ID, ChatterboxTTS
from .optimizations.optimized_t3_inference import optimized_inference


class OptimizedChatterboxTTS(ChatterboxTTS):
    """
    Optimized version of ChatterboxTTS with:
    - torch.compile support
    - GPU-based resampling
    - Mixed precision (BF16)
    - Fused operations
    - Optional watermarking
    - Reduced CPU-GPU transfers
    - Flash Attention support
    """

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str,
        conds: Conditionals = None,
        enable_compilation=True,
        use_mixed_precision=True,
        enable_watermark=True,
    ):
        super().__init__(t3, s3gen, ve, tokenizer, device, conds)

        self.enable_compilation = enable_compilation
        self.use_mixed_precision = use_mixed_precision
        self.enable_watermark = enable_watermark

        # Create GPU-based resamplers
        self._setup_gpu_resamplers()

        # Apply compilation
        if enable_compilation:
            self._compile_models()

    def _setup_gpu_resamplers(self):
        """Setup GPU-based resampling for faster inference"""
        # Resampler for S3GEN_SR (24kHz) to S3_SR (16kHz)
        self.resample_24_to_16 = torchaudio.transforms.Resample(
            orig_freq=S3GEN_SR, new_freq=S3_SR
        ).to(self.device)

        # Resampler for arbitrary SR to S3GEN_SR
        # We'll create these on-demand with caching

    def _compile_models(self):
        """Apply torch.compile to critical components"""
        if not hasattr(torch, "compile"):
            print("⚠️  torch.compile not available, skipping compilation")
            return

        try:
            # Compile S3Gen components
            print("Compiling S3Gen flow model...")
            self.s3gen.flow = torch.compile(
                self.s3gen.flow, mode="reduce-overhead", fullgraph=False
            )

            print("Compiling S3Gen mel2wav...")
            self.s3gen.mel2wav = torch.compile(
                self.s3gen.mel2wav, mode="reduce-overhead", fullgraph=False
            )

            # Compile voice encoder
            print("Compiling VoiceEncoder...")
            self.ve = torch.compile(self.ve, mode="reduce-overhead", fullgraph=False)

            print("✅ Model compilation complete!")

        except Exception as e:
            print(f"⚠️  Compilation failed: {e}")

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        """
        Optimized conditional preparation with GPU-based resampling
        """
        # Load audio directly to tensor
        s3gen_ref_wav, sr = torchaudio.load(wav_fpath)

        # Move to GPU immediately
        s3gen_ref_wav = s3gen_ref_wav.to(self.device)

        # Resample to S3GEN_SR if needed (GPU-based)
        if sr != S3GEN_SR:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=S3GEN_SR
            ).to(self.device)
            s3gen_ref_wav = resampler(s3gen_ref_wav)

        # Convert to mono if stereo
        if s3gen_ref_wav.shape[0] > 1:
            s3gen_ref_wav = s3gen_ref_wav.mean(dim=0, keepdim=True)

        # Flatten
        s3gen_ref_wav = s3gen_ref_wav.squeeze(0)

        # Resample to 16kHz for voice encoder (GPU-based)
        ref_16k_wav = self.resample_24_to_16(s3gen_ref_wav.unsqueeze(0)).squeeze(0)

        # Trim to max length
        s3gen_ref_wav = s3gen_ref_wav[: self.DEC_COND_LEN]
        ref_16k_wav = ref_16k_wav[: self.ENC_COND_LEN]

        # Convert to numpy only when needed for specific components
        s3gen_ref_wav_np = s3gen_ref_wav.cpu().numpy()
        ref_16k_wav_np = ref_16k_wav.cpu().numpy()

        # S3Gen embedding
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav_np, S3GEN_SR, device=self.device)

        # Speech conditional tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward(
                [ref_16k_wav_np], max_len=plen
            )
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(
                self.device
            )

        # Voice encoder embedding
        ve_embed = torch.from_numpy(
            self.ve.embeds_from_wavs([ref_16k_wav_np], sample_rate=S3_SR)
        )
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)

        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    @torch.inference_mode()
    def generate(
        self,
        text,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        verbose=False,  # NEW: Control progress bar
    ):
        """
        Optimized generate with:
        - Mixed precision inference
        - GPU-based preprocessing
        - Fused operations
        - Optional watermarking
        """
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert (
                self.conds is not None
            ), "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Normalize and tokenize
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

        # Add special tokens
        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        # T3 inference with optimizations
        with torch.cuda.amp.autocast(
            enabled=self.use_mixed_precision, dtype=torch.bfloat16
        ):
            # Use optimized inference
            speech_tokens = optimized_inference(
                self.t3,
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                verbose=verbose,
            )

        # Extract conditional batch
        speech_tokens = speech_tokens[0]

        # Clean tokens
        speech_tokens = drop_invalid_tokens(speech_tokens)
        speech_tokens = speech_tokens[speech_tokens < 6561]
        speech_tokens = speech_tokens.to(self.device)

        # S3Gen inference with mixed precision
        with torch.cuda.amp.autocast(
            enabled=self.use_mixed_precision, dtype=torch.bfloat16
        ):
            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )

        # Convert to CPU for final processing
        wav = wav.squeeze(0).detach().cpu().numpy()

        # Optional watermarking
        if self.enable_watermark:
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
            return torch.from_numpy(watermarked_wav).unsqueeze(0)
        else:
            return torch.from_numpy(wav).unsqueeze(0)

    @classmethod
    def from_pretrained(
        cls,
        device,
        enable_compilation=True,
        use_mixed_precision=True,
        enable_watermark=True,
    ) -> "OptimizedChatterboxTTS":
        """Load optimized model from HuggingFace"""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        # Check MPS availability
        if device == "mps" and not torch.backends.mps.is_available():
            print("MPS not available, falling back to CPU")
            device = "cpu"

        # Download weights
        for fpath in [
            "ve.safetensors",
            "t3_cfg.safetensors",
            "s3gen.safetensors",
            "tokenizer.json",
            "conds.pt",
        ]:
            hf_hub_download(repo_id=REPO_ID, filename=fpath)

        # Load from local cache
        ckpt_dir = Path(hf_hub_download(repo_id=REPO_ID, filename="conds.pt")).parent

        # Load models
        if device in ["cpu", "mps"]:
            map_location = torch.device("cpu")
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
        ve.to(device).eval()

        t3 = T3()
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
        s3gen.to(device).eval()

        tokenizer = EnTokenizer(str(ckpt_dir / "tokenizer.json"))

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(
                device
            )

        return cls(
            t3,
            s3gen,
            ve,
            tokenizer,
            device,
            conds=conds,
            enable_compilation=enable_compilation,
            use_mixed_precision=use_mixed_precision,
            enable_watermark=enable_watermark,
        )

    @classmethod
    def from_local(
        cls,
        ckpt_dir,
        device,
        enable_compilation=True,
        use_mixed_precision=True,
        enable_watermark=True,
    ) -> "OptimizedChatterboxTTS":
        """Load from local checkpoint directory"""
        from safetensors.torch import load_file

        ckpt_dir = Path(ckpt_dir)

        if device in ["cpu", "mps"]:
            map_location = torch.device("cpu")
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
        ve.to(device).eval()

        t3 = T3()
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
        s3gen.to(device).eval()

        tokenizer = EnTokenizer(str(ckpt_dir / "tokenizer.json"))

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(
                device
            )

        return cls(
            t3,
            s3gen,
            ve,
            tokenizer,
            device,
            conds=conds,
            enable_compilation=enable_compilation,
            use_mixed_precision=use_mixed_precision,
            enable_watermark=enable_watermark,
        )
