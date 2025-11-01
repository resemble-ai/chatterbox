from dataclasses import dataclass
from pathlib import Path
import tempfile
import subprocess
import os
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed


import librosa
import torch
import perth
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import re
import numpy as np
import torchaudio

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond
from .text_utils import split_text_into_segments


REPO_ID = "ResembleAI/chatterbox"


def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxTTS':
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(
            load_file(ckpt_dir / "ve.safetensors")
        )
        ve.to(device).eval()

        t3 = T3()
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
        s3gen.to(device).eval()

        tokenizer = EnTokenizer(
            str(ckpt_dir / "tokenizer.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxTTS':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

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
        use_auto_editor=False,
        ae_threshold=0.06,
        ae_margin=0.2,
        disable_watermark=False,
        max_segment_length=300,
        max_workers=None,  # None means auto-detect optimal value
    ):
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Long text processing: automatically determine if text needs to be split based on length
        if len(text) > max_segment_length:
            # Use async batch generation
            if max_workers is None:
                max_workers = 3  # Simple default value
            return self._generate_long_text_async(
                text, 
                max_segment_length=max_segment_length,
                cfg_weight=cfg_weight,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                use_auto_editor=use_auto_editor,
                ae_threshold=ae_threshold,
                ae_margin=ae_margin,
                disable_watermark=disable_watermark,
                max_workers=max_workers
            )

        # Parse pause tags BEFORE applying punc_norm to preserve the tags
        segments = parse_pause_tags(text)
        
        # Single segment processing (simplified logic for single text without pauses)
        if len(segments) == 1 and segments[0][1] == 0.0:  # Single text, no pause
            text_segment = segments[0][0]
            segment_audio = self._generate_single_segment(
                text_segment, cfg_weight, temperature, repetition_penalty, min_p, top_p, disable_watermark
            )
            
            # Clean artifacts (if enabled)
            if use_auto_editor:
                # Save temporary audio file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_audio_path = temp_file.name
                torchaudio.save(temp_audio_path, segment_audio, self.sr)
                
                # Clean artifacts
                cleaned_audio_path = self._clean_artifacts(temp_audio_path, ae_threshold, ae_margin)
                
                # Load cleaned audio
                if cleaned_audio_path != temp_audio_path:
                    try:
                        cleaned_audio, _ = torchaudio.load(cleaned_audio_path)
                        # Clean temporary files
                        if os.path.exists(temp_audio_path):
                            os.unlink(temp_audio_path)
                        if os.path.exists(cleaned_audio_path):
                            os.unlink(cleaned_audio_path)
                        return cleaned_audio
                    except Exception as e:
                        print(f"[WARNING] Unable to load cleaned audio: {e}")
                        # Clean temporary files
                        if os.path.exists(temp_audio_path):
                            os.unlink(temp_audio_path)
                        if os.path.exists(cleaned_audio_path):
                            os.unlink(cleaned_audio_path)
                else:
                    # Cleaning failed, use original audio
                    if os.path.exists(temp_audio_path):
                        os.unlink(temp_audio_path)
                    
            return segment_audio
        
        # Process text with pauses - generate and clean each segment first, then add pauses
        audio_segments = []
        temp_files_to_cleanup = []
        
        try:
            for text_segment, pause_duration in segments:
                if text_segment.strip():  # Non-empty text segment
                    # 1. Generate audio segment
                    segment_audio = self._generate_single_segment(
                        text_segment, cfg_weight, temperature, repetition_penalty, min_p, top_p, disable_watermark
                    )
                    
                    # 2. Clean artifacts for this segment
                    if use_auto_editor:
                        # Save temporary audio file
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                            temp_audio_path = temp_file.name
                        temp_files_to_cleanup.append(temp_audio_path)
                        torchaudio.save(temp_audio_path, segment_audio, self.sr)
                        
                        # Clean artifacts
                        cleaned_audio_path = self._clean_artifacts(temp_audio_path, ae_threshold, ae_margin)
                        if cleaned_audio_path != temp_audio_path:
                            temp_files_to_cleanup.append(cleaned_audio_path)
                        
                        # Load cleaned audio
                        try:
                            if cleaned_audio_path != temp_audio_path:
                                cleaned_audio, _ = torchaudio.load(cleaned_audio_path)
                                segment_audio = cleaned_audio
                        except Exception as e:
                            print(f"[WARNING] Unable to load cleaned audio segment: {e}")
                            # Continue using original audio segment
                    
                    audio_segments.append(segment_audio.squeeze(0))
                
                # 3. Add pause (after artifact cleaning)
                if pause_duration > 0:
                    silence = create_silence(pause_duration, self.sr)
                    audio_segments.append(silence.squeeze(0))
            
            # 4. Concatenate all audio segments
            if audio_segments:
                final_audio = torch.cat(audio_segments, dim=0)
                return final_audio.unsqueeze(0)
            else:
                # If no valid audio segments, return brief silence
                return create_silence(0.1, self.sr)
                
        finally:
            # Clean up all temporary files
            for temp_file in temp_files_to_cleanup:
                if os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except:
                        pass  # Ignore cleanup errors

    def _generate_long_text_async(
        self,
        text,
        max_segment_length=300,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        use_auto_editor=False,
        ae_threshold=0.06,
        ae_margin=0.2,
        disable_watermark=False,
        max_workers=3
    ):
        """
        Async generation of long text audio - core optimization only keeps async batch generation
        """
        # Split text into short paragraphs
        text_segments = split_text_into_segments(text, max_segment_length)
        
        if not text_segments:
            return create_silence(0.1, self.sr)
        
        # Preprocess text segments, preserve pause tag information
        all_segments = []  # [(text, pause_duration), ...]
        for segment_text in text_segments:
            # Process pause tags, preserve pause information
            segments = parse_pause_tags(segment_text)
            all_segments.extend(segments)
        
        # Separate text and pause information
        text_parts = []
        pause_info = []
        for text_part, pause_duration in all_segments:
            if text_part.strip():
                text_parts.append(text_part.strip())
                pause_info.append(pause_duration)
        
        # Core: async batch generation
        audio_segments = self._generate_segments_async(
            text_parts, cfg_weight, temperature, 
            repetition_penalty, min_p, top_p, disable_watermark, max_workers
        )
        
        # Apply audio cleaning (if enabled)
        if use_auto_editor:
            audio_segments = self._clean_audio_segments_batch(
                audio_segments, ae_threshold, ae_margin
            )
        
        # Merge audio, including pauses
        final_audio_parts = []
        for i, audio in enumerate(audio_segments):
            if audio is not None:
                final_audio_parts.append(audio.squeeze(0))
                # Add pause (if any)
                if i < len(pause_info) and pause_info[i] > 0:
                    silence = create_silence(pause_info[i], self.sr)
                    final_audio_parts.append(silence.squeeze(0))
        
        if final_audio_parts:
            final_audio = torch.cat(final_audio_parts, dim=0)
            return final_audio.unsqueeze(0)
        else:
            return create_silence(0.1, self.sr)









    def _generate_segments_async(self, text_list, cfg_weight, temperature, repetition_penalty, min_p, top_p, disable_watermark, max_workers):
        """Async parallel generation of text segments - core performance optimization"""
        audio_results = [None] * len(text_list)
        result_queue = queue.Queue()
        
        def generate_worker(index, text):
            try:
                if text and text.strip():
                    # Preprocess text
                    text = punc_norm(text.strip())
                    # Generate audio
                    audio = self._generate_single_segment(
                        text, cfg_weight, temperature, repetition_penalty, min_p, top_p, disable_watermark
                    )
                else:
                    audio = create_silence(0.1, self.sr)
                result_queue.put((index, audio, None))
            except Exception as e:
                result_queue.put((index, None, str(e)))
        
        # Concurrent execution
        with ThreadPoolExecutor(max_workers=min(max_workers, len(text_list))) as executor:
            futures = [executor.submit(generate_worker, i, text) for i, text in enumerate(text_list)]
            
            # Wait for completion
            for future in as_completed(futures):
                future.result()
        
        # Collect results
        while not result_queue.empty():
            index, audio, error = result_queue.get()
            if error:
                audio_results[index] = create_silence(0.1, self.sr)
            else:
                audio_results[index] = audio
        
        return audio_results

    def _clean_audio_segments_batch(self, audio_segments, ae_threshold, ae_margin):
        """Batch clean artifacts from audio segments"""
        cleaned_segments = []
        temp_files_to_cleanup = []
        
        try:
            for i, audio in enumerate(audio_segments):
                if audio is not None:
                    # Save temporary audio file
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=f'_segment_{i}.wav', delete=False) as temp_file:
                        temp_audio_path = temp_file.name
                    temp_files_to_cleanup.append(temp_audio_path)
                    torchaudio.save(temp_audio_path, audio, self.sr)
                    
                    # Clean artifacts
                    cleaned_audio_path = self._clean_artifacts(temp_audio_path, ae_threshold, ae_margin)
                    if cleaned_audio_path != temp_audio_path:
                        temp_files_to_cleanup.append(cleaned_audio_path)
                    
                    # Load cleaned audio
                    try:
                        if cleaned_audio_path != temp_audio_path:
                            cleaned_audio, _ = torchaudio.load(cleaned_audio_path)
                            cleaned_segments.append(cleaned_audio)
                        else:
                            # Cleaning failed, use original audio
                            cleaned_segments.append(audio)
                    except Exception as e:
                        print(f"[WARNING] Unable to load cleaned audio segment {i}: {e}")
                        cleaned_segments.append(audio)
                else:
                    cleaned_segments.append(audio)
            
            return cleaned_segments
            
        finally:
            # Clean up all temporary files
            for temp_file in temp_files_to_cleanup:
                if os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except:
                        pass  # Ignore cleanup errors

    def _generate_single_segment(self, text, cfg_weight, temperature, repetition_penalty=1.2, min_p=0.05, top_p=1.0, disable_watermark=False):
        """Generate audio for a single text segment"""
        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,  # TODO: use the value in config
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )
            # Extract only the conditional batch.
            speech_tokens = speech_tokens[0]

            # TODO: output becomes 1D
            speech_tokens = drop_invalid_tokens(speech_tokens)
            
            speech_tokens = speech_tokens[speech_tokens < 6561]

            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            
            if not disable_watermark:
                watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
                return torch.from_numpy(watermarked_wav).unsqueeze(0)
            else:
                return torch.from_numpy(wav).unsqueeze(0)

    def _clean_artifacts(self, audio_path: str, threshold: float = 0.06, margin: float = 0.2) -> str:
        """
        Clean artifacts from audio using auto-editor
        
        Args:
            audio_path: Path to input audio file
            threshold: Volume threshold, values below this are considered silence/artifacts
            margin: Boundary protection time in seconds
            
        Returns:
            Path to cleaned audio file
        """
        import subprocess
        import tempfile
        import os
        
        # Create output file
        output_file = tempfile.NamedTemporaryFile(suffix='_cleaned.wav', delete=False)
        output_file.close()
        
        try:
            # Build auto-editor command (adapted for 28.0.0 version)
            cmd = [
                "auto-editor",
                audio_path,
                "--edit", f"audio:threshold={threshold}",
                "--margin", f"{margin}s",
                "--output-file", output_file.name
            ]
            
            print(f"[INFO] Cleaning artifacts: {' '.join(cmd)}")
            
            # Execute auto-editor
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                check=True
            )
            
            if os.path.exists(output_file.name) and os.path.getsize(output_file.name) > 0:
                print(f"[INFO] Artifact cleaning completed: {output_file.name}")
                return output_file.name
            else:
                raise RuntimeError("auto-editor did not generate a valid output file")
                
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] auto-editor execution failed: {e}")
            print(f"[ERROR] stderr: {e.stderr}")
            print(f"[ERROR] stdout: {e.stdout}")
            # Clean up failed output file
            if os.path.exists(output_file.name):
                os.unlink(output_file.name)
            return audio_path  # Return original file
            
        except Exception as e:
            print(f"[ERROR] Exception occurred during artifact cleaning: {e}")
            # Clean up failed output file
            if os.path.exists(output_file.name):
                os.unlink(output_file.name)
            return audio_path  # Return original file


def parse_pause_tags(text: str):
    """
    Parse pause tags in text and return text segments with corresponding pause durations
    
    Args:
        text: Text containing pause tags like "Hello[pause:0.5s]world[pause:1.0s]end"
    
    Returns:
        segments: [(text_segment, pause_duration), ...]
        Example: [("Hello", 0.5), ("world", 1.0), ("end", 0.0)]
    """
    if not text:
        return [("", 0.0)]
    
    # Regular expression to match pause tags
    pause_pattern = r'\[pause:([\d.]+)s\]'
    
    segments = []
    last_end = 0
    
    # Find all pause tags
    for match in re.finditer(pause_pattern, text):
        # Extract text before the pause tag
        text_segment = text[last_end:match.start()].strip()
        if text_segment:
            segments.append((text_segment, 0.0))
        
        # Extract pause duration
        pause_duration = float(match.group(1))
        # Ensure pause duration is a multiple of 0.1s
        pause_duration = round(pause_duration / 0.1) * 0.1
        segments.append(("", pause_duration))
        
        last_end = match.end()
    
    # Add the final text segment
    final_text = text[last_end:].strip()
    if final_text:
        segments.append((final_text, 0.0))
    
    # If no segments found, return original text
    if not segments:
        segments = [(text, 0.0)]
    
    return segments


def create_silence(duration_seconds: float, sample_rate: int) -> torch.Tensor:
    """
    Create silence of specified duration
    
    Args:
        duration_seconds: Duration of silence in seconds
        sample_rate: Sample rate
    
    Returns:
        Silent audio tensor
    """
    num_samples = int(duration_seconds * sample_rate)
    return torch.zeros(1, num_samples)