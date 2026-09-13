"""
SRT Narrator module.
Generates audio narration from SRT subtitle files using Chatterbox TTS.
"""

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Any
import torch
import torchaudio

from .emotion_control import apply_speaking_rate

@dataclass
class SubtitleEntry:
    """Represents a single subtitle entry."""
    index: int
    start_seconds: float
    end_seconds: float
    text: str
    
    @property
    def duration(self) -> float:
        return self.end_seconds - self.start_seconds

def _time_str_to_seconds(time_str: str) -> float:
    """Converts SRT time string (HH:MM:SS,MMM) to seconds."""
    time_str = time_str.strip()
    match = re.match(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", time_str)
    if not match:
        raise ValueError(f"Invalid time format: {time_str}")
    
    h, m, s, ms = match.groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

def parse_srt(srt_path: str) -> List[SubtitleEntry]:
    """
    Parses an SRT file into a list of SubtitleEntry objects.
    
    Args:
        srt_path: Path to the SRT file.
        
    Returns:
        List of SubtitleEntry objects.
    """
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    blocks = content.strip().split('\n\n')
    entries = []
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            index_line = lines[0].strip()
            # Remove any BOM that might be present
            if index_line.startswith('\ufeff'):
                index_line = index_line[1:]
            
            try:
                index = int(index_line)
            except ValueError:
                continue
                
            time_line = lines[1].strip()
            times = time_line.split(' --> ')
            if len(times) != 2:
                continue
                
            start_seconds = _time_str_to_seconds(times[0])
            end_seconds = _time_str_to_seconds(times[1])
            
            text = ' '.join(line.strip() for line in lines[2:])
            
            entries.append(SubtitleEntry(
                index=index,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                text=text
            ))
            
    return sorted(entries, key=lambda x: x.start_seconds)

def _fit_audio_to_duration(wav: torch.Tensor, target_duration: float, sample_rate: int) -> torch.Tensor:
    """
    Speeds up or pads audio to fit the target duration.
    
    Args:
        wav: Audio tensor.
        target_duration: Target duration in seconds.
        sample_rate: Sample rate.
        
    Returns:
        Audio tensor fitted to the target duration.
    """
    current_duration = wav.shape[-1] / sample_rate
    
    if current_duration > target_duration:
        # Audio is too long, we need to speed it up
        rate = current_duration / target_duration
        wav = apply_speaking_rate(wav, rate, sample_rate)
        
        # In case of small rounding errors, trim to exactly target duration
        target_samples = int(target_duration * sample_rate)
        if wav.shape[-1] > target_samples:
            wav = wav[..., :target_samples]
    elif current_duration < target_duration:
        # Audio is too short, we pad with silence at the end
        padding_duration = target_duration - current_duration
        padding_samples = int(padding_duration * sample_rate)
        padding = torch.zeros((wav.shape[0], padding_samples) if wav.dim() > 1 else (padding_samples,), dtype=wav.dtype, device=wav.device)
        wav = torch.cat([wav, padding], dim=-1)
        
    return wav

class SRTNarrator:
    """Generates complete audio narrations from SRT files."""
    
    def __init__(self, model: Any, sample_rate: Optional[int] = None):
        """
        Initializes the narrator.
        
        Args:
            model: Chatterbox TTS model instance.
            sample_rate: Target sample rate. If None, uses model.sr.
        """
        self.model = model
        self.sample_rate = sample_rate or getattr(model, 'sr', 24000)
        
    def narrate(self, srt_path: str, output_path: str = 'narrated.wav', lang: str = 'en', voice_ref: Optional[str] = None, **kwargs) -> torch.Tensor:
        """
        Generates a continuous audio narration for the entire SRT file.
        
        Args:
            srt_path: Path to the input SRT file.
            output_path: Path to save the resulting audio.
            lang: Language for generation (if supported by model).
            voice_ref: Path to the reference audio for voice cloning.
            **kwargs: Additional generation arguments (like exaggeration, etc.).
            
        Returns:
            The complete generated audio tensor.
        """
        entries = parse_srt(srt_path)
        if not entries:
            raise ValueError(f"No valid subtitles found in {srt_path}")
            
        total_duration = entries[-1].end_seconds
        total_samples = int(total_duration * self.sample_rate)
        
        # Initialize an empty canvas for the full audio
        full_audio = torch.zeros(1, total_samples)
        
        for entry in entries:
            # Prepare arguments
            gen_kwargs = {'text': entry.text}
            
            # Accommodate different model APIs
            if hasattr(self.model.__class__, 'generate'):
                import inspect
                sig = inspect.signature(self.model.generate)
                if 'audio_prompt_path' in sig.parameters:
                    gen_kwargs['audio_prompt_path'] = voice_ref
                if 'lang' in sig.parameters:
                    gen_kwargs['lang'] = lang
                    
                # Add extra kwargs
                gen_kwargs.update(kwargs)
                
                # Generate audio
                with torch.no_grad():
                    wav = self.model.generate(**gen_kwargs)
            else:
                raise RuntimeError("Model does not have a recognizable generate method.")
                
            # Ensure shape is (1, L)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
                
            # Fit audio to the exact duration of the subtitle
            fitted_wav = _fit_audio_to_duration(wav, entry.duration, self.sample_rate)
            
            # Insert into full audio
            start_sample = int(entry.start_seconds * self.sample_rate)
            end_sample = start_sample + fitted_wav.shape[-1]
            
            # Ensure we don't exceed the buffer
            if end_sample > total_samples:
                diff = end_sample - total_samples
                fitted_wav = fitted_wav[..., :-diff]
                end_sample = total_samples
                
            full_audio[0, start_sample:end_sample] = fitted_wav[0]
            
        # Save output
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
            torchaudio.save(output_path, full_audio.cpu(), self.sample_rate)
            
        return full_audio
        
    def narrate_segments(self, srt_path: str, output_dir: str, lang: str = 'en', voice_ref: Optional[str] = None, **kwargs) -> List[str]:
        """
        Generates individual audio files for each subtitle entry.
        
        Args:
            srt_path: Path to the input SRT file.
            output_dir: Directory to save the segments.
            lang: Language for generation.
            voice_ref: Path to the reference audio.
            **kwargs: Additional generation arguments.
            
        Returns:
            List of paths to the generated segment files.
        """
        entries = parse_srt(srt_path)
        if not entries:
            raise ValueError(f"No valid subtitles found in {srt_path}")
            
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        
        for entry in entries:
            gen_kwargs = {'text': entry.text}
            
            if hasattr(self.model.__class__, 'generate'):
                import inspect
                sig = inspect.signature(self.model.generate)
                if 'audio_prompt_path' in sig.parameters:
                    gen_kwargs['audio_prompt_path'] = voice_ref
                if 'lang' in sig.parameters:
                    gen_kwargs['lang'] = lang
                gen_kwargs.update(kwargs)
                
                with torch.no_grad():
                    wav = self.model.generate(**gen_kwargs)
            else:
                raise RuntimeError("Model does not have a recognizable generate method.")
                
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
                
            fitted_wav = _fit_audio_to_duration(wav, entry.duration, self.sample_rate)
            
            out_name = f"segment_{entry.index:04d}_{entry.start_seconds:.3f}_{entry.end_seconds:.3f}.wav"
            out_path = os.path.join(output_dir, out_name)
            
            torchaudio.save(out_path, fitted_wav.cpu(), self.sample_rate)
            saved_paths.append(out_path)
            
        return saved_paths
