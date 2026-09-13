import re
import torch
import torchaudio
from typing import Generator, Tuple, Optional, Any, Iterable


class SentenceSplitter:
    """
    Splits text into sentences using regular expressions.
    Handles standard punctuation, common abbreviations, and minimum sentence lengths.
    """

    def __init__(self, min_length: int = 5):
        self.min_length = min_length
        # Punctuation to split on: . ! ? and Asian variants 。 ？！
        self.punctuation_pattern = r'([\.!\?。？！]+[\s]*)'
        
        # Simple abbreviations to avoid splitting on
        self.abbreviations = {
            "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.", 
            "e.g.", "i.e.", "etc.", "st.", "inc.", "ltd."
        }

    def split(self, text: str) -> list[str]:
        """
        Splits the given text into a list of sentences.
        """
        if not text:
            return []
            
        # Split using punctuation and keep the punctuation attached to the preceding text
        # by combining pairs from the split result
        parts = re.split(self.punctuation_pattern, text)
        sentences = []
        
        current_sentence = ""
        for i in range(0, len(parts), 2):
            part = parts[i]
            punct = parts[i+1] if i+1 < len(parts) else ""
            
            chunk = part + punct
            current_sentence += chunk
            
            # Check if we should split here
            should_split = True
            
            # Avoid splitting on common abbreviations
            lower_chunk = chunk.strip().lower()
            for abbr in self.abbreviations:
                if lower_chunk.endswith(abbr):
                    should_split = False
                    break
                    
            if should_split and len(current_sentence.strip()) >= self.min_length:
                sentences.append(current_sentence.strip())
                current_sentence = ""
                
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
            
        # Fallback if no splits were made but text is long, just return original
        if not sentences and text.strip():
            return [text.strip()]
            
        return sentences


class StreamingTTS:
    """
    Wrapper for Chatterbox models to generate audio in a streaming fashion by 
    processing text sentence-by-sentence.
    """

    def __init__(self, model: Any, sentence_splitter: Optional[SentenceSplitter] = None):
        if not hasattr(model, 'generate') or not hasattr(model, 'sr'):
            raise TypeError("Model must have a 'generate' method and an 'sr' (sample rate) attribute.")
            
        self.model = model
        self.sentence_splitter = sentence_splitter or SentenceSplitter()
        
    def generate_stream(self, text: str, **kwargs) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """
        Generator that yields (sentence, audio_tensor) tuples.
        """
        sentences = self.sentence_splitter.split(text)
        for sentence in sentences:
            if not sentence.strip():
                continue
            audio = self.model.generate(sentence, **kwargs)
            yield sentence, audio

    def generate_stream_audio(self, text: str, **kwargs) -> Generator[torch.Tensor, None, None]:
        """
        Generator that yields only audio tensor chunks.
        """
        for _, audio in self.generate_stream(text, **kwargs):
            yield audio

    def generate_full(self, text: str, gap_seconds: float = 0.3, **kwargs) -> torch.Tensor:
        """
        Generates full audio for the text by concatenating sentence chunks 
        with gaps between them.
        """
        chunks = list(self.generate_stream_audio(text, **kwargs))
        if not chunks:
            # Return an empty tensor of shape [1, 0] if no audio was generated
            return torch.zeros((1, 0))
            
        gap_samples = int(gap_seconds * self.model.sr)
        gap_tensor = torch.zeros((1, gap_samples), device=chunks[0].device, dtype=chunks[0].dtype)
        
        result_chunks = []
        for i, chunk in enumerate(chunks):
            # Ensure chunk is 2D: [1, seq_len]
            if chunk.dim() == 1:
                chunk = chunk.unsqueeze(0)
                
            result_chunks.append(chunk)
            if i < len(chunks) - 1 and gap_samples > 0:
                result_chunks.append(gap_tensor)
                
        return torch.cat(result_chunks, dim=-1)


def stream_to_file(stream: Iterable[torch.Tensor], output_path: str, sample_rate: int) -> None:
    """
    Saves an audio stream (iterable of tensors) to a WAV file.
    """
    chunks = []
    for chunk in stream:
        if chunk.dim() == 1:
            chunk = chunk.unsqueeze(0)
        chunks.append(chunk)
        
    if not chunks:
        # Create empty file
        torchaudio.save(output_path, torch.zeros((1, 0)), sample_rate)
        return
        
    full_audio = torch.cat(chunks, dim=-1)
    
    # Ensure CPU for saving
    full_audio = full_audio.cpu()
    
    torchaudio.save(output_path, full_audio, sample_rate)
