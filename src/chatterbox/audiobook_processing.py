"""
Enhanced audiobook processing utilities for Chatterbox TTS.

This module provides advanced text processing, voice management, and audiobook creation features.
"""

import re
import os
import wave
import json
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import time

# Optional imports for enhanced functionality
try:
    import librosa
    import soundfile as sf
    AUDIO_ENHANCED = True
except ImportError:
    AUDIO_ENHANCED = False

class AudiobookProcessor:
    """Main class for audiobook processing and management."""
    
    def __init__(self, voice_library_path: str = "voice_library"):
        self.voice_library_path = Path(voice_library_path)
        self.voice_library_path.mkdir(exist_ok=True)
        
    def chunk_text_by_sentences(self, text: str, max_words: int = 50) -> List[str]:
        """Split text into chunks, breaking at sentence boundaries after reaching max_words."""
        sentences = re.split(r'([.!?]+\s*)', text)
        
        chunks = []
        current_chunk = ""
        current_word_count = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i].strip()
            if not sentence:
                i += 1
                continue
                
            if i + 1 < len(sentences) and re.match(r'[.!?]+\s*', sentences[i + 1]):
                sentence += sentences[i + 1]
                i += 2
            else:
                i += 1
            
            sentence_words = len(sentence.split())
            
            if current_word_count > 0 and current_word_count + sentence_words > max_words:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_word_count = sentence_words
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_word_count += sentence_words
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def adaptive_chunk_text(self, text: str, max_words: int = 50) -> List[str]:
        """Adaptively chunk text with error handling."""
        return self.chunk_text_by_sentences(text, max_words)
    
    def parse_multi_voice_text(self, text: str) -> List[Dict[str, str]]:
        """Parse text with multi-voice format markers like [Character] dialogue."""
        segments = []
        parts = re.split(r'(\[[^\]]+\])', text)
        
        current_character = None
        buffer = ""
        
        for part in parts:
            if not part:
                continue
                
            part_stripped = part.strip()
            if re.match(r'^\[[^\]]+\]$', part_stripped):
                if current_character and buffer.strip():
                    segments.append({
                        'character': current_character,
                        'text': buffer.strip()
                    })
                current_character = part_stripped[1:-1]
                buffer = ""
            else:
                if current_character is None and part_stripped:
                    segments.append({
                        'character': "Narrator",
                        'text': part_stripped
                    })
                    buffer = ""
                elif current_character:
                    buffer += part
        
        if current_character and buffer.strip():
            segments.append({
                'character': current_character,
                'text': buffer.strip()
            })
        elif not current_character and buffer.strip() and not segments:
            segments.append({
                'character': "Narrator",
                'text': buffer.strip()
            })
        
        return segments
    
    def count_line_breaks(self, text: str) -> int:
        """Count line breaks for pause calculation."""
        return text.count('\n')
    
    def add_pause_segments(self, text: str) -> List[Dict[str, Any]]:
        """Add pause segments based on line breaks for natural speech flow."""
        segments = self.parse_multi_voice_text(text)
        enhanced_segments = []
        
        for segment in segments:
            text = segment['text']
            line_breaks = self.count_line_breaks(text)
            pause_duration = line_breaks * 0.1  # 0.1 seconds per line break
            
            enhanced_segments.append({
                'character': segment['character'],
                'text': text,
                'pause_duration': pause_duration,
                'line_breaks': line_breaks
            })
        
        return enhanced_segments
    
    def load_text_file(self, file_path: str) -> Tuple[str, str]:
        """Load text content from a file with encoding detection."""
        if not file_path:
            return "", "No file selected"
        
        try:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            
            if not content.strip():
                return "", "File is empty"
            
            return content.strip(), f"✅ Loaded {len(content.split())} words from file"
        
        except FileNotFoundError:
            return "", "❌ File not found"
        except Exception as e:
            return "", f"❌ Error reading file: {str(e)}"
    
    def validate_audiobook_input(self, text_content: str, project_name: str) -> Tuple[bool, str]:
        """Validate input for audiobook creation."""
        if not text_content or not text_content.strip():
            return False, "❌ Please provide text content or upload a text file"
        
        if not project_name or not project_name.strip():
            return False, "❌ Please provide a project name"
        
        word_count = len(text_content.split())
        if word_count < 10:
            return False, "❌ Text content too short (minimum 10 words)"
        
        if word_count > 100000:
            return False, "❌ Text content too long (maximum 100,000 words)"
        
        return True, ""
    
    def save_voice_profile(self, voice_name: str, display_name: str, description: str, 
                          audio_file: str, exaggeration: float, cfg_weight: float, 
                          temperature: float, **kwargs) -> str:
        """Save a voice profile to the voice library."""
        profile_dir = self.voice_library_path / voice_name
        profile_dir.mkdir(exist_ok=True)
        
        config = {
            'voice_name': voice_name,
            'display_name': display_name,
            'description': description,
            'audio_file': os.path.basename(audio_file) if audio_file else None,
            'exaggeration': exaggeration,
            'cfg_weight': cfg_weight,
            'temperature': temperature,
            'created_date': str(time.time()),
            'version': '2.0'
        }
        
        # Add additional parameters
        for key, value in kwargs.items():
            config[key] = value
        
        config_file = profile_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return f"✅ Voice profile '{display_name or voice_name}' saved successfully!"
    
    def load_voice_profile(self, voice_name: str) -> Dict[str, Any]:
        """Load a voice profile from the voice library."""
        if not voice_name:
            return {}
        
        profile_dir = self.voice_library_path / voice_name
        config_file = profile_dir / "config.json"
        
        if not config_file.exists():
            return {}
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            return {}
    
    def list_voice_profiles(self) -> List[str]:
        """List all available voice profiles."""
        if not self.voice_library_path.exists():
            return []
        
        profiles = []
        for item in self.voice_library_path.iterdir():
            if item.is_dir() and (item / "config.json").exists():
                profiles.append(item.name)
        
        return sorted(profiles)
    
    def delete_voice_profile(self, voice_name: str) -> str:
        """Delete a voice profile from the library."""
        if not voice_name:
            return "❌ No voice selected"
        
        profile_dir = self.voice_library_path / voice_name
        if profile_dir.exists():
            try:
                shutil.rmtree(profile_dir)
                return f"✅ Voice profile '{voice_name}' deleted successfully!"
            except Exception as e:
                return f"❌ Error deleting voice profile: {str(e)}"
        else:
            return f"❌ Voice profile '{voice_name}' not found"
