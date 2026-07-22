"""
Text processing utility functions

Contains long text splitting, sentence merging and other text processing functions
Support for multiple languages including English, Chinese, Japanese, Korean, etc.
"""

import re
from typing import List, Optional, Tuple
import logging


def detect_language(text: str) -> str:
    """
    Simple language detection based on character patterns
    
    Parameters:
        text: Text to detect language for
        
    Returns:
        Language code: 'zh' (Chinese), 'ja' (Japanese), 'ko' (Korean), 'en' (English/others)
    """
    # Count different character types
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))  # CJK Unified Ideographs
    japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))  # Hiragana + Katakana  
    korean_chars = len(re.findall(r'[\uac00-\ud7af]', text))  # Hangul
    
    total_chars = len(re.sub(r'\s+', '', text))
    
    if total_chars == 0:
        return 'en'
    
    # Calculate ratios
    chinese_ratio = chinese_chars / total_chars
    japanese_ratio = japanese_chars / total_chars
    korean_ratio = korean_chars / total_chars
    
    # Determine primary language
    if chinese_ratio > 0.3:
        return 'zh'
    elif japanese_ratio > 0.3:
        return 'ja'
    elif korean_ratio > 0.3:
        return 'ko'
    else:
        return 'en'


def get_sentence_separators(lang: str) -> str:
    """
    Get sentence separator pattern for different languages
    
    Parameters:
        lang: Language code
        
    Returns:
        Regex pattern for sentence separators
    """
    if lang == 'zh':
        # Chinese punctuation
        return r'(?<=[。！？；])\s*'
    elif lang == 'ja':
        # Japanese punctuation
        return r'(?<=[。！？])\s*'
    elif lang == 'ko':
        # Korean punctuation  
        return r'(?<=[\.!?。！？])\s*'
    else:
        # English and other languages
        return r'(?<=[.!?])\s+'


def get_punctuation_pattern(lang: str) -> str:
    """
    Get punctuation pattern for word boundary splitting
    
    Parameters:
        lang: Language code
        
    Returns:
        Regex pattern for punctuation marks
    """
    if lang == 'zh':
        return r'(?<=[。！？，；：、])\s*'
    elif lang == 'ja': 
        return r'(?<=[。！？、，])\s*'
    elif lang == 'ko':
        return r'(?<=[\.!?,;。！？，；])\s*'
    else:
        return r'(?<=[.!?,;:])\s+'


def split_by_word_boundary(text: str, max_len: int, lang: Optional[str] = None) -> List[str]:
    """
    Split text by word boundaries to ensure words are not broken in the middle
    Support for multiple languages
    
    Parameters:
        text: Text to split
        max_len: Maximum length of each segment
        lang: Language code (auto-detect if None)
        
    Returns:
        List of text segments
    """
    if lang is None:
        lang = detect_language(text)
    
    # First try to split at punctuation marks
    punct_pattern = get_punctuation_pattern(lang)
    punct_splits = re.split(punct_pattern, text)
    
    segments = []
    for split in punct_splits:
        if len(split) <= max_len:
            segments.append(split)
            continue
            
        # If the segment after punctuation splitting is still too long
        if lang in ['zh', 'ja', 'ko']:
            # For CJK languages, split by character for fine-grained control
            segments.extend(_split_cjk_text(split, max_len))
        else:
            # For space-separated languages, split by word boundaries
            segments.extend(_split_spaced_text(split, max_len))
            
    return segments


def _split_cjk_text(text: str, max_len: int) -> List[str]:
    """
    Split CJK text by characters while trying to preserve semantic units
    
    Parameters:
        text: CJK text to split
        max_len: Maximum length of each segment
        
    Returns:
        List of text segments
    """
    segments = []
    current = ""
    
    # Try to break at common phrase boundaries
    phrase_boundaries = r'[，、；：]'  # Common CJK phrase separators
    
    i = 0
    while i < len(text):
        char = text[i]
        
        # Check if adding this character would exceed max length
        if len(current + char) > max_len:
            if current:
                segments.append(current)
                current = char
            else:
                # Single character exceeds max length, add it anyway
                segments.append(char)
                current = ""
        else:
            current += char
            
        # Try to break at phrase boundaries if we're getting close to max length
        if (len(current) > max_len * 0.8 and 
            re.search(phrase_boundaries, char) and 
            i < len(text) - 1):
            segments.append(current)
            current = ""
            
        i += 1
    
    # Add the remaining part
    if current:
        segments.append(current)
        
    return segments


def _split_spaced_text(text: str, max_len: int) -> List[str]:
    """
    Split space-separated text by word boundaries
    
    Parameters:
        text: Text to split
        max_len: Maximum length of each segment
        
    Returns:
        List of text segments
    """
    segments = []
    words = text.split()
    current = ""
    
    for word in words:
        # If the word itself exceeds maximum length, try to keep the complete word
        if len(word) > max_len:
            if current:
                segments.append(current)
            segments.append(word)
            current = ""
            continue
            
        # Check if adding this word would exceed maximum length
        potential_segment = (current + " " + word).strip() if current else word
        if len(potential_segment) > max_len:
            if current:
                segments.append(current)
            current = word
        else:
            current = potential_segment
    
    # Add the remaining part
    if current:
        segments.append(current)
        
    return segments


def merge_short_sentences(sentences: List[str], max_length: int, min_length: int = 20, lang: Optional[str] = None) -> List[str]:
    """
    Merge short sentences to the next sentence, ensuring not to exceed maximum length limit
    Support for multiple languages
    
    Parameters:
        sentences: List of sentences
        max_length: Maximum length limit after merging
        min_length: Length threshold for short sentences
        lang: Language code (auto-detect if None)
    
    Returns:
        List of merged sentences
    """
    if not sentences:
        return []
    
    # Auto-detect language from first non-empty sentence
    if lang is None:
        for sentence in sentences:
            if sentence.strip():
                lang = detect_language(sentence)
                break
        else:
            lang = 'en'
    
    # Adjust min_length for different languages
    # CJK languages are more information-dense
    if lang in ['zh', 'ja', 'ko']:
        min_length = max(10, min_length // 2)
    
    result = []
    i = 0
    
    while i < len(sentences):
        current = sentences[i].strip()
        
        # Skip empty sentences
        if not current:
            i += 1
            continue
            
        # If current sentence length is greater than or equal to min_length, add directly to result
        if len(current) >= min_length:
            result.append(current)
            i += 1
            continue
            
        # Current sentence is short, try to merge with subsequent sentences
        merged = current
        j = i + 1
        while j < len(sentences) and len(merged) < min_length:
            next_sentence = sentences[j].strip()
            if not next_sentence:
                j += 1
                continue
                
            # Check if merging would exceed maximum length
            separator = "" if lang in ['zh', 'ja', 'ko'] else " "
            potential_merge = merged + separator + next_sentence
            if len(potential_merge) <= max_length:
                merged = potential_merge
                j += 1
            else:
                # If merging would exceed maximum length, stop merging
                break
        
        # Add merged result
        if merged:
            result.append(merged)
        
        # Update index
        i = j if j > i else i + 1
    
    return result


def split_text_into_segments(text: str, max_length: int = 300, logger: Optional[logging.Logger] = None) -> List[str]:
    """
    Split text into segments suitable for TTS processing
    Support for multiple languages including English, Chinese, Japanese, Korean
    
    Parameters:
        text: Text to split
        max_length: Maximum character count for each segment
        logger: Optional logger
        
    Returns:
        List of split segments
    """
    if not text or not text.strip():
        return []
    
    text = text.strip()
    
    # Auto-detect language
    lang = detect_language(text)
    
    # If text length doesn't exceed maximum length, return directly
    if len(text) <= max_length:
        return [text]
    
    if logger:
        logger.debug(f"Text length {len(text)} exceeds maximum limit {max_length}, starting split (language: {lang})")
    
    # First split by paragraphs (double newlines)
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    
    segments = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_length:
            segments.append(paragraph)
            continue
        
        # Paragraph too long, split further
        # 1. Try to split by sentences using language-appropriate patterns
        sentence_pattern = get_sentence_separators(lang)
        sentences = re.split(sentence_pattern, paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 2. Merge short sentences to avoid producing too many fragments
        # Adjust min_length based on language characteristics
        min_length = 30 if lang == 'en' else 15  # CJK languages are more dense
        sentences = merge_short_sentences(sentences, max_length, min_length=min_length, lang=lang)
        
        # 3. Handle sentences that are still too long
        for sentence in sentences:
            if len(sentence) <= max_length:
                segments.append(sentence)
            else:
                # Use word boundary splitting with language support
                word_segments = split_by_word_boundary(sentence, max_length, lang=lang)
                segments.extend(word_segments)
    
    # Clean empty paragraphs
    segments = [s.strip() for s in segments if s.strip()]
    
    if logger:
        logger.debug(f"Text splitting completed, total {len(segments)} segments (language: {lang})")
    
    return segments 