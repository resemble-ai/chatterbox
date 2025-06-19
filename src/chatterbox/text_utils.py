"""
Text processing utility functions

Contains long text splitting, sentence merging and other text processing functions
"""

import re
from typing import List, Optional
import logging


def split_by_word_boundary(text: str, max_len: int) -> List[str]:
    """
    Split text by word boundaries to ensure words are not broken in the middle
    
    Parameters:
        text: Text to split
        max_len: Maximum length of each segment
        
    Returns:
        List of text segments
    """
    # First try to split at punctuation marks
    punct_pattern = r'(?<=[.!?,;:])\s+'
    punct_splits = re.split(punct_pattern, text)
    
    segments = []
    for split in punct_splits:
        if len(split) <= max_len:
            segments.append(split)
            continue
            
        # If the segment after punctuation splitting is still too long, split by word boundaries
        words = split.split()
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


def merge_short_sentences(sentences: List[str], max_length: int, min_length: int = 20) -> List[str]:
    """
    Merge short sentences to the next sentence, ensuring not to exceed maximum length limit
    
    Parameters:
        sentences: List of sentences
        max_length: Maximum length limit after merging
        min_length: Length threshold for short sentences
    
    Returns:
        List of merged sentences
    """
    if not sentences:
        return []
    
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
            potential_merge = merged + " " + next_sentence
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
    
    # If text length doesn't exceed maximum length, return directly
    if len(text) <= max_length:
        return [text]
    
    if logger:
        logger.debug(f"Text length {len(text)} exceeds maximum limit {max_length}, starting split")
    
    # First split by paragraphs (double newlines)
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    
    segments = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_length:
            segments.append(paragraph)
            continue
        
        # Paragraph too long, split further
        # 1. Try to split by sentences (period, question mark, exclamation mark)
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 2. Merge short sentences to avoid producing too many fragments
        sentences = merge_short_sentences(sentences, max_length, min_length=30)
        
        # 3. Handle sentences that are still too long
        for sentence in sentences:
            if len(sentence) <= max_length:
                segments.append(sentence)
            else:
                # Use word boundary splitting
                word_segments = split_by_word_boundary(sentence, max_length)
                segments.extend(word_segments)
    
    # Clean empty paragraphs
    segments = [s.strip() for s in segments if s.strip()]
    
    if logger:
        logger.debug(f"Text splitting completed, total {len(segments)} segments")
    
    return segments 