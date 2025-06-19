"""
文本处理工具函数

包含长文本拆分、句子合并等文本处理功能
"""

import re
from typing import List, Optional
import logging


def split_by_word_boundary(text: str, max_len: int) -> List[str]:
    """
    按单词边界拆分文本，确保不会在单词中间断开
    
    参数:
        text: 要拆分的文本
        max_len: 每个分段的最大长度
        
    返回:
        文本分段列表
    """
    # 首先尝试在标点符号处分割
    punct_pattern = r'(?<=[.!?,;:])\s+'
    punct_splits = re.split(punct_pattern, text)
    
    segments = []
    for split in punct_splits:
        if len(split) <= max_len:
            segments.append(split)
            continue
            
        # 如果标点分割后的片段仍然过长，按单词边界拆分
        words = split.split()
        current = ""
        
        for word in words:
            # 如果单词本身超过最大长度，尝试保留完整单词
            if len(word) > max_len:
                if current:
                    segments.append(current)
                segments.append(word)
                current = ""
                continue
                
            # 检查添加这个单词是否会超过最大长度
            potential_segment = (current + " " + word).strip() if current else word
            if len(potential_segment) > max_len:
                if current:
                    segments.append(current)
                current = word
            else:
                current = potential_segment
        
        # 添加最后剩余的部分
        if current:
            segments.append(current)
            
    return segments


def merge_short_sentences(sentences: List[str], max_length: int, min_length: int = 20) -> List[str]:
    """
    合并短句子到下一句，确保不超过最大长度限制
    
    参数:
        sentences: 句子列表
        max_length: 合并后句子的最大长度限制
        min_length: 短句子的长度阈值
    
    返回:
        合并后的句子列表
    """
    if not sentences:
        return []
    
    result = []
    i = 0
    
    while i < len(sentences):
        current = sentences[i].strip()
        
        # 跳过空句子
        if not current:
            i += 1
            continue
            
        # 如果当前句子长度大于等于min_length，直接添加到结果中
        if len(current) >= min_length:
            result.append(current)
            i += 1
            continue
            
        # 当前句子是短句，尝试与后面的句子合并
        merged = current
        j = i + 1
        while j < len(sentences) and len(merged) < min_length:
            next_sentence = sentences[j].strip()
            if not next_sentence:
                j += 1
                continue
                
            # 检查合并后是否超过最大长度
            potential_merge = merged + " " + next_sentence
            if len(potential_merge) <= max_length:
                merged = potential_merge
                j += 1
            else:
                # 如果合并会超过最大长度，停止合并
                break
        
        # 添加合并后的结果
        if merged:
            result.append(merged)
        
        # 更新索引
        i = j if j > i else i + 1
    
    return result


def split_text_into_segments(text: str, max_length: int = 300, logger: Optional[logging.Logger] = None) -> List[str]:
    """
    将文本拆分成适合TTS处理的段落
    
    参数:
        text: 要拆分的文本
        max_length: 每个段落的最大字符数
        logger: 可选的日志记录器
        
    返回:
        拆分后的段落列表
    """
    if not text or not text.strip():
        return []
    
    text = text.strip()
    
    # 如果文本长度不超过最大长度，直接返回
    if len(text) <= max_length:
        return [text]
    
    if logger:
        logger.debug(f"文本长度 {len(text)} 超过最大限制 {max_length}，开始拆分")
    
    # 首先按段落分割（双换行符）
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    
    segments = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_length:
            segments.append(paragraph)
            continue
        
        # 段落过长，进一步拆分
        # 1. 尝试按句子分割（句号、问号、感叹号）
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 2. 合并短句子，避免产生太多碎片
        sentences = merge_short_sentences(sentences, max_length, min_length=30)
        
        # 3. 处理仍然过长的句子
        for sentence in sentences:
            if len(sentence) <= max_length:
                segments.append(sentence)
            else:
                # 使用单词边界拆分
                word_segments = split_by_word_boundary(sentence, max_length)
                segments.extend(word_segments)
    
    # 清理空段落
    segments = [s.strip() for s in segments if s.strip()]
    
    if logger:
        logger.debug(f"文本拆分完成，共 {len(segments)} 个段落")
    
    return segments 