#!/usr/bin/env python3
"""
Shared Text Processing Utilities for SkyrimNet TTS Applications
Contains text analysis, tokenization, and splitting functions
"""

import re
from loguru import logger
from typing import Any, Dict, List, Optional


# =============================================================================
# TEXT LENGTH ANALYSIS AND VALIDATION
# =============================================================================

def check_text_length(text: str, model: Any, language: str = "en", char_limit: Optional[int] = None, max_tokens: Optional[int] = None) -> Dict:
    """
    Check if text exceeds model limits and recommend text splitting using actual tokenization.
    
    Args:
        text: Input text to check
        model: Loaded VoxCPM model (for tokenization)
        language: Language code for text
        char_limit: Custom character limit (uses tokenization-based limit if None)
        max_tokens: Custom token limit (uses model defaults if None)
        
    Returns:
        dict: {
            'should_split': bool,
            'char_count': int,
            'token_count': int, 
            'char_limit': int,
            'token_limit': int,
            'chars_per_token': float,
            'recommendation': str
        }
    """
    try:
        # Get actual token count using model's tokenizer
        tokens = model.tts_model.text_tokenizer(text)
        token_count = len(tokens)
        char_count = len(text)
        chars_per_token = char_count / token_count if token_count > 0 else 4.5
        
        # VoxCPM model limits from config
        model_limits = {
            'max_position_embeddings': 32768,  # Absolute theoretical limit
            'default_max_length': 4096,       # Model config default (verified from actual config)
            'conservative_limit': 2800,       # Practical limit for TTS quality (based on actual tokenizer behavior ~3.5 chars/token)
            'prompt_buffer': 200              # Reserve tokens for prompt text (reduced based on actual usage)
        }
        
        # Determine token limit
        if max_tokens is None:
            # Use conservative limit by default (leaves room for prompt + generation)
            max_tokens = model_limits['conservative_limit'] - model_limits['prompt_buffer']
        
        # Determine character limit based on tokenization
        if char_limit is None:
            char_limit = int(max_tokens * chars_per_token)
        
        # Determine if splitting is needed
        should_split = token_count > max_tokens
        
        # Generate recommendation
        if token_count <= model_limits['conservative_limit']:
            recommendation = "OK - Text length is optimal"
        elif token_count <= model_limits['default_max_length']:
            recommendation = "WARNING - Text is long but within limits - consider splitting for better performance"
        elif token_count <= model_limits['max_position_embeddings']:
            recommendation = "WARNING - Text exceeds recommended limits - splitting strongly recommended"
        else:
            recommendation = "ERROR - Text exceeds model capacity - must split before processing"
        logger.info(f"Text length check: {char_count} chars, {token_count} tokens (limit: {max_tokens} tokens). {recommendation}")
        return {
            'should_split': should_split,
            'char_count': char_count,
            'token_count': token_count,
            'char_limit': char_limit,
            'token_limit': max_tokens,
            'chars_per_token': chars_per_token,
            'recommendation': recommendation,
            'model_limits': model_limits
        }
        
    except Exception as e:
        logger.warning(f"Could not use tokenizer for text length check: {e}")
        # Fallback to character-based estimation
        char_count = len(text)
        estimated_chars_per_token = 3.5  # Based on actual VoxCPM tokenizer behavior
        estimated_tokens = char_count / estimated_chars_per_token
        
        if char_limit is None:
            char_limit = int(2600 * estimated_chars_per_token)  # ~9,100 chars (2800-200 buffer)
        if max_tokens is None:
            max_tokens = 2600  # Conservative fallback (2800-200 buffer)
        
        should_split = char_count > char_limit
        
        return {
            'should_split': should_split,
            'char_count': char_count,
            'token_count': int(estimated_tokens),
            'char_limit': char_limit,
            'token_limit': max_tokens,
            'chars_per_token': estimated_chars_per_token,
            'recommendation': "WARNING - Estimated limits (tokenizer unavailable)",
            'model_limits': {'fallback': True}
        }


# =============================================================================
# TEXT SPLITTING FUNCTIONS
# =============================================================================

def get_optimal_text_split_points(text: str, model: Any, max_tokens: int = 3200, use_regex_splitter: bool = True) -> List[str]:
    """
    Find optimal points to split text while respecting sentence boundaries.
    Uses a proven regex pattern for reliable sentence splitting.
    
    Args:
        text: Input text to split
        model: Loaded VoxCPM model (for tokenization)
        max_tokens: Maximum tokens per chunk
        use_regex_splitter: Whether to use the proven regex sentence splitter
        
    Returns:
        list: List of text chunks that respect model limits
    """
    # Check if splitting is needed
    length_check = check_text_length(text, model, max_tokens=max_tokens)
    
    if not length_check['should_split']:
        return [text]
    
    if use_regex_splitter:
        # Use the proven regex approach for reliable sentence splitting
        return _regex_text_split(text, model, max_tokens)
    else:
        # Fallback to simple sentence-based splitting
        return _simple_text_split(text, model, max_tokens)


def _regex_text_split(text: str, model: Any, max_tokens: int) -> List[str]:
    """
    Regex-based text splitting using a proven pattern for sentence boundaries.
    Handles abbreviations, quotes, and complex punctuation correctly.
    """
    # Proven regex pattern for sentence splitting that handles edge cases
    # (?<=[.?!])\s*(?![.\w"\'\d]|[,!]|\*)
    sentence_pattern = r'(?<=[.?!])\s*(?![.\w"\'\d]|[,!]|\*)'
    
    # Split into sentences using the regex
    sentences = [s.strip() for s in re.split(sentence_pattern, text) if s.strip()]
    
    # If no sentences found or pattern didn't work, fallback to simple split
    if len(sentences) <= 1:
        return _simple_text_split(text, model, max_tokens)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Test adding this sentence
        test_chunk = current_chunk + " " + sentence if current_chunk else sentence
        test_check = check_text_length(test_chunk.strip(), model, max_tokens=max_tokens)
        
        if test_check['should_split'] and current_chunk:
            # Current chunk is full, start new chunk
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk = test_chunk
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def _simple_text_split(text: str, model: Any, max_tokens: int) -> List[str]:
    """
    Simple sentence-based text splitting fallback.
    """
    # Split into sentences
    sentence_endings = r'[.!?]+\s+'
    sentences = re.split(sentence_endings, text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Test adding this sentence
        test_chunk = current_chunk + " " + sentence if current_chunk else sentence
        test_check = check_text_length(test_chunk.strip(), model, max_tokens=max_tokens)
        
        if test_check['should_split'] and current_chunk:
            # Current chunk is full, start new chunk
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk = test_chunk
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


# =============================================================================
# TEXT ANALYSIS UTILITIES
# =============================================================================

def estimate_token_count(text: str, chars_per_token: float = 3.5) -> int:
    """
    Estimate token count based on character length.
    
    Args:
        text: Input text
        chars_per_token: Average characters per token (default: 3.5 based on VoxCPM tokenizer)
        
    Returns:
        int: Estimated token count
    """
    return int(len(text) / chars_per_token)


def get_text_statistics(text: str, model: Optional[Any] = None) -> Dict:
    """
    Get comprehensive text statistics including tokenization if model is available.
    
    Args:
        text: Input text to analyze
        model: Optional VoxCPM model for accurate tokenization
        
    Returns:
        dict: Text statistics including char count, estimated/actual tokens, etc.
    """
    char_count = len(text)
    word_count = len(text.split())
    
    stats = {
        'char_count': char_count,
        'word_count': word_count,
        'estimated_tokens': estimate_token_count(text),
        'chars_per_word': char_count / word_count if word_count > 0 else 0
    }
    
    if model is not None:
        try:
            tokens = model.tts_model.text_tokenizer(text)
            actual_tokens = len(tokens)
            stats.update({
                'actual_tokens': actual_tokens,
                'chars_per_token': char_count / actual_tokens if actual_tokens > 0 else 3.5,
                'words_per_token': word_count / actual_tokens if actual_tokens > 0 else 1.0
            })
        except Exception as e:
            logger.warning(f"Could not get actual token count: {e}")
    
    return stats