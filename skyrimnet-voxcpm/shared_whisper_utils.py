#!/usr/bin/env python3
"""
Shared Whisper Utilities Module for SkyrimNet TTS Applications
Handles audio transcription for VoxCPM reference text generation
"""

import time
from pathlib import Path
from typing import Optional
import torch
from faster_whisper import WhisperModel
from loguru import logger

# Local imports - Handle both direct and module execution
try:
    from .shared_config import get_models_dir
except ImportError:
    from shared_config import get_models_dir


# Global Whisper model instance
# Whisper Configuration
# Using distil-whisper CT2 format for faster inference with good quality
# Note: The -ct2 suffix indicates this is pre-converted to CTranslate2 format
WHISPER_MODEL_NAME = "Numbat/faster-skyrim-whisper-base.en"
#WHISPER_MODEL_NAME = "distil-whisper/distil-large-v3.5-ct2"
WHISPER_ENGINE: Optional[WhisperModel] = None


def initialize_whisper_model(device: str = None) -> WhisperModel:
    """
    Initialize Whisper model for transcription.
    Downloads CTranslate2 format model to models/whisper directory.
    
    Args:
        device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
        
    Returns:
        WhisperModel: Initialized Whisper model
    """
    global WHISPER_ENGINE
    
    if WHISPER_ENGINE is None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        device , device_index = device.split(":") if ":" in device else (device, 0)
        device_index = int(device_index)
        compute_type = "bfloat16" if device == "cuda" else "int8"
        
        # Use models directory for Whisper cache
        # Note: faster-whisper will download the CT2 format to this location
        models_dir = get_models_dir()
        whisper_cache = str(Path(models_dir) / "whisper")
        
        logger.info(f"Loading Whisper model '{WHISPER_MODEL_NAME}' on {device}...")
        logger.info(f"Model cache directory: {whisper_cache}")
        
        # Create cache directory
        Path(whisper_cache).mkdir(parents=True, exist_ok=True)
        
        try:
            WHISPER_ENGINE = WhisperModel(
                WHISPER_MODEL_NAME,
                device=device,
                device_index=device_index,
                compute_type=compute_type,
                download_root=whisper_cache
            )
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            logger.info("Attempting to download Whisper model...")
            # Let faster-whisper handle the download
            WHISPER_ENGINE = WhisperModel(
                WHISPER_MODEL_NAME,
                device=device,
                compute_type=compute_type,
                download_root=whisper_cache
            )
        logger.info("Whisper model loaded successfully")
    
    return WHISPER_ENGINE


def transcribe_audio_with_whisper(
    audio_path: str,
    language: str = "en",
    use_cache: bool = True
) -> str:
    """
    Transcribe audio using Whisper model with optional caching.
    
    Args:
        audio_path: Path to audio file to transcribe
        language: Language code for transcription
        use_cache: Whether to use cached transcriptions (future enhancement)
        
    Returns:
        str: Transcribed text
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        Exception: If transcription fails
    """
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Initialize Whisper if needed
    if WHISPER_ENGINE is None:
        initialize_whisper_model()
    
    filename = Path(audio_path).stem
    logger.info(f"Transcribing audio file: {filename}")
    start_time = time.perf_counter()
    
    try:
        # Transcribe audio
        texts = []
        segments, info = WHISPER_ENGINE.transcribe(
            audio_path,
            beam_size=10,
            vad_filter=True,
            without_timestamps=True,
            language=language if language != "zh-cn" else "zh"
        )
        
        for segment in segments:
            texts.append(segment.text.strip())
        
        transcription = ' '.join(texts).strip()
        
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        
        logger.info(
            f"Transcription completed in {elapsed_ms:.2f}ms: "
            f"'{transcription[:100]}{'...' if len(transcription) > 100 else ''}'"
        )
        
        return transcription
        
    except Exception as e:
        logger.error(f"Failed to transcribe audio '{filename}': {str(e)}")
        # Return a fallback message
        return "This is the voice you should use for the generation."


def get_whisper_device() -> str:
    """
    Get the device where Whisper model is loaded.
    
    Returns:
        str: Device name ('cuda' or 'cpu')
    """
    if WHISPER_ENGINE is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    # Whisper uses device attribute
    return getattr(WHISPER_ENGINE, 'device', 'cpu')
