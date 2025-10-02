# =============================================================================
import torch
import numpy as np
import time
from loguru import logger
from typing import Optional, Any
from pathlib import Path
# Local imports - Handle both direct and module execution
try:
    from .shared_cache_utils import get_reference_audio_and_text, save_torchaudio_wav
    from .shared_config import DEFAULT_TTS_PARAMS
except ImportError:
    from shared_cache_utils import get_reference_audio_and_text, save_torchaudio_wav
    from shared_config import DEFAULT_TTS_PARAMS
# =============================================================================

def generate_audio_file(

    model: Any,
    language: str,
    speaker_wav: str,
    text: str,
    uuid: Optional[int] = None,
    stream: bool = False,
    **inference_kwargs
) -> Path:
    """
    Generate audio file using VoxCPM cached prompt system (our new approach).
    
    This function uses the cached prompt approach similar to XTTS latents for efficiency.
    Optimized to work directly with tensors and minimize device transfers.
    
    Args:
        model: The VoxCPM model to use for inference
        language: Language code for synthesis
        speaker_wav: Speaker reference (file path or speaker name)
        text: Text to synthesize
        uuid: Optional UUID for caching and file naming
        stream: Whether to use streaming mode
        **inference_kwargs: Additional parameters for model._generate_cached()
            Supported kwargs:
            - cfg_value: Guidance scale (default: 2.0)
            - inference_timesteps: Number of timesteps (default: 10)
            - normalize: Whether to normalize text (default: True)
            - max_length: Maximum generation length (default: 4096)
    
    Returns:
        Path: Path to the generated audio file
    """
    
    func_start_time = time.perf_counter()

    logger.info(f"Generating cached audio for text='{text[:50]}...', speaker='{Path(speaker_wav).stem}', language='{language}', uuid={uuid}")
    logger.debug(f"Inference kwargs: {inference_kwargs}")
    # Get prompt cache using the cache manager (returns tensor-based cache)
    try:
        # Import here to avoid circular imports
        try:
            from .shared_cache_utils import get_cached_prompt_cache
        except ImportError:
            from shared_cache_utils import get_cached_prompt_cache
        
        prompt_cache = get_cached_prompt_cache(
            model=model,
            language=language,
            speaker_audio=speaker_wav,
            speaker_audio_uuid=uuid
        )
        
        if prompt_cache is None:
            raise ValueError(f"Failed to build prompt cache for speaker: {speaker_wav}")
    
    except Exception as e:
        logger.error(f"Failed to get prompt cache for {speaker_wav}: {e}")
        raise
    
    
    # Prepare inference parameters with VoxCPM defaults
    inference_params = {
        'text': text,
        'prompt_cache': prompt_cache,
        'cfg_value': 2.0,
        'inference_timesteps': 10,
        'normalize': True,
        'max_length': 4096,
        'streaming': stream
    }
    
    # Override defaults with provided kwargs
    for key, value in inference_kwargs.items():
        if key in inference_params and value is not None:
            inference_params[key] = value
        elif key not in inference_params:
            logger.warning(f"Ignoring unknown inference parameter for VoxCPM cached generation: {key}={value}")
    
    # Generate audio using the tensor-optimized method
    try:
        # Call the tensor-optimized wrapper method to avoid numpy conversion
        generate_result = model._generate_with_prompt_cache_tensor(**inference_params)     
        wav_tensor = next(generate_result)
        
    except Exception as e:
        logger.error(f"VoxCPM cached generation failed: {e}")
        raise
    
    # VoxCPM outputs at 16kHz
    output_sample_rate = model.tts_model.sample_rate    
    
    # Save audio file (wav_tensor is already a torch tensor)
    wav_out_path = save_torchaudio_wav(
        wav_tensor=wav_tensor,
        sr=output_sample_rate,
        audio_path=speaker_wav,
        uuid=uuid
    )
    wav_length_s = wav_tensor.size(1) / output_sample_rate

    func_end_time = time.perf_counter()
    total_duration_s = func_end_time - func_start_time
    
    if speaker_wav:
        logger.info(
            f"Generated cached audio saved to {wav_out_path.name} for {Path(speaker_wav).stem} "
            f"length: {wav_length_s:.2f}s, execution time: {total_duration_s:.2f}s, "
            f"Speed: {wav_length_s/total_duration_s:.2f}x"
        )
    else:
        logger.info(f"Cached audio generation completed in {total_duration_s:.2f}s")

    del wav_tensor
    
    return wav_out_path
