#!/usr/bin/env python3
"""
Shared Model Management Module for SkyrimNet TTS Applications
Contains common model loading, initialization, and management functions
"""

import os
import torch
from pathlib import Path
from loguru import logger
from typing import Optional, Any

# VoxCPM imports
try:
    from .voxcpm.core import VoxCPM
    from .shared_whisper_utils import initialize_whisper_model
except ImportError:
    from voxcpm.core import VoxCPM
    from shared_whisper_utils import initialize_whisper_model

# Local imports for cache initialization - Handle both direct and module execution
try:
    from .shared_config import get_models_dir
except ImportError:
    from shared_config import get_models_dir


# =============================================================================
# MODEL LOADING AND MANAGEMENT
# =============================================================================

def load_model(model_name="openbmb/VoxCPM-0.5B", use_cpu=False, optimize=True):
    """
    Load VoxCPM model with configuration
    Downloads to TTS_HOME/voxcpm directory.
    
    Args:
        model_name: Name/path of the model to load (default: "openbmb/VoxCPM-0.5B")
        use_cpu: Whether to force CPU mode instead of CUDA
        optimize: Whether to enable torch.compile optimization
        
    Returns:
        VoxCPM: Loaded and configured model
        
    Raises:
        Exception: If model loading fails
    """
    logger.info(f"Loading VoxCPM model: {model_name}, use_cpu: {use_cpu}, optimize: {optimize}")
    
    try:
        # Use TTS_HOME for model cache
        models_dir = get_models_dir()
        voxcpm_cache = str(Path(models_dir) / "voxcpm")
        
        logger.info(f"Model cache directory: {voxcpm_cache}")
        
        # Load VoxCPM model from pretrained
        model = VoxCPM.from_pretrained(
            hf_model_id=model_name,
            cache_dir=voxcpm_cache,
            optimize=optimize
        )
        
        # Initialize Whisper model for reference text transcription
        logger.info("Initializing Whisper model for reference audio transcription...")
        device = "cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        initialize_whisper_model(device=device)
        
        logger.info("VoxCPM model loading completed successfully")
        return model
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Failed to load model '{model_name}': {str(e)}")
        raise


def setup_model_seed(seed=None, randomize=False):
    """
    Set up random seed for reproducible model inference
    
    Args:
        seed: Random seed (int). If None, generates random seed
        
    Returns:
        int: The seed that was set
    """
    if randomize:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    elif seed is None:
        seed = 20250527

    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    logger.debug(f"Model seed set to: {seed}")
    return seed


def get_model_device(model):
    """
    Get the device where the model is currently loaded
    
    Args:
        model: The loaded VoxCPM model
        
    Returns:
        torch.device: Device where model is located
    """
    # VoxCPM uses tts_model internally
    try:
        if hasattr(model, 'tts_model'):
            return next(model.tts_model.parameters()).device
        return torch.device("cpu")
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def validate_model_state(model):
    """
    Validate that model is properly loaded and ready for inference
    
    Args:
        model: The model to validate
        
    Returns:
        bool: True if model is ready
        
    Raises:
        RuntimeError: If model is not properly initialized
    """
    if model is None:
        raise RuntimeError("Model is None - not loaded")
    
    try:
        device = get_model_device(model)
        logger.debug(f"Model validation: device={device}")
        return True
    except Exception as e:
        raise RuntimeError(f"Model validation failed: {str(e)}")


# =============================================================================
# MODEL INFERENCE HELPERS
# =============================================================================


def prepare_inference_params(cfg_value=2.0, inference_timesteps=10, 
                           normalize=True, denoise=True, **kwargs):
    """
    Prepare and validate inference parameters for VoxCPM
    
    Args:
        cfg_value: Guidance scale for generation (default: 2.0)
        inference_timesteps: Number of inference steps (default: 10)
        normalize: Whether to normalize text (default: True)
        denoise: Whether to denoise reference audio (default: True)
        **kwargs: Additional parameters
        
    Returns:
        dict: Validated inference parameters
    """
    params = {
        "cfg_value": float(cfg_value),
        "inference_timesteps": int(inference_timesteps),
        "normalize": bool(normalize),
        "denoise": bool(denoise)
    }
    
    # Add any additional parameters
    params.update(kwargs)
    
    logger.debug(f"VoxCPM inference parameters: {params}")
    return params


def initialize_model_with_cache(
    use_cpu: bool = False, 
    seed: Optional[int] = None,
    validate: bool = True,
    optimize: bool = True
) -> Any:
    """
    Complete model initialization with caching setup.
    
    Args:
        use_cpu: Whether to use CPU instead of CUDA
        seed: Random seed for reproducibility (optional)
        validate: Whether to validate model state after loading
        optimize: Whether to enable torch.compile optimization
        
    Returns:
        Loaded and initialized VoxCPM model
        
    Raises:
        Exception: If model loading fails
    """
    logger.info("Starting VoxCPM model initialization...")
    
    try:
        # Load VoxCPM model
        model = load_model(use_cpu=use_cpu, optimize=optimize)
        
        setup_model_seed(seed)
        
        # Validate model state
        if validate:
            validate_model_state(model)
        
        # Initialize prompt cache using the same pattern as XTTS latent cache
        try:
            from .shared_cache_utils import init_prompt_cache  
        except ImportError:
            from shared_cache_utils import init_prompt_cache
        init_prompt_cache(model=model)
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to initialize VoxCPM model: {e}")
        raise