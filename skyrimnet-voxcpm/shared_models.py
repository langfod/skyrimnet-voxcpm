#!/usr/bin/env python3
"""
Shared Model Management Module for SkyrimNet TTS Applications
Contains common model loading, initialization, and management functions
"""

import os
import torch
from loguru import logger
from typing import TYPE_CHECKING, Optional

# VoxCPM imports
try:
    from .shared_config import get_models_dir
    from .shared_whisper_utils import initialize_whisper_model
    from .voxcpm.core import VoxCPM
except ImportError:
    from shared_whisper_utils import initialize_whisper_model
    from shared_config import get_models_dir  
    from voxcpm.core import VoxCPM

# =============================================================================
# MODEL LOADING AND MANAGEMENT
# =============================================================================
def _resolve_model_dir(HF_REPO_ID) -> str:
    """
    Resolve model directory:
    1) Use local checkpoint directory if exists
    2) If HF_REPO_ID env is set, download into models/{repo}
    3) Fallback to 'models'
    """
    models_dir = get_models_dir()
    repo_id = HF_REPO_ID.strip()
    if len(repo_id) > 0:
        target_dir = os.path.join(models_dir, repo_id.replace("/", "__"))
        if not os.path.isdir(target_dir):
            try:
                from huggingface_hub import snapshot_download  # type: ignore
                os.makedirs(target_dir, exist_ok=True)
                print(f"Downloading model from HF repo '{repo_id}' to '{target_dir}' ...")
                models_dir = snapshot_download(repo_id=repo_id, local_dir=target_dir, local_dir_use_symlinks=False)
                print(f"Model downloaded to '{models_dir}'")
            except Exception as e:
                print(f"Warning: HF download failed: {e}. Falling back to 'data'.")
                exit(1)
                return models_dir
        return target_dir
    return models_dir


def _resolve_modelscope_dir(HF_REPO_ID) -> str:
    """
    Resolve model directory:
    1) Use local checkpoint directory if exists
    2) If HF_REPO_ID env is set, download into models/{repo}
    3) Fallback to 'models'
    """
    models_dir = get_models_dir()
    repo_id = HF_REPO_ID.strip()
    if len(repo_id) > 0:
        target_dir = os.path.join(models_dir, repo_id.replace("/", "__"))
        if not os.path.isdir(target_dir):
            try:
                from modelscope import snapshot_download   # type: ignore
                #os.makedirs(target_dir, exist_ok=True)
                print(f"Downloading model from ModelScope repo '{repo_id}' to '{target_dir}' ...")
                models_dir = snapshot_download(repo_id=repo_id, local_dir=target_dir)
            except Exception as e:
                print(f"Warning: ModelScope download failed: {e}. Falling back to 'data'.")
                return models_dir
        return target_dir
    return models_dir


def load_model(model_name="openbmb/VoxCPM1.5", device="auto", optimize=True) -> "VoxCPM":
    """
    Load VoxCPM model with configuration
    Downloads to TTS_HOME/voxcpm directory.
    
    Args:
        model_name: Name/path of the model to load (default: "openbmb/VoxCPM-0.5B")
        device: Device to load the model on ("auto", "cpu", "cuda:0", etc.)
        optimize: Whether to enable torch.compile optimization
        
    Returns:
        VoxCPM: Loaded and configured model
        
    Raises:
        Exception: If model loading fails
    """
    device = device if device is not None else ("cpu" if not torch.cuda.is_available() else "cuda:0")
    logger.info(f"Loading VoxCPM model: {model_name}, device: {device}, optimize: {optimize}")
    
    try:
        denoiser_models_dir = _resolve_modelscope_dir("iic/speech_zipenhancer_ans_multiloss_16k_base")
        logger.info(f"Denoiser model directory: {denoiser_models_dir}")
        # Use TTS_HOME for model cache
        models_dir = _resolve_model_dir(model_name)
        
        logger.info(f"Model cache directory: {models_dir}")
        
        # Load VoxCPM model from pretrained
        #model = VoxCPM(voxcpm_model_path=models_dir,zipenhancer_model_path="models\\iic_speech_zipenhancer_ans_multiloss_16k_base")
        torch.set_default_device(device if device != "auto" else ("cpu" if not torch.cuda.is_available() else "cuda:0"))
        model = VoxCPM(voxcpm_model_path=models_dir,enable_denoiser=False,zipenhancer_model_path=denoiser_models_dir)

        # Initialize Whisper model for reference text transcription
        logger.info("Initializing Whisper model for reference audio transcription...")
        device = device if device != "auto" else ("cpu" if not torch.cuda.is_available() else "cuda:0")
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
        cfg_value: Guidance scale for generation (default: 1.6)
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
    device: str = "auto",
    seed: Optional[int] = 20250527,
    validate: bool = True,
    optimize: bool = True
) -> "VoxCPM":
    """
    Complete model initialization with caching setup.
    
    Args:
        device: Device to load the model on ("auto", "cpu", "cuda:0", etc.)
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
        model = load_model(device=device, optimize=optimize)
        
        setup_model_seed(seed)
        
        # Validate model state
        if validate:
            validate_model_state(model)
        
        try:
            from .shared_cache_utils import init_prompt_cache  
        except ImportError:
            from shared_cache_utils import init_prompt_cache
        init_prompt_cache(model=model)
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to initialize VoxCPM model: {e}")
        raise