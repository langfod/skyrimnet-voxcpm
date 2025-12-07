#!/usr/bin/env python3
"""
Shared Configuration Module for SkyrimNet TTS Applications
Contains common environment setup, constants, and configuration settings
"""

import os
import sys

import torch


# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# =============================================================================
# COMMON CONSTANTS
# =============================================================================

# Models directory for VoxCPM and Whisper downloads
MODELS_DIR = "models"


def setup_environment():
    """Setup common environment variables and system configuration"""
    
    # Fix torch.compile C++ compilation issues on Windows
    if sys.platform == "win32":
        os.environ["TORCH_COMPILE_CPP_FORCE_X64"] = "1"
        os.environ["DISTUTILS_USE_SDK"] = "1" 
        os.environ["MSSdk"] = "1"
    # Enable TF32 for better performance on Ampere+ GPUs
    torch.backends.fp32_precision = "tf32"
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.set_float32_matmul_precision("high")
    


    import warnings
    warnings.filterwarnings("ignore", module='Setuptools.*', append=True)
    warnings.filterwarnings("ignore", module='numbpysbd.*', append=True)
    warnings.filterwarnings("ignore", module='jieba.*', append=True)
    warnings.filterwarnings("ignore", module='jamo.*', append=True)
    warnings.filterwarnings("ignore", module='g2pkk.*', append=True)


def get_models_dir():
    """
    Get the models directory path.
    
    Returns:
        str: Path to models directory
    """
    from pathlib import Path
    
    models_dir = Path(MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    return str(models_dir.absolute())


# =============================================================================
# ADDITIONAL CONSTANTS
# =============================================================================

# Supported language codes for TTS

# Cache configuration defaults
DEFAULT_CACHE_CONFIG = {
    "ENABLE_DISK_CACHE": True,
    "ENABLE_MEMORY_CACHE": True
}

# Default TTS inference parameters for VoxCPM
DEFAULT_TTS_PARAMS = {
    "CFG_VALUE": 1.6,
    "INFERENCE_TIMESTEPS": 10,
    "NORMALIZE": False,
    "DENOISE": False,
    "MAX_LENGTH": 4096
}

# Text splitting threshold per language (characters)
DEFAULT_CHAR_LIMITS = 250


# =============================================================================
# CONFIGURATION HELPERS
# =============================================================================


def get_cache_config(enable_disk=None, enable_memory=None):
    """
    Get cache configuration with optional overrides
    
    Args:
        enable_disk: Override for disk cache (None to use default)
        enable_memory: Override for memory cache (None to use default)
        
    Returns:
        dict: Cache configuration
    """
    config = DEFAULT_CACHE_CONFIG.copy()
    
    if enable_disk is not None:
        config["ENABLE_DISK_CACHE"] = enable_disk
    
    if enable_memory is not None:
        config["ENABLE_MEMORY_CACHE"] = enable_memory
    
    return config