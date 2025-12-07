#!/usr/bin/env python3
"""
SkyrimNet TTS Package
A unified Text-to-Speech application with FastAPI backend and Gradio UI
"""

__version__ = "1.0.0"
__author__ = "SkyrimNet Team"
__description__ = "Unified VoXCPM application with API and web interface"

import sys

# Handle imports - try relative first, fall back to absolute for PyInstaller
try:
    from .shared_config import setup_environment
    from .shared_models import load_model
    from .shared_args import parse_api_args, parse_gradio_args
except ImportError:
    # PyInstaller or direct execution - use absolute imports
    from shared_config import setup_environment
    from shared_models import load_model
    from shared_args import parse_api_args, parse_gradio_args

__all__ = [
    "setup_environment", 
    "load_model",
    "parse_api_args",
    "parse_gradio_args",
]
