"""
PyInstaller runtime hook to fix torch.compile/dynamo/inductor in frozen apps.

Problem: torch._inductor.cudagraph_trees uses thread-local storage (TLS) that may
not be properly initialized in PyInstaller's frozen environment.

Fix: Set environment variables and provide a patch function to call later.
"""

import os
import sys

# Check if running as frozen PyInstaller executable
if getattr(sys, 'frozen', False):
    # Get temp directory for caches
    temp_dir = os.environ.get('TEMP', os.environ.get('TMP', '/tmp'))
    
    # Set up proper cache directories for torch compile artifacts
    torch_cache = os.path.join(temp_dir, 'torch_compile_cache')
    os.makedirs(torch_cache, exist_ok=True)
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = torch_cache
    
    # Configure triton cache directory  
    triton_cache = os.path.join(temp_dir, 'triton_cache')
    os.makedirs(triton_cache, exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = triton_cache
    
    print(f"[PyInstaller] torch.compile cache: {torch_cache}")
    print(f"[PyInstaller] triton cache: {triton_cache}")

