# -*- mode: python ; coding: utf-8 -*-
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules




datas = []
hiddenimports = []

# =============================================================================
# CRITICAL: Include all application source files
# =============================================================================
# Manually add all Python source files from skyrimnet-voxcpm package
# Copy to root of bundle so absolute imports work (shared_args, shared_models, etc.)
import glob

app_src_dir = Path("skyrimnet-voxcpm")
for py_file in app_src_dir.glob("*.py"):
    # Add each .py file to the ROOT of the bundle (not a subdirectory)
    # This allows "from shared_args import ..." to work in PyInstaller
    datas.append((str(py_file), "."))

# Also include the voxcpm subdirectory at root level
voxcpm_dir = app_src_dir / "voxcpm"
if voxcpm_dir.exists():
    for py_file in voxcpm_dir.rglob("*.py"):
        rel_path = py_file.relative_to(app_src_dir)
        datas.append((str(py_file), str(rel_path.parent)))

# Include speakers directory at root level
#speakers_dir = app_src_dir / "speakers"
#if speakers_dir.exists():
#    for file in speakers_dir.rglob("*"):
#        if file.is_file():
#            rel_path = file.relative_to(app_src_dir)
#            datas.append((str(file), str(rel_path.parent)))

print(f"Added {len([d for d in datas if d[1] == '.' or 'voxcpm' in d[1] or 'speakers' in d[1]])} application files")

# Core application data files with better exclusions
#datas += collect_data_files("gradio_client", excludes=[
#    "*.md", "*.txt", "*.rst", "test*", "*test*", "example*", "*example*"
#])
#datas += collect_data_files("gradio", excludes=[
#    "*.md", "*.txt", "*.rst", "test*", "*test*", "demo*"
#])

# Keep these smaller libraries as-is (minimal benefit to optimize)
#datas += collect_data_files("groovy")
#datas += collect_data_files("safehttpx")

# CRITICAL: Include faster_whisper and ctranslate2 data files
faster_whisper_datas = collect_data_files("faster_whisper", excludes=[
    "test*", "*test*", "tests/*", "*/tests/*",
    "example*", "*example*", "examples/*", "*/examples/*",
    "*.md", "*.txt", "*.rst", "docs/*", "*/docs/*"
])
datas += faster_whisper_datas

# ctranslate2 is required by faster_whisper
try:
    datas += collect_data_files("ctranslate2", excludes=[
        "test*", "*test*", "tests/*", "*/tests/*",
        "example*", "*example*", "examples/*", "*/examples/*",
        "*.md", "*.txt", "*.rst", "docs/*", "*/docs/*"
    ])
except Exception as e:
    print(f"Warning: Could not collect ctranslate2 data files: {e}")

# CRITICAL: Include comprehensive spaCy data files
import os

MODEL_SUPPORTED_LANGS = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja"] 


#print(f"Total spaCy data files added: {len([d for d in datas if 'spacy' in d[1]])}")

# Include setuptools data files (needed for jaraco.text and other components)
datas += collect_data_files("setuptools", excludes=[
    "test*", "*test*", "tests/*", "*/tests/*",
    "example*", "*example*", "examples/*", "*/examples/*",
    "docs/*", "*/docs/*"
])


# CRITICAL: PyTorch with enhanced exclusions to prevent bloat
datas += collect_data_files("torch", excludes=[
    "*.cpp", "*.cu", "*.c", "*.h", "*.cuh",
    "test*", "*test*", "tests/*", "*/tests/*",
    "example*", "*example*", "examples/*", "*/examples/*",
    "*.md", "*.txt", "*.rst", "docs/*", "*/docs/*",
    # CRITICAL: Exclude massive library files that cause bloat
    #"*.lib", "lib/*.lib", "libs/*.lib",
    # Exclude huge CUDA runtime libraries
    "lib/libtorch_cuda.so*", "lib/libtorch_cpu.a", "lib/dnnl.lib",
    # Multi-GPU solvers (not needed for single-GPU TTS)
    "lib/cusolverMg64_11.dll",   # 179 MB - Multi-GPU solvers
    # NOTE: cuDNN and other CUDA DLL exclusions are handled in post-processing
    # because collect_data_files() doesn't reliably exclude binary dependencies
])

datas += collect_data_files("torchaudio", excludes=[
    "*.cpp", "*.cu", "*.c", "*.h", "*.cuh",
    "test*", "*test*", "tests/*", "*/tests/*",
    "example*", "*example*", "examples/*", "*/examples/*",
    "*.md", "*.txt", "*.rst", "docs/*", "*/docs/*"
])


datas += collect_data_files("transformers", excludes=[
    "test*", "*test*", "tests/*", "*/tests/*",
    "example*", "*example*", "examples/*", "*/examples/*",
    "*.md", "*.txt", "*.rst", "docs/*", "*/docs/*"
])

# CRITICAL: Include Triton backend data files (driver.py and other backend modules)
# Exclude AMD backend entirely - we only need NVIDIA CUDA backend
datas += collect_data_files("triton", excludes=[
    "test*", "*test*", "tests/*", "*/tests/*",
    "example*", "*example*", "examples/*", "*/examples/*",
    "*.md", "*.txt", "*.rst", "docs/*", "*/docs/*",
    "backends/amd/*",  # Exclude AMD backend
    "backends/amd"
])

# CRITICAL: Include Python development headers for Triton JIT compilation
import sys
import sysconfig
from pathlib import Path

# Get Python installation paths
python_base = Path(sys.base_prefix)
python_include = Path(sysconfig.get_path('include'))
python_stdlib = Path(sysconfig.get_path('stdlib'))

# Include Python headers (Python.h and related files)
if python_include.exists():
    for header_file in python_include.rglob('*.h'):
        rel_path = header_file.relative_to(python_include)
        datas.append((str(header_file), f'include/{rel_path.parent}'))
    print(f"Added Python headers from: {python_include}")

# Include Python libs directory (python3X.lib for linking)
python_libs = python_base / 'libs'
if python_libs.exists():
    for lib_file in python_libs.glob('*.lib'):
        datas.append((str(lib_file), 'libs'))
    print(f"Added Python libs from: {python_libs}")

# Include essential distutils files (needed for compilation)
try:
    datas += collect_data_files("distutils", excludes=[
        "test*", "*test*", "tests/*", "*/tests/*",
        "example*", "*example*", "examples/*", "*/examples/*"
    ])
except:
    # distutils might be built-in, try setuptools._distutils
    try:
        datas += collect_data_files("setuptools._distutils", excludes=[
            "test*", "*test*", "tests/*", "*/tests/*"
        ])
    except:
        pass


# Essential PyTorch modules only
hiddenimports += collect_submodules('torch.nn.functional')


# Fix torch._dynamo import issues (keep minimal)
hiddenimports += collect_submodules('torch._dynamo.polyfills')

# Fix transformers import issues (reduce scope)
hiddenimports += collect_submodules('transformers.generation.utils')
hiddenimports += collect_submodules('transformers.utils')

# CRITICAL: Include faster_whisper and its dependencies
hiddenimports += collect_submodules('faster_whisper')
hiddenimports += [
    'faster_whisper',
    'faster_whisper.transcribe',
    'faster_whisper.audio',
    'faster_whisper.feature_extractor',
    'faster_whisper.tokenizer',
    'faster_whisper.vad',
    'ctranslate2',
]

# CRITICAL: Include FastAPI and its dependencies
hiddenimports += collect_submodules('fastapi')
hiddenimports += collect_submodules('starlette')
hiddenimports += [
    'fastapi',
    'fastapi.middleware',
    'fastapi.middleware.cors',
    'fastapi.responses',
    'fastapi.staticfiles',
    'starlette.middleware',
    'starlette.middleware.cors',
    'starlette.responses',
    'uvicorn',
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
]

# Application modules - use underscores for valid Python module names
hiddenimports += [
    "shared_args",
    "shared_app_utils",
    "shared_audio_utils",
    "shared_cache_utils",
    "shared_models",
    "shared_whisper_utils",
]


# Fix specific import errors (minimal set)
hiddenimports += [
    'torch._dynamo.polyfills.fx',
    # Triton - comprehensive imports for JIT and NVIDIA driver
    "triton",
    "triton.compiler",
    "triton.compiler.compiler",
    "triton.tools",
    "triton.language",
    "triton.runtime",
    "triton.runtime.jit",
    "triton.runtime.driver",
    "triton.runtime.autotuner",
    "triton.backends",
    "triton.backends.nvidia",
    "triton.backends.nvidia.driver",
    "triton.backends.nvidia.compiler",
]

# CRITICAL: Include compilation modules for Triton JIT
hiddenimports += [
    'distutils',
    'distutils.util',
    'distutils.spawn',
    'distutils.version',
    'setuptools._distutils',
    'sysconfig',
    'subprocess',
    'tempfile',
    # Platform-specific compilation support
    'msvcrt',  # Windows-specific
]

# =============================================================================
# EXCLUDE PROBLEMATIC MODULES (from original)
# =============================================================================

excludedimports = [
    'ninja',
    #'torch.utils.cpp_extension',
    # Exclude AMD backend (we only use NVIDIA CUDA)
    'triton.backends.amd', 
]

# =============================================================================
# ANALYSIS (Keep original configuration mostly)
# =============================================================================

a = Analysis(
    ["skyrimnet-voxcpm\\skyrimnet_api.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=["pyinstaller-hooks"],
    hooksconfig={},
    runtime_hooks=[
        # CRITICAL: Disable dynamo/inductor FIRST before any torch imports
        "pyinstaller-hooks/rthook_disable_torch_dynamo.py",
        "pyinstaller-hooks/rthook_disable_typeguard.py",
        "pyinstaller-hooks/rthook_setup_cuda_path.py",
        "pyinstaller-hooks/rthook_triton_nvidia_only.py",
    ],
    excludes=excludedimports,
    noarchive=False,
    optimize=1,  # Conservative optimization
    module_collection_mode={ 
        #'gradio': 'py+pyz',
        'torch': 'py+pyz',
        # CRITICAL: torch._dynamo needs source files for introspection
        'torch._dynamo': 'py',
        'torch._dynamo.polyfills': 'py',
        'torch._dynamo.variables': 'py',
        'torch._inductor': 'py',
        'torch.compiler': 'py',
        # Triton itself needs source files for JIT - include all subpackages
        'triton': 'py',
        'triton.runtime': 'py',
        'triton.runtime.jit': 'py',
        'triton.runtime.driver': 'py',
        'triton.runtime.autotuner': 'py',
        'triton.language': 'py',
        'triton.compiler': 'py',
        'triton.backends': 'py',
        'triton.backends.nvidia': 'py',
        'triton.backends.nvidia.driver': 'py',
    },
    cipher=None,
    upx=True,
)

# =============================================================================
# NVIDIA CUDA DLL EXCLUSION - Remove system-provided CUDA libraries
# =============================================================================
# These DLLs will be loaded from the system CUDA installation
# CUDA_PATH environment variable points to: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
# DLLs are available in: %CUDA_PATH%\bin\

cuda_dlls_to_exclude = [
    # ULTRA-CONSERVATIVE APPROACH: Only exclude the absolutely safe CUDA runtime DLL
    # Testing shows shm.dll loading issues - reducing exclusions to minimal set
    
    # CUDA Runtime (available in system CUDA installation)
    'cudart64_12.dll',                          # 0.6 MB - CUDA Runtime - SAFE TO EXCLUDE
    
    # Temporarily removing other exclusions to debug shm.dll loading issue
    # Will re-add after confirming application starts correctly
    
    # NOTE: Keep these PyTorch-required DLLs that were previously excluded:
    'cublas64_12.dll', # (97.8 MB) - Required by torch_cuda.dll
    'cublaslt64_12.dll', # (638 MB) - Required by torch_cuda.dll
    'cufft64_11.dll', # (274 MB) - Required by torch_cuda.dll
    'cufftw64_11.dll', # (0.2 MB) - FFTW Interface
    'curand64_10.dll', # (75.5 MB) - Random Number Generation
    'cusolver64_11.dll', # (270 MB) - Required by torch_cuda.dll
    'cusparse64_12.dll', # (455.4 MB) - Required by torch_cuda.dll
    'nvrtc64_120_0.dll', # (85.7 MB) - Runtime Compilation
    # - cudnn64_9.dll (0.3 MB) - Required by torch_cuda.dll
    # - cudnn_cnn64_9.dll (4.4 MB) - Core CNN operations
    # - cudnn_ops64_9.dll (120.6 MB) - Core operations
    # - cudnn_engines_runtime_compiled64_9.dll (19.3 MB) - Runtime engines
    # - cudnn_graph64_9.dll (2.3 MB) - Graph operations
]

# Additional post-processing to remove bloat and CUDA system libraries
a.datas = [x for x in a.datas if not any([
    # Remove .lib files (redundant with collect_data_files exclusions, but some may slip through)
    x[0].lower().endswith('.lib') and 'dnnl' in x[0].lower(),
    x[0].lower().endswith('.lib') and any(huge in x[0].lower() for huge in ['cublas', 'cudnn', 'cufft', 'cusolver']),
    
    # CRITICAL: Exclude NVIDIA CUDA system DLLs - users will have these installed
    any(cuda_dll.lower() in x[0].lower() for cuda_dll in cuda_dlls_to_exclude),
])]

# CRITICAL: Remove NVIDIA CUDA DLLs from binaries as well (they get pulled in as binary dependencies)
a.binaries = [x for x in a.binaries if not any([
    # Exclude all NVIDIA CUDA system DLLs - users will have these installed via CUDA toolkit
    any(cuda_dll.lower() in x[0].lower() for cuda_dll in cuda_dlls_to_exclude),
])]

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# =============================================================================
# EXECUTABLE CONFIGURATION (Keep original structure)
# =============================================================================

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Skyrimnet-VoxCPM',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,  
    upx=False,    # Disable UPX for now to avoid issues
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    optimize=2,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,   # Disable strip
    upx=False,     # Disable UPX compression to avoid massive files
    upx_exclude=[],
    name='Skyrimnet-VoxCPM',
    optimize=2,
)