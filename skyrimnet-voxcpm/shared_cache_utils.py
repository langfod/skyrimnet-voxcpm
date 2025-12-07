import functools
import threading
import psutil
import torch
from datetime import datetime
import os
import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Any, List
from loguru import logger


if TYPE_CHECKING:
    try:
        from .voxcpm.core import VoxCPM
    except ImportError:
        from voxcpm.core import VoxCPM

class PromptCacheManager:
    """In-memory cache manager for VoxCPM prompt caches."""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()  # Reentrant lock for nested operations

    def get(self, language: str, cache_key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._cache.get(language, {}).get(cache_key)

    def set(self, language: str, cache_key: str, prompt_cache: Dict[str, Any]) -> None:
        with self._lock:
            if language not in self._cache:
                self._cache[language] = {}
            self._cache[language][cache_key] = prompt_cache

    def get_all_keys(self) -> List[Tuple[str, str]]:
        with self._lock:
            keys = []
            for lang, lang_cache in self._cache.items():
                for key in lang_cache.keys():
                    keys.append((lang, key))
            return keys

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            total_entries = sum(len(lang_cache)
                                for lang_cache in self._cache.values())
            return {
                'total_entries': total_entries,
                'languages': len(self._cache),
                'languages_list': list(self._cache.keys())
            }


# Global cache manager instance
cache_manager = PromptCacheManager()


@functools.cache
def get_process_creation_time():
    """Get the process creation time as a datetime object"""
    p = psutil.Process(os.getpid())
    return datetime.fromtimestamp(p.create_time())


@functools.cache
def get_latent_dir(language: str = "en") -> Path:
    """Get or create the reference audio cache directory (renamed for compatibility)"""
    cache_dir = Path("latents_pt").joinpath(language)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

@functools.cache
def get_speakers_dir(language: str = "en") -> Path:
    """Get or create the speakers directory"""        
    speakers_dir = Path("speakers").joinpath(language)
    speakers_dir.mkdir(parents=True, exist_ok=True)
    return speakers_dir

@functools.cache
def get_cache_key(audio_path) -> Optional[str]:
    """Generate a cache key based on audio file"""
    if audio_path is None:
        return None

    cache_prefix = Path(audio_path).stem
    return cache_prefix


def _load_transcription_metadata(metadata_filename: Path) -> Optional[Dict]:
    """Load transcription metadata from JSON file."""
    try:
        if metadata_filename.is_file():
            with open(metadata_filename, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load metadata from {metadata_filename}: {e}")
    return None


def _get_model_device(model: "VoxCPM") -> torch.device:
    """Safely get the device from a model, handling different model types."""
    try:
        # Try direct device access first (VoxCPM wrapper)
        if hasattr(model, 'device'):
            device = model.device
            # Ensure it's a torch.device object
            if isinstance(device, str):
                return torch.device(device)
            return device
            
        # Try tts_model device access (VoxCPM internal model)
        elif hasattr(model, 'tts_model') and hasattr(model.tts_model, 'device'):
            device = model.tts_model.device
            if isinstance(device, str):
                return torch.device(device)
            return device
            
        # Fall back to getting device from parameters
        elif hasattr(model, 'parameters'):
            try:
                return next(model.parameters()).device
            except StopIteration:
                # Model has no parameters
                pass
                
        # Try tts_model parameters as fallback
        elif hasattr(model, 'tts_model') and hasattr(model.tts_model, 'parameters'):
            try:
                return next(model.tts_model.parameters()).device
            except StopIteration:
                pass
                
        # Final fallback - use CUDA if available, otherwise CPU
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
            
    except Exception as e:
        logger.debug(f"Could not determine model device ({e}), falling back to auto-detection")
        # Smart fallback - prefer CUDA if available
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')


def resolve_speaker_audio_path(speaker_audio: str, language: str = "en") -> Optional[str]:
    """
    Resolve speaker audio path - handles both full paths and speaker names.
    
    Args:
        speaker_audio: Full path or speaker name (without extension)
        language: Language code for speaker directory
        
    Returns:
        str: Full path to audio file, or None if not found
    """
    if speaker_audio is None:
        return None
    
    # If it's already a full path and exists, return it
    audio_path = Path(speaker_audio)
    if audio_path.is_file():
        return str(audio_path.absolute())
    
    # Otherwise, try to find it in the speakers directory
    speakers_dir = get_speakers_dir(language=language)
    
    # Try different extensions
    for ext in ['.wav', '.mp3', '.flac', '.ogg']:
        candidate = speakers_dir / f"{speaker_audio}{ext}"
        if candidate.is_file():
            return str(candidate.absolute())
    
    # If not found, return original path (will fail later with clear error)
    logger.warning(f"Could not find speaker audio file: {speaker_audio}")
    return speaker_audio


def get_cached_prompt_cache(model: "VoxCPM" = None, language: str= "en", speaker_audio: str=None) -> Optional[Dict[str, Any]]:
    """
    Get or build VoxCPM prompt cache for a speaker.
    
    Args:
        model: VoxCPM model instance
        language: Language code
        speaker_audio: Path to speaker audio file or speaker name
        
    Returns:
        Prompt cache dict that can be used with VoxCPM generation
    """
    if speaker_audio is None or model is None:
        return None
    

    
    cache_file_key = get_cache_key(speaker_audio)
    
    # Check in-memory cache first
    cached = cache_manager.get(language, cache_file_key)
    if cached:
        logger.info(f"Using cached prompt cache for {Path(speaker_audio).stem}")
        return cached

    # Check disk cache
    latent_dir = get_latent_dir(language=language)
    cache_filename = latent_dir / f"{cache_file_key}.pt"
    metadata_filename = latent_dir / f"{cache_file_key}.json"
    
    if cache_filename.is_file():
        logger.info(f"Loading cached prompt cache from {cache_filename}")
        try:
            # Determine target device for loading
            target_device = _get_model_device(model)
            
            # Load cache with proper device mapping
            prompt_cache = torch.load(cache_filename, map_location=target_device)
            
            # Verify cache is properly loaded and on correct device
            if prompt_cache is None:
                raise ValueError("Loaded cache is None")
                
            # If cache contains tensors, verify they're on the correct device
            cache_device_verified = True
            for key, value in prompt_cache.items():
                if isinstance(value, torch.Tensor):
                    if value.device != target_device:
                        logger.debug(f"Moving cache tensor '{key}' from {value.device} to {target_device}")
                        prompt_cache[key] = value.to(target_device)
                        cache_device_verified = False
            
            if not cache_device_verified:
                logger.debug(f"Corrected device placement for cached tensors")
            
            # Store in memory cache for future use
            cache_manager.set(language, cache_file_key, prompt_cache)
            
            # Load and log transcription metadata if available
            metadata = _load_transcription_metadata(metadata_filename)
            if metadata:
                transcription_preview = metadata.get('transcription', 'N/A')[:50]
                logger.debug(f"Loaded prompt cache with transcription: '{transcription_preview}...'")
            
            logger.debug(f"Loaded prompt cache from disk for {Path(speaker_audio).stem}")
            return prompt_cache
            
        except Exception as e:
            logger.warning(f"Failed to load cached prompt cache {cache_filename}: {e}")
            # Continue to rebuild cache below

    # Build prompt cache using VoxCPM
    logger.info(f"Building prompt cache for: {speaker_audio}")
    
    # Resolve speaker audio path (handles both full paths and speaker names)
    resolved_audio_path = resolve_speaker_audio_path(speaker_audio, language)
    if resolved_audio_path is None:
        logger.error(f"Could not resolve speaker audio: {speaker_audio}")
        return None
    try:
        # First transcribe the audio
        try:
            from .shared_whisper_utils import transcribe_audio_with_whisper
        except ImportError:
            from shared_whisper_utils import transcribe_audio_with_whisper
        
        transcription = transcribe_audio_with_whisper(
            audio_path=resolved_audio_path,
            language=language,
            use_cache=True
        )

        #audio, sr = load_wav_as_tensor(resolved_audio_path)
        #audio_tensor = model.denoiser.enhance_tensor(audio)
        #prompt_tensor = {
        #    'audio_tensor': audio_tensor,
        #    'sample_rate': sr
        #}
        ## Build VoxCPM prompt cache (call low-level method to avoid circular dependency)
        #prompt_cache = model.tts_model.build_prompt_cache(
        #    prompt_text=transcription,
        #    prompt_tensor=prompt_tensor
        #)
        prompt_cache = model.tts_model.build_prompt_cache(
            prompt_text=transcription,
            prompt_wav_path=resolved_audio_path
        )
        
        # Save to disk cache with transcription metadata
        latent_dir = get_latent_dir(language=language)
        cache_filename = latent_dir / f"{cache_file_key}.pt"
        metadata_filename = latent_dir / f"{cache_file_key}.json"
        
        # Save prompt cache to disk
        try:
            torch.save(prompt_cache, cache_filename)
            logger.debug(f"Saved prompt cache to {cache_filename}")
            
            # Save transcription metadata
            metadata = {
                'transcription': transcription,
                'audio_path': resolved_audio_path,
                'language': language,
                'created_at': datetime.now().isoformat()
            }
            with open(metadata_filename, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved transcription metadata to {metadata_filename}")
            
        except Exception as e:
            logger.warning(f"Failed to save prompt cache to disk: {e}")
        
        # Store in memory cache
        cache_manager.set(language, cache_file_key, prompt_cache)
        
        logger.info(f"Built and cached prompt cache for {Path(resolved_audio_path).stem}")
        return prompt_cache
        
    except Exception as e:
        logger.error(f"Failed to build prompt cache for {resolved_audio_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_reference_audio_and_text(model, language: str, speaker_audio: str) -> Tuple[str, str]:
    """
    Legacy function for compatibility. Get reference audio path and its transcription.
    
    This function is kept for backward compatibility but now uses the whisper transcription
    without JSON caching as requested.
    
    Args:
        model: VoxCPM model (for compatibility)
        language: Language code
        speaker_audio: Path to speaker audio file or speaker name
        
    Returns:
        Tuple[str, str]: (audio_path, transcription_text)
    """
    if speaker_audio is None:
        return None, None
    
    # Resolve speaker audio path (handles both full paths and speaker names)
    resolved_audio_path = resolve_speaker_audio_path(speaker_audio, language)
    if resolved_audio_path is None:
        logger.error(f"Could not resolve speaker audio: {speaker_audio}")
        return None, None
    
    # Transcribe the audio (no JSON caching)
    logger.info(f"Transcribing reference audio: {resolved_audio_path}")
    try:
        try:
            from .shared_whisper_utils import transcribe_audio_with_whisper
        except ImportError:
            from shared_whisper_utils import transcribe_audio_with_whisper
        
        transcription = transcribe_audio_with_whisper(
            audio_path=resolved_audio_path,
            language=language,
            use_cache=True  # Let whisper handle its own caching
        )
        
        return resolved_audio_path, transcription
        
    except Exception as e:
        logger.error(f"Failed to transcribe audio {resolved_audio_path}: {e}")
        return resolved_audio_path, ""

def init_prompt_cache(model, supported_languages: List[str] = ["en"]) -> None:
    """
    Initialize VoxCPM prompt cache from disk for all supported languages.
    """
    cached_prompts = {}
    for lang in supported_languages:
        latent_dir = get_latent_dir(language=lang)
        cached_prompts[lang] = []
        
        # Load existing .pt files from latents directory (VoxCPM prompt caches)
        for filename in latent_dir.glob("*.pt"):
            try:
                base_name = filename.stem
                cached_prompts[lang].append(base_name)
                
                # Determine target device for loading
                target_device = _get_model_device(model)
                
                # Load the prompt cache into memory with proper device
                prompt_cache = torch.load(filename, map_location=target_device)
                
                # Verify and fix device placement for cache tensors
                if prompt_cache is not None:
                    for key, value in prompt_cache.items():
                        if isinstance(value, torch.Tensor) and value.device != target_device:
                            prompt_cache[key] = value.to(target_device)
                
                cache_manager.set(lang, base_name, prompt_cache)
                
                # Also load corresponding transcription metadata if available
                metadata_filename = latent_dir / f"{base_name}.json"
                metadata = _load_transcription_metadata(metadata_filename)
                if metadata:
                    logger.debug(f"Loaded prompt cache + metadata: {filename}")
                else:
                    logger.debug(f"Loaded prompt cache (no metadata): {filename}")
                
            except Exception as e:
                logger.error(f"Failed to load cache file {filename}: {e}")
        
        speaker_dir = get_speakers_dir(language=lang)  
        print(f"Loading speakers from: {speaker_dir}")      
        # Get all speaker files and organize by base name
        speaker_files = {}
        for file_path in speaker_dir.iterdir():
            print(f"Found speaker file: {file_path}")
            if file_path.is_file() and file_path.suffix in ['.wav']:
                base_name = file_path.stem
                if base_name not in speaker_files:
                    speaker_files[base_name] = {}
                speaker_files[base_name][file_path.suffix] = file_path
        
        # Process each speaker, preferring .wav over .json
        for base_name, files in speaker_files.items():
            try:
                if '.wav' in files:
                    # .wav files - compute latents from audio
                    speaker_wav_path = files['.wav']
                    logger.info(f"Processing .wav file: {speaker_wav_path}")
                    get_cached_prompt_cache(model=model, speaker_audio=str(speaker_wav_path), language=lang)                                   
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Failed to load latents for speaker {base_name}: {e}") 
    
    stats = cache_manager.get_stats()
    logger.info(
        f"Initialized prompt cache with {stats['total_entries']} entries across languages: {stats['languages_list']}")


def get_prompt_cache_keys() -> List[Tuple[str, str]]:
    """Return a list of all cached prompt cache keys."""
    return cache_manager.get_all_keys()


def get_prompt_cache_stats() -> Dict[str, int]:
    """Get statistics about the prompt cache."""
    return cache_manager.get_stats()


@functools.cache
def get_wavout_dir():
    formatted_start_time = get_process_creation_time().strftime("%Y%m%d_%H%M%S")
    wavout_dir = Path("output_temp").joinpath(formatted_start_time)
    wavout_dir.mkdir(parents=True, exist_ok=True)
    return wavout_dir



