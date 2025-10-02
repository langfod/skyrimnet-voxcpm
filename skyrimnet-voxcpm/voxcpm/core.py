import os
import re
import tempfile
import torch
import numpy as np
from typing import Generator
from huggingface_hub import snapshot_download
from .model.voxcpm import VoxCPMModel

class VoxCPM:
    """
    VoxCPM Text-to-Speech Model with original and cached prompt approaches.
    
    Original VoxCPM methods (unchanged):
    - generate(text, prompt_wav_path, prompt_text) - Direct generation
    - generate_streaming(text, prompt_wav_path, prompt_text) - Streaming generation
    
    New cached prompt methods (our additions, like XTTS latents):
    - generate_cached(text, language, speaker_audio) - Auto-cached generation
    - generate_streaming_cached(text, language, speaker_audio) - Streaming cached
    - build_prompt_cache_from_audio(language, speaker_audio) - Manual cache building
    - generate_with_prompt_cache(text, prompt_cache) - Use pre-built cache
    
    The cached approach scans speakers/ folder and builds/reuses prompt caches
    for efficiency when generating multiple audio clips with the same speaker.
    """
    
    def __init__(self,
            voxcpm_model_path : str,
            optimize: bool = True,
        ):
        """Initialize VoxCPM TTS pipeline.

        Args:
            voxcpm_model_path: Local filesystem path to the VoxCPM model assets
                (weights, configs, etc.). Typically the directory returned by
                a prior download step.
            optimize: Whether to optimize the model with torch.compile. True by default, but can be disabled for debugging.
        """
        print(f"voxcpm_model_path: {voxcpm_model_path}, optimize: {optimize}")
        self.tts_model = VoxCPMModel.from_local(voxcpm_model_path, optimize=optimize)
        self.text_normalizer = None
        # Force default mode to avoid Triton "int too large to convert to C long" error on Windows
        if optimize:
            self.tts_model.optimize(force_mode="default")
        #print("Warm up VoxCPMModel...")
        #try:
        #    self.tts_model.generate(
        #        target_text="Hello, this is the first test sentence.",
        #        max_len=10,
        #    )
        #    print("✓ VoxCPM warmup completed successfully with optimization")
        #except Exception as warmup_e:
        #    if "Python int too large to convert to C long" in str(warmup_e):
        #        print(f"✗ Warmup failed due to Triton overflow: {type(warmup_e).__name__}")
        #        print("🔄 Attempting fallback: re-optimizing with default mode...")
        #        
        #        # Try fallback with default mode
        #        try:
        #            self.tts_model.optimize(force_mode="default")
        #            self.tts_model.generate(
        #                target_text="Hello, this is the first test sentence.",
        #                max_len=10,
        #            )
        #            print("✓ VoxCPM warmup completed successfully with default mode")
        #        except Exception as default_e:
        #            print(f"✗ Default mode also failed: {type(default_e).__name__}")
        #            print("🔄 Final fallback: disabling torch.compile optimization...")
        #            
        #            # Final fallback: disable optimization completely
        #            self.tts_model.optimize(disable=True)
        #            try:
        #                self.tts_model.generate(
        #                    target_text="Hello, this is the first test sentence.",
        #                    max_len=10,
        #                )
        #                print("✓ VoxCPM warmup completed successfully without optimization")
        #            except Exception as final_e:
        #                print(f"✗ Even unoptimized warmup failed: {final_e}")
        #                raise
        #    else:
        #        print(f"✗ Warmup failed with unexpected error: {warmup_e}")
        #        raise

    @classmethod
    def from_pretrained(cls,
            hf_model_id: str = "openbmb/VoxCPM-0.5B",
            cache_dir: str = None,
            local_files_only: bool = False,
            **kwargs,
        ):
        """Instantiate ``VoxCPM`` from a Hugging Face Hub snapshot.

        Args:
            hf_model_id: Explicit Hugging Face repository id (e.g. "org/repo") or local path.
            cache_dir: Custom cache directory for the snapshot.
            local_files_only: If True, only use local files and do not attempt
                to download.
        Kwargs:
            Additional keyword arguments passed to the ``VoxCPM`` constructor.

        Returns:
            VoxCPM: Initialized instance whose ``voxcpm_model_path`` points to
            the downloaded snapshot directory.

        Raises:
            ValueError: If neither a valid ``hf_model_id`` nor a resolvable
                ``hf_model_id`` is provided.
        """
        repo_id = hf_model_id
        if not repo_id:
            raise ValueError("You must provide hf_model_id")
        
        # Load from local path if provided
        if os.path.isdir(repo_id):
            local_path = repo_id
        else:
            # Otherwise, try from_pretrained (Hub); exit on failure
            local_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )

        return cls(
            voxcpm_model_path=local_path,
            **kwargs,
        )

    def generate(self, *args, **kwargs) -> np.ndarray:
        return next(self._generate(*args, streaming=False, **kwargs))

    def generate_streaming(self, *args, **kwargs) -> Generator[np.ndarray, None, None]:
        return self._generate(*args, streaming=True, **kwargs)

    def _generate(self, 
            text : str,
            prompt_wav_path : str = None,
            prompt_text : str = None,
            cfg_value : float = 2.0,    
            inference_timesteps : int = 10,
            max_length : int = 4096,
            normalize : bool = True,
            retry_badcase : bool = True,
            retry_badcase_max_times : int = 3,
            retry_badcase_ratio_threshold : float = 6.0,
            streaming: bool = False,
        ) -> Generator[np.ndarray, None, None]:
        """Synthesize speech for the given text and return a single waveform.

        This method optionally builds and reuses a prompt cache. If an external
        prompt (``prompt_wav_path`` + ``prompt_text``) is provided, it will be
        used for all sub-sentences. Otherwise, the prompt cache is built from
        the first generated result and reused for the remaining text chunks.

        Args:
            text: Input text. Can include newlines; each non-empty line is
                treated as a sub-sentence.
            prompt_wav_path: Path to a reference audio file for prompting.
            prompt_text: Text content corresponding to the prompt audio.
            cfg_value: Guidance scale for the generation model.
            inference_timesteps: Number of inference steps.
            max_length: Maximum token length during generation.
            normalize: Whether to run text normalization before generation.
            retry_badcase: Whether to retry badcase.
            retry_badcase_max_times: Maximum number of times to retry badcase.
            retry_badcase_ratio_threshold: Threshold for audio-to-text ratio.
            streaming: Whether to return a generator of audio chunks.
        Returns:
            Generator of numpy.ndarray: 1D waveform array (float32) on CPU. 
            Yields audio chunks for each generations step if ``streaming=True``,
            otherwise yields a single array containing the final audio.
        """
        if not text.strip() or not isinstance(text, str):
            raise ValueError("target text must be a non-empty string")
        
        if prompt_wav_path is not None:
            if not os.path.exists(prompt_wav_path):
                raise FileNotFoundError(f"prompt_wav_path does not exist: {prompt_wav_path}")
        
        if (prompt_wav_path is None) != (prompt_text is None):
            raise ValueError("prompt_wav_path and prompt_text must both be provided or both be None")
        
        text = text.replace("\n", " ")
        text = re.sub(r'\s+', ' ', text)
        
        if prompt_wav_path is not None and prompt_text is not None:
            fixed_prompt_cache = self.tts_model.build_prompt_cache(
                prompt_wav_path=prompt_wav_path,
                prompt_text=prompt_text
            )
        else:
            fixed_prompt_cache = None  # will be built from the first inference
        
        if normalize:
            if self.text_normalizer is None:
                from .utils.text_normalize import TextNormalizer
                self.text_normalizer = TextNormalizer()
            text = self.text_normalizer.normalize(text)
        
        generate_result = self.tts_model._generate_with_prompt_cache(
                        target_text=text,
                        prompt_cache=fixed_prompt_cache,
                        min_len=2,
                        max_len=max_length,
                        inference_timesteps=inference_timesteps,
                        cfg_value=cfg_value,
                        retry_badcase=retry_badcase,
                        retry_badcase_max_times=retry_badcase_max_times,
                        retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                        streaming=streaming,
                    )
    
        for wav, _, _ in generate_result:
            yield wav.squeeze(0).cpu().numpy()

    # ===== New cached prompt functions (our additions) =====
    
    def generate_cached(self, *args, **kwargs) -> np.ndarray:
        """Generate audio using cached prompt system (non-streaming)."""
        return next(self._generate_cached(*args, streaming=False, **kwargs))

    def generate_streaming_cached(self, *args, **kwargs) -> Generator[np.ndarray, None, None]:
        """Generate audio using cached prompt system (streaming)."""
        return self._generate_cached(*args, streaming=True, **kwargs)

    def build_prompt_cache_from_audio(self, language: str, speaker_audio: str, speaker_audio_uuid: int = None):
        """
        Build a prompt cache from speaker audio that can be reused for multiple generations.
        
        This method uses the shared cache system to get or build prompt caches efficiently.
        
        Args:
            language: Language code for the speaker
            speaker_audio: Path to speaker audio file or speaker name
            speaker_audio_uuid: Optional UUID for caching
            
        Returns:
            dict: Prompt cache that can be passed to _generate_with_prompt_cache()
            
        Raises:
            FileNotFoundError: If the speaker audio file cannot be found
            ValueError: If cache building fails
        """
        # Import here to avoid circular imports
        try:
            from ..shared_cache_utils import get_cached_prompt_cache
        except ImportError:
            from shared_cache_utils import get_cached_prompt_cache
        
        # Get or build prompt cache using the shared cache system
        prompt_cache = get_cached_prompt_cache(
            model=self,  # Pass self as model
            language=language,
            speaker_audio=speaker_audio,
            speaker_audio_uuid=speaker_audio_uuid
        )
        
        if prompt_cache is None:
            raise ValueError(f"Failed to build prompt cache for speaker: {speaker_audio}")
        
        return prompt_cache

    def _generate_cached(self, 
            text: str,
            language: str,
            speaker_audio: str,
            speaker_audio_uuid: int = None,
            cfg_value: float = 2.0,    
            inference_timesteps: int = 10,
            max_length: int = 4096,
            normalize: bool = True,
            retry_badcase: bool = True,
            retry_badcase_max_times: int = 3,
            retry_badcase_ratio_threshold: float = 6.0,
            streaming: bool = False,
        ) -> Generator[np.ndarray, None, None]:
        """
        Generate audio using cached prompt system.
        
        This is our new method that uses the cached prompt approach similar to XTTS latents.
        It automatically gets or builds a prompt cache for the speaker and then generates audio.
        
        Args:
            text: Input text to synthesize
            language: Language code for the speaker 
            speaker_audio: Path to speaker audio file or speaker name
            speaker_audio_uuid: Optional UUID for caching
            cfg_value: Guidance scale for the generation model
            inference_timesteps: Number of inference steps
            max_length: Maximum token length during generation
            normalize: Whether to run text normalization before generation
            retry_badcase: Whether to retry badcase
            retry_badcase_max_times: Maximum number of times to retry badcase
            retry_badcase_ratio_threshold: Threshold for audio-to-text ratio
            streaming: Whether to return a generator of audio chunks
            
        Returns:
            Generator of numpy.ndarray: 1D waveform array (float32) on CPU
        """
        if not text.strip() or not isinstance(text, str):
            raise ValueError("target text must be a non-empty string")
        
        # Get or build prompt cache for this speaker
        prompt_cache = self.build_prompt_cache_from_audio(
            language=language,
            speaker_audio=speaker_audio,
            speaker_audio_uuid=speaker_audio_uuid
        )
        
        # Use the original _generate_with_prompt_cache method
        return self._generate_with_prompt_cache(
            text=text,
            prompt_cache=prompt_cache,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            max_length=max_length,
            normalize=normalize,
            retry_badcase=retry_badcase,
            retry_badcase_max_times=retry_badcase_max_times,
            retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
            streaming=streaming
        )

    def generate_with_prompt_cache(self, *args, **kwargs) -> np.ndarray:
        """Generate audio using a pre-built prompt cache (non-streaming)."""
        return next(self._generate_with_prompt_cache(*args, streaming=False, **kwargs))

    def generate_streaming_with_prompt_cache(self, *args, **kwargs) -> Generator[np.ndarray, None, None]:
        """Generate audio using a pre-built prompt cache (streaming)."""
        return self._generate_with_prompt_cache(*args, streaming=True, **kwargs)

    def _generate_with_prompt_cache(self, 
            text: str,
            prompt_cache: dict,
            cfg_value: float = 2.0,    
            inference_timesteps: int = 10,
            max_length: int = 4096,
            normalize: bool = True,
            retry_badcase: bool = True,
            retry_badcase_max_times: int = 3,
            retry_badcase_ratio_threshold: float = 6.0,
            streaming: bool = False,
        ) -> Generator[np.ndarray, None, None]:
        """
        Synthesize speech using a pre-built prompt cache.
        
        This method is more efficient when generating multiple audio clips with the same
        speaker since it reuses the pre-computed prompt cache instead of rebuilding it
        from audio files each time.
        
        Args:
            text: Input text. Can include newlines; each non-empty line is
                treated as a sub-sentence.
            prompt_cache: Pre-built prompt cache from build_prompt_cache_from_audio()
            cfg_value: Guidance scale for the generation model.
            inference_timesteps: Number of inference steps.
            max_length: Maximum token length during generation.
            normalize: Whether to run text normalization before generation.
            retry_badcase: Whether to retry badcase.
            retry_badcase_max_times: Maximum number of times to retry badcase.
            retry_badcase_ratio_threshold: Threshold for audio-to-text ratio.
            streaming: Whether to return a generator of audio chunks.
            
        Returns:
            Generator of numpy.ndarray: 1D waveform array (float32) on CPU. 
            Yields audio chunks for each generations step if ``streaming=True``,
            otherwise yields a single array containing the final audio.
            
        Raises:
            ValueError: If text is empty or prompt_cache is None
        """
        if not text.strip() or not isinstance(text, str):
            raise ValueError("target text must be a non-empty string")
        
        if prompt_cache is None:
            raise ValueError("prompt_cache cannot be None")
        
        text = text.replace("\n", " ")
        text = re.sub(r'\s+', ' ', text)
        
        if normalize:
            if self.text_normalizer is None:
                from .utils.text_normalize import TextNormalizer
                self.text_normalizer = TextNormalizer()
            text = self.text_normalizer.normalize(text)
        
        generate_result = self.tts_model._generate_with_prompt_cache(
                        target_text=text,
                        prompt_cache=prompt_cache,
                        min_len=2,
                        max_len=max_length,
                        inference_timesteps=inference_timesteps,
                        cfg_value=cfg_value,
                        retry_badcase=retry_badcase,
                        retry_badcase_max_times=retry_badcase_max_times,
                        retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                        streaming=streaming,
                    )
    
        for wav, _, _ in generate_result:
            yield wav.squeeze(0).cpu().numpy()

    def _generate_with_prompt_cache_tensor(self, 
            text: str,
            prompt_cache: dict,
            cfg_value: float = 2.0,    
            inference_timesteps: int = 10,
            max_length: int = 4096,
            normalize: bool = True,
            retry_badcase: bool = True,
            retry_badcase_max_times: int = 3,
            retry_badcase_ratio_threshold: float = 6.0,
            streaming: bool = False,
        ):
        """
        Tensor-optimized version of _generate_with_prompt_cache.
        
        This method returns torch tensors directly from the VoxCPM model without any 
        additional conversions, avoiding unnecessary CPU transfers or dimension changes.
        The VoxCPM model already returns properly shaped tensors on CPU.
        
        Args:
            text: Input text. Can include newlines; each non-empty line is
                treated as a sub-sentence.
            prompt_cache: Pre-built prompt cache from build_prompt_cache_from_audio()
            cfg_value: Guidance scale for the generation model.
            inference_timesteps: Number of inference steps.
            max_length: Maximum token length during generation.
            normalize: Whether to run text normalization before generation.
            retry_badcase: Whether to retry badcase.
            retry_badcase_max_times: Maximum number of times to retry badcase.
            retry_badcase_ratio_threshold: Threshold for audio-to-text ratio.
            streaming: Whether to return a generator of audio chunks.
            
        Returns:
            Generator of torch.Tensor: 1D waveform tensor (float32) on CPU with shape [audio_length]. 
            Yields audio tensors directly from VoxCPM model for each generation step if ``streaming=True``,
            otherwise yields a single tensor containing the final audio.
            
        Raises:
            ValueError: If text is empty or prompt_cache is None
        """
        if not text.strip() or not isinstance(text, str):
            raise ValueError("target text must be a non-empty string")
        
        if prompt_cache is None:
            raise ValueError("prompt_cache cannot be None")
        
        ## Clean up text
        #text = text.replace("\n", " ")
        #text = re.sub(r'\s+', ' ', text).strip()
        #
        ## Apply text normalization if requested
        #if normalize:
        #    if self.text_normalizer is None:
        #        from .utils.text_normalize import TextNormalizer
        #        self.text_normalizer = TextNormalizer()
        #    text = self.text_normalizer.normalize(text)
        #
        # Use enhanced VoxCPM model method which handles chunking internally
        generate_result = self.tts_model._generate_with_prompt_cache(
                        target_text=text,
                        prompt_cache=prompt_cache,
                        min_len=2,
                        max_len=max_length,
                        inference_timesteps=inference_timesteps,
                        cfg_value=cfg_value,
                        retry_badcase=retry_badcase,
                        retry_badcase_max_times=retry_badcase_max_times,
                        retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                        streaming=streaming,
                    )
    
        # Return tensor directly without numpy conversion
        # VoxCPM model already returns tensors on CPU with proper dimensions
        for wav, _, _ in generate_result:
            yield wav