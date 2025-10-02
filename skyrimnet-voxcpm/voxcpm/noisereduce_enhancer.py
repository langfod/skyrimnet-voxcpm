"""
NoiseReduceEnhancer Module - Lightweight Audio Denoising

Provides a lightweight alternative to ZipEnhancer using the noisereduce library.
Compatible interface with ZipEnhancer for easy switching.
"""

import os
import tempfile
from typing import Optional
import numpy as np

class NoiseReduceEnhancer:
    """Lightweight audio denoiser using noisereduce library"""
    
    def __init__(self, stationary: bool = True, prop_decrease: float = 1.0):
        """
        Initialize NoiseReduceEnhancer
        
        Args:
            stationary: Whether to use stationary noise reduction (faster)
            prop_decrease: The proportion to decrease the noise by (1.0 = remove completely)
        """
        self.stationary = stationary
        self.prop_decrease = prop_decrease
        
    def enhance(self, input_path: str, output_path: Optional[str] = None, 
                normalize_loudness: bool = True) -> str:
        """
        Audio denoising enhancement using noisereduce
        
        Args:
            input_path: Input audio file path
            output_path: Output audio file path (optional, creates temp file by default)
            normalize_loudness: Whether to perform loudness normalization
        Returns:
            str: Output audio file path
        Raises:
            RuntimeError: If processing fails
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input audio file does not exist: {input_path}")
            
        # Create temporary file if no output path is specified
        if output_path is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                output_path = tmp_file.name
                
        try:
            # Import here to avoid dependency issues if not installed
            import noisereduce as nr
            import soundfile as sf
            
            # Load audio
            data, sample_rate = sf.read(input_path)
            
            # Apply noise reduction
            if self.stationary:
                # Faster stationary noise reduction
                reduced_noise = nr.reduce_noise(
                    y=data, 
                    sr=sample_rate, 
                    stationary=True,
                    prop_decrease=self.prop_decrease
                )
            else:
                # More thorough non-stationary noise reduction
                reduced_noise = nr.reduce_noise(
                    y=data, 
                    sr=sample_rate, 
                    stationary=False,
                    prop_decrease=self.prop_decrease
                )
            
            # Optional loudness normalization
            if normalize_loudness:
                # Simple RMS normalization to -20dB
                rms = np.sqrt(np.mean(reduced_noise**2))
                if rms > 0:
                    target_rms = 10**(-20/20)  # -20dB
                    reduced_noise = reduced_noise * (target_rms / rms)
                    # Clip to prevent overflow
                    reduced_noise = np.clip(reduced_noise, -1.0, 1.0)
            
            # Save enhanced audio
            sf.write(output_path, reduced_noise, sample_rate)
            return output_path
            
        except Exception as e:
            # Clean up possibly created temporary files
            if output_path and os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except OSError:
                    pass
            raise RuntimeError(f"Audio denoising processing failed: {e}")