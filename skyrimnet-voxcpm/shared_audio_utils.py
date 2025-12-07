# =============================================================================
import wave
import struct
import torch
import time
from loguru import logger
from typing import TYPE_CHECKING
from pathlib import Path
from datetime import datetime

# Local imports - Handle both direct and module execution
try:
    from .shared_cache_utils import get_wavout_dir, get_cache_key
except ImportError:
    from shared_cache_utils import get_wavout_dir, get_cache_key

if TYPE_CHECKING:
    try:
        from .voxcpm.core import VoxCPM
    except ImportError:
        from voxcpm.core import VoxCPM
# =============================================================================

def save_tensor_as_wav(tensor_data, filename, sample_rate=44100, n_channels=1, sampwidth=2):
    """
    Saves a PyTorch tensor as a WAV file.

    Args:
        tensor_data (torch.Tensor): The audio data as a PyTorch tensor.
                                     Expected to be a 1D tensor for mono,
                                     or a 2D tensor with shape [num_samples, num_channels] for multi-channel.
        filename (str): The name of the output WAV file.
        sample_rate (int): The sample rate of the audio.
        n_channels (int): The number of audio channels.
        sampwidth (int): The sample width in bytes (e.g., 2 for 16-bit PCM).
    """
    # Convert tensor to appropriate data type and scale if necessary
    # For 16-bit PCM (sampwidth=2), values typically range from -32768 to 32767
    if sampwidth == 2:
        # Assuming float tensor in range [-1, 1], scale to 16-bit PCM range
        # Clamp to avoid overflow when converting to int16
        scaled_data = (tensor_data * 32767).clamp(-32768, 32767).to(torch.int16)
    else:
        raise ValueError("Unsupported sample width. Only 2 bytes (16-bit PCM) is implemented.")

    # Flatten the tensor data for writing (still on GPU if input was on GPU)
    if n_channels > 1:
        # Interleave channels if multi-channel
        scaled_data = scaled_data.transpose(0, 1).contiguous().view(-1)
    else:
        scaled_data = scaled_data.view(-1)

    # .contiguous() ensures memory layout is correct, then get underlying storage as bytes
    if scaled_data.is_cuda:
        scaled_data = scaled_data.cpu()
    
    # Use struct.pack with format for entire array at once
    # '<' = little-endian, 'h' = signed short (2 bytes), repeated for all samples
    num_samples = scaled_data.numel()
    packed_data = struct.pack(f'<{num_samples}h', *scaled_data.tolist())

    with wave.open(str(filename), 'wb') as wav_file:
        wav_file.setnchannels(n_channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(packed_data)

def load_wav_as_tensor(file_path, normalize=True):
    """
    Loads a WAV file and returns its audio data as a PyTorch tensor.
    
    Args:
        file_path: Path to the WAV file
        normalize: If True, returns float32 tensor normalized to [-1, 1] range.
                   If False, returns original integer dtype tensor.
    
    Returns:
        tuple: (audio_tensor, framerate)
            - audio_tensor: Shape (n_channels, n_frames), dtype float32 if normalize=True
            - framerate: Sample rate of the audio
    """
    with wave.open(str(file_path), 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        
        # Read all audio frames as raw bytes
        # Note: Some WAV files have invalid nframes (-1 or very large values),
        # so we read all frames and calculate actual count from bytes read
        frames_bytes = wf.readframes(wf.getnframes())
        
        # Calculate actual frame count from bytes read (more reliable than getnframes)
        n_frames = len(frames_bytes) // (n_channels * sampwidth)

        # Determine the format string and normalization factor based on sample width
        if sampwidth == 1:  # 8-bit unsigned
            format_string = f'{n_frames * n_channels}B'
            dtype = torch.uint8
            max_val = 255.0
            offset = 128  # 8-bit audio is unsigned, center is 128
        elif sampwidth == 2:  # 16-bit signed
            format_string = f'<{n_frames * n_channels}h'
            dtype = torch.int16
            max_val = 32768.0
            offset = 0
        elif sampwidth == 3:  # 24-bit signed
            # 24-bit requires manual unpacking (3 bytes per sample)
            audio_data_list = []
            for i in range(0, len(frames_bytes), 3):
                # Little-endian: sign-extend 24-bit to 32-bit
                sample = frames_bytes[i] | (frames_bytes[i+1] << 8) | (frames_bytes[i+2] << 16)
                # Sign extend if negative (bit 23 is set)
                if sample & 0x800000:
                    sample -= 0x1000000
                audio_data_list.append(sample)
            dtype = torch.int32
            max_val = 8388608.0  # 2^23
            offset = 0
        elif sampwidth == 4:  # 32-bit signed
            format_string = f'<{n_frames * n_channels}i'
            dtype = torch.int32
            max_val = 2147483648.0  # 2^31
            offset = 0
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

        # Unpack raw bytes into a list of integers (except for 24-bit which is already done)
        if sampwidth != 3:
            audio_data_list = list(struct.unpack(format_string, frames_bytes))

        # Convert to tensor
        audio_tensor = torch.tensor(audio_data_list, dtype=dtype)

        # Reshape to (n_channels, n_frames)
        audio_tensor = audio_tensor.reshape(n_channels, n_frames)

        # Normalize to float32 in range [-1, 1] if requested
        if normalize:
            audio_tensor = audio_tensor.to(torch.float32)
            if offset != 0:
                audio_tensor = (audio_tensor - offset) / (max_val / 2)
            else:
                audio_tensor = audio_tensor / max_val

        return audio_tensor, framerate


def generate_audio_file(

    model: "VoxCPM",
    language: str,
    speaker_wav: str,
    text: str,
    **inference_kwargs
) -> Path:
    """
    Generate audio file using VoxCPM cached prompt system (our new approach).

    Optimized to work directly with tensors and minimize device transfers.

    Args:
        model: The VoxCPM model to use for inference
        language: Language code for synthesis
        speaker_wav: Speaker reference (file path or speaker name)
        text: Text to synthesize
        stream: Whether to use streaming mode
        **inference_kwargs: Additional parameters for model._generate_cached()
            Supported kwargs:
            - cfg_value: Guidance scale (default: 1.6)
            - inference_timesteps: Number of timesteps (default: 10)
            - normalize: Whether to normalize text (default: True)
            - max_length: Maximum generation length (default: 4096)

    Returns:
        Path: Path to the generated audio file
    """

    func_start_time = time.perf_counter()

    logger.info(
        f"Generating cached audio for text='{text[:50]}...', speaker='{Path(speaker_wav).stem}', language='{language}'")
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
        )

        if prompt_cache is None:
            raise ValueError(
                f"Failed to build prompt cache for speaker: {speaker_wav}")

    except Exception as e:
        logger.error(f"Failed to get prompt cache for {speaker_wav}: {e}")
        raise

    #logger.warning("prompt cache keys: " + ", ".join(prompt_cache.keys()))
    # Prepare inference parameters with VoxCPM defaults
    inference_params = {
        'text': text,
        'prompt_cache': prompt_cache,
        'cfg_value': 2.0,
        'inference_timesteps': 10,
        'normalize': False,
        'max_len': 4096,
        'streaming': False
    }

    # Override defaults with provided kwargs
    for key, value in inference_kwargs.items():
        if key in inference_params and value is not None:
            inference_params[key] = value
        elif key not in inference_params:
            logger.warning(
                f"Ignoring unknown inference parameter for VoxCPM cached generation: {key}={value}")
    logger.info(f"Using cfg value: {inference_params['cfg_value']}, inference_timesteps: {inference_params['inference_timesteps']}")
    # Generate audio using the tensor-optimized method
    try:
        # Call the tensor-optimized wrapper method to avoid numpy conversion
        generate_result = model._generate_with_prompt_cache_tensor(
            **inference_params)
        wav_tensor = next(generate_result)

    except Exception as e:
        logger.error(f"VoxCPM cached generation failed: {e}")
        raise

    # VoxCPM outputs at 16kHz
    output_sample_rate = model.tts_model.sample_rate

    # Save audio file (wav_tensor is already a torch tensor)
    wav_out_path = save_audio_wav(
        wav_tensor=wav_tensor,
        sr=output_sample_rate,
        audio_path=speaker_wav,
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
        logger.info(
            f"Cached audio generation completed in {total_duration_s:.2f}s")

    del wav_tensor

    return wav_out_path

def save_audio_wav(wav_tensor, sr, audio_path) -> Path:
    """Save a tensor as a WAV file"""

    formatted_now_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{formatted_now_time}_{get_cache_key(audio_path)}"
    path = Path(get_wavout_dir(), f"{filename}.wav")
    save_tensor_as_wav(tensor_data=wav_tensor, filename=path, sample_rate=sr)
    del wav_tensor
    return path #.resolve()
