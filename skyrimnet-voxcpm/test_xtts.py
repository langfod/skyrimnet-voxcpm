#!/usr/bin/env python3
"""
SkyrimNet TTS VoxCPM Test Application Entry Point
Test VoxCPM model integration with reference audio transcription
"""

import sys
import os
from pathlib import Path
from typing import TYPE_CHECKING

# Add current directory to Python path for relative imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from loguru import logger
# Import shared modules - Handle both direct execution and module execution
try:
    # Try relative imports first (for module execution: python -m skyrimnet-voxcpm)
    from .shared_args import parse_api_args
    from .shared_app_utils import setup_application_logging, initialize_application_environment
    from .shared_models import initialize_model_with_cache
    from .shared_cache_utils import get_wavout_dir, get_latent_dir, get_speakers_dir, get_cached_prompt_cache
    from .shared_audio_utils import generate_audio_file

except ImportError:
    # Fall back to absolute imports (for direct execution or PyInstaller)
    from shared_args import parse_api_args
    from shared_app_utils import setup_application_logging, initialize_application_environment
    from shared_models import initialize_model_with_cache
    from shared_cache_utils import get_wavout_dir, get_latent_dir, get_speakers_dir, get_cached_prompt_cache
    from shared_audio_utils import generate_audio_file


if TYPE_CHECKING:
    try:
        from .voxcpm.core import VoxCPM
    except ImportError:
        from voxcpm.core import VoxCPM

def initialize_logging():
    """Initialize logging configuration using shared utility"""
    setup_application_logging(log_file_path='logs/skyrimnet_voxcpm_test.log', log_to_file=False)


def initialize_configuration():
    """Initialize environment and configuration"""
    output_temp = get_wavout_dir().parent.absolute()
    reference_audio_dir = get_latent_dir().parent.absolute()
    speakers_dir = get_speakers_dir().parent.absolute()

    os.environ["GRADIO_ALLOWED_PATHS"] = f'assets,{output_temp},{reference_audio_dir},{speakers_dir}'


def initialize_model(device="auto", optimize=True):
    """Initialize and load the VoxCPM TTS model using shared utility"""
    return initialize_model_with_cache(
        device=device,
        seed=20250527,
        validate=True,
        optimize=optimize
    )


if __name__ == "__main__":
    # Parse command line arguments
    extra_args = {
        "--no-optimize": {
            "action": "store_false",
            "dest": "optimize",
            "default": True,
            "help": "Disable torch.compile optimization"
        }
    }
    args = parse_api_args("SkyrimNet TTS VoxCPM Test Application", extra_args)
    
    # Initialize application environment using shared utility
    initialize_application_environment("SkyrimNet TTS VoxCPM Test")
    
    # Initialize logging using shared utility
    initialize_logging()
    
    # Initialize configuration
    initialize_configuration()
    
    # Initialize VoxCPM model using shared utility
    try:
        model = initialize_model(
            device=args.device,
            optimize=args.optimize
        )
    except Exception as e:
        logger.error(f"VoxCPM model initialization failed: {e}")
        sys.exit(1)
    
    # Test audio generation with VoxCPM parameters

    try:
        text = "Now let's make my mum's favourite. So 33 mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible."

        #text = "Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup."

        speaker_audio = "assets\\malebrute.wav"
        language = "en"
        speaker_audio_uuid = None

        logger.info(f"Testing VoxCPM with text: '{text}'")
        logger.info(f"Speaker: {speaker_audio}, Language: {language}")

        #prompt_text = transcribe_audio_with_whisper(speaker_audio)
        prompt_dict = get_cached_prompt_cache(model=model, language=language, speaker_audio=speaker_audio)

        inference_kwargs = {
            'cfg_value': 2.0,
            'inference_timesteps': 10,
            'normalize': True,
            'max_len': 4096,
            'streaming': False,
            'prompt_cache' :prompt_dict,
            'text': text,
        }
 
        # model is voxcpm.core VoxCPM
        #wav_tensor0 = next(model._generate_with_prompt_cache_tensor(**inference_kwargs)) 
        wav_tensor = next(model._generate_with_prompt_cache_tensor(**inference_kwargs)) 
        del wav_tensor
        wav_out_path =  generate_audio_file(model=model, language='en', speaker_wav="maleuniqueghost.wav", text=text)
        wav_out_path =  generate_audio_file(model=model, language='en', speaker_wav="maleuniqueghost.wav", text=text,inference_timesteps= 8)
        wav_out_path =  generate_audio_file(model=model, language='en', speaker_wav="maleuniqueghost.wav", text=text, inference_timesteps=5)
        wav_out_path =  generate_audio_file(model=model, language='en', speaker_wav="maleuniqueghost.wav", text=text, inference_timesteps=1)
        
        #wav_out_path =  generate_audio_file(model=model, language='en', speaker_wav="ghost_resemble_denoised.wav", text=text)
        #wav_out_path =  generate_audio_file(model=model, language='en', speaker_wav="ghost_resemble_enahnced.wav", text=text)
        #wav_out_path =  generate_audio_file(model=model, language='en', speaker_wav="ghost_df_noisy.wav", text=text)
        #wav_out_path =  generate_audio_file(model=model, language='en', speaker_wav="ghost_df_enhanced.wav", text=text)
        # Note: VoxCPM outputs at model.tts_model.sample_rate (16000 Hz)
        sample_rate = model.tts_model.sample_rate
        #del wav_tensor
        #del wav_tensor0
        logger.info(f"✓ Audio generated and saved to: {wav_out_path}")

        #wav_out_path =  generate_audio_file(model=model, language='en', speaker_wav="kimshinvoicetypeenglish", text=text)
        logger.info(f"✓ Audio generated and saved to: {wav_out_path}")
        logger.info("✓ VoxCPM test completed successfully!")
        
    except Exception as e:
        logger.error(f"✗ Audio generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Application shutdown requested by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

# Example: Load VoxCPM model
#voxcpm = VoxCPM.from_pretrained()