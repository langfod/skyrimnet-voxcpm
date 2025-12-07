#!/usr/bin/env python3
"""
Text-to-Speech Application with Gradio Interface
Enhanced with disk and memory caching for speaker embeddings
"""

# Standard library imports
from pathlib import Path
import uuid


# Third-party imports
import gradio as gr
from loguru import logger

# Handle both direct execution and module execution
try:
    # Try relative imports first (for module execution: python -m skyrimnet-voxcpm)
    from .shared_cache_utils import get_wavout_dir, get_latent_dir, get_speakers_dir
    from .shared_config import DEFAULT_CACHE_CONFIG
    from .shared_args import parse_gradio_args
    from .shared_audio_utils import generate_audio_file
    from .shared_app_utils import initialize_application_environment
    from .shared_models import initialize_model_with_cache, setup_model_seed
except ImportError:
    # Fall back to absolute imports 
    from shared_cache_utils import get_wavout_dir, get_latent_dir, get_speakers_dir
    from shared_config import DEFAULT_CACHE_CONFIG
    from shared_args import parse_gradio_args
    from shared_audio_utils import generate_audio_file
    from shared_app_utils import initialize_application_environment
    from shared_models import initialize_model_with_cache, setup_model_seed

# =============================================================================
# GLOBAL CONFIGURATION AND CONSTANTS
# =============================================================================

# Global model state
CURRENT_MODEL_TYPE = None
CURRENT_MODEL = None
IGNORE_PING = None
SILENCE_AUDIO_PATH = "assets/silence_100ms.wav"
# Cache flags - defaults that can be overridden by skyrimnet_config.txt
ENABLE_DISK_CACHE = DEFAULT_CACHE_CONFIG["ENABLE_DISK_CACHE"]
ENABLE_MEMORY_CACHE = DEFAULT_CACHE_CONFIG["ENABLE_MEMORY_CACHE"]
_CONFIG_CACHE = None
_CONFIG_FILE_PATH = "skyrimnet_config.txt"
# Testing flag - when True, bypasses config loading and uses all API values
_USE_API_MODE = False
_FROM_GRADIO = False
STREAM = True
# =============================================================================
# COMMAND LINE ARGUMENT PARSING
# =============================================================================

args = parse_gradio_args("VoxCPM Text-to-Speech Application with Gradio Interface")

# =============================================================================
# Support Functions
# =============================================================================

def load_skyrimnet_config():
    """Load configuration from skyrimnet_config.txt - simplified version"""
    global _CONFIG_CACHE, ENABLE_MEMORY_CACHE, ENABLE_DISK_CACHE
    
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    
    # Simple config loading - just get user overrides from file
    config_overrides = {}
    if Path(_CONFIG_FILE_PATH).exists():
        try:
            with open(_CONFIG_FILE_PATH, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip().lower()
                            value = value.strip()
                            
                            # Parse TTS parameters
                            if key == 'cfg_scale':
                                if value.lower() != "default":
                                    try:
                                        config_overrides[key] = float(value)
                                    except ValueError:
                                        logger.warning(f"Invalid value for {key}: {value}")
                            elif key == 'inference_timesteps':
                                if value.lower() != "default":
                                    try:
                                        config_overrides[key] = int(float(value))  # Convert to int, allowing decimal input
                                    except ValueError:
                                        logger.warning(f"Invalid value for {key}: {value}")
                            # Parse cache flags
                            elif key == 'enable_memory_cache':
                                ENABLE_MEMORY_CACHE = value.lower() in ['true', '1', 'yes', 'on']
                            elif key == 'enable_disk_cache':
                                ENABLE_DISK_CACHE = value.lower() in ['true', '1', 'yes', 'on']
        except Exception as e:
            logger.warning(f"Error loading config file {_CONFIG_FILE_PATH}: {e}")
    
    _CONFIG_CACHE = config_overrides
    return _CONFIG_CACHE

def get_config_override(param_name, api_value):
    """Get config override value, only using API value in API mode"""
    config_overrides = load_skyrimnet_config()
    config_value = config_overrides.get(param_name)
    
    # Only use API values if explicitly in API mode or from Gradio web interface
    if (_USE_API_MODE or _FROM_GRADIO) and api_value is not None:
        return api_value
    
    # Otherwise, only return config file value (ignore API value)
    return config_value


# =============================================================================
# MAIN APPLICATION FUNCTIONS
# =============================================================================

def generate_audio(model_choice:str=None, text:str=None, language:str="en", speaker_audio:str=None, prefix_audio:str=None,
                    e1:float=None, e2:float=None, e3:float=None, e4:float=None, e5:float=None, e6:float=None, e7:float=None, e8:float=None,
                  vq_single:float=None, fmax:int=None, pitch_std:float=None, speaking_rate:float=None, dnsmos_ovrl:float=None, speaker_noised:float=None, cfg_scale:float=None, top_p:float=None,
                  min_k:float=None, min_p:float=None, linear:float=None, confidence:float=None, quadratic:float=None, seed:int=None, randomize_seed:bool=None, unconditional_keys:float=None
                  ) -> tuple[Path, int]:
    """
    Generates audio based on the provided UI parameters with enhanced caching.
    """
    global IGNORE_PING
    #print(locals())
    if isinstance(speaker_audio, dict) and 'path' in speaker_audio:
        speaker_audio = speaker_audio['path']
    logger.info(f"inputs: text={text}, speaker_audio={Path(speaker_audio).stem if speaker_audio else 'None'}, seed={seed}")

    #if text == "ping" and (speaker_audio is None or speaker_audio == 'maleeventoned' or speaker_audio == 'player voice'):
    #logger.info("Ping received.")
    #if text == "ping" and (speaker_audio is None or speaker_audio == 'maleeventoned' or speaker_audio == 'player voice') and not IGNORE_PING:
    #    IGNORE_PING = True
    #    #logger.info("Ping sending silence audio.")
    #    return "assets/silence_100ms.wav", seed
    #IGNORE_PING = True

    job_id = seed      
    global IGNORE_PING

    if isinstance(speaker_audio, dict) and 'path' in speaker_audio:
        speaker_audio = speaker_audio['path']
    logger.info(f"inputs: text={text}, language={language}, speaker_audio={Path(speaker_audio).stem if speaker_audio else 'None'}, seed={seed}")

    if text == "ping":
       if IGNORE_PING is None:
          IGNORE_PING = "pending"
       else:
          logger.info("Ping request received, sending silence audio.")
          return SILENCE_AUDIO_PATH, job_id

    setup_model_seed(randomize=randomize_seed)

    language, _ = language.split("-") if language is not None and "-" in language else ("en", None)

    # Get parameter overrides - only pass non-None values
    inference_kwargs = {}   

    # Only use API parameters when explicitly from Gradio web interface or API mode is enabled
    use_api_params = _FROM_GRADIO or _USE_API_MODE

    # Convert parameters if we're using API mode 
    if use_api_params and not _FROM_GRADIO:
        # Convert parameters for Gradio API calls
        cfg_scale = float(cfg_scale) if cfg_scale is not None else None
        confidence = int(confidence) if confidence is not None else None
    elif _FROM_GRADIO or _USE_API_MODE:
        # Convert parameters for web UI calls
        cfg_scale = float(cfg_scale)
        confidence = int(confidence)
    
    if use_api_params:
        # Use API parameters directly when in API mode (Gradio web interface)
        if cfg_scale is not None:
            inference_kwargs['cfg_value'] = cfg_scale
        if confidence is not None:
            inference_kwargs['inference_timesteps'] = confidence
        logger.info(f"Using API parameters: cfg_scale={cfg_scale}, inference_timesteps={confidence}")
    else:
        cfg_scale_override = get_config_override('cfg_scale', cfg_scale)
        if cfg_scale_override is not None:
            inference_kwargs['cfg_value'] = cfg_scale_override

        inference_timesteps_override = get_config_override('inference_timesteps', confidence)
        if inference_timesteps_override is not None:
            inference_kwargs['inference_timesteps'] = int(inference_timesteps_override)
    
    # Always pass the stream parameter
    logger.debug(f"Inference kwargs: {inference_kwargs}")
    # Use shared audio generation function with only necessary kwargs
    wav_out_path = generate_audio_file(
        model=CURRENT_MODEL,
        language=language,
        speaker_wav=speaker_audio,
        text=text,
        **inference_kwargs
    )

    if IGNORE_PING == "pending":
        IGNORE_PING = True
        Path(wav_out_path).unlink(missing_ok=True)
        wav_out_path = SILENCE_AUDIO_PATH
    
    return wav_out_path, job_id


def generate_gradio_audio(model_choice, text, speaker_audio, prefix_audio, 
                cfg_scale, inference_timesteps, uuid_number) -> tuple[Path, int]:
    global _FROM_GRADIO
    _FROM_GRADIO = True
    wav_out_path, speaker_audio_uuid = generate_audio(model_choice=model_choice, text=text, speaker_audio=speaker_audio, prefix_audio=prefix_audio,
                           cfg_scale=cfg_scale, confidence=inference_timesteps, seed=uuid_number)
    _FROM_GRADIO = False
    return wav_out_path, speaker_audio_uuid

def build_interface():

    """Build and return the Gradio interface with cache management."""
    output_temp = get_wavout_dir().parent.absolute()
    latents_dir = get_latent_dir().parent.absolute()
    speakers_dir = get_speakers_dir().parent.absolute()

    
    gr.set_static_paths([output_temp, latents_dir, speakers_dir])
    with gr.Blocks(analytics_enabled=False, title="VoxCPM") as demo:
        gr.Markdown("# VoxCPM with Speaker Embedding Cache")

        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Text to Synthesize",
                    value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
                    lines=4)

            with gr.Column():
                speaker_audio = gr.Audio(label="Optional Speaker Audio (for cloning)", type="filepath",sources=["upload", "microphone"])

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Conditioning Parameters")
                cfg_scale = gr.Slider(0, 5, value=2.0, step=0.1, label="CFG Scale")
                inference_timesteps = gr.Slider(1, 50, value=10, step=1, label="Inference Timesteps (Simmering Time: Quality vs. Speed)")

        with gr.Column():
            generate_button = gr.Button("Generate Audio", variant="primary")
            output_audio = gr.Audio(label="Generated Audio", type="filepath", autoplay=True)

        model_choice = gr.Textbox(visible=False)
        language = gr.Textbox(visible=False)
        #prefix_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File", value=None, visible=False)
        emotion1 = gr.Number(visible=False)
        emotion2 = gr.Number(visible=False)
        emotion3 = gr.Number(visible=False)
        emotion4 = gr.Number(visible=False)
        emotion5 = gr.Number(visible=False)
        emotion6 = gr.Number(visible=False)
        emotion7 = gr.Number(visible=False)
        emotion8 = gr.Number(visible=False)
        vq_single = gr.Number(visible=False)
        fmax = gr.Number(visible=False)
        pitch_std = gr.Number(visible=False)
        speaking_rate = gr.Number(visible=False)
        dnsmos_ovrl = gr.Number(visible=False)
        speaker_noised = gr.Checkbox(visible=False)
        cfg_scale_input = gr.Number(visible=False)
        min_k = gr.Number(visible=False)
        min_p = gr.Number(visible=False)
        linear = gr.Number(visible=False)
        confidence = gr.Number(visible=False)
        quadratic = gr.Number(visible=False)
        randomize_seed = gr.Checkbox(visible=False)
        unconditional_keys = gr.Textbox(visible=False)
        uuid_number = gr.Number(visible=False, value=uuid.uuid4())
        speed_input = gr.Number(visible=False)
        top_p_input = gr.Number(visible=False)
        top_k_input = gr.Number(visible=False)
        temperature_input = gr.Number(visible=False)
        repetition_penalty_input = gr.Number(visible=False)
        prefix_audio = gr.Audio( label="Optional Reference Audio (for style)", type="filepath",sources=["upload", "microphone"],visible=False)

        # Web UI button - uses visible sliders with generate_gradio_audio
        generate_button.click(fn=generate_gradio_audio,
            inputs=[model_choice, text, speaker_audio, prefix_audio, 
                cfg_scale, inference_timesteps, uuid_number],
                 outputs=[output_audio, uuid_number])
        
        # API-only button - uses hidden Number components with generate_audio
        # This is the endpoint that external API calls should use
        api_button = gr.Button(visible=False)
        api_button.click(fn=generate_audio,
            inputs=[model_choice, text, language, speaker_audio, prefix_audio, emotion1, emotion2, emotion3, emotion4, emotion5, emotion6, emotion7, emotion8,
                  vq_single, fmax, pitch_std, speaking_rate, dnsmos_ovrl, speaker_noised, cfg_scale_input, top_p_input,
                  min_k, min_p, linear, confidence, quadratic, uuid_number, randomize_seed, unconditional_keys],
                  outputs=[output_audio, uuid_number])
        
        # Expose only the API function for external calls
        gr.api(fn=generate_audio, api_name="generate_audio")
    return demo


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Initialize application environment
    initialize_application_environment("SkyrimNet VoxCPM Text-to-Speech Application with Gradio Interface")
    
    # Load model with standardized initialization
    CURRENT_MODEL = initialize_model_with_cache(device=args.device)

    demo = build_interface()
    demo.launch(server_name=args.server, server_port=args.port, share=args.share, inbrowser=args.inbrowser, debug=True, show_api=True)