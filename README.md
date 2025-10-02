

Windows setup meant for use with SkyrimNet either locally (or local secondary PC) install using VoxCPM using either the "XTTS" or "Zonos" endpoints. 
- should support Blackwell cards
- model files in `models` folder
- latents in `latents_pt` folder by language type
- wav files in speakers/en will be converted once on startup (only en)
- output files saved in `output_temp` folder under process timestamp folders
- default server is at http://localhost:7860
- Gradio UI is available there also.

- Whisper might work using '/v1/audio/transcriptions' but not tested

Only checked with English.

Based on [VoXCPM](https://github.com/OpenBMB/VoxCPM)

Assumes that [Python 3.12](https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe) is already installed. 

To install other needed files:

`1_Install.bat` 

To run:

`2_Start.bat` 

This should start in a high priority process window.

or
`2_Start_CPU.bat` (for CPU only)


Note:
Currently hardcoded as the default values in the SkyrimNet Zonos configuration dont work well.

    cfg_scale=2.0,  
    inference_timesteps=10,  

See skyrimnet_config.txt on how to change these.

---

To run by hand:

py -3.12 -m venv .venv
.venv\scripts\activate
pip install -r requirements.txt

python -m skyrimnet-voxcpm

