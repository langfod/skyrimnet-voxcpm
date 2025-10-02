"""
VoxCPM: A Tokenizer-free speech generation model

This module contains the main VoxCPM model implementation, including configuration classes
and the core VoxCPMModel for text-to-speech generation.

Copyright 2025 OpenBMB
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
from typing import Tuple, Union, Generator, List
import warnings

import torch
import torch.nn as nn
import torchaudio
from einops import rearrange
from pydantic import BaseModel
from tqdm import tqdm
from transformers import LlamaTokenizerFast

from ..modules.audiovae import AudioVAE
from ..modules.layers import ScalarQuantizationLayer
from ..modules.locdit import CfmConfig, UnifiedCFM, VoxCPMLocDiT
from ..modules.locenc import VoxCPMLocEnc
from ..modules.minicpm4 import MiniCPM4Config, MiniCPMModel
from .utils import get_dtype, mask_multichar_chinese_tokens


class VoxCPMEncoderConfig(BaseModel):
    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 4
    kv_channels: int = None


class VoxCPMDitConfig(BaseModel):
    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 4
    kv_channels: int = None

    cfm_config: CfmConfig


class VoxCPMConfig(BaseModel):
    lm_config: MiniCPM4Config
    patch_size: int = 2
    feat_dim: int = 64
    residual_lm_num_layers: int = 6
    scalar_quantization_latent_dim: int = 256
    scalar_quantization_scale: int = 9

    encoder_config: VoxCPMEncoderConfig
    dit_config: VoxCPMDitConfig

    max_length: int = 4096
    device: str = "cuda"
    dtype: str = "bfloat16"


class VoxCPMModel(nn.Module):
    def __init__(
        self,
        config: VoxCPMConfig,
        tokenizer: LlamaTokenizerFast,
        audio_vae: AudioVAE,
    ):
        super().__init__()
        self.config = config
        self.feat_dim = config.feat_dim
        self.patch_size = config.patch_size
        self.device = config.device
        if not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        print(f"Running on device: {self.device}, dtype: {self.config.dtype}")

        # Text-Semantic LM
        self.base_lm = MiniCPMModel(config.lm_config)
        self.base_lm.setup_cache(1, config.max_length, self.device, get_dtype(self.config.dtype))

        self.text_tokenizer = mask_multichar_chinese_tokens(tokenizer)
        self.audio_start_token = 101
        self.audio_end_token = 102

        # Residual Acoustic LM
        residual_lm_config = config.lm_config.model_copy(deep=True)
        residual_lm_config.num_hidden_layers = config.residual_lm_num_layers
        residual_lm_config.vocab_size = 0
        self.residual_lm = MiniCPMModel(residual_lm_config)
        self.residual_lm.setup_cache(1, config.max_length, self.device, get_dtype(self.config.dtype))

        # Local Encoder
        encoder_config = config.lm_config.model_copy(deep=True)
        encoder_config.hidden_size = config.encoder_config.hidden_dim
        encoder_config.intermediate_size = config.encoder_config.ffn_dim
        encoder_config.num_attention_heads = config.encoder_config.num_heads
        encoder_config.num_hidden_layers = config.encoder_config.num_layers
        encoder_config.kv_channels = config.encoder_config.kv_channels
        encoder_config.vocab_size = 0
        self.feat_encoder = VoxCPMLocEnc(encoder_config, input_dim=config.feat_dim)

        # Local DiT
        decoder_config = config.lm_config.model_copy(deep=True)
        decoder_config.hidden_size = config.dit_config.hidden_dim
        decoder_config.intermediate_size = config.dit_config.ffn_dim
        decoder_config.num_attention_heads = config.dit_config.num_heads
        decoder_config.num_hidden_layers = config.dit_config.num_layers
        decoder_config.kv_channels = config.dit_config.kv_channels
        decoder_config.vocab_size = 0
        self.feat_decoder = UnifiedCFM(
            in_channels=config.feat_dim,
            cfm_params=config.dit_config.cfm_config,
            estimator=VoxCPMLocDiT(decoder_config, in_channels=config.feat_dim),
        )

        # Projection layers
        self.fsq_layer = ScalarQuantizationLayer(
            config.lm_config.hidden_size, 
            config.lm_config.hidden_size, 
            config.scalar_quantization_latent_dim, 
            config.scalar_quantization_scale
        )
        self.enc_to_lm_proj = nn.Linear(config.encoder_config.hidden_dim, config.lm_config.hidden_size)
        self.lm_to_dit_proj = nn.Linear(config.lm_config.hidden_size, config.dit_config.hidden_dim)
        self.res_to_dit_proj = nn.Linear(config.lm_config.hidden_size, config.dit_config.hidden_dim)

        # Stop Predictor
        self.stop_proj = nn.Linear(config.lm_config.hidden_size, config.lm_config.hidden_size)
        self.stop_actn = nn.SiLU()
        self.stop_head = nn.Linear(config.lm_config.hidden_size, 2, bias=False)

        # Audio VAE
        self.audio_vae = audio_vae
        self.chunk_size = audio_vae.chunk_size
        self.sample_rate = audio_vae.sample_rate

    
    def optimize(self, disable: bool = False, force_mode: str = None):
        """
        Optimize VoxCPM model with torch.compile using progressive fallback strategy.
        
        Tries compilation modes in order of decreasing aggressiveness:
        1. reduce-overhead (most aggressive, may fail on Windows with large tensors)
        2. default (moderate optimization, better compatibility)
        3. disable optimization (fallback)
        
        Args:
            disable: If True, skip optimization entirely
            force_mode: If specified, force a specific compilation mode
        """
        try:
            if disable:
                print("Disabling torch.compile optimization")
                # Store references to original functions if not already stored
                if not hasattr(self, '_original_base_lm_forward_step'):
                    self._original_base_lm_forward_step = self.base_lm.forward_step.__func__ if hasattr(self.base_lm.forward_step, '__func__') else self.base_lm.forward_step
                    self._original_residual_lm_forward_step = self.residual_lm.forward_step.__func__ if hasattr(self.residual_lm.forward_step, '__func__') else self.residual_lm.forward_step
                    self._original_feat_encoder = self.feat_encoder
                    self._original_feat_decoder_estimator = self.feat_decoder.estimator
                
                # Reset to original functions
                self.base_lm.forward_step = self._original_base_lm_forward_step
                self.residual_lm.forward_step = self._original_residual_lm_forward_step
                self.feat_encoder_step = self._original_feat_encoder
                self.feat_decoder.estimator = self._original_feat_decoder_estimator
                return self
                
            if self.device != "cuda":
                raise ValueError("VoxCPMModel can only be optimized on CUDA device")
            try:
                import triton
            except:
                raise ValueError("triton is not installed")
            
            # Store original functions before compilation attempt
            if not hasattr(self, '_original_base_lm_forward_step'):
                self._original_base_lm_forward_step = self.base_lm.forward_step
                self._original_residual_lm_forward_step = self.residual_lm.forward_step
                self._original_feat_encoder = self.feat_encoder
                self._original_feat_decoder_estimator = self.feat_decoder.estimator
            
            # Handle forced mode directly without loop
            if force_mode:
                mode = force_mode
                fullgraph = False
                print(f"Attempting torch.compile with forced mode='{mode}'")
                
                # Compile all components with forced mode
                self.base_lm.forward_step = torch.compile(
                    self._original_base_lm_forward_step, 
                    mode=mode, 
                    fullgraph=fullgraph
                )
                self.residual_lm.forward_step = torch.compile(
                    self._original_residual_lm_forward_step, 
                    mode=mode, 
                    fullgraph=fullgraph
                )
                self.feat_encoder_step = torch.compile(
                    self._original_feat_encoder, 
                    mode=mode, 
                    fullgraph=fullgraph
                )
                self.feat_decoder.estimator = torch.compile(
                    self._original_feat_decoder_estimator, 
                    mode=mode, 
                    fullgraph=fullgraph
                )
                
                print(f"✓ Successfully optimized VoxCPMModel with forced mode='{mode}'")
                return self
            
            # Try fallback modes only when no force_mode is specified
            compilation_modes = [
                ("reduce-overhead", True, "Maximum optimization (may fail on Windows)"),
                ("default", False, "Standard optimization (better compatibility)"),
            ]
            
            for mode, fullgraph, description in compilation_modes:
                try:
                    print(f"Attempting torch.compile with mode='{mode}' - {description}")
                    
                    # Try compiling all components with current mode
                    self.base_lm.forward_step = torch.compile(
                        self._original_base_lm_forward_step, 
                        mode=mode, 
                        fullgraph=fullgraph
                    )
                    self.residual_lm.forward_step = torch.compile(
                        self._original_residual_lm_forward_step, 
                        mode=mode, 
                        fullgraph=fullgraph
                    )
                    self.feat_encoder_step = torch.compile(
                        self._original_feat_encoder, 
                        mode=mode, 
                        fullgraph=fullgraph
                    )
                    self.feat_decoder.estimator = torch.compile(
                        self._original_feat_decoder_estimator, 
                        mode=mode, 
                        fullgraph=fullgraph
                    )
                    
                    print(f"✓ Successfully optimized VoxCPMModel with mode='{mode}'")
                    return self
                    
                except Exception as compile_e:
                    print(f"✗ torch.compile mode='{mode}' failed: {compile_e}")
                    # Reset to original functions before trying next mode
                    self.base_lm.forward_step = self._original_base_lm_forward_step
                    self.residual_lm.forward_step = self._original_residual_lm_forward_step
                    self.feat_encoder_step = self._original_feat_encoder
                    self.feat_decoder.estimator = self._original_feat_decoder_estimator
                    continue
            
            # If all compilation modes fail, use original functions
            raise ValueError("All torch.compile modes failed")
            
        except Exception as e:
            print(f"Error: {e}")
            print("Warning: VoxCPMModel can not be optimized by torch.compile, using original forward_step functions")
            # Ensure we fall back to original functions
            if hasattr(self, '_original_base_lm_forward_step'):
                self.base_lm.forward_step = self._original_base_lm_forward_step
                self.residual_lm.forward_step = self._original_residual_lm_forward_step
                self.feat_encoder_step = self._original_feat_encoder
                self.feat_decoder.estimator = self._original_feat_decoder_estimator
        return self


    def generate(self, *args, **kwargs) -> torch.Tensor:
        return next(self._generate(*args, streaming=False, **kwargs))

    def generate_streaming(self, *args, **kwargs) -> Generator[torch.Tensor, None, None]:
        return self._generate(*args, streaming=True, **kwargs)

    @torch.inference_mode()
    def _generate(
        self,
        target_text: str,
        prompt_text: str = "",
        prompt_wav_path: str = "",
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        retry_badcase: bool = False,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0, # setting acceptable ratio of audio length to text length (for badcase detection)
        streaming: bool = False,
    ) -> Generator[torch.Tensor, None, None]:
        if retry_badcase and streaming:
            warnings.warn("Retry on bad cases is not supported in streaming mode, setting retry_badcase=False.")
            retry_badcase = False
        if len(prompt_wav_path) == 0:
            text = target_text
            text_token = torch.LongTensor(self.text_tokenizer(text))
            text_token = torch.cat(
                [
                    text_token,
                    torch.tensor(
                        [self.audio_start_token],
                        dtype=torch.int32,
                        device=text_token.device,
                    ),
                ],
                dim=-1,
            )
            text_length = text_token.shape[0]

            audio_feat = torch.zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )
            text_mask = torch.ones(text_length).type(torch.int32).to(text_token.device)
            audio_mask = torch.zeros(text_length).type(torch.int32).to(text_token.device)

        else:
            text = prompt_text + target_text
            text_token = torch.LongTensor(self.text_tokenizer(text))
            text_token = torch.cat(
                [
                    text_token,
                    torch.tensor([self.audio_start_token], dtype=torch.int32, device=text_token.device),
                ],
                dim=-1,
            )
            text_length = text_token.shape[0]

            audio, sr = torchaudio.load(prompt_wav_path)
            if audio.size(0) > 1:
                audio = audio.mean(dim=0, keepdim=True)
                
            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

            patch_len = self.patch_size * self.chunk_size

            if audio.size(1) % patch_len != 0:
                audio = torch.nn.functional.pad(audio, (0, patch_len - audio.size(1) % patch_len))

            # (B, D, T) - keep on device for consistency
            audio_feat = self.audio_vae.encode(audio.to(self.device), self.sample_rate)

            audio_feat = audio_feat.view(
                self.audio_vae.latent_dim,
                -1,
                self.patch_size,
            ).permute(1, 2, 0)
            audio_feat = audio_feat[:-1, ...] # trick: remove the last padding token
            audio_length = audio_feat.size(0)
            text_pad_token = torch.zeros(audio_length, dtype=torch.int32, device=text_token.device)
            text_token = torch.cat([text_token, text_pad_token])
            audio_pad_feat = torch.zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )
            audio_feat = torch.cat([audio_pad_feat, audio_feat], dim=0)
            text_mask = (
                torch.cat([torch.ones(text_length), torch.zeros(audio_length)]).type(torch.int32).to(text_token.device)
            )
            audio_mask = (
                torch.cat([torch.zeros(text_length), torch.ones(audio_length)]).type(torch.int32).to(text_token.device)
            )

        text_token = text_token.unsqueeze(0).to(self.device)
        text_mask = text_mask.unsqueeze(0).to(self.device)
        audio_feat = audio_feat.unsqueeze(0).to(self.device).to(get_dtype(self.config.dtype))
        audio_mask = audio_mask.unsqueeze(0).to(self.device)

        target_text_length = len(self.text_tokenizer(target_text))
        
        retry_badcase_times = 0
        while retry_badcase_times < retry_badcase_max_times:
            inference_result = self._inference(
                text_token,
                text_mask,
                audio_feat,
                audio_mask,
                min_len=min_len,
                max_len=int(target_text_length * retry_badcase_ratio_threshold + 10) if retry_badcase else max_len,
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                streaming=streaming,
                keep_on_device=True,  # Keep on GPU for audio_vae.decode()
            )
            if streaming:
                patch_len = self.patch_size * self.chunk_size
                for latent_pred, _ in inference_result:
                    decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32))
                    decode_audio = decode_audio[..., -patch_len:].squeeze(1).cpu()
                    yield decode_audio
                break
            else:
                latent_pred, pred_audio_feat = next(inference_result)
                if retry_badcase:
                    if pred_audio_feat.shape[0] >= target_text_length * retry_badcase_ratio_threshold:
                        print(f"  Badcase detected, audio_text_ratio={pred_audio_feat.shape[0] / target_text_length}, retrying...")
                        retry_badcase_times += 1
                        continue
                    else:
                        break
                else:
                    break   
                
        if not streaming:
            decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32)).squeeze(1).cpu()  
            decode_audio = decode_audio[..., 640:-640] # trick: trim the start and end of the audio
            yield decode_audio        
    
    @torch.inference_mode()
    def build_prompt_cache(
        self,
        prompt_text: str,
        prompt_wav_path: str,
    ):
        """
        Build prompt cache for subsequent fast generation.
        
        Args:
            prompt_text: prompt text (required)
            prompt_wav_path: prompt audio path (required)
            
        Returns:
            prompt_cache: dict with text tokens and audio features
        """
        if not prompt_text or not prompt_wav_path:
            raise ValueError("prompt_text and prompt_wav_path are required")
        
        # build text tokens
        text_token = torch.LongTensor(self.text_tokenizer(prompt_text))

        # load audio
        audio, sr = torchaudio.load(prompt_wav_path)
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
            
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

        patch_len = self.patch_size * self.chunk_size

        if audio.size(1) % patch_len != 0:
            audio = torch.nn.functional.pad(audio, (0, patch_len - audio.size(1) % patch_len))

        # extract audio features (keep on model device for performance)
        audio_feat = self.audio_vae.encode(audio.to(self.device), self.sample_rate)

        audio_feat = audio_feat.view(
            self.audio_vae.latent_dim,
            -1,
            self.patch_size,
        ).permute(1, 2, 0) # (D, T, P)
        audio_feat = audio_feat[:-1, ...] # trick: remove the last padding token
        # build prompt cache
        prompt_cache = {
            "text_token": text_token,
            "audio_feat": audio_feat,
        }
        
        return prompt_cache

    
    def merge_prompt_cache(
        self,
        original_cache: dict,
        new_text_token: torch.Tensor,
        new_audio_feat: torch.Tensor,
    ):
        """
        Merge original prompt cache with newly generated content to stabilize voice.
        
        Args:
            original_cache: original prompt cache
            new_text_token: newly generated text tokens
            new_audio_feat: newly generated audio features
            
        Returns:
            merged_cache: merged cache
        """
        if original_cache is None:
            return {
                "text_token": new_text_token,
                "audio_feat": new_audio_feat,
            }
        original_text_token = original_cache["text_token"]
        original_audio_feat = original_cache["audio_feat"]
        merged_text_token = torch.cat([original_text_token, new_text_token], dim=0)
        merged_audio_feat = torch.cat([original_audio_feat, new_audio_feat], dim=0)

        # build new cache
        merged_cache = {
            "text_token": merged_text_token,
            "audio_feat": merged_audio_feat,
        }
        
        return merged_cache

    def generate_with_prompt_cache(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return next(self._generate_with_prompt_cache(*args, streaming=False, **kwargs))

    def generate_with_prompt_cache_streaming(
        self, *args, **kwargs
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]], None, None]:
        return self._generate_with_prompt_cache(*args, streaming=True, **kwargs)

    @torch.inference_mode()
    def _generate_with_prompt_cache(
        self,
        target_text: Union[str, List[str]],
        prompt_cache: dict,
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        retry_badcase: bool = False,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        streaming: bool = False,
        auto_chunk: bool = True,
        chunk_max_tokens: int = 3200,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]], None, None]:
        """
        Generate audio using pre-built prompt cache with automatic text chunking support.
        
        Args:
            target_text: Text to convert to speech (str) or pre-split chunks (List[str])
            prompt_cache: Cache built by build_prompt_cache (can be None)
            min_len: Minimum audio length to avoid very short audio
            max_len: Maximum audio length per chunk
            inference_timesteps: Number of diffusion sampling steps
            cfg_value: Classifier-free guidance value
            retry_badcase: Whether to retry on bad cases
            retry_badcase_max_times: Maximum retry attempts
            retry_badcase_ratio_threshold: Threshold for audio-to-text ratio
            streaming: Whether to return a generator of audio chunks
            auto_chunk: Whether to automatically split long text (only applies to str input)
            chunk_max_tokens: Maximum tokens per chunk when auto_chunk=True
            
        Returns:
            Generator of Tuple containing:
                - Decoded audio tensor for the current step if ``streaming=True``, else final decoded audio tensor
                - Tensor of new text tokens
                - New audio features up to the current step as a List if ``streaming=True``, else as a concatenated Tensor
        """
        if retry_badcase and streaming:
            warnings.warn("Retry on bad cases is not supported in streaming mode, setting retry_badcase=False.")
            retry_badcase = False
        
        # Handle text chunking
        if isinstance(target_text, list):
            # Already chunked - process each chunk
            text_chunks = target_text
        elif auto_chunk:
            # Check if text needs chunking using tokenizer
            temp_tokens = self.text_tokenizer(target_text)
            if len(temp_tokens) > chunk_max_tokens:
                # Split the text into chunks using shared utilities
                from ...shared_text_utils import get_optimal_text_split_points
                text_chunks = get_optimal_text_split_points(target_text, max_tokens=chunk_max_tokens, model=self)
            else:
                text_chunks = [target_text]
        else:
            # Single text, no chunking
            text_chunks = [target_text]
        
        # If multiple chunks, process sequentially and merge results
        if len(text_chunks) > 1:
            accumulated_audio = []
            accumulated_text_tokens = []
            accumulated_audio_features = []
            current_cache = prompt_cache
            
            for chunk_idx, chunk_text in enumerate(text_chunks):
                # Generate audio for this chunk
                chunk_generator = self._generate_with_prompt_cache(
                    target_text=chunk_text,
                    prompt_cache=current_cache,
                    min_len=min_len,
                    max_len=max_len,
                    inference_timesteps=inference_timesteps,
                    cfg_value=cfg_value,
                    retry_badcase=retry_badcase,
                    retry_badcase_max_times=retry_badcase_max_times,
                    retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                    streaming=False,  # Process chunks individually, not streaming
                    auto_chunk=False,  # Prevent recursive chunking
                    chunk_max_tokens=chunk_max_tokens
                )
                
                chunk_audio, chunk_text_tokens, chunk_audio_feat = next(chunk_generator)
                
                # Accumulate results
                accumulated_audio.append(chunk_audio)
                accumulated_text_tokens.append(chunk_text_tokens)
                if isinstance(chunk_audio_feat, list):
                    accumulated_audio_features.extend(chunk_audio_feat)
                else:
                    accumulated_audio_features.append(chunk_audio_feat)
                
                # Build new cache for next chunk using merge_prompt_cache
                if chunk_idx < len(text_chunks) - 1:  # Not the last chunk
                    if current_cache is None:
                        # Create initial cache from this chunk's output
                        chunk_text_token = self.text_tokenizer(chunk_text)
                        current_cache = {
                            "text_token": torch.LongTensor(chunk_text_token),
                            "audio_feat": chunk_audio_feat if not isinstance(chunk_audio_feat, list) else torch.cat(chunk_audio_feat, dim=0)
                        }
                    else:
                        # Merge current cache with this chunk's output
                        chunk_text_token = self.text_tokenizer(chunk_text)
                        new_cache = {
                            "text_token": torch.LongTensor(chunk_text_token),
                            "audio_feat": chunk_audio_feat if not isinstance(chunk_audio_feat, list) else torch.cat(chunk_audio_feat, dim=0)
                        }
                        current_cache = self.merge_prompt_cache(current_cache, new_cache)
                
                # Yield intermediate results if streaming
                if streaming:
                    yield (
                        chunk_audio,
                        chunk_text_tokens, 
                        accumulated_audio_features.copy()
                    )
            
            # Combine all chunks
            final_audio = torch.cat(accumulated_audio, dim=-1)  # Concatenate along time dimension
            final_text_tokens = torch.cat(accumulated_text_tokens, dim=0)
            final_audio_features = accumulated_audio_features if streaming else torch.cat(accumulated_audio_features, dim=0)
            
            # Yield final result
            yield (final_audio, final_text_tokens, final_audio_features)
            return
        
        # Single chunk processing (original logic)
        target_text = text_chunks[0]
        
        # get prompt from cache
        if prompt_cache is None:
            prompt_text_token = torch.empty(0, dtype=torch.int32)
            prompt_audio_feat = torch.empty((0, self.patch_size, self.audio_vae.latent_dim), dtype=torch.float32)
        else:
            prompt_text_token = prompt_cache["text_token"]
            prompt_audio_feat = prompt_cache["audio_feat"]
        # build target text tokens (ensure same device as prompt tokens)
        target_text_token = torch.LongTensor(self.text_tokenizer(target_text))
        if len(prompt_text_token) > 0:
            target_text_token = target_text_token.to(prompt_text_token.device)
        else:
            target_text_token = target_text_token.to(self.device)
        text_token = torch.cat([prompt_text_token, target_text_token], dim=0)
        text_token = torch.cat(
            [
                text_token,
                torch.tensor(
                    [self.audio_start_token],
                    dtype=torch.int32,
                    device=text_token.device,
                ),
            ],
            dim=-1,
        )

        audio_length = prompt_audio_feat.size(0)
        text_length = text_token.shape[0]
        
        # Create tensors directly on target device to avoid transfers
        device = self.device
        dtype = get_dtype(self.config.dtype)
        
        text_pad_token = torch.zeros(audio_length, dtype=torch.int32, device=device)
        audio_pad_feat = torch.zeros(
            (text_length, self.patch_size, self.audio_vae.latent_dim),
            dtype=dtype,
            device=device,
        )
        
        # Move inputs to device once and concatenate (optimize device transfers)
        text_token = torch.cat([text_token.to(device), text_pad_token])
        prompt_audio_feat_gpu = prompt_audio_feat.to(device, dtype)  # Single combined transfer
        audio_feat = torch.cat([audio_pad_feat, prompt_audio_feat_gpu], dim=0)
        
        # Create masks directly on device
        text_mask = torch.cat([
            torch.ones(text_length, dtype=torch.int32, device=device), 
            torch.zeros(audio_length, dtype=torch.int32, device=device)
        ])
        audio_mask = torch.cat([
            torch.zeros(text_length, dtype=torch.int32, device=device), 
            torch.ones(audio_length, dtype=torch.int32, device=device)
        ])

        # Add batch dimension (no additional device transfer needed)
        text_token = text_token.unsqueeze(0)
        text_mask = text_mask.unsqueeze(0)
        audio_feat = audio_feat.unsqueeze(0)
        audio_mask = audio_mask.unsqueeze(0)
    
        # run inference
        target_text_length = len(self.text_tokenizer(target_text))
        retry_badcase_times = 0
        while retry_badcase_times < retry_badcase_max_times:
            inference_result = self._inference(
                text_token,
                text_mask,
                audio_feat,
                audio_mask,
                min_len=min_len,
                max_len=int(target_text_length * retry_badcase_ratio_threshold + 10) if retry_badcase else max_len,
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                streaming=streaming,
                keep_on_device=True,  # Keep on GPU for audio_vae.decode()
            )
            if streaming:
                patch_len = self.patch_size * self.chunk_size
                for latent_pred, pred_audio_feat in inference_result:
                    decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32))
                    decode_audio = decode_audio[..., -patch_len:].squeeze(1).cpu()
                    yield (
                        decode_audio,
                        target_text_token,
                        pred_audio_feat
                    )
                break
            else:
                latent_pred, pred_audio_feat = next(inference_result)
                if retry_badcase:
                    if pred_audio_feat.shape[0] >= target_text_length * retry_badcase_ratio_threshold:
                        print(f"  Badcase detected, audio_text_ratio={pred_audio_feat.shape[0] / target_text_length}, retrying...")
                        retry_badcase_times += 1
                        continue
                    else:
                        break
                else:
                    break
        if not streaming:
            decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32)).squeeze(1)
            decode_audio = decode_audio[..., 640:-640]  # trick: trim the start and end of the audio
            # Move to CPU only at the very end to avoid unnecessary GPU/CPU transfers
            #decode_audio = decode_audio.cpu()

            yield (
                decode_audio,
                target_text_token,
                pred_audio_feat
            )

    def inference(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        return next(self._inference(*args, streaming=False, **kwargs))
    
    def inference_streaming(self, *args, **kwargs) -> Generator[Tuple[torch.Tensor, List[torch.Tensor]], None, None]:
        return self._inference(*args, streaming=True, **kwargs)

    @torch.inference_mode()
    def _inference(
        self,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        feat: torch.Tensor,
        feat_mask: torch.Tensor,
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        streaming: bool = False,
        keep_on_device: bool = False,
    ) -> Generator[Tuple[torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]], None, None]:
        """Core inference method for audio generation.
        
        This is the main inference loop that generates audio features
        using the language model and diffusion transformer.
        
        Args:
            text: Input text tokens
            text_mask: Mask for text tokens
            feat: Input audio features
            feat_mask: Mask for audio features
            min_len: Minimum generation length
            max_len: Maximum generation length
            inference_timesteps: Number of diffusion steps
            cfg_value: Classifier-free guidance value
            streaming: Whether to yield each step latent feature or just the final result
            keep_on_device: If True, keep results on GPU to avoid unnecessary transfers
            
        Returns:
            Generator of Tuple containing:
                - Predicted latent feature at the current step if ``streaming=True``, else final latent features
                - Predicted audio feature sequence so far as a List if ``streaming=True``, else as a concatenated Tensor
                  (on GPU if keep_on_device=True, else on CPU)
        """
        B, T, P, D = feat.shape

        feat_embed = self.feat_encoder(feat)  # [b, t, h_feat]
        feat_embed = self.enc_to_lm_proj(feat_embed)
        
        if self.config.lm_config.use_mup:
            scale_emb = self.config.lm_config.scale_emb
        else:
            scale_emb = 1.0
       
        text_embed = self.base_lm.embed_tokens(text) * scale_emb
        combined_embed = text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed

        prefix_feat_cond = feat[:, -1, ...]  # b, p, d
        pred_feat_seq = []  # b, t, p, d
        curr_embed = None

        enc_outputs, kv_cache_tuple = self.base_lm(
            inputs_embeds=combined_embed,
            is_causal=True,
        )
        self.base_lm.kv_cache.fill_caches(kv_cache_tuple)
        
        enc_outputs = self.fsq_layer(enc_outputs) * feat_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
        lm_hidden = enc_outputs[:, -1, :]

         
        residual_enc_outputs, residual_kv_cache_tuple = self.residual_lm(
            inputs_embeds=enc_outputs + feat_mask.unsqueeze(-1) * feat_embed,
            is_causal=True,
        )
        self.residual_lm.kv_cache.fill_caches(residual_kv_cache_tuple)
        residual_hidden = residual_enc_outputs[:, -1, :]


        for i in tqdm(range(max_len)):
            dit_hidden_1 = self.lm_to_dit_proj(lm_hidden)  # [b, h_dit]
            dit_hidden_2 = self.res_to_dit_proj(residual_hidden)  # [b, h_dit]
            dit_hidden = dit_hidden_1 + dit_hidden_2  # [b, h_dit]

            pred_feat = self.feat_decoder(
                mu=dit_hidden,
                patch_size=self.patch_size,
                cond=prefix_feat_cond.transpose(1, 2).contiguous(),
                n_timesteps=inference_timesteps,
                cfg_value=cfg_value,
            ).transpose(
                1, 2
            )  # [b, p, d]
            
            curr_embed = self.feat_encoder_step(pred_feat.unsqueeze(1))  # b, 1, c
            curr_embed = self.enc_to_lm_proj(curr_embed)
            
            pred_feat_seq.append(pred_feat.unsqueeze(1))  # b, 1, p, d
            prefix_feat_cond = pred_feat

            if streaming:
                # return the last three predicted latent features to provide enough context for smooth decoding
                pred_feat_chunk = torch.cat(pred_feat_seq[-3:], dim=1)
                feat_pred = rearrange(pred_feat_chunk, "b t p d -> b d (t p)", b=B, p=self.patch_size)
                yield feat_pred, pred_feat_seq
            
            stop_flag = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden))).argmax(dim=-1)[0].cpu().item()
            if i > min_len and stop_flag == 1:
                break
    
            # Use consistent device (self.device) to avoid device mismatches
            lm_hidden = self.base_lm.forward_step(
                curr_embed[:, 0, :], torch.tensor([self.base_lm.kv_cache.step()], device=self.device)
            ).clone()
           

            lm_hidden = self.fsq_layer(lm_hidden)
            residual_hidden = self.residual_lm.forward_step(
                lm_hidden + curr_embed[:, 0, :], torch.tensor([self.residual_lm.kv_cache.step()], device=self.device)
            ).clone()
                
        if not streaming:
            pred_feat_seq = torch.cat(pred_feat_seq, dim=1)  # b, t, p, d

            feat_pred = rearrange(pred_feat_seq, "b t p d -> b d (t p)", b=B, p=self.patch_size)
            # Conditionally move to CPU based on keep_on_device flag
            if keep_on_device:
                # Keep tensors on GPU for further processing (e.g., in _generate_with_prompt_cache)
                yield feat_pred, pred_feat_seq.squeeze(0)
            else:
                # Move to CPU for final result (traditional behavior)
                yield feat_pred, pred_feat_seq.squeeze(0).cpu()

    @classmethod
    def from_local(cls, path: str, optimize: bool = True):
        config = VoxCPMConfig.model_validate_json(open(os.path.join(path, "config.json")).read())

        tokenizer = LlamaTokenizerFast.from_pretrained(path)

        audio_vae = AudioVAE()
        vae_state_dict = torch.load(
            os.path.join(path, "audiovae.pth"),
            map_location="cpu",
            weights_only=True,
        )["state_dict"]

        model = cls(config, tokenizer, audio_vae)
        lm_dtype = get_dtype(model.config.dtype)
        model = model.to(lm_dtype)
        model.audio_vae = model.audio_vae.to(torch.float32)

        model_state_dict = torch.load(
            os.path.join(path, "pytorch_model.bin"),
            map_location="cpu",
            weights_only=True,
        )["state_dict"]

        for kw, val in vae_state_dict.items():
            model_state_dict[f"audio_vae.{kw}"] = val
        model.load_state_dict(model_state_dict, strict=True)
        return model.to(model.device).eval().optimize(disable=not optimize)
