# Copyright (c) 2026 Wonderful AI (authors: Xiang Lyu, Zhihao Du, Abdallah Farag)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random
from typing import Dict, Optional

logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
from torch.nn import functional as F
from .utils.mask import make_pad_mask
from .configs import CFM_PARAMS
from omegaconf import DictConfig


logger = logging.getLogger(__name__)


def _repeat_batch_dim(tnsr, B, ndim):
    "repeat batch dimension if it's equal to 1"
    if tnsr is not None:
        # add missing batch dim if needed
        while tnsr.ndim < ndim:
            tnsr = tnsr[None]
        # repeat batch dim as needed
        if B > 1 and tnsr.size(0) == 1:
            tnsr = tnsr.repeat(B, *([1] * (ndim - 1)))
        assert tnsr.ndim == ndim, f"Expected {ndim=}, got {tnsr.ndim=}"
    return tnsr


class CausalMaskedDiffWithXvec(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 6561,
                 input_frame_rate: int = 25,
                 only_mask_loss: bool = True,
                 token_mel_ratio: int = 2,
                 pre_lookahead_len: int = 3,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1,
                                       'cfm_params': DictConfig(
                                           {'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',
                                            'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7,
                                            'reg_loss_type': 'l1'}),
                                       'decoder_params': {'channels': [256, 256], 'dropout': 0.0,
                                                          'attention_head_dim': 64,
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8,
                                                          'act_fn': 'gelu'}},
                 mel_feat_conf: Dict = {'n_fft': 1024, 'num_mels': 80, 'sampling_rate': 22050,
                                        'hop_size': 256, 'win_size': 1024, 'fmin': 0, 'fmax': 8000}):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len

    # NOTE: copied in from cosyvoice repo
    def compute_loss(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        token = batch['speech_token'].to(device)
        token_len = batch['speech_token_len'].to(device)
        feat = batch['speech_feat'].to(device)  # (B, 80, T)
        feat_len = batch['speech_feat_len'].to(device)
        embedding = batch['embedding'].to(device)

        # NOTE unified training, static_chunk_size > 0 or = 0
        # streaming = True if random.random() < 0.5 else False

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)  # (B, T, 1)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask  # (B, T, emb)

        # text encode
        h, h_lengths = self.encoder(token, token_len)  # (B, T, C) -> (B, 2T, C)
        h = self.encoder_proj(h)

        # get conditions
        conds = torch.zeros(feat.shape, device=token.device)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :, :index] = feat[i, :, :index]

        mask = (~make_pad_mask(h_lengths.sum(dim=-1).squeeze(dim=1))).to(h)
        loss, _ = self.decoder.compute_loss(
            feat.contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding,
            cond=conds,
            # streaming=streaming,
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  finalize,
                  n_timesteps=10,
                  noised_mels=None,
                  meanflow=False,
                  encoder_cache=None):
        """
        Inference with optional encoder output caching for streaming.
        
        Args:
            encoder_cache: Optional dict with keys:
                - 'h': Cached encoder output (B, cached_mel_len, hidden_dim)
                - 'h_proj': Cached projected encoder output (B, cached_mel_len, output_size)
                - 'num_tokens': Number of tokens (including prompt) that produced this cache
                - 'prompt_len': Length of prompt tokens in the cached sequence
            
        Returns:
            feat: Generated mel spectrogram features
            new_cache: Updated encoder cache dict (None if finalize=True)
        """
        # token: (B, n_toks)
        # token_len: (B,)
        B = token.size(0)

        # xvec projection
        embedding = torch.atleast_2d(embedding)
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)  # (1 or B, emb_dim)

        # adjust shapes (batching logic)
        prompt_token = _repeat_batch_dim(prompt_token, B, ndim=2)  # (B, n_prompt)
        prompt_token_len = _repeat_batch_dim(prompt_token_len, B, ndim=1)  # (B,)
        prompt_feat = _repeat_batch_dim(prompt_feat, B, ndim=3)  # (B, n_feat, feat_dim=80)
        prompt_feat_len = _repeat_batch_dim(prompt_feat_len, B, ndim=1)  # (B,) or None
        embedding = _repeat_batch_dim(embedding, B, ndim=2)  # (B, emb_dim)

        # concat text and prompt_text
        full_token, full_token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(full_token_len)).unsqueeze(-1).to(embedding)

        if (full_token >= self.vocab_size).any():
            logger.error(f"{full_token.max()}>{self.vocab_size}\n out-of-range special tokens found in flow, fix inputs!")
        embedded_token = self.input_embedding(full_token.long()) * mask

        # Encoder with caching support
        current_num_tokens = full_token.size(1)
        prompt_len = prompt_token.size(1)
        
        # Check if we can use cached encoder output
        use_cache = (
            encoder_cache is not None 
            and encoder_cache.get('h') is not None
            and encoder_cache.get('num_tokens', 0) > 0
        )
        
        if use_cache:
            cached_num_tokens = encoder_cache['num_tokens']
            cached_h = encoder_cache['h']
            cached_h_proj = encoder_cache['h_proj']
            
            if current_num_tokens == cached_num_tokens:
                # Same number of tokens - reuse entire cache
                h = cached_h
                h_proj = cached_h_proj
                logger.debug(f"Encoder cache HIT: reusing {cached_num_tokens} tokens")
            elif current_num_tokens > cached_num_tokens:
                # New tokens added - encode only the new tokens and concatenate
                new_tokens_start = cached_num_tokens
                new_embedded_tokens = embedded_token[:, new_tokens_start:]
                new_token_len = full_token_len - cached_num_tokens
                
                # Encode only new tokens
                new_h, new_h_masks = self.encoder(new_embedded_tokens, new_token_len)
                
                # Concatenate with cached encoder output
                h = torch.cat([cached_h, new_h], dim=1)
                
                # Project new encoder output and concatenate
                new_h_proj = self.encoder_proj(new_h)
                h_proj = torch.cat([cached_h_proj, new_h_proj], dim=1)
                
                logger.debug(f"Encoder cache PARTIAL: cached {cached_num_tokens}, encoded {current_num_tokens - cached_num_tokens} new tokens")
            else:
                # Fewer tokens than cached (shouldn't happen in normal streaming) - re-encode all
                h, h_masks = self.encoder(embedded_token, full_token_len)
                h_proj = self.encoder_proj(h)
                logger.debug(f"Encoder cache MISS: token count decreased, re-encoding all {current_num_tokens} tokens")
        else:
            # No cache - encode all tokens
            h, h_masks = self.encoder(embedded_token, full_token_len)
            h_proj = self.encoder_proj(h)
            logger.debug(f"Encoder cache MISS: encoding all {current_num_tokens} tokens")
        
        # Store full (untrimmed) h for cache before any trimming
        h_full = h
        h_proj_full = h_proj
        
        # Apply lookahead trimming for non-final chunks
        if finalize is False:
            trim_amount = self.pre_lookahead_len * self.token_mel_ratio
            h_proj = h_proj[:, :-trim_amount]
        
        # Calculate lengths and mel dimensions
        h_lengths = torch.tensor([h_proj.size(1)], device=h_proj.device)
        if h_lengths.size(0) != B:
            h_lengths = h_lengths.expand(B)
        
        mel_len1 = prompt_feat.shape[1]
        mel_len2 = h_proj.shape[1] - mel_len1

        # Get conditions
        conds = torch.zeros([B, mel_len1 + mel_len2, self.output_size], device=embedded_token.device).to(h_proj.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(h_lengths)).unsqueeze(1).to(h_proj)

        if mask.shape[0] != B:
            mask = mask.repeat(B, 1, 1)

        feat, _ = self.decoder(
            mu=h_proj.transpose(1, 2).contiguous(),
            mask=mask,
            spks=embedding,
            cond=conds,
            n_timesteps=n_timesteps,
            noised_mels=noised_mels,
            meanflow=meanflow,
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        
        # Build new cache (only if not finalizing)
        new_cache = None
        if not finalize:
            new_cache = {
                'h': h_full,
                'h_proj': h_proj_full,
                'num_tokens': current_num_tokens,
                'prompt_len': prompt_len,
            }
        
        return feat, new_cache
