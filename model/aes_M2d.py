# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import logging
from typing import Dict
from torch import nn
import torch

from utils import create_mlp_block
from wavlm import WavLM, WavLMConfig
from muq import MuQ
from portable_m2d import PortableM2D
from huggingface_hub import PyTorchModelHubMixin

logging = logging.getLogger(__name__)

DEFAULT_AUDIO_CFG = WavLMConfig({
    "extractor_mode": "default",
    "encoder_layers": 12,
    "encoder_embed_dim": 768,
    "encoder_ffn_embed_dim": 3072,
    "encoder_attention_heads": 12,
    "activation_fn": "gelu",
    "dropout": 0.1,
    "attention_dropout": 0.1,
    "activation_dropout": 0.0,
    "encoder_layerdrop": 0.05,
    "dropout_input": 0.1,
    "dropout_features": 0.1,
    "layer_norm_first": False,
    "conv_feature_layers": "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
    "conv_bias": False,
    "feature_grad_mult": 0.1,
    "mask_length": 10,
    "mask_prob": 0.8,
    "mask_selection": "static",
    "mask_other": 0.0,
    "no_mask_overlap": False,
    "mask_min_space": 1,
    "mask_channel_length": 10,
    "mask_channel_prob": 0.0,
    "mask_channel_selection": "static",
    "mask_channel_other": 0.0,
    "no_mask_channel_overlap": False,
    "mask_channel_min_space": 1,
    "conv_pos": 128,
    "conv_pos_groups": 16,
    "relative_position_embedding": True,
    "num_buckets": 320,
    "max_distance": 800,
    "gru_rel_pos": True,
    "normalize": False,
})

AXES_NAME = ["CE", "CU", "PC", "PQ"]

@dataclass(eq=False)
class AesMultiOutput(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/facebookresearch/audiobox-aesthetics",
    pipeline_tag="audio-classification",
    license="cc-by-4.0",
):
    proj_num_layer: int = 1
    proj_ln: bool = False
    proj_act_fn: str = "gelu"
    proj_dropout: float = 0
    nth_layer: int = 13
    use_weighted_layer_sum: bool = True
    precision: str = "32"
    normalize_embed: bool = True
    output_dim: int = 1
    freeze_encoder: bool = True
    m2d_ckpt: str = "your/path/to/m2d_clap_vit_base-80x1001p16x16-240128_AS-FT_enconly/weights_ep67it3124-0.48558.pth"

    def __post_init__(self):
        super().__init__()
        # WavLM backbone
        self.wavlm_model = WavLM(DEFAULT_AUDIO_CFG)
        wavlm_out_dim = self.wavlm_model.cfg.encoder_embed_dim
        # MuQ backbone
        self.muq_model = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
        muq_out_dim = self.muq_model.config.encoder_dim
        # M2D backbone
        self.m2d_model = PortableM2D(self.m2d_ckpt).eval()
        # infer the frame‐dim by a single dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 10 * 16000)            # 1×10 s mono waveform
            seq   = self.m2d_model(dummy)                # [1, F, D]
        m2d_out_dim = seq.shape[-1]                     # hidden dim D

        # combined embedding size
        combined_dim = wavlm_out_dim + muq_out_dim + m2d_out_dim

        self.proj_layer = nn.ModuleDict({
            x: nn.Sequential(*create_mlp_block(
                combined_dim,
                self.output_dim,
                self.proj_num_layer,
                self.proj_act_fn,
                self.proj_ln,
                dropout=self.proj_dropout,
            ))
            for x in AXES_NAME
        })
        if self.use_weighted_layer_sum:
            self.layer_weights = nn.ParameterDict({
                x: torch.nn.Parameter(torch.ones(self.nth_layer) / self.nth_layer)
                for x in AXES_NAME
            })

        precision_map = {
            "64": torch.float64,
            "32": torch.float32,
            "16": torch.half,
            "bf16": torch.bfloat16,
        }
        self.precision = precision_map[self.precision]
        self.enable_autocast = self.precision in {torch.half, torch.bfloat16}

    def forward(self, batch):
        assert batch["wav"].ndim == 3
        wav = batch["wav"].squeeze(1)

        if "mask" in batch:
            padding_mask = ~batch["mask"].squeeze(1)
        else:
            padding_mask = torch.zeros_like(wav, dtype=torch.bool)

        with torch.amp.autocast(
            device_type=wav.device.type,
            dtype=self.precision,
            enabled=self.enable_autocast,
        ), torch.set_grad_enabled(self.training):

            # WavLM features
            if self.wavlm_model.cfg.normalize:
                wav = torch.nn.functional.layer_norm(wav, wav.shape)
            with torch.set_grad_enabled(self.training and not self.freeze_encoder):
                (_, all_outputs), embed_mask = self.wavlm_model.extract_features(
                    source=wav,
                    padding_mask=padding_mask,
                    output_layer=self.nth_layer,
                    ret_layer_results=True,
                )
            all_outputs = torch.stack([o[0] for o in all_outputs], dim=-1)
            if self.use_weighted_layer_sum:
                weights = torch.nn.functional.softmax(self.layer_weights[AXES_NAME[0]], dim=-1)
                wavlm_emb = torch.einsum("tbcl,l->btc", all_outputs, weights)
            else:
                wavlm_emb = all_outputs[-1][0].transpose(1, 0)
            mask = (~embed_mask).unsqueeze(-1).type_as(wavlm_emb)
            wavlm_emb = (wavlm_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            if self.normalize_embed:
                wavlm_emb = torch.nn.functional.normalize(wavlm_emb, dim=-1)

            # MuQ features
            muq_out = self.muq_model(wav, output_hidden_states=True)
            muq_last = muq_out.hidden_states[-1]
            muq_emb = muq_last.mean(dim=1)
            if self.normalize_embed:
                muq_emb = torch.nn.functional.normalize(muq_emb, dim=-1)

            # M2D features
            m2d_seq = self.m2d_model(wav)          # [B, T, D]
            m2d_emb = m2d_seq.mean(dim=1)          # [B, D]
            if self.normalize_embed:
                m2d_emb = torch.nn.functional.normalize(m2d_emb, dim=-1)

            # concatenate and project
            combined = torch.cat([wavlm_emb, muq_emb, m2d_emb], dim=-1)
            preds = {
                name: self.proj_layer[name](combined).squeeze(-1)
                for name in AXES_NAME
            }

        return preds
