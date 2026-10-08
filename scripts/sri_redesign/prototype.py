"""Isolated SRI predictor prototype; no data, training or checkpoint entry point.

Values and binary visibility enter as [B,L,2C]. Hidden values are discarded
before interpolation. A single patch embedding, temporal stack and decoder
are shared by all variables. time_channel adds variable attention after each
temporal block; time_time uses the same second-block parameters along time.
Independence/equivariance statements concern the predictor:
the retained dense temporal refiner mixes variables and uses variable-specific
weights. Refiner-off bypasses that module but retains its registered parameters.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from models.patchtst import TransformerEncoderLayer
from utils.interp import linear_interp_torch


class SharedPatchSRI(nn.Module):
    def __init__(
        self, channels: int, variant: str = "time_channel", *,
        seq_len: int = 96, patch_len: int = 16, stride: int = 8,
        d_model: int = 64, n_heads: int = 4, d_ff: int = 128,
        layers: int = 3, dropout: float = 0.1, refiner: bool = True,
        learnable_refine_scale: bool = True,
    ):
        super().__init__()
        sizes = (channels, seq_len, patch_len, stride, d_model, n_heads, d_ff, layers)
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in sizes):
            raise ValueError("Dimensions and layer count must be positive integers")
        if stride > patch_len or d_model % n_heads or not 0 <= dropout < 1:
            raise ValueError("Require stride <= patch_len, D divisible by heads, 0 <= dropout < 1")
        if variant not in ("time_only", "time_time", "time_channel"):
            raise ValueError("variant must be time_only, time_time or time_channel")
        self.out_channels = channels
        self.variant = variant
        self.seq_len, self.patch_len, self.stride = seq_len, patch_len, stride
        self.d_model, self.n_heads, self.d_ff = d_model, n_heads, d_ff
        self.num_layers, self.dropout = layers, float(dropout)
        self.num_patches = 1 + math.ceil(max(0, seq_len - patch_len) / stride)
        self.padded_len = patch_len + (self.num_patches - 1) * stride
        self.right_padding = self.padded_len - seq_len
        self.refiner_enabled = bool(refiner)

        # Every variable supplies exactly two feature channels: skeleton, mask.
        self.patch_embed = nn.Conv1d(2, d_model, patch_len, stride=stride)
        self.time_pos = nn.Parameter(torch.empty(1, 1, self.num_patches, d_model))
        nn.init.normal_(self.time_pos, std=0.02)
        self.input_dropout = nn.Dropout(dropout)
        self.time_blocks = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(layers)
        ])
        # Identical module registration/RNG order in time_time and time_channel.
        self.second_blocks = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(layers if variant != "time_only" else 0)
        ])
        # Keep PyTorch's nonzero random initialization, including the decoder.
        self.patch_decoder = nn.Linear(d_model, patch_len)
        self.temporal_refine = nn.Sequential(
            nn.Conv1d(2 * channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=1),
        )
        self.refine_scale = nn.Parameter(
            torch.tensor(0.3), requires_grad=learnable_refine_scale,
        )
        coverage = torch.zeros(self.padded_len)
        for p in range(self.num_patches):
            coverage[p * stride:p * stride + patch_len] += 1
        self.register_buffer("patch_coverage", coverage, persistent=False)

    def _prepare(self, x_cat: torch.Tensor):
        if x_cat.ndim != 3 or x_cat.shape[1:] != (self.seq_len, 2 * self.out_channels):
            raise ValueError(f"Expected [B,{self.seq_len},{2 * self.out_channels}]")
        if x_cat.shape[0] < 1 or not x_cat.is_floating_point():
            raise ValueError("Use a nonempty floating-point batch")
        values, mask = x_cat.split(self.out_channels, dim=-1)
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("Visibility mask must contain only zero and one")
        # where also discards NaN/Inf sentinels and cuts all hidden-value gradients.
        visible = torch.where(mask.bool(), values, torch.zeros_like(values))
        if not bool(torch.isfinite(visible).all()):
            raise ValueError("Observed values must be finite")
        skeleton = linear_interp_torch(visible, mask)
        return visible, mask, skeleton

    def _encode_skeleton(self, skeleton: torch.Tensor, mask: torch.Tensor):
        b, _, c = skeleton.shape
        values_bc = skeleton.transpose(1, 2).reshape(b * c, 1, self.seq_len)
        mask_bc = mask.transpose(1, 2).reshape(b * c, 1, self.seq_len)
        if self.right_padding:
            # Complete the last patch: extend the skeleton, mark padding missing.
            values_bc = F.pad(values_bc, (0, self.right_padding), mode="replicate")
            mask_bc = F.pad(mask_bc, (0, self.right_padding), value=0)
        features = torch.cat([values_bc, mask_bc], dim=1)
        z = self.patch_embed(features).transpose(1, 2)
        z = z.reshape(b, c, self.num_patches, self.d_model)
        z = self.input_dropout(z + self.time_pos)
        for i, time_block in enumerate(self.time_blocks):
            z = time_block(z.reshape(b * c, self.num_patches, self.d_model))
            z = z.reshape(b, c, self.num_patches, self.d_model)
            if self.variant == "time_channel":
                across_c = z.permute(0, 2, 1, 3).reshape(b * self.num_patches, c, self.d_model)
                across_c = self.second_blocks[i](across_c)
                z = across_c.reshape(b, self.num_patches, c, self.d_model).permute(0, 2, 1, 3)
            elif self.variant == "time_time":
                z = self.second_blocks[i](z.reshape(b * c, self.num_patches, self.d_model))
                z = z.reshape(b, c, self.num_patches, self.d_model)
        return z  # [B,C,P,D]; no variable embedding or variable-specific norm.

    def encode(self, x_cat: torch.Tensor) -> torch.Tensor:
        """Return shared per-variable patch representations [B,C,P,D]."""
        _, mask, skeleton = self._prepare(x_cat)
        return self._encode_skeleton(skeleton, mask)

    def overlap_average(self, patches: torch.Tensor) -> torch.Tensor:
        """[B,C,P,patch_len] -> [B,L,C], with every real time step covered."""
        expected = (self.out_channels, self.num_patches, self.patch_len)
        if patches.ndim != 4 or patches.shape[1:] != expected:
            raise ValueError(f"Expected decoded patches [B,{expected}]")
        b, c = patches.shape[:2]
        columns = patches.reshape(b * c, self.num_patches, self.patch_len).transpose(1, 2)
        total = F.fold(columns, output_size=(1, self.padded_len),
                       kernel_size=(1, self.patch_len), stride=(1, self.stride))
        average = total.reshape(b, c, self.padded_len) / self.patch_coverage
        return average[..., :self.seq_len].transpose(1, 2)

    def forward_recon(self, x_cat: torch.Tensor) -> torch.Tensor:
        """Predict R in skeleton residual coordinates, before masking/composition."""
        return self.overlap_average(self.patch_decoder(self.encode(x_cat)))

    def forward_recon_refine(self, x_cat: torch.Tensor):
        """Return (S + (1-M)*R, stage1 + alpha*(1-M)*refiner([stage1,M]))."""
        visible, mask, skeleton = self._prepare(x_cat)
        z = self._encode_skeleton(skeleton, mask)
        residual = self.overlap_average(self.patch_decoder(z))
        missing = 1 - mask
        stage1 = skeleton + missing * residual
        stage1 = torch.where(mask.bool(), visible, stage1)
        if not self.refiner_enabled:
            return stage1, stage1
        features = torch.cat([stage1, mask], dim=-1).transpose(1, 2)
        delta = self.temporal_refine(features).transpose(1, 2)
        stage2 = stage1 + self.refine_scale * missing * delta
        stage2 = torch.where(mask.bool(), visible, stage2)
        return stage1, stage2

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        return self.forward_recon_refine(x_cat)[1]


def build_model(channels: int, variant: str = "time_channel", **kwargs) -> SharedPatchSRI:
    """Build a fresh instance; defaults are L96/K16/s8/D64/h4/FF128/3 layers."""
    return SharedPatchSRI(channels, variant, **kwargs)
