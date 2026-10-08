# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.interp import linear_interp_torch, locf_torch, observed_mean_torch, seasonal_lag_torch


class PatchEmbedding1D(nn.Module):
    """Conv1d patchify: [B,L,in_ch] -> [B,Lp,d_model]."""
    def __init__(self, in_channels: int, d_model: int, patch_len: int, stride: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Conv1d(in_channels, d_model, kernel_size=patch_len, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)           # [B,in_ch,L]
        z = self.proj(x)                 # [B,d_model,Lp]
        z = z.permute(0, 2, 1)           # [B,Lp,d_model]
        return z


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.mha(x, x, x, need_weights=False)
        x = self.norm1(x + self.drop(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.drop(ff_out))
        return x


class PatchTSTBackbone(nn.Module):
    def __init__(self, in_channels_2c: int, d_model: int, n_heads: int,
                 e_layers: int, d_ff: int, dropout: float,
                 patch_len: int, stride: int, max_patches: int = 2048):
        super().__init__()
        self.embed = PatchEmbedding1D(in_channels_2c, d_model, patch_len, stride)
        self.max_patches = max_patches
        self.pos = nn.Parameter(torch.zeros(1, max_patches, d_model))
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(e_layers)
        ])
        self.drop = nn.Dropout(dropout)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        z = self.embed(x_cat)            # [B,Lp,d_model]
        Lp = z.size(1)
        if Lp > self.max_patches:
            z = z[:, :self.max_patches, :]
            Lp = self.max_patches
        z = z + self.pos[:, :Lp, :]
        z = self.drop(z)
        for layer in self.layers:
            z = layer(z)
        return z


class PatchTSTReconContrast(nn.Module):
    def __init__(self, in_channels_2c: int, out_channels: int,
                 seq_len: int, d_model: int, n_heads: int, e_layers: int,
                 d_ff: int, dropout: float, patch_len: int, stride: int,
                 proj_dim: int = 128):
        super().__init__()
        self.seq_len = seq_len
        self.out_channels = out_channels
        self.patch_len = patch_len
        self.stride = stride

        # backbone + decoder
        self.backbone = PatchTSTBackbone(
            in_channels_2c, d_model, n_heads, e_layers, d_ff,
            dropout, patch_len, stride
        )
        self.patch_decoder = nn.Linear(d_model, patch_len * out_channels)

        # projection head for SSL
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim)
        )

        # ===== temporal refinement module =====
        # input = concat([stage1_pred, input_mask]) => [B,L,2C]
        # conv works on time dimension
        self.temporal_refine = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.refine_scale = nn.Parameter(torch.tensor(0.3))
        self.skeleton_router = None
        self._skeleton_dropout = 0.0
        self._use_boundary_mean_skeleton = False
        self._boundary_width = 3
        self._distance_aware_refine = False
        self._distance_refine_clip = 48.0
        self._region_disentangled_refine = False
        self._region_refine_width = 3
        self._length_conditioned_refine = False
        self._length_refine_clip = 48.0
        self._length_refine_gate_scale = 0.5
        self._length_adaptive_refine = False
        self._length_adaptive_clip = 48.0
        self._length_adaptive_temperature = 0.15
        self._length_adaptive_n_experts = 3
        self._length_adaptive_delta_refine = False
        self._length_adaptive_delta_scale = 0.1
        self._spectral_residual_refine = False
        self._spectral_keep_ratio = 0.25
        self._spectral_delta_scale = 0.1
        self._structural_consistency_gate = False
        self._structural_gate_scale = 0.5
        self._structural_residual_bound = False
        self._srb_base_radius = 0.05
        self._srb_spread_scale = 1.0
        self._srb_distance_scale = 0.2
        self._uncertainty_head = False
        self._uncertainty_quantiles = (0.05, 0.5, 0.95)
        self._cross_channel_skeleton = False
        self._skeleton_bootstrap = False
        self._skeleton_bootstrap_iters = 1
        self._skeleton_bootstrap_grad = False
        self._bridge_residual_scale = False
        self._bridge_ref_len = 24.0

    def enable_uncertainty_head(self, quantiles=(0.05, 0.5, 0.95)) -> None:
        """Opt-in residual quantile head for split-conformal calibration."""
        qs = tuple(float(q) for q in quantiles)
        if not qs or any(q <= 0.0 or q >= 1.0 for q in qs):
            raise ValueError("uncertainty quantiles must be inside (0, 1)")
        if any(qs[i] >= qs[i + 1] for i in range(len(qs) - 1)):
            raise ValueError("uncertainty quantiles must be strictly increasing")

        self._uncertainty_quantiles = qs
        self._uncertainty_head = True
        self.quantile_decoder = nn.Linear(
            self.patch_decoder.in_features,
            self.patch_len * self.out_channels * len(qs),
        )

    def enable_cross_channel_skeleton(self, coef, bias, valid) -> None:
        """Opt-in pairwise cross-channel draft skeleton.

        `coef`, `bias`, and `valid` are indexed as [target_channel, source_channel].
        Missing target values use only source channels whose input_mask is observed
        at the same time step; positions without a valid observed source fall back
        to the linear interpolation skeleton.
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        coef_t = torch.as_tensor(coef, device=device, dtype=dtype)
        bias_t = torch.as_tensor(bias, device=device, dtype=dtype)
        valid_t = torch.as_tensor(valid, device=device, dtype=dtype)
        expected = (self.out_channels, self.out_channels)
        if tuple(coef_t.shape) != expected or tuple(bias_t.shape) != expected or tuple(valid_t.shape) != expected:
            raise ValueError(f"cross-channel skeleton tensors must have shape {expected}")
        eye = torch.eye(self.out_channels, device=device, dtype=dtype)
        valid_t = (valid_t > 0.5).to(dtype) * (1.0 - eye)

        for name, tensor in (
            ("cross_channel_coef", coef_t),
            ("cross_channel_bias", bias_t),
            ("cross_channel_valid", valid_t),
        ):
            if name in self._buffers:
                self._buffers[name] = tensor
            else:
                self.register_buffer(name, tensor)
        self._cross_channel_skeleton = True

    def enable_skeleton_bootstrap(self, iters: int = 1, allow_grad: bool = False) -> None:
        """
        Opt-in iterative re-skeletonization (skeleton bootstrap).

        After the stage-one composition, the composed sequence replaces the
        deterministic linear skeleton as the coordinate system for one or more
        additional aligned passes: missing positions of the backbone input are
        re-filled with the previous composition, the shared backbone/decoder
        predict a fresh residual relative to that refreshed skeleton, and the
        composition is rebuilt as refreshed-skeleton + residual. Observed
        positions never change, no new parameters are introduced (the backbone
        and patch decoder are reused across passes), and by default the
        refreshed skeleton is detached so each pass treats it as a fixed
        coordinate system.
        """
        self._skeleton_bootstrap = True
        self._skeleton_bootstrap_iters = int(max(1, iters))
        self._skeleton_bootstrap_grad = bool(allow_grad)

    def enable_bridge_residual_scale(self, ref_len: float = 24.0) -> None:
        """
        Opt-in Brownian-bridge residual reparameterization.

        The stage-one residual prediction is multiplied by a deterministic
        per-position envelope sqrt(left*right/(left+right)), where left/right
        are the distances (inclusive run counts) to the two observed gap
        boundaries. This is the standard-deviation profile of a Brownian
        bridge pinned at the gap boundaries, so the network learns a
        gap-scale-standardized residual whose distribution is comparable
        across block lengths. The envelope is normalized by its value at the
        center of a gap of length `ref_len`, keeping the effective residual
        scale unchanged at the reference block length. No parameters are added.
        """
        self._bridge_residual_scale = True
        self._bridge_ref_len = float(max(1.0, ref_len))

    def enable_multi_skeleton(
        self,
        hidden: int = 32,
        use_seasonal: bool = False,
        seasonal_period: int = 24,
        linear_prior: float = 0.0,
    ) -> None:
        """Enable a mask-aware router over a small bank of deterministic skeletons."""
        n_skeletons = 4 if use_seasonal else 3
        self.skeleton_router = nn.Sequential(
            nn.Conv1d(self.out_channels * 4, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, self.out_channels * n_skeletons, kernel_size=1),
        )
        if linear_prior != 0.0:
            final = self.skeleton_router[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            with torch.no_grad():
                bias = final.bias.view(n_skeletons, self.out_channels)
                bias[0, :].fill_(float(linear_prior))
        self._multi_skeleton = True
        self._multi_skeleton_use_seasonal = bool(use_seasonal)
        self._multi_skeleton_n = n_skeletons
        self._seasonal_period = int(seasonal_period)

    def set_skeleton_dropout(self, p: float) -> None:
        self._skeleton_dropout = float(p)

    def enable_boundary_mean_skeleton(self, boundary_width: int = 3) -> None:
        self._use_boundary_mean_skeleton = True
        self._boundary_width = int(boundary_width)

    def enable_distance_aware_refine(self, distance_clip: float = 48.0) -> None:
        """
        Add mask-distance channels to the temporal refinement stage.

        The first conv is initialized so the new path is equivalent to the
        original temporal_refine at step 0; distance channels start with zero
        weights and can be learned during finetuning.
        """
        self.distance_refine = nn.Sequential(
            nn.Conv1d(self.out_channels * 4, self.out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
        )
        self.reset_distance_refine_from_temporal()
        self._distance_aware_refine = True
        self._distance_refine_clip = float(max(1.0, distance_clip))
        self.distance_refine.to(next(self.parameters()).device)

    def reset_distance_refine_from_temporal(self) -> None:
        """Initialize distance-aware refinement as an exact extension of temporal_refine."""
        if not hasattr(self, "distance_refine"):
            return
        with torch.no_grad():
            self.distance_refine[0].weight.zero_()
            self.distance_refine[0].weight[:, : self.out_channels * 2, :].copy_(
                self.temporal_refine[0].weight
            )
            self.distance_refine[0].bias.copy_(self.temporal_refine[0].bias)
            self.distance_refine[2].weight.copy_(self.temporal_refine[2].weight)
            self.distance_refine[2].bias.copy_(self.temporal_refine[2].bias)

    def enable_region_disentangled_refine(self, boundary_width: int = 3) -> None:
        """
        Use separate refinement experts for boundary and interior target regions.

        Both experts are initialized from the original temporal_refine module, so
        enabling the feature starts from the same function as the old single-head
        refinement before finetuning updates the copied heads independently.
        """
        self.boundary_refine = nn.Sequential(
            nn.Conv1d(self.out_channels * 2, self.out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
        )
        self.interior_refine = nn.Sequential(
            nn.Conv1d(self.out_channels * 2, self.out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
        )
        self.reset_region_refine_from_temporal()
        self._region_disentangled_refine = True
        self._region_refine_width = int(max(1, boundary_width))
        device = next(self.parameters()).device
        self.boundary_refine.to(device)
        self.interior_refine.to(device)

    def reset_region_refine_from_temporal(self) -> None:
        """Copy temporal_refine weights into both region-specific refine heads."""
        for name in ("boundary_refine", "interior_refine"):
            if not hasattr(self, name):
                continue
            module = getattr(self, name)
            with torch.no_grad():
                module[0].weight.copy_(self.temporal_refine[0].weight)
                module[0].bias.copy_(self.temporal_refine[0].bias)
                module[2].weight.copy_(self.temporal_refine[2].weight)
                module[2].bias.copy_(self.temporal_refine[2].bias)

    def enable_length_conditioned_refine(
        self,
        length_clip: float = 48.0,
        gate_scale: float = 0.5,
    ) -> None:
        """
        Gate the refinement residual with missing-run length features.

        The final gate layer is initialized to zero, making the multiplicative
        gate exactly 1.0 at step 0. This keeps old behavior unchanged before
        finetuning learns length-specific residual amplitudes.
        """
        self.length_refine_gate = nn.Sequential(
            nn.Conv1d(self.out_channels * 2, self.out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.out_channels, self.out_channels, kernel_size=1),
        )
        self.reset_length_refine_gate_to_identity()
        self._length_conditioned_refine = True
        self._length_refine_clip = float(max(1.0, length_clip))
        self._length_refine_gate_scale = float(max(0.0, gate_scale))
        self.length_refine_gate.to(next(self.parameters()).device)

    def reset_length_refine_gate_to_identity(self) -> None:
        """Initialize length-conditioned residual gate as an identity gate."""
        if not hasattr(self, "length_refine_gate"):
            return
        with torch.no_grad():
            self.length_refine_gate[2].weight.zero_()
            self.length_refine_gate[2].bias.zero_()

    def enable_length_adaptive_refine(
        self,
        n_experts: int = 3,
        length_clip: float = 48.0,
        temperature: float = 0.15,
    ) -> None:
        """
        Use deterministic missing-length routed refinement experts.

        Each expert starts as a copy of the original temporal_refine module, so
        the initial mixed residual is exactly equivalent to the old single-head
        refinement. During finetuning, short/mid/long gaps update different
        copied heads through fixed soft routing based on the mask-derived gap
        length.
        """
        n_experts = int(max(2, n_experts))
        self.length_refine_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(self.out_channels * 2, self.out_channels, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv1d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
                )
                for _ in range(n_experts)
            ]
        )
        self._length_adaptive_refine = True
        self._length_adaptive_clip = float(max(1.0, length_clip))
        self._length_adaptive_temperature = float(max(1e-3, temperature))
        self._length_adaptive_n_experts = n_experts
        self.reset_length_adaptive_refine_from_temporal()
        self.length_refine_experts.to(next(self.parameters()).device)

    def enable_length_adaptive_delta_refine(
        self,
        n_experts: int = 3,
        length_clip: float = 48.0,
        temperature: float = 0.15,
        delta_scale: float = 0.1,
        delta_init: str = "zero_output",
    ) -> None:
        """
        Add zero-initialized length-specific residual adapters on top of temporal_refine.

        Unlike length_adaptive_refine, this keeps the shared refinement head as
        the main path and only learns small routed deltas. The initial function
        is exactly equivalent to the base model.
        """
        n_experts = int(max(2, n_experts))
        self.length_delta_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(self.out_channels * 2, self.out_channels, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv1d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
                )
                for _ in range(n_experts)
            ]
        )
        self._length_adaptive_delta_refine = True
        self._length_adaptive_clip = float(max(1.0, length_clip))
        self._length_adaptive_temperature = float(max(1e-3, temperature))
        self._length_adaptive_n_experts = n_experts
        if delta_init not in {"zero_all", "zero_output"}:
            raise ValueError(f"Unknown length-adaptive delta init: {delta_init}")
        self._length_adaptive_delta_scale = float(max(0.0, delta_scale))
        self._length_adaptive_delta_init = delta_init
        self.reset_length_adaptive_delta_to_zero()
        self.length_delta_experts.to(next(self.parameters()).device)

    def reset_length_adaptive_delta_to_zero(self) -> None:
        """Initialize length-specific delta adapters as exact zero functions."""
        if not hasattr(self, "length_delta_experts"):
            return
        delta_init = getattr(self, "_length_adaptive_delta_init", "zero_output")
        for expert in self.length_delta_experts:
            with torch.no_grad():
                if delta_init == "zero_output":
                    expert[0].weight.copy_(self.temporal_refine[0].weight)
                    expert[0].bias.copy_(self.temporal_refine[0].bias)
                else:
                    expert[0].weight.zero_()
                    expert[0].bias.zero_()
                expert[2].weight.zero_()
                expert[2].bias.zero_()

    def reset_length_adaptive_refine_from_temporal(self) -> None:
        """Copy temporal_refine weights into all length-adaptive experts."""
        if not hasattr(self, "length_refine_experts"):
            return
        for expert in self.length_refine_experts:
            with torch.no_grad():
                expert[0].weight.copy_(self.temporal_refine[0].weight)
                expert[0].bias.copy_(self.temporal_refine[0].bias)
                expert[2].weight.copy_(self.temporal_refine[2].weight)
                expert[2].bias.copy_(self.temporal_refine[2].bias)

    def enable_spectral_residual_refine(
        self,
        keep_ratio: float = 0.25,
        delta_scale: float = 0.1,
    ) -> None:
        """
        Add a frequency-aware residual correction branch.

        The branch receives the structural prediction, its low-frequency
        projection, the corresponding high-frequency residual, and the mask.
        Its output layer is zero-initialized, so enabling this module starts
        from exactly the same function as the base residual refinement.
        """
        self.spectral_refine = nn.Sequential(
            nn.Conv1d(self.out_channels * 4, self.out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
        )
        self._spectral_residual_refine = True
        self._spectral_keep_ratio = float(min(max(keep_ratio, 1e-3), 1.0))
        self._spectral_delta_scale = float(max(0.0, delta_scale))
        self.reset_spectral_residual_refine_to_zero()
        self.spectral_refine.to(next(self.parameters()).device)

    def reset_spectral_residual_refine_to_zero(self) -> None:
        """Initialize spectral residual correction as an exact zero function."""
        if not hasattr(self, "spectral_refine"):
            return
        with torch.no_grad():
            self.spectral_refine[2].weight.zero_()
            self.spectral_refine[2].bias.zero_()

    def enable_structural_consistency_gate(
        self,
        gate_scale: float = 0.5,
    ) -> None:
        """
        Gate the shared refinement residual using deterministic skeleton agreement.

        The gate receives stage1, mask-derived geometry, and disagreement among
        linear/LOCF/mean skeletons. It starts as an identity gate, so enabling
        the module leaves the base model function unchanged before finetuning.
        """
        self.structural_gate = nn.Sequential(
            nn.Conv1d(self.out_channels * 8, self.out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.out_channels, self.out_channels, kernel_size=1),
        )
        self._structural_consistency_gate = True
        self._structural_gate_scale = float(max(0.0, gate_scale))
        self.reset_structural_consistency_gate_to_identity()
        self.structural_gate.to(next(self.parameters()).device)

    def reset_structural_consistency_gate_to_identity(self) -> None:
        """Initialize the structural consistency gate as an exact identity."""
        if not hasattr(self, "structural_gate"):
            return
        with torch.no_grad():
            self.structural_gate[2].weight.zero_()
            self.structural_gate[2].bias.zero_()

    def enable_structural_residual_bound(
        self,
        base_radius: float = 0.05,
        spread_scale: float = 1.0,
        distance_scale: float = 0.2,
    ) -> None:
        """Enable zero-parameter clipping of refinement deltas by structure."""
        self._structural_residual_bound = True
        self._srb_base_radius = float(max(0.0, base_radius))
        self._srb_spread_scale = float(max(0.0, spread_scale))
        self._srb_distance_scale = float(max(0.0, distance_scale))

    def apply_structural_residual_bound(
        self,
        stage1: torch.Tensor,
        residual_delta: torch.Tensor,
        x_in: torch.Tensor,
        input_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Clip refinement deltas on missing positions using deterministic structure.

        `residual_delta` is expected to be the already scaled refinement delta
        that would be added to `stage1`.
        """
        if not getattr(self, "_structural_residual_bound", False):
            return residual_delta

        miss_mask = 1.0 - input_mask
        linear_base = linear_interp_torch(x_in, input_mask)
        locf_base = locf_torch(x_in, input_mask)
        mean_base = observed_mean_torch(x_in, input_mask)
        bank = torch.stack([linear_base, locf_base, mean_base], dim=2)
        skeleton_spread = bank.std(dim=2, unbiased=False) * miss_mask
        distance_feat = self._mask_distance_features(input_mask)
        distance_norm = distance_feat[:, :, : self.out_channels]
        radius = (
            float(getattr(self, "_srb_base_radius", 0.05))
            + float(getattr(self, "_srb_spread_scale", 1.0)) * skeleton_spread
            + float(getattr(self, "_srb_distance_scale", 0.2)) * distance_norm
        ).clamp_min(0.0)
        return residual_delta.clamp(min=-radius, max=radius) * miss_mask

    def _spectral_lowpass(self, x: torch.Tensor) -> torch.Tensor:
        """Return a low-frequency reconstruction along the temporal dimension."""
        dtype = x.dtype
        x_float = x.float()
        freq = torch.fft.rfft(x_float, dim=1)
        n_freq = freq.size(1)
        keep = max(1, int(round(n_freq * float(getattr(self, "_spectral_keep_ratio", 0.25)))))
        keep = min(keep, n_freq)
        mask = torch.zeros(n_freq, device=x.device, dtype=x_float.dtype)
        mask[:keep] = 1.0
        filtered = torch.fft.irfft(freq * mask.view(1, n_freq, 1), n=x.size(1), dim=1)
        return filtered.to(dtype=dtype)

    def _structural_consistency_features(
        self,
        stage1: torch.Tensor,
        x_in: torch.Tensor,
        input_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return local structure features for residual gating."""
        miss_mask = 1.0 - input_mask
        linear_base = linear_interp_torch(x_in, input_mask)
        locf_base = locf_torch(x_in, input_mask)
        mean_base = observed_mean_torch(x_in, input_mask)
        bank = torch.stack([linear_base, locf_base, mean_base], dim=2)
        skeleton_spread = bank.std(dim=2, unbiased=False) * miss_mask
        stage1_delta = (stage1 - linear_base).abs() * miss_mask
        distance_feat = self._mask_distance_features(input_mask)
        gap_feat = self._mask_gap_length_features(input_mask)
        return torch.cat(
            [
                stage1,
                input_mask,
                skeleton_spread,
                stage1_delta,
                distance_feat,
                gap_feat,
            ],
            dim=-1,
        )

    def _structural_consistency_residual_gate(
        self,
        stage1: torch.Tensor,
        x_in: torch.Tensor,
        input_mask: torch.Tensor,
    ) -> torch.Tensor:
        feat = self._structural_consistency_features(stage1, x_in, input_mask)
        logits = self.structural_gate(feat.transpose(1, 2)).transpose(1, 2)
        scale = float(getattr(self, "_structural_gate_scale", 0.5))
        return 1.0 + scale * torch.tanh(logits)

    def _length_expert_weights(self, input_mask: torch.Tensor) -> torch.Tensor:
        """Return fixed soft weights over length experts, shape [B,L,E,C]."""
        gap_len_norm = self._mask_gap_length_features(input_mask)[..., : self.out_channels]
        clip = float(getattr(self, "_length_adaptive_clip", 48.0))
        gap_len_norm = gap_len_norm * (float(getattr(self, "_length_refine_clip", 48.0)) / clip)
        gap_len_norm = gap_len_norm.clamp(0.0, 1.0)

        n_experts = int(getattr(self, "_length_adaptive_n_experts", 3))
        if n_experts == 3:
            centers = torch.tensor([12.0 / clip, 24.0 / clip, 48.0 / clip], device=input_mask.device, dtype=input_mask.dtype)
            centers = centers.clamp(0.0, 1.0)
        else:
            centers = torch.linspace(0.0, 1.0, steps=n_experts, device=input_mask.device, dtype=input_mask.dtype)
        temperature = float(getattr(self, "_length_adaptive_temperature", 0.15))
        logits = -(gap_len_norm.unsqueeze(2) - centers.view(1, 1, n_experts, 1)).abs() / temperature
        return logits.softmax(dim=2)

    def _region_refine_masks(self, input_mask: torch.Tensor):
        width = max(1, int(getattr(self, "_region_refine_width", 3)))
        near_obs = F.max_pool1d(
            input_mask.transpose(1, 2),
            kernel_size=2 * width + 1,
            stride=1,
            padding=width,
        ).transpose(1, 2)
        miss_mask = 1.0 - input_mask
        boundary_mask = (near_obs > 0.5).float() * miss_mask
        interior_mask = (near_obs <= 0.5).float() * miss_mask
        return boundary_mask, interior_mask

    def _compose_boundary_mean_skeleton(self, x_in: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        linear_base = linear_interp_torch(x_in, input_mask)
        mean_base = observed_mean_torch(x_in, input_mask)
        width = max(1, int(getattr(self, "_boundary_width", 3)))
        near_obs = F.max_pool1d(
            input_mask.transpose(1, 2),
            kernel_size=2 * width + 1,
            stride=1,
            padding=width,
        ).transpose(1, 2)
        use_linear = (near_obs > 0.5).float()
        skeleton = use_linear * linear_base + (1.0 - use_linear) * mean_base
        return torch.where(input_mask > 0.5, x_in, skeleton)

    def _compose_cross_channel_skeleton(
        self,
        linear_base: torch.Tensor,
        x_in: torch.Tensor,
        input_mask: torch.Tensor,
    ) -> torch.Tensor:
        coef = self.cross_channel_coef.to(device=x_in.device, dtype=x_in.dtype)
        bias = self.cross_channel_bias.to(device=x_in.device, dtype=x_in.dtype)
        valid = self.cross_channel_valid.to(device=x_in.device, dtype=x_in.dtype)

        source_values = x_in.unsqueeze(-2)
        source_mask = input_mask.unsqueeze(-2)
        pred = source_values * coef.view(1, 1, self.out_channels, self.out_channels)
        pred = pred + bias.view(1, 1, self.out_channels, self.out_channels)
        weights = source_mask * valid.view(1, 1, self.out_channels, self.out_channels)
        denom = weights.sum(dim=-1)
        cross = (pred * weights).sum(dim=-1) / denom.clamp_min(1e-8)
        skeleton = torch.where(denom > 0.5, cross, linear_base)
        return torch.where(input_mask > 0.5, x_in, skeleton)

    def _compose_skeleton_bank(self, x_in: torch.Tensor, input_mask: torch.Tensor):
        linear_base = linear_interp_torch(x_in, input_mask)
        if getattr(self, "_cross_channel_skeleton", False):
            if not all(hasattr(self, name) for name in ("cross_channel_coef", "cross_channel_bias", "cross_channel_valid")):
                raise RuntimeError("cross_channel_skeleton is enabled but coefficients are not initialized")
            return self._compose_cross_channel_skeleton(linear_base, x_in, input_mask)
        if getattr(self, "_use_boundary_mean_skeleton", False):
            return self._compose_boundary_mean_skeleton(x_in, input_mask)
        if not getattr(self, "_multi_skeleton", False):
            return linear_base
        if self.skeleton_router is None:
            raise RuntimeError("multi_skeleton is enabled but skeleton_router is not initialized")

        locf_base = locf_torch(x_in, input_mask)
        mean_base = observed_mean_torch(x_in, input_mask)
        skeletons = [linear_base, locf_base, mean_base]
        if getattr(self, "_multi_skeleton_use_seasonal", False):
            seasonal_base = seasonal_lag_torch(
                x_in, input_mask, period=getattr(self, "_seasonal_period", 24)
            )
            skeletons.append(seasonal_base)
        bank = torch.stack(skeletons, dim=-2)  # [B,L,K,C]

        router_in = torch.cat([x_in, input_mask, linear_base, locf_base], dim=-1).transpose(1, 2)
        logits = self.skeleton_router(router_in).transpose(1, 2)
        B, L, _ = logits.shape
        weights = logits.view(B, L, len(skeletons), self.out_channels).softmax(dim=2)
        dropout_p = getattr(self, "_skeleton_dropout", 0.0)
        if self.training and dropout_p > 0.0 and len(skeletons) > 1:
            keep = (torch.rand(B, L, len(skeletons), 1, device=weights.device) > dropout_p).float()
            all_dropped = keep.sum(dim=2, keepdim=True) < 0.5
            keep = torch.where(all_dropped, torch.ones_like(keep), keep)
            weights = weights * keep
            weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-8)
        mixed = (bank * weights).sum(dim=2)
        return torch.where(input_mask > 0.5, x_in, mixed)

    def skeleton_diagnostics(self, x_cat: torch.Tensor):
        """Return skeleton bank and router weights for analysis."""
        B, L, twoC = x_cat.shape
        C = twoC // 2
        x_in = x_cat[:, :, :C]
        input_mask = x_cat[:, :, C:]

        linear_base = linear_interp_torch(x_in, input_mask)
        locf_base = locf_torch(x_in, input_mask)
        mean_base = observed_mean_torch(x_in, input_mask)
        names = ["linear", "locf", "mean"]
        skeletons = [linear_base, locf_base, mean_base]
        if getattr(self, "_multi_skeleton_use_seasonal", False):
            skeletons.append(
                seasonal_lag_torch(x_in, input_mask, period=getattr(self, "_seasonal_period", 24))
            )
            names.append("seasonal")
        bank = torch.stack(skeletons, dim=-2)

        if not getattr(self, "_multi_skeleton", False):
            weights = torch.zeros(B, L, len(skeletons), C, device=x_cat.device, dtype=x_cat.dtype)
            weights[:, :, 0, :] = 1.0
            return names, bank, weights
        if self.skeleton_router is None:
            raise RuntimeError("multi_skeleton is enabled but skeleton_router is not initialized")

        router_in = torch.cat([x_in, input_mask, linear_base, locf_base], dim=-1).transpose(1, 2)
        logits = self.skeleton_router(router_in).transpose(1, 2)
        weights = logits.view(B, L, len(skeletons), C).softmax(dim=2)
        return names, bank, weights

    def _prepare_backbone_input(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Optionally replace zero-filled missing values with the linear-interp
        skeleton before feeding the sequence into the backbone.

        This keeps the mask channels unchanged but makes the value channels
        consistent with the residual-learning objective used in stage1.
        """
        if not getattr(self, "_fill_backbone_with_linear", False):
            return x_cat

        b, l, two_c = x_cat.shape
        c = two_c // 2
        x_in = x_cat[:, :, :c]
        input_mask = x_cat[:, :, c:]
        skeleton = self._compose_skeleton_bank(x_in, input_mask)
        x_filled = input_mask * x_in + (1.0 - input_mask) * skeleton
        return torch.cat([x_filled, input_mask], dim=-1)

    def _decode_patch_sequence(self, patch_flat: torch.Tensor, L: int, tail_shape: tuple[int, ...]) -> torch.Tensor:
        B, Lp = patch_flat.shape[:2]
        patch = patch_flat.view(B, Lp, self.patch_len, *tail_shape)

        out = torch.zeros(B, L, *tail_shape, device=patch_flat.device, dtype=patch_flat.dtype)
        cnt_shape = (B, L) + (1,) * len(tail_shape)
        cnt = torch.zeros(*cnt_shape, device=patch_flat.device, dtype=patch_flat.dtype)

        for i in range(Lp):
            st = i * self.stride
            ed = st + self.patch_len
            if ed > L:
                break
            out[:, st:ed, ...] += patch[:, i, ...]
            cnt[:, st:ed, ...] += 1.0

        return out / (cnt + 1e-6)

    def _compose_stage1_from_patch_pred(self, x_cat: torch.Tensor, patch_pred: torch.Tensor) -> torch.Tensor:
        if not getattr(self, "_use_linear_base", True):
            return patch_pred

        B, L, twoC = x_cat.shape
        C = twoC // 2
        x_in = x_cat[:, :, :C]
        input_mask = x_cat[:, :, C:]
        miss_mask = 1.0 - input_mask

        skeleton = self._compose_skeleton_bank(x_in, input_mask)   # [B,L,C]
        if getattr(self, "_bridge_residual_scale", False):
            patch_pred = patch_pred * self._bridge_residual_envelope(input_mask)
        return skeleton + patch_pred * miss_mask

    def _bridge_residual_envelope(self, input_mask: torch.Tensor) -> torch.Tensor:
        """
        Deterministic per-position residual envelope from gap geometry.

        For a missing run pinned by observed values on both sides, a position
        with inclusive boundary distances (left, right), left + right = n + 1,
        gets the Brownian-bridge standard-deviation profile
        sqrt(left * right / (left + right)). A run that touches a window edge
        has no real pin on that side; treating the edge as a pin would suppress
        the residual exactly where constant extrapolation is worst, so the
        open side's distance is replaced by the window length, which lets the
        formula degrade to the one-sided Brownian-motion profile ~sqrt(d)
        growing away from the single real anchor. The envelope is normalized
        by its value at the center of a two-sided gap of length
        `_bridge_ref_len` and is zero at observed positions.
        """
        B, L, C = input_mask.shape
        miss = (input_mask <= 0.5).float()

        left = torch.zeros_like(input_mask)
        right = torch.zeros_like(input_mask)
        run = torch.zeros(B, C, device=input_mask.device, dtype=input_mask.dtype)
        for t in range(L):
            run = (run + 1.0) * miss[:, t, :]
            left[:, t, :] = run
        run = torch.zeros(B, C, device=input_mask.device, dtype=input_mask.dtype)
        for t in range(L - 1, -1, -1):
            run = (run + 1.0) * miss[:, t, :]
            right[:, t, :] = run

        # left[t] == t+1 means positions 0..t are all missing (no real left
        # pin inside the window); right[t] == L-t is the mirrored case.
        t_idx = torch.arange(1, L + 1, device=input_mask.device, dtype=input_mask.dtype).view(1, L, 1)
        big = torch.full_like(left, float(L))
        left_eff = torch.where((left >= t_idx - 0.5) & (miss > 0.5), big, left)
        right_eff = torch.where((right >= (float(L) + 1.0 - t_idx) - 0.5) & (miss > 0.5), big, right)

        env = (left_eff * right_eff / (left_eff + right_eff).clamp_min(1.0)).sqrt()
        ref = float(getattr(self, "_bridge_ref_len", 24.0))
        half = (ref + 1.0) / 2.0
        g_ref = math.sqrt(half * half / (ref + 1.0))
        return (env / max(g_ref, 1e-6)) * miss

    def _forward_recon_prefilled(self, x_cat_filled: torch.Tensor) -> torch.Tensor:
        """Backbone+decoder pass on an already-filled input (no skeleton re-fill)."""
        B, L, twoC = x_cat_filled.shape
        if L != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got L={L}")
        z = self.backbone(x_cat_filled)
        patch_flat = self.patch_decoder(z)
        return self._decode_patch_sequence(patch_flat, L, (self.out_channels,))

    def _bootstrap_stage1(self, x_cat: torch.Tensor, stage1: torch.Tensor) -> torch.Tensor:
        """
        Iterative re-skeletonization: re-fill missing positions with the
        previous composition, rerun the shared backbone on the refreshed
        coordinate system, and rebuild the composition. Observed positions are
        pinned to the true input values at every pass. In the default detached
        mode only the final pass needs a gradient graph; intermediate passes
        run under no_grad.
        """
        B, L, twoC = x_cat.shape
        C = twoC // 2
        x_in = x_cat[:, :, :C]
        input_mask = x_cat[:, :, C:]
        miss_mask = 1.0 - input_mask

        envelope = None
        if getattr(self, "_bridge_residual_scale", False):
            envelope = self._bridge_residual_envelope(input_mask)

        current = stage1
        n_iters = int(max(1, getattr(self, "_skeleton_bootstrap_iters", 1)))
        allow_grad = bool(getattr(self, "_skeleton_bootstrap_grad", False))
        for i in range(n_iters):
            skeleton = current if allow_grad else current.detach()
            skeleton = torch.where(input_mask > 0.5, x_in, skeleton)
            x_cat_boot = torch.cat([skeleton, input_mask], dim=-1)
            if not allow_grad and i < n_iters - 1:
                with torch.no_grad():
                    residual = self._forward_recon_prefilled(x_cat_boot)
            else:
                residual = self._forward_recon_prefilled(x_cat_boot)
            if envelope is not None:
                residual = residual * envelope
            current = skeleton + residual * miss_mask
        return current

    def _backbone_latent(self, x_cat: torch.Tensor) -> torch.Tensor:
        B, L, twoC = x_cat.shape
        if L != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got L={L}")
        x_backbone = self._prepare_backbone_input(x_cat)
        return self.backbone(x_backbone)

    def forward_recon(self, x_cat: torch.Tensor) -> torch.Tensor:
        B, L, twoC = x_cat.shape
        z = self._backbone_latent(x_cat)                              # [B,Lp,D]
        patch_flat = self.patch_decoder(z)                            # [B,Lp,patch_len*C]
        return self._decode_patch_sequence(patch_flat, L, (self.out_channels,))

    def compose_stage1(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Stage 1 prediction.

        Default mode (use_linear_base=True):
          stage1 = linear_interp(x_in, mask)   # boundary-anchored skeleton
                 + forward_recon(x_cat) * miss_mask
          ↳ forward_recon's output is interpreted as a residual on the linear
            baseline (only added at missing positions); at observed positions
            stage1 == x_in by construction.

        Ablation mode (use_linear_base=False):
            stage1 = forward_recon(x_cat)        # raw absolute-value prediction
        """
        patch_pred = self.forward_recon(x_cat)   # [B,L,C]
        return self._compose_stage1_from_patch_pred(x_cat, patch_pred)

    def _refine_stage1(self, x_cat: torch.Tensor, stage1: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_no_refine", False):
            return stage1

        B, L, twoC = x_cat.shape
        C = twoC // 2
        input_mask = x_cat[:, :, C:]         # [B,L,C]
        x_in = x_cat[:, :, :C]

        if getattr(self, "_length_adaptive_refine", False):
            refine_feat = torch.cat([stage1, input_mask], dim=-1)   # [B,L,2C]
            refine_in = refine_feat.transpose(1, 2)                 # [B,2C,L]
            expert_residuals = [
                expert(refine_in).transpose(1, 2)
                for expert in self.length_refine_experts
            ]
            residual_stack = torch.stack(expert_residuals, dim=2)   # [B,L,E,C]
            weights = self._length_expert_weights(input_mask)
            residual = (residual_stack * weights).sum(dim=2)
        elif getattr(self, "_region_disentangled_refine", False):
            refine_feat = torch.cat([stage1, input_mask], dim=-1)   # [B,L,2C]
            refine_in = refine_feat.transpose(1, 2)                 # [B,2C,L]
            boundary_residual = self.boundary_refine(refine_in).transpose(1, 2)
            interior_residual = self.interior_refine(refine_in).transpose(1, 2)
            boundary_mask, interior_mask = self._region_refine_masks(input_mask)
            residual = boundary_residual * boundary_mask + interior_residual * interior_mask
        elif getattr(self, "_distance_aware_refine", False):
            distance_feat = self._mask_distance_features(input_mask)
            refine_feat = torch.cat([stage1, input_mask, distance_feat], dim=-1)  # [B,L,4C]
            refine_in = refine_feat.transpose(1, 2)                              # [B,4C,L]
            residual = self.distance_refine(refine_in).transpose(1, 2)           # [B,L,C]
        else:
            refine_feat = torch.cat([stage1, input_mask], dim=-1)   # [B,L,2C]
            refine_in = refine_feat.transpose(1, 2)                 # [B,2C,L]
            residual = self.temporal_refine(refine_in).transpose(1, 2)  # [B,L,C]

        if getattr(self, "_length_adaptive_delta_refine", False):
            refine_feat = torch.cat([stage1, input_mask], dim=-1)
            refine_in = refine_feat.transpose(1, 2)
            delta_stack = torch.stack(
                [
                    expert(refine_in).transpose(1, 2)
                    for expert in self.length_delta_experts
                ],
                dim=2,
            )
            weights = self._length_expert_weights(input_mask)
            delta = (delta_stack * weights).sum(dim=2)
            residual = residual + float(getattr(self, "_length_adaptive_delta_scale", 0.1)) * delta

        if getattr(self, "_spectral_residual_refine", False):
            low_freq = self._spectral_lowpass(stage1)
            high_freq = stage1 - low_freq
            spectral_feat = torch.cat([stage1, low_freq, high_freq, input_mask], dim=-1)
            spectral_delta = self.spectral_refine(spectral_feat.transpose(1, 2)).transpose(1, 2)
            residual = residual + float(getattr(self, "_spectral_delta_scale", 0.1)) * spectral_delta

        miss_mask = 1.0 - input_mask
        if getattr(self, "_structural_consistency_gate", False):
            gate = self._structural_consistency_residual_gate(stage1, x_in, input_mask)
            residual = residual * gate

        if getattr(self, "_length_conditioned_refine", False):
            gap_feat = self._mask_gap_length_features(input_mask)
            gate_logits = self.length_refine_gate(gap_feat.transpose(1, 2)).transpose(1, 2)
            gate_scale = float(getattr(self, "_length_refine_gate_scale", 0.5))
            gate = 1.0 + gate_scale * torch.tanh(gate_logits)
            residual = residual * gate

        refine_scale = self.refine_scale if getattr(self, "_learnable_refine_scale", False) else 0.3
        residual_delta = refine_scale * residual
        residual_delta = self.apply_structural_residual_bound(stage1, residual_delta, x_in, input_mask)
        stage2 = stage1 + residual_delta * miss_mask
        return stage2

    def forward_recon_refine(self, x_cat: torch.Tensor):
        """
        Residual-structured imputation:
          stage1 = structural prediction from skeleton + PatchTST residual
          stage2 = stage1 + temporal refinement residual on missing positions

        Optional refinement variants add mask-distance, region, missing-length,
        or spectral residual features without changing the observed positions.

        If self._skeleton_bootstrap is True, the stage-one composition is
        re-skeletonized before refinement: the composed output becomes the
        skeleton for one or more additional shared-weight backbone passes. The
        returned stage1 is the first-pass composition so the auxiliary loss
        keeps supervising the pass whose input distribution matches the
        deterministic linear skeleton, while stage2 refines the bootstrapped
        composition.

        If self._no_refine is True, stage2 is short-circuited to stage1
        (used for ablation: refine on/off).

        Returns: (stage1, stage2), both [B, L, C].
        """
        stage1 = self.compose_stage1(x_cat)  # [B,L,C]
        stage1_final = stage1
        if getattr(self, "_skeleton_bootstrap", False):
            stage1_final = self._bootstrap_stage1(x_cat, stage1)
        stage2 = self._refine_stage1(x_cat, stage1_final)
        return stage1, stage2

    def forward_recon_quantile_residuals(self, x_cat: torch.Tensor) -> torch.Tensor:
        if not getattr(self, "_uncertainty_head", False) or not hasattr(self, "quantile_decoder"):
            raise RuntimeError("uncertainty head is not enabled")
        B, L, twoC = x_cat.shape
        z = self._backbone_latent(x_cat)
        q_flat = self.quantile_decoder(z)
        return self._decode_patch_sequence(q_flat, L, (self.out_channels, len(self._uncertainty_quantiles)))

    def forward_recon_refine_quantiles(self, x_cat: torch.Tensor):
        if not getattr(self, "_uncertainty_head", False) or not hasattr(self, "quantile_decoder"):
            raise RuntimeError("uncertainty head is not enabled")

        B, L, twoC = x_cat.shape
        C = twoC // 2
        z = self._backbone_latent(x_cat)
        patch_flat = self.patch_decoder(z)
        patch_pred = self._decode_patch_sequence(patch_flat, L, (self.out_channels,))
        stage1 = self._compose_stage1_from_patch_pred(x_cat, patch_pred)
        stage2 = self._refine_stage1(x_cat, stage1)

        q_flat = self.quantile_decoder(z)
        q_delta = self._decode_patch_sequence(q_flat, L, (self.out_channels, len(self._uncertainty_quantiles)))
        input_mask = x_cat[:, :, C:]
        miss_mask = 1.0 - input_mask
        quantile_pred = stage2.unsqueeze(-1) + q_delta * miss_mask.unsqueeze(-1)
        return stage1, stage2, quantile_pred

    def encode_repr(self, x_cat: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x_cat)                                 # [B,Lp,D]
        g = z.mean(dim=1)                                        # [B,D]
        p = self.proj(g)                                         # [B,proj_dim]
        p = F.normalize(p, dim=-1)
        return p

    def _mask_distance_features(self, input_mask: torch.Tensor) -> torch.Tensor:
        """Return normalized distance-to-observed and boundary-proximity features."""
        B, L, C = input_mask.shape
        obs = input_mask > 0.5
        large = float(L + 1)
        dist_forward = torch.empty_like(input_mask)
        dist_backward = torch.empty_like(input_mask)

        last = torch.full((B, C), large, device=input_mask.device, dtype=input_mask.dtype)
        for t in range(L):
            last = torch.where(obs[:, t, :], torch.zeros_like(last), last + 1.0)
            dist_forward[:, t, :] = last

        last = torch.full((B, C), large, device=input_mask.device, dtype=input_mask.dtype)
        for t in range(L - 1, -1, -1):
            last = torch.where(obs[:, t, :], torch.zeros_like(last), last + 1.0)
            dist_backward[:, t, :] = last

        miss_mask = 1.0 - input_mask
        clip = float(getattr(self, "_distance_refine_clip", 48.0))
        dist = torch.minimum(dist_forward, dist_backward).clamp(max=clip)
        dist_norm = (dist / clip) * miss_mask
        boundary_proximity = (1.0 - dist_norm).clamp(min=0.0, max=1.0) * miss_mask
        return torch.cat([dist_norm, boundary_proximity], dim=-1)

    def _mask_gap_length_features(self, input_mask: torch.Tensor) -> torch.Tensor:
        """Return missing-run length and center-position features inferred from the mask."""
        B, L, C = input_mask.shape
        miss = (input_mask <= 0.5).float()
        left = torch.zeros_like(input_mask)
        right = torch.zeros_like(input_mask)

        run = torch.zeros(B, C, device=input_mask.device, dtype=input_mask.dtype)
        for t in range(L):
            run = (run + 1.0) * miss[:, t, :]
            left[:, t, :] = run

        run = torch.zeros(B, C, device=input_mask.device, dtype=input_mask.dtype)
        for t in range(L - 1, -1, -1):
            run = (run + 1.0) * miss[:, t, :]
            right[:, t, :] = run

        gap_len = (left + right - 1.0).clamp(min=0.0) * miss
        clip = float(getattr(self, "_length_refine_clip", 48.0))
        gap_len_norm = (gap_len.clamp(max=clip) / clip) * miss

        # 1 near the middle of a missing run, lower near observed boundaries.
        center_proximity = (1.0 - (left - right).abs() / (gap_len + 1e-6)).clamp(0.0, 1.0) * miss
        return torch.cat([gap_len_norm, center_proximity], dim=-1)
