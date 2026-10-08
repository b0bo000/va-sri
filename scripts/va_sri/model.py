"""VA-SRI: variable-axis skeleton-residual imputer with visibility-guided
variable attention (VGA).

Built on the 2026-09-27 variable-axis predictor
(`scripts.sri_variable_axis_ssl.model.build_model`). With VGA disabled the
forward pass is the original one, so earlier no-SSL/legacy-SSL runs remain
valid reference arms.

VGA adds, in every variable-attention block and head h, an additive logit bias
    beta_h * log(v[c'] + eps) + G_h[c, c']
where v[c'] is the visible fraction of key variable c' inside the patch and
G is a learnable variable-pair prior. Both are zero-initialised, so a freshly
enabled VGA model computes exactly the same function as the base model.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from scripts.sri_variable_axis_ssl.model import build_model as build_base

VGA_MODES = ("off", "beta", "beta_g")
LOW_RANK_THRESHOLD = 32   # use G = U V^T above this many variables
LOW_RANK = 8


def run_layer(layer, x: torch.Tensor, attn_bias: torch.Tensor | None) -> torch.Tensor:
    """models.patchtst.TransformerEncoderLayer.forward with an additive mask."""
    if attn_bias is None:
        return layer(x)
    attn_out, _ = layer.mha(x, x, x, attn_mask=attn_bias, need_weights=False)
    x = layer.norm1(x + layer.drop(attn_out))
    return layer.norm2(x + layer.drop(layer.ff(x)))


class VASRI(nn.Module):
    def __init__(self, channels: int, *, seed: int, identity: bool = True,
                 vga: str = "off", eps: float = 1e-3,
                 input_fill: str = "skeleton", output: str = "residual"):
        super().__init__()
        if vga not in VGA_MODES:
            raise ValueError(f"vga must be one of {VGA_MODES}")
        # Factorial switches (defaults = VA-SRI). "zero" feeds the zero-filled
        # values instead of the skeleton; "absolute" predicts values directly
        # instead of a residual on the skeleton and does not copy observations.
        if input_fill not in ("skeleton", "zero") or output not in ("residual", "absolute"):
            raise ValueError("input_fill in {skeleton,zero}, output in {residual,absolute}")
        self.input_fill, self.output = input_fill, output
        # Same construction (and RNG consumption) as the 09-27 experiments.
        self.base = build_base(channels, identity=identity, seed=seed)
        self.channels, self.vga, self.eps = channels, vga, eps
        heads = self.base.n_heads
        layers = self.base.num_layers
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(40_000_000 + seed)
            if vga != "off":
                self.vga_beta = nn.Parameter(torch.zeros(layers, heads))
            if vga == "beta_g":
                if channels <= LOW_RANK_THRESHOLD:
                    self.vga_g = nn.Parameter(torch.zeros(layers, heads, channels, channels))
                else:
                    # U random, V zero: G starts at 0 but receives gradients.
                    self.vga_u = nn.Parameter(0.02 * torch.randn(layers, heads, channels, LOW_RANK))
                    self.vga_v = nn.Parameter(torch.zeros(layers, heads, channels, LOW_RANK))

    # ------------------------------------------------------------------ utils
    def patch_visibility(self, mask: torch.Tensor) -> torch.Tensor:
        """[B,L,C] visibility -> [B,C,P] visible fraction per patch."""
        b = self.base
        bsz, _, c = mask.shape
        m = mask.transpose(1, 2).reshape(bsz * c, 1, b.seq_len)
        if b.right_padding:
            m = F.pad(m, (0, b.right_padding), value=0)
        v = F.avg_pool1d(m, b.patch_len, stride=b.stride)
        return v.reshape(bsz, c, b.num_patches)

    def pair_prior(self, layer: int) -> torch.Tensor | None:
        if self.vga != "beta_g":
            return None
        if hasattr(self, "vga_g"):
            return self.vga_g[layer]
        return self.vga_u[layer] @ self.vga_v[layer].transpose(-1, -2)

    def variable_bias(self, layer: int, vis: torch.Tensor) -> torch.Tensor | None:
        """Additive mask [B*P*H, C, C] for the variable-attention block."""
        if self.vga == "off":
            return None
        bsz, c, p = vis.shape
        heads = self.base.n_heads
        logv = torch.log(vis.permute(0, 2, 1) + self.eps)              # [B,P,C] keys
        bias = self.vga_beta[layer].view(1, 1, heads, 1, 1) * logv[:, :, None, None, :]
        bias = bias.expand(bsz, p, heads, c, c)
        prior = self.pair_prior(layer)
        if prior is not None:
            bias = bias + prior[None, None]
        return bias.reshape(bsz * p * heads, c, c)

    # ---------------------------------------------------------------- forward
    def encode(self, skeleton: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b = self.base
        bsz, _, c = skeleton.shape
        values = skeleton.transpose(1, 2).reshape(bsz * c, 1, b.seq_len)
        vmask = mask.transpose(1, 2).reshape(bsz * c, 1, b.seq_len)
        if b.right_padding:
            values = F.pad(values, (0, b.right_padding), mode="replicate")
            vmask = F.pad(vmask, (0, b.right_padding), value=0)
        z = b.patch_embed(torch.cat([values, vmask], dim=1)).transpose(1, 2)
        z = z.reshape(bsz, c, b.num_patches, b.d_model)
        z = b.input_dropout(z + b.time_pos)
        vis = self.patch_visibility(mask) if self.vga != "off" else None
        for i, time_block in enumerate(b.time_blocks):
            z = time_block(z.reshape(bsz * c, b.num_patches, b.d_model))
            z = z.reshape(bsz, c, b.num_patches, b.d_model)
            across = z.permute(0, 2, 1, 3).reshape(bsz * b.num_patches, c, b.d_model)
            across = run_layer(b.second_blocks[i], across,
                               self.variable_bias(i, vis) if vis is not None else None)
            z = across.reshape(bsz, b.num_patches, c, b.d_model).permute(0, 2, 1, 3)
        return z

    def compose_stage1(self, x_cat: torch.Tensor) -> torch.Tensor:
        visible, mask, skeleton = self.base._prepare(x_cat)
        inp = skeleton if self.input_fill == "skeleton" else visible
        pred = self.base.overlap_average(self.base.patch_decoder(self.encode(inp, mask)))
        if self.output == "absolute":
            return pred
        stage1 = skeleton + (1 - mask) * pred
        return torch.where(mask.bool(), visible, stage1)

    def forward_recon_refine(self, x_cat: torch.Tensor):
        b = self.base
        visible, mask, _ = b._prepare(x_cat)
        stage1 = self.compose_stage1(x_cat)
        features = torch.cat([stage1, mask], dim=-1).transpose(1, 2)
        delta = b.temporal_refine(features).transpose(1, 2)
        stage2 = stage1 + b.refine_scale * (1 - mask) * delta
        if self.output == "residual":
            stage2 = torch.where(mask.bool(), visible, stage2)
        return stage1, stage2

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        return self.forward_recon_refine(x_cat)[1]


# Parameters shared across datasets during (multi-dataset) pretraining.
# Variable identity, all VGA parameters, the refiner and the unused SSL
# projector are finetune-only (zero/fresh initialised), so one pretrained
# trunk serves every VGA setting and dataset.
def is_shared_param(name: str) -> bool:
    excluded = ("base.input_dropout.embedding", "base.temporal_refine.",
                "base.refine_scale", "base.proj.", "vga_")
    return not any(name.startswith(e) or name == e for e in excluded)


def shared_state(model: VASRI) -> dict:
    return {k: v for k, v in model.state_dict().items() if is_shared_param(k)}


def load_shared(model: VASRI, state: dict) -> list[str]:
    own = model.state_dict()
    loaded = []
    for k, v in state.items():
        if k in own and is_shared_param(k):
            if own[k].shape != v.shape:
                raise ValueError(f"shape mismatch for {k}: {own[k].shape} vs {v.shape}")
            own[k] = v
            loaded.append(k)
    model.load_state_dict(own, strict=True)
    return loaded
