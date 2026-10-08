"""Add an SSL interface to the existing variable-axis SRI predictor.

This is a fresh prototype, not a replacement for the V10 paper model. The
default predictor retains the previous TC D64/FF128 configuration. Each
variable keeps its own patch tokens; optional identity vectors are inserted
before input dropout. Both SSL views and stage-one reconstruction use the
observed-only linear skeleton. In particular, the SSL representation input
differs from the original joint-channel SRI's raw augmented-view path.

The projector is initialized from a separate CPU random stream and never
changes the predictor/data-loader RNG. Only synthetic CPU interface checks
are provided alongside this file; there is no training or evaluation entry.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from scripts.sri_redesign.prototype import SharedPatchSRI


class VariableIdentityDropout(nn.Module):
    """Preserve [B,C,P,D]; add a zero-initialized identity to each variable."""

    def __init__(self, dropout: nn.Module, channels: int, reference: torch.Tensor):
        super().__init__()
        self.embedding = nn.Parameter(reference.new_zeros(channels, reference.shape[-1]))
        self.dropout = dropout

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 4 or (tokens.shape[1], tokens.shape[-1]) != tuple(self.embedding.shape):
            raise ValueError('Expected [B,C,P,D] matching the variable identity table')
        return self.dropout(tokens + self.embedding[None, :, None, :])


class VariableAxisSSL(SharedPatchSRI):
    """TC predictor plus projection/compose interfaces used by SSL training."""

    def __init__(self, channels: int, *, identity: bool, projection_seed: int,
                 proj_dim: int = 128, **kwargs):
        if not isinstance(identity, bool):
            raise ValueError('identity must be an explicit boolean')
        if not isinstance(proj_dim, int) or isinstance(proj_dim, bool) or proj_dim < 1:
            raise ValueError('proj_dim must be a positive integer')
        super().__init__(channels, variant='time_channel', **kwargs)
        self.identity_enabled = identity
        self.proj_dim = proj_dim
        if identity:
            self.input_dropout = VariableIdentityDropout(self.input_dropout, channels, self.time_pos)
        # Parent construction is on CPU; isolate the extra head's RNG from
        # both common predictor initialization and the next data-loader draw.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(projection_seed)
            self.proj = nn.Sequential(
                nn.Linear(self.d_model, self.d_model), nn.GELU(),
                nn.Linear(self.d_model, proj_dim),
            )

    def compose_stage1(self, x_cat: torch.Tensor) -> torch.Tensor:
        """Observed-preserving skeleton + residual, with no refiner execution."""
        visible, mask, skeleton = self._prepare(x_cat)
        tokens = self._encode_skeleton(skeleton, mask)
        residual = self.overlap_average(self.patch_decoder(tokens))
        stage1 = skeleton + (1 - mask) * residual
        return torch.where(mask.bool(), visible, stage1)

    def encode_repr(self, x_cat: torch.Tensor) -> torch.Tensor:
        """Normalized [B,proj_dim] SSL vectors from skeleton-filled tokens."""
        tokens = self.encode(x_cat)  # [B,C,P,D], all views use _prepare().
        pooled = tokens.mean(dim=(1, 2))
        return F.normalize(self.proj(pooled), dim=-1)


def build_model(channels: int = 7, *, identity: bool = True, seed: int = 42,
                d_model: int = 64, d_ff: int = 128, **kwargs) -> VariableAxisSSL:
    """Fresh paired ID-on/off models; retain previous TC refiner initialization."""
    torch.random.default_generator.manual_seed(seed)
    model = VariableAxisSSL(channels, identity=identity, projection_seed=30_000_000 + seed,
                            d_model=d_model, d_ff=d_ff, **kwargs)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(20_000_000 + seed)
        refiner = nn.Sequential(
            nn.Conv1d(2 * channels, channels, 3, padding=1), nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=1),
        )
        model.temporal_refine.load_state_dict(refiner.state_dict(), strict=True)
    return model


def parameter_counts(model: VariableAxisSSL) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    projection = sum(p.numel() for p in model.proj.parameters())
    identity = model.input_dropout.embedding.numel() if model.identity_enabled else 0
    refiner = sum(p.numel() for p in model.temporal_refine.parameters()) + model.refine_scale.numel()
    return {'total': total, 'projection': projection, 'identity': identity,
            'predictor_and_refiner': total - projection,
            'predictor_without_identity_or_refiner': total - projection - identity - refiner,
            'refiner_with_scale': refiner}
