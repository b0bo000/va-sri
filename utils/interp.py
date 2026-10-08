# -*- coding: utf-8 -*-
"""Vectorized linear interpolation on GPU/CPU for [B, L, C] tensors."""
import torch


def linear_interp_torch(x_in: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
    """
    Per-(batch, channel) linear interpolation along time, with constant
    extrapolation at the edges. Vectorized — runs on GPU.

    x_in:       [B, L, C]   value tensor (0 at missing positions is fine)
    input_mask: [B, L, C]   1 = observed, 0 = missing

    Returns: [B, L, C], where every position has a value:
      - observed positions are kept as-is (= x_in)
      - missing positions interpolate linearly between the nearest observed neighbors
      - leading/trailing missing positions copy the first/last observed value
      - if a (batch, channel) is entirely missing, output is 0
    """
    B, L, C = x_in.shape
    device = x_in.device
    dtype = x_in.dtype

    obs = input_mask > 0.5
    t = torch.arange(L, device=device, dtype=dtype).view(1, L, 1).expand(B, L, C)

    # left_idx[b,t,c] = largest s <= t with obs[b,s,c], else -1
    obs_or_neg = torch.where(obs, t, torch.full_like(t, -1.0))
    left_idx, _ = obs_or_neg.cummax(dim=1)
    has_left = left_idx >= 0

    # right_idx[b,t,c] = smallest s >= t with obs[b,s,c], else L (sentinel)
    BIG = float(L)
    obs_or_big = torch.where(obs, t, torch.full_like(t, BIG))
    right_idx_rev, _ = obs_or_big.flip(dims=[1]).cummin(dim=1)
    right_idx = right_idx_rev.flip(dims=[1])
    has_right = right_idx < L

    # safe gather indices
    left_idx_safe = torch.clamp(left_idx, 0, L - 1).long()
    right_idx_safe = torch.clamp(right_idx, 0, L - 1).long()

    # gather x at left/right via [B, C, L] layout
    x_BCL = x_in.permute(0, 2, 1).contiguous()
    left_BCL = left_idx_safe.permute(0, 2, 1).contiguous()
    right_BCL = right_idx_safe.permute(0, 2, 1).contiguous()
    x_left = torch.gather(x_BCL, dim=2, index=left_BCL).permute(0, 2, 1)
    x_right = torch.gather(x_BCL, dim=2, index=right_BCL).permute(0, 2, 1)

    # linear interp; clamp alpha for numerical safety
    diff = right_idx - left_idx
    diff_safe = torch.where(diff > 0, diff, torch.ones_like(diff))
    alpha = ((t - left_idx) / diff_safe).clamp(0.0, 1.0)
    interp = x_left * (1.0 - alpha) + x_right * alpha

    # resolve edge cases
    out = torch.where(
        has_left & has_right,
        interp,
        torch.where(
            has_left,
            x_left,
            torch.where(has_right, x_right, torch.zeros_like(interp)),
        ),
    )

    # at observed positions, use the actual input value (avoids tiny float mismatches)
    out = torch.where(obs, x_in, out)
    return out


def locf_torch(x_in: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
    """Forward-fill with backward-fill fallback for leading missing values."""
    B, L, C = x_in.shape
    device = x_in.device
    dtype = x_in.dtype
    obs = input_mask > 0.5
    t = torch.arange(L, device=device, dtype=dtype).view(1, L, 1).expand(B, L, C)

    obs_or_neg = torch.where(obs, t, torch.full_like(t, -1.0))
    left_idx, _ = obs_or_neg.cummax(dim=1)
    has_left = left_idx >= 0
    left_idx_safe = torch.clamp(left_idx, 0, L - 1).long()

    x_BCL = x_in.permute(0, 2, 1).contiguous()
    left_BCL = left_idx_safe.permute(0, 2, 1).contiguous()
    x_left = torch.gather(x_BCL, dim=2, index=left_BCL).permute(0, 2, 1)

    obs_or_big = torch.where(obs, t, torch.full_like(t, float(L)))
    right_idx_rev, _ = obs_or_big.flip(dims=[1]).cummin(dim=1)
    right_idx = right_idx_rev.flip(dims=[1])
    has_right = right_idx < L
    right_idx_safe = torch.clamp(right_idx, 0, L - 1).long()
    right_BCL = right_idx_safe.permute(0, 2, 1).contiguous()
    x_right = torch.gather(x_BCL, dim=2, index=right_BCL).permute(0, 2, 1)

    out = torch.where(has_left, x_left, torch.where(has_right, x_right, torch.zeros_like(x_in)))
    return torch.where(obs, x_in, out)


def observed_mean_torch(x_in: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
    """Per-window, per-channel observed mean fill."""
    denom = input_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (x_in * input_mask).sum(dim=1, keepdim=True) / denom
    out = mean.expand_as(x_in)
    return torch.where(input_mask > 0.5, x_in, out)


def seasonal_lag_torch(x_in: torch.Tensor, input_mask: torch.Tensor, period: int = 24) -> torch.Tensor:
    """Fill from same-channel seasonal lag, then seasonal lead, with mean fallback."""
    B, L, C = x_in.shape
    period = int(period)
    if period <= 0 or period >= L:
        return observed_mean_torch(x_in, input_mask)

    mean_base = observed_mean_torch(x_in, input_mask)
    out = mean_base.clone()

    lag_val = torch.zeros_like(x_in)
    lag_mask = torch.zeros_like(input_mask)
    lag_val[:, period:, :] = x_in[:, :-period, :]
    lag_mask[:, period:, :] = input_mask[:, :-period, :]

    lead_val = torch.zeros_like(x_in)
    lead_mask = torch.zeros_like(input_mask)
    lead_val[:, :-period, :] = x_in[:, period:, :]
    lead_mask[:, :-period, :] = input_mask[:, period:, :]

    out = torch.where(lead_mask > 0.5, lead_val, out)
    out = torch.where(lag_mask > 0.5, lag_val, out)
    return torch.where(input_mask > 0.5, x_in, out)
