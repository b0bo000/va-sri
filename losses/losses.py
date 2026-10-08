# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

def masked_mse_loss(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((pred - true) ** 2 * mask).sum() / (mask.sum() + 1e-8)

def weighted_masked_mse_loss(pred: torch.Tensor, true: torch.Tensor,
                            mask: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    w = w.view(1, 1, -1)  # [1,1,C]
    return (((pred - true) ** 2) * mask * w).sum() / ((mask * w).sum() + 1e-8)

def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    """
    SimCLR NT-Xent loss:
      z1,z2: [B,D] normalized
    """
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                    # [2B,D]
    sim = torch.mm(z, z.t()) / tau                    # [2B,2B]
    sim.fill_diagonal_(-1e9)                          # remove self similarity

    pos = torch.arange(B, device=z.device)
    labels = torch.cat([pos + B, pos], dim=0)         # positives index
    return F.cross_entropy(sim, labels)
