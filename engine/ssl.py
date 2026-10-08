# -*- coding: utf-8 -*-
from typing import Optional
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from losses.losses import nt_xent, masked_mse_loss


def augment_xcat(x_cat: torch.Tensor,
                 jitter_std: float = 0.02,
                 scale_std: float = 0.1,
                 time_mask_ratio: float = 0.0) -> torch.Tensor:
    """
    x_cat: [B,L,2C] -> augmented [B,L,2C]
    """
    B, L, twoC = x_cat.shape
    C = twoC // 2
    val = x_cat[:, :, :C].clone()
    msk = x_cat[:, :, C:].clone()

    if jitter_std > 0:
        val = val + torch.randn_like(val) * jitter_std

    if scale_std > 0:
        s = 1.0 + torch.randn(B, 1, C, device=val.device) * scale_std
        val = val * s

    # 建议当前先把 time_mask_ratio 设为 0.0
    # 避免增强阶段再引入“所有变量同一段一起遮挡”的偏差
    if time_mask_ratio > 0:
        mask_len = max(1, int(L * time_mask_ratio))
        for b in range(B):
            st = torch.randint(0, L - mask_len + 1, (1,), device=val.device).item()
            ed = st + mask_len
            val[b, st:ed, :] = 0.0
            msk[b, st:ed, :] = 0.0

    return torch.cat([val, msk], dim=-1)


def make_recon_mask(B: int, L: int, C: int, block_ratio: float = 0.25,
                    device: Optional[torch.device] = None) -> torch.Tensor:
    """
    single_block-style reconstruction mask:
    for each sample, choose one variable and mask one contiguous block.
    """
    device = device if device is not None else torch.device("cpu")
    mask = torch.zeros(B, L, C, device=device)
    block_len = max(1, int(L * block_ratio))

    for b in range(B):
        c = torch.randint(0, C, (1,), device=device).item()
        s = torch.randint(0, L - block_len + 1, (1,), device=device).item()
        mask[b, s:s + block_len, c] = 1.0

    return mask


def ssl_pretrain(model, train_loader: DataLoader, device: torch.device,
                 epochs: int = 30, lr: float = 3e-4,
                 lambda_recon: float = 0.2, tau: float = 0.2,
                 contrastive_weight: float = 1.0,
                 jitter_std: float = 0.02, scale_std: float = 0.1, time_mask_ratio: float = 0.0,
                 recon_mask_ratio: float = 0.4,
                 save_path: str = "ckpt_ssl.pt") -> None:
    """
    Contrastive (NT-Xent) + masked reconstruction pretraining.
    model is expected to have encode_repr() and forward_recon().
    """
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = total_ctr = total_rec = 0.0
        n_batches = 0

        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x, obs = batch
                obs = obs.to(device).float()
            else:
                x = batch
                obs = None
            x = x.to(device).float()                # x: [B,L,C]
            B, L, C = x.shape

            if obs is None:
                obs = torch.ones(B, L, C, device=device)
            x_cat = torch.cat([x, obs], dim=-1)      # [B,L,2C]

            # two augmented views
            v1 = augment_xcat(x_cat, jitter_std, scale_std, time_mask_ratio)
            v2 = augment_xcat(x_cat, jitter_std, scale_std, time_mask_ratio)

            # contrastive branch
            z1 = model.encode_repr(v1)
            z2 = model.encode_repr(v2)
            loss_ctr = nt_xent(z1, z2, tau=tau)

            # reconstruction branch (task-aligned single_block mask)
            target_clean = x_cat[:, :, :C]           # [B,L,C]
            recon_mask = make_recon_mask(B, L, C, block_ratio=recon_mask_ratio, device=device)
            recon_mask = recon_mask * obs

            v1_in = v1.clone()
            keep = 1.0 - recon_mask
            v1_in[:, :, :C] = v1_in[:, :, :C] * keep
            v1_in[:, :, C:] = v1_in[:, :, C:] * keep

            # Use the same stage1 composition as finetune so SSL trains the
            # backbone+decoder under the same target semantics (residual on
            # linear baseline when _use_linear_base is True, else absolute).
            pred = model.compose_stage1(v1_in)
            loss_rec = masked_mse_loss(pred, target_clean, recon_mask)

            loss = float(contrastive_weight) * loss_ctr + lambda_recon * loss_rec

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            total_ctr += loss_ctr.item()
            total_rec += loss_rec.item()
            n_batches += 1

        sched.step()
        print(f"[SSL] Epoch {ep:03d} | loss={total_loss/max(1,n_batches):.6f} "
              f"| ctr={total_ctr/max(1,n_batches):.6f} | rec={total_rec/max(1,n_batches):.6f}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, save_path)
    print(f"[SSL] saved: {save_path}")
