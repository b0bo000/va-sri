"""Cache test-split outputs for the figures (ETTh2, seed 42): truth X, target mask T, skeleton S,
VA-SRI first reconstruction H and output Xhat, tuned SAITS output. No training.

  python -m scripts.va_sri.example_cache
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from pypots.imputation import SAITS

from data.datasets import ImputationDataset
from scripts.va_sri.data import load_splits
from scripts.va_sri.model import VASRI
from scripts.va_sri.train import make_loader, mask_kwargs
from utils.interp import linear_interp_torch

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "experiments/va_sri/runs/ft_etth2_s42_pre_p2ett_s42_beta_g_final2"
SAITS_CKPT = ROOT / "experiments/va_sri/final/saits_ckpt_etth2_s42/best.pt"
CACHE = ROOT / "experiments/va_sri/final/example_cache_etth2_s42.npz"
VAR = 6          # OT


def load():
    if not CACHE.exists():
        build()
    return dict(np.load(CACHE))


def pick(d, quantiles=(0.25, 0.5, 0.75)):
    """Windows at given percentiles of the skeleton's error on OT (model-independent selection)."""
    m = d["T"][:, :, VAR]
    err = ((d["S"][:, :, VAR] - d["X"][:, :, VAR]) ** 2 * m).sum(1) / np.maximum(m.sum(1), 1)
    idx = np.where(m.sum(1) > 0)[0]
    order = idx[np.argsort(err[idx])]
    return [int(order[int(q * (len(order) - 1))]) for q in quantiles]


@torch.no_grad()
def build():
    torch.set_num_threads(4)
    cfg = json.loads((RUN / "config.json").read_text())
    _, _, test, _ = load_splits("ETTh2")
    c = test.shape[1]
    loader = make_loader(ImputationDataset(test, mode="test", **mask_kwargs(argparse.Namespace(**cfg))), False, 0)
    va = VASRI(c, seed=cfg["seed"], identity=not cfg["no_identity"], vga=cfg["vga"])
    va.load_state_dict(torch.load(RUN / "best.pt", map_location="cpu", weights_only=True)["model"])
    va.eval()
    sa = SAITS(n_steps=96, n_features=c, n_layers=3, d_model=128, d_ffn=256, n_heads=4, d_k=32, d_v=32,
               dropout=0.1, attn_dropout=0.0, ORT_weight=1.0, MIT_weight=1.0, batch_size=64, epochs=1,
               device="cpu").model
    sa.load_state_dict(torch.load(SAITS_CKPT, map_location="cpu", weights_only=False)["model"])
    sa.eval()
    out = {k: [] for k in ("X", "T", "M", "S", "H", "VA", "SA")}
    for x, truth, target, _ in loader:
        x = x.float()
        h, pv = va.forward_recon_refine(x)
        ps = sa.forward({"X": x[..., :c], "missing_mask": x[..., c:]}, diagonal_attention_mask=True,
                        training=False)["imputed_data"]
        for k, v in (("X", truth), ("T", target), ("M", x[..., c:]), ("S", linear_interp_torch(x[..., :c], x[..., c:])),
                     ("H", h), ("VA", pv), ("SA", ps)):
            out[k].append(v.numpy().astype(np.float32))
    np.savez_compressed(CACHE, **{k: np.concatenate(v) for k, v in out.items()})
    print("wrote", CACHE)


if __name__ == "__main__":
    build()
