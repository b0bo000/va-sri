"""Test std-MSE and original-scale MSE of linear interpolation (the skeleton) under random point
missingness, r in {12.5, 25, 37.5, 50}%, seeds 42-44 (same masks as the trained models). No training.
Writes experiments/va_sri/final/point_skeleton.json.

  CUDA_VISIBLE_DEVICES= python -m scripts.va_sri.point_skeleton
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.datasets import ImputationDataset
from scripts.va_sri.data import load_splits
from scripts.va_sri.train import make_loader, mask_kwargs
from utils.interp import linear_interp_torch

OUT = Path(__file__).resolve().parents[2] / "experiments/va_sri/final/point_skeleton.json"
RATIOS = (0.125, 0.25, 0.375, 0.5)


def main():
    res = {}
    for ds in ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"):
        _, _, test, scaler = load_splits(ds)
        c = test.shape[1]
        std = torch.as_tensor(scaler.std, dtype=torch.float64)
        for r in RATIOS:
            for seed in (42, 43, 44):
                a = argparse.Namespace(seed=seed, mask_mode="point", point_ratio=r, block_len=24)
                e2 = e2o = n = 0.0
                for x, truth, target, _ in make_loader(ImputationDataset(test, mode="test", **mask_kwargs(a)), False, 0):
                    x, t, tr = x.float(), target.double(), truth.double()
                    err = linear_interp_torch(x[..., :c], x[..., c:]).double() - tr
                    e2 += float((err ** 2 * t).sum()); e2o += float(((err * std) ** 2 * t).sum()); n += float(t.sum())
                res[f"{ds}|{r}|{seed}"] = {"std_mse": e2 / n, "mse": e2o / n}
            print(ds, r, flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
