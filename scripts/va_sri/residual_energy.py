"""Energy of the residual target: test std-MSE of the training mean, of the skeleton and of VA-SRI
for gaps of 12, 24 and 48 steps (seed-42 masks). No training; writes final/skeleton_energy.json.

  CUDA_VISIBLE_DEVICES= python -m scripts.va_sri.residual_energy
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

OUT = Path(__file__).resolve().parents[2] / "experiments/va_sri/final/skeleton_energy.json"


def main():
    res = {}
    for ds in ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"):
        _, _, test, _ = load_splits(ds)
        c = test.shape[1]
        res[ds] = {}
        for bl in (12, 24, 48):
            a = argparse.Namespace(seed=42, mask_mode="block", block_len=bl, point_ratio=0.03)
            sk = mz = n = 0.0
            for x, truth, target, _ in make_loader(ImputationDataset(test, mode="test", **mask_kwargs(a)), False, 0):
                x, t, tr = x.float(), target.double(), truth.double()
                s = linear_interp_torch(x[..., :c], x[..., c:]).double()
                sk += float(((s - tr) ** 2 * t).sum()); mz += float((tr ** 2 * t).sum()); n += float(t.sum())
            res[ds][str(bl)] = (sk / n, mz / n)        # (skeleton, training mean = 0 after standardization)
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
