"""Linear-interpolation (skeleton-only) baseline on the test split, same masks as the main
comparison (seeds 42-46, block 24). No training.  python -m scripts.va_sri.linear_baseline"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import torch

from data.datasets import ImputationDataset
from scripts.va_sri.data import load_splits
from scripts.va_sri.train import make_loader, mask_kwargs
from utils.interp import linear_interp_torch

ROOT = Path(__file__).resolve().parents[2]
DS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"]


@torch.no_grad()
def evaluate(loader, std, c):
    tot = torch.zeros(5, dtype=torch.float64)
    for x, truth, target, _ in loader:
        x, truth, target = x.float(), truth.float(), target.float()
        pred = linear_interp_torch(x[..., :c], x[..., c:])
        err = pred.double() - truth.double()
        m = target.double()
        tot += torch.stack(((err * std).square().mul(m).sum(), (err * std).abs().mul(m).sum(),
                            err.square().mul(m).sum(), err.abs().mul(m).sum(), m.sum()))
    mse, mae, smse, smae, n = tot.tolist()
    return dict(mse=mse / n, mae=mae / n, std_mse=smse / n, std_mae=smae / n)


def main():
    torch.set_num_threads(4)
    out = {}
    for d in DS:
        _, _, test, scaler = load_splits(d)
        std = torch.as_tensor(scaler.std, dtype=torch.float64)
        res = {}
        for s in (42, 43, 44, 45, 46):
            ds = ImputationDataset(test, mode="test", **mask_kwargs(argparse.Namespace(seed=s)))
            res[s] = evaluate(make_loader(ds, False, 0), std, test.shape[1])
        out[d] = res
        print(d, {k: f"{st.mean(r[k] for r in res.values()):.4f}" for k in ("std_mse", "std_mae", "mse", "mae")}, flush=True)
    p = ROOT / "experiments/va_sri/final/linear_baseline.json"
    p.write_text(json.dumps(out, indent=2))
    print("wrote", p)


if __name__ == "__main__":
    main()
