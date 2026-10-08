"""Error by position inside the gap for the factorial arms (no training; CPU inference from best.pt).

Every hidden step of a gap is assigned to the boundary (first and last quarter of the gap) or the
interior (middle half). The standardized squared error is summed per bin for each arm and for the
skeleton itself. The total over both bins must reproduce the recorded test std-MSE of the run.

  CUDA_VISIBLE_DEVICES= python -m scripts.va_sri.gap_position [--lengths 24 12 48]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.datasets import ImputationDataset
from scripts.va_sri.data import load_splits
from scripts.va_sri.model import VASRI
from scripts.va_sri.train import make_loader, mask_kwargs
from utils.interp import linear_interp_torch

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments/va_sri/runs"
OUT = ROOT / "experiments/va_sri/final/gap_position.json"
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
SEEDS = (42, 43, 44)
ARMS = ("e00", "e10", "e01", "e11")


def run_name(ds, seed, arm, bl):
    if bl == 24:   # the 24-step factorial: the full arm is the no-pretraining ablation run
        tag = "nopre" if arm == "e11" else arm
        return f"ft_{ds.lower()}_s{seed}_none_beta_g_f2_{tag}"
    return f"ft_{ds.lower()}_s{seed}_none_beta_g_f2_{arm}_b{bl}"


def positions(target):
    """target: (B, L, C) 0/1 with one contiguous gap per variable -> bin id 0 boundary, 1 interior, -1 none."""
    t = target.bool()
    L = t.shape[1]
    steps = torch.arange(L).view(1, L, 1)
    first = torch.where(t, steps, L).amin(1, keepdim=True)
    last = torch.where(t, steps, -1).amax(1, keepdim=True)
    rel = (steps - first) / (last - first).clamp(min=1)
    bins = torch.where((rel >= 0.25) & (rel <= 0.75), 1, 0)
    return torch.where(t, bins, -1)


@torch.no_grad()
def evaluate(run: Path, with_skeleton: bool):
    cfg = json.loads((run / "config.json").read_text())
    args = argparse.Namespace(**cfg)
    _, _, test, _ = load_splits(args.dataset)
    c = test.shape[1]
    model = VASRI(c, seed=args.seed, identity=not args.no_identity, vga=args.vga,
                  input_fill=args.input_fill, output=args.output)
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True)["model"])
    model.eval()
    loader = make_loader(ImputationDataset(test, mode="test", **mask_kwargs(args)), False, 0)
    sums = {"model": [0.0, 0.0], "skeleton": [0.0, 0.0]}
    counts = [0, 0]
    bl = args.block_len
    prof = {"model": np.zeros(bl), "skeleton": np.zeros(bl)}
    pcount = np.zeros(bl)
    for x, truth, target, _ in loader:
        x, truth, target = x.float(), truth.float(), target.float()
        _, pred = model.forward_recon_refine(x)
        pos = positions(target)
        e_m = (pred.double() - truth.double()).square()
        e_s = (linear_interp_torch(x[..., :c], x[..., c:]).double() - truth.double()).square() if with_skeleton else None
        t = target.bool()
        steps = torch.arange(t.shape[1]).view(1, -1, 1)
        first = torch.where(t, steps, t.shape[1]).amin(1, keepdim=True)
        k_in = (steps - first).expand_as(t)
        for k in range(bl):
            sel = t & (k_in == k)
            prof["model"][k] += float(e_m[sel].sum())
            if with_skeleton:
                prof["skeleton"][k] += float(e_s[sel].sum())
            pcount[k] += int(sel.sum())
        for k in (0, 1):
            sel = pos == k
            sums["model"][k] += float(e_m[sel].sum())
            if with_skeleton:
                sums["skeleton"][k] += float(e_s[sel].sum())
            counts[k] += int(sel.sum())
    total = (sums["model"][0] + sums["model"][1]) / (counts[0] + counts[1])
    recorded = json.loads((run / "result.json").read_text())["test"]["std_mse"]
    assert abs(total / recorded - 1) < 1e-3, (run.name, total, recorded)   # CPU re-evaluation vs GPU run
    out = {"boundary": sums["model"][0] / counts[0], "interior": sums["model"][1] / counts[1],
           "total": total, "counts": counts, "profile": (prof["model"] / pcount).tolist()}
    if with_skeleton:
        out["skel_boundary"] = sums["skeleton"][0] / counts[0]
        out["skel_interior"] = sums["skeleton"][1] / counts[1]
        out["skel_profile"] = (prof["skeleton"] / pcount).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[24])
    a = ap.parse_args()
    torch.set_num_threads(4)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for bl in a.lengths:
        for ds in DATASETS:
            for seed in SEEDS:
                for arm in ARMS:
                    name = run_name(ds, seed, arm, bl)
                    if (name in res and "profile" in res[name]) or not (RUNS / name / "result.json").exists():
                        continue
                    res[name] = evaluate(RUNS / name, with_skeleton=(arm == "e11"))
                    OUT.write_text(json.dumps(res, indent=1))
                    print(name, {k: round(v, 5) for k, v in res[name].items() if isinstance(v, float)}, flush=True)
    print("wrote", OUT, len(res))


if __name__ == "__main__":
    main()
