"""Figure 2 data: skeleton-residual model vs the same encoder without the skeleton (factorial arms
e11 and e00, seed 42, 24-step gaps, test split). CPU inference from best.pt, no training.

Per dataset one variable; windows at the 25th/50th/75th percentile of the skeleton's error on that
variable (independent of either model). Writes figures/src/grid_<d>_<k>.csv and grid_data.tex.

  CUDA_VISIBLE_DEVICES= python -m scripts.va_sri.example_grid
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from data.datasets import ImputationDataset
from scripts.va_sri.data import load_splits
from scripts.va_sri.figstyle import OUT
from scripts.va_sri.model import VASRI
from scripts.va_sri.train import make_loader, mask_kwargs
from scripts.va_sri.gap_position import RUNS
from utils.interp import linear_interp_torch

SRC = OUT / "src"
PANELS = (("ETTh2", 6, "OT"), ("ETTm1", 6, "OT"), ("Weather", 4, "rh"))
QUANTILES = (0.25, 0.5, 0.75)


def load_model(run):
    cfg = argparse.Namespace(**json.loads((run / "config.json").read_text()))
    _, _, test, _ = load_splits(cfg.dataset)
    m = VASRI(test.shape[1], seed=cfg.seed, identity=not cfg.no_identity, vga=cfg.vga,
              input_fill=cfg.input_fill, output=cfg.output)
    m.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True)["model"])
    return m.eval(), cfg, test


@torch.no_grad()
def collect(ds, var):
    full, cfg, test = load_model(RUNS / f"ft_{ds.lower()}_s42_none_beta_g_f2_nopre")
    zero, _, _ = load_model(RUNS / f"ft_{ds.lower()}_s42_none_beta_g_f2_e00")
    c = test.shape[1]
    loader = make_loader(ImputationDataset(test, mode="test", **mask_kwargs(cfg)), False, 0)
    out = {k: [] for k in ("X", "T", "S", "F", "Z")}
    for x, truth, target, _ in loader:
        x = x.float()
        out["X"].append(truth[..., var].numpy()); out["T"].append(target[..., var].numpy())
        out["S"].append(linear_interp_torch(x[..., :c], x[..., c:])[..., var].numpy())
        out["F"].append(full.forward_recon_refine(x)[1][..., var].numpy())
        out["Z"].append(zero.forward_recon_refine(x)[1][..., var].numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    torch.set_num_threads(4)
    macros = []
    for i, (ds, var, vname) in enumerate(PANELS):
        d = collect(ds, var)
        m = d["T"] > 0
        has = m.sum(1) > 0
        err = lambda P: ((P - d["X"]) ** 2 * m).sum(1) / np.maximum(m.sum(1), 1)
        es = err(d["S"])
        idx = np.where(has)[0]
        order = idx[np.argsort(es[idx])]
        ef, ez = err(d["F"]), err(d["Z"])
        macros.append(f"% {ds} {vname}: over {has.sum()} windows, full<zero-input in {np.mean(ef[idx] < ez[idx]):.3f}")
        L = "ABC"[i]
        macros.append(f"\\def\\gridName{L}{{{ds}, {vname}}}")
        for k, q in enumerate(QUANTILES):
            w = int(order[int(q * (len(order) - 1))])
            gi = np.where(m[w])[0]
            a, b = max(gi.min() - 14, 0), min(gi.max() + 15, m.shape[1])
            seg = np.full(m.shape[1], np.nan)
            seg[gi.min() - 1:gi.max() + 2] = 1
            with open(SRC / f"grid_{i}_{k}.csv", "w") as f:
                f.write("t x s f z\n")
                for t in range(a, b):
                    vals = [d["X"][w, t]] + [d[key][w, t] * seg[t] for key in ("S", "F", "Z")]
                    f.write(f"{t} " + " ".join("nan" if np.isnan(v) else f"{v:.5g}" for v in vals) + "\n")
            macros.append(f"\\expandafter\\def\\csname grid{L}{k}\\endcsname{{{gi.min() - 0.5}/{gi.max() + 0.5}/{a}/{b - 1}/"
                          f"{ef[w]:.2g}/{ez[w]:.2g}}}")
        print(ds, vname, "windows", has.sum(), "full better than zero-input:", f"{np.mean(ef[idx] < ez[idx]):.3f}", flush=True)
    (SRC / "grid_data.tex").write_text("\n".join(macros) + "\n")
    print("wrote", SRC / "grid_data.tex")


if __name__ == "__main__":
    main()
