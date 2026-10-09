"""Learned visibility bias of the final VA-SRI checkpoints (5 datasets x seeds 42-46): sign and size of
beta, cross-seed agreement of Gamma (mean over layers and heads). Writes final/vga_params.json.

  python -m scripts.va_sri.vga_stats
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "experiments/va_sri/runs/ft_{d}_s{s}_pre_p2ett_s{s}_beta_g_final2/best.pt"
OUT = ROOT / "experiments/va_sri/final/vga_params.json"
ETT = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")


def main():
    out = {}
    for d in ("etth1", "etth2", "ettm1", "ettm2", "weather"):
        sds = [torch.load(str(RUN).format(d=d, s=s), map_location="cpu", weights_only=True)["model"] for s in range(42, 47)]
        B = np.stack([sd["vga_beta"].numpy() for sd in sds])
        G = np.stack([sd["vga_g"].mean((0, 1)).numpy() for sd in sds])
        off = ~np.eye(G.shape[1], dtype=bool)
        cor = [np.corrcoef(G[i][off], G[j][off])[0, 1] for i, j in itertools.combinations(range(len(G)), 2)]
        r = {"beta_pos": float(np.mean(B > 0)), "beta_median": float(np.median(B)), "beta_max": float(B.max()),
             "corr_min": float(min(cor)), "corr_max": float(max(cor)),
             "diag": float(np.mean([np.diag(g).mean() for g in G])), "off": float(np.mean([g[off].mean() for g in G]))}
        if d.startswith("ett"):
            gm = G.mean(0)
            pairs = sorted(((gm[i, j] + gm[j, i]) / 2, ETT[i], ETT[j]) for i in range(7) for j in range(i + 1, 7))
            r["neg"] = [(a, b, float(v)) for v, a, b in pairs[:3]]
            r["pos"] = [(a, b, float(v)) for v, a, b in pairs[-3:]]
        out[d] = r
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
