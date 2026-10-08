"""Per-variable test MAE of VA-SRI vs tuned SAITS-blocksup (block 24, seeds 42-46), no training.

VA-SRI is re-evaluated from best.pt (test was already evaluated for these runs); SAITS saves no
checkpoint, so its per-variable original-scale MAE is read from the log ("per-var MAE" line of the
test block). Only MAE is available for both, so the decomposition is in MAE.

  python -m scripts.va_sri.per_variable
"""
from __future__ import annotations

import argparse
import ast
import json
import statistics as st
from pathlib import Path

import torch

from data.datasets import ImputationDataset
from scripts.va_sri.data import load_splits
from scripts.va_sri.model import VASRI
from scripts.va_sri.train import make_loader, mask_kwargs

ROOT = Path(__file__).resolve().parents[2]
RUN = "experiments/va_sri/runs/ft_{d}_s{s}_pre_p2ett_s{s}_beta_g_final2"
SAITS = "experiments/va_sri/final/saits_plain_final_{d}_s{s}/log"
SEEDS = (42, 43, 44, 45, 46)


@torch.no_grad()
def per_var_mae(run):
    cfg = json.loads((run / "config.json").read_text())
    _, _, test, scaler = load_splits(cfg["dataset"])
    c = test.shape[1]
    model = VASRI(c, seed=cfg["seed"], identity=not cfg["no_identity"], vga=cfg["vga"])
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True)["model"])
    model.eval()
    std = torch.as_tensor(scaler.std, dtype=torch.float64)
    ds = ImputationDataset(test, mode="test", **mask_kwargs(argparse.Namespace(**cfg)))
    s, n = torch.zeros(c, dtype=torch.float64), torch.zeros(c, dtype=torch.float64)
    for x, truth, target, _ in make_loader(ds, False, 0):
        _, pred = model.forward_recon_refine(x.float())
        m = target.double()
        s += ((pred.double() - truth.double()).abs() * std * m).sum((0, 1))
        n += m.sum((0, 1))
    total = float(s.sum() / n.sum())
    ref = json.loads((run / "result.json").read_text())["test"]["mae"]
    assert abs(total - ref) <= 1e-3 * ref, (total, ref)
    return (s / n).tolist(), scaler.std.tolist()


def saits_per_var(log):
    lines = [l for l in Path(log).read_text().splitlines() if l.startswith("[SAITS") and "per-var MAE" in l]
    return ast.literal_eval(lines[-1].split("per-var MAE:")[1].strip())


def main():
    torch.set_num_threads(4)
    out = {}
    for d in ("etth1", "ettm1"):
        rows = []
        for s in SEEDS:
            va, std = per_var_mae(ROOT / RUN.format(d=d, s=s))
            sa = saits_per_var(ROOT / SAITS.format(d=d, s=s))
            rows.append((va, sa))
        c = len(std)
        var = [x * x for x in std]
        share = [v / sum(var) for v in var]
        res = []
        for j in sorted(range(c), key=lambda j: -share[j]):
            rel = [100 * (r[0][j] / r[1][j] - 1) for r in rows]
            res.append(dict(var=j, variance_share=share[j], va_mae=st.mean(r[0][j] for r in rows),
                            saits_mae=st.mean(r[1][j] for r in rows), rel=st.mean(rel), wins=sum(x < 0 for x in rel)))
            print(f"{d} var{j} share {share[j]:.3f}  VA {res[-1]['va_mae']:.4f}  SAITS {res[-1]['saits_mae']:.4f}  "
                  f"{res[-1]['rel']:+.1f}%  wins {res[-1]['wins']}/5")
        out[d] = res
    (ROOT / "experiments/va_sri/final/per_variable_mae.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
