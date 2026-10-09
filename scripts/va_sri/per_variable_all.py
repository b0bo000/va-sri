"""Per-variable original-scale test MAE of VA-SRI vs SAITS-blocksup on all five datasets (24-step gaps,
seeds 42-46). VA-SRI is re-evaluated on CPU from best.pt (no training); SAITS keeps no checkpoint, so its
per-variable MAE is read from the test block of its log, for the tuned loss and for its own MAE loss.
Writes experiments/va_sri/final/per_variable_all.json.

  CUDA_VISIBLE_DEVICES= python -m scripts.va_sri.per_variable_all
"""
from __future__ import annotations

import json
import statistics as st

import torch

from scripts.va_sri.per_variable import ROOT, RUN, SEEDS, per_var_mae, saits_per_var

SAITS = {"tuned": "experiments/va_sri/final/saits_plain_msemae_{d}_s{s}/log",
         "mae": "experiments/va_sri/final/saits_plain_final_{d}_s{s}/log"}
OUT = ROOT / "experiments/va_sri/final/per_variable_all.json"


def main():
    torch.set_num_threads(4)
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for d in ("etth1", "etth2", "ettm1", "ettm2", "weather"):
        if d in out:
            continue
        va, std = [], None
        for s in SEEDS:
            v, std = per_var_mae(ROOT / RUN.format(d=d, s=s))
            va.append(v)
            print(d, s, "done", flush=True)
        var = [x * x for x in std]
        res = {"variance_share": [x / sum(var) for x in var], "va": [st.mean(r[j] for r in va) for j in range(len(std))]}
        for key, pat in SAITS.items():
            sa = [saits_per_var(ROOT / pat.format(d=d, s=s)) for s in SEEDS]
            rel = [[100 * (va[i][j] / sa[i][j] - 1) for i in range(len(SEEDS))] for j in range(len(std))]
            res[key] = {"mae": [st.mean(r[j] for r in sa) for j in range(len(std))],
                        "rel": [st.mean(x) for x in rel], "wins": [sum(v < 0 for v in x) for x in rel]}
        out[d] = res
        OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
