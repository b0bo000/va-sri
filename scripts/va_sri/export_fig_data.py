"""Export the data behind Figures 1-3 for the TikZ/pgfplots sources in output/va_sri_paper/figures/src.

Writes CSV files (NaN = no point) and fig_data.tex (macros: gap bounds, token visibility, MSE values).

  python -m scripts.va_sri.export_fig_data
"""
from __future__ import annotations

import numpy as np

from scripts.va_sri.draw_gap_length import LENGTHS, data
from scripts.va_sri.example_cache import VAR, load, pick
from scripts.va_sri.figstyle import OUT
from scripts.va_sri.example_cache import ROOT
from scripts.va_sri.render_paper import DS, NAME

SRC = OUT / "src"
ARCH_WINDOW, ARCH_VARS = 14, (0, 2, 6)          # ETTh2 test window; HUFL, MUFL, OT
QUANTILES = (0.25, 0.5, 0.75)


def csv(path, cols):
    names = list(cols)
    arr = np.column_stack([np.asarray(cols[n], dtype=float) for n in names])
    with open(path, "w") as f:
        f.write(" ".join(names) + "\n")
        for row in arr:
            f.write(" ".join("nan" if np.isnan(v) else f"{v:.5g}" for v in row) + "\n")


def seg_only(y, gap):
    gi = np.where(gap)[0]
    out = np.full_like(y, np.nan, dtype=float)
    out[gi.min() - 1:gi.max() + 2] = y[gi.min() - 1:gi.max() + 2]
    return out


def main():
    SRC.mkdir(parents=True, exist_ok=True)
    d = load()
    t = np.arange(d["X"].shape[1])
    macros = []

    # Figure 1: three variables of one window; values min-max scaled per variable to [0, 1]
    w = ARCH_WINDOW
    cols = {"t": t}
    for k, v in enumerate(ARCH_VARS):
        x, s, y, gap = d["X"][w, :, v], d["S"][w, :, v], d["VA"][w, :, v], d["T"][w, :, v] > 0
        lo, hi = x.min(), x.max()
        sc = lambda z: (z - lo) / (hi - lo)
        cols[f"x{k}"] = np.where(gap, np.nan, sc(x))
        cols[f"s{k}"] = seg_only(sc(s), gap)
        cols[f"g{k}"] = seg_only(sc(x), gap)
        cols[f"y{k}"] = seg_only(sc(y), gap)
        gi = np.where(gap)[0]
        L = "ABC"[k]
        macros.append(f"\\def\\archGap{L}lo{{{gi.min() - 0.5}}}\\def\\archGap{L}hi{{{gi.max() + 0.5}}}")
        vis = [1 - gap[p * 8:p * 8 + 16].mean() for p in range(11)]
        macros.append(f"\\def\\archVis{L}{{{','.join(f'{p}/{q:.4f}' for p, q in enumerate(vis))}}}")
    csv(SRC / "arch.csv", cols)

    # Figure 1(b): learned pair prior Gamma (ETTh2, mean over layers and heads, seed 42) and the visible
    # fraction v of every variable in one patch of the same window
    import torch
    sd = torch.load(ROOT / "experiments/va_sri/runs/ft_etth2_s42_pre_p2ett_s42_beta_g_final2/best.pt",
                    map_location="cpu", weights_only=True)["model"]
    G = sd["vga_g"].mean((0, 1)).numpy()
    with open(SRC / "gamma.csv", "w") as f:
        f.write("x y z\n")
        for i in range(7):
            for j in range(7):
                f.write(f"{j} {i} {G[i, j]:.4f}\n")
            f.write("\n")
    gaps_all = d["T"][w] > 0                                   # (L, 7)
    p_hid = 5                                                  # patch in which MUFL is fully hidden
    v = [1 - gaps_all[p_hid * 8:p_hid * 8 + 16, c].mean() for c in range(7)]
    macros.append("\\def\\visVec{" + ",".join(f"{c}/{x:.3f}" for c, x in enumerate(v)) + "}")
    macros.append(f"\\def\\gammaMax{{{abs(G).max():.2f}}}")

    # Figure 2: OT at the 25th/50th/75th percentile of skeleton error
    for k, w in enumerate(pick(d, QUANTILES)):
        gap = d["T"][w, :, VAR] > 0
        gi = np.where(gap)[0]
        a, b = max(gi.min() - 14, 0), min(gi.max() + 15, len(t))
        X, S, V, A = (d[n][w, :, VAR] for n in ("X", "S", "VA", "SA"))
        e = lambda P: float(((P - X) ** 2)[gap].mean())
        sl = slice(a, b)
        csv(SRC / f"example_{k}.csv", {"t": t[sl], "x": X[sl], "s": seg_only(S, gap)[sl],
                                        "va": seg_only(V, gap)[sl], "sa": seg_only(A, gap)[sl]})
        L = "ABC"[k]
        macros.append(f"\\def\\exGap{L}lo{{{gi.min() - 0.5}}}\\def\\exGap{L}hi{{{gi.max() + 0.5}}}"
                      f"\\def\\exMin{L}{{{a}}}\\def\\exMax{L}{{{b - 1}}}"
                      f"\\def\\exVa{L}{{{e(V):.3f}}}\\def\\exSa{L}{{{e(A):.3f}}}")

    # Figure 3: relative difference VA-SRI vs SAITS per dataset and gap length
    for metric, tag in (("std_mse", "std"), ("mse", "raw")):
        res = data(metric)
        cols = {"i": np.arange(len(DS))}
        for j, bl in enumerate(LENGTHS):
            cols[f"m{bl}"] = [res[ds][j][0] for ds in DS]
            cols[f"e{bl}"] = [res[ds][j][1] for ds in DS]
        csv(SRC / f"gap_{tag}.csv", cols)
    macros.append("\\def\\gapNames{" + ",".join(NAME[ds] for ds in DS) + "}")

    # Figure 3(c,d): error along the gap, factorial arms at 24 steps (scripts/va_sri/gap_position.py)
    import json
    gp = json.loads((ROOT / "experiments/va_sri/final/gap_position.json").read_text())
    prof = lambda ds, arm, key="profile": np.mean([gp[f"ft_{ds}_s{s}_none_beta_g_f2_{arm}"][key] for s in (42, 43, 44)], 0)
    cols = {"k": np.arange(1, 25)}
    for ds in DS:
        f, z = prof(ds, "nopre"), prof(ds, "e00")
        cols[f"f_{ds}"], cols[f"z_{ds}"], cols[f"s_{ds}"] = f, z, prof(ds, "nopre", "skel_profile")
        cols[f"rel_{ds}"] = 100 * (1 - f / z)
    csv(SRC / "profile.csv", cols)

    (SRC / "fig_data.tex").write_text("\n".join(macros) + "\n")
    print("wrote", SRC)


if __name__ == "__main__":
    main()
