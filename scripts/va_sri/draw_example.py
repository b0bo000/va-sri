"""Figure 2: three hidden 24-step blocks of OT (ETTh2, test split, seed 42).

Windows are chosen without looking at either model: they sit at the 25th, 50th and 75th percentile of
the skeleton's error over all windows with OT hidden. Uses the cache of scripts/va_sri/example_cache.py.

  python -m scripts.va_sri.draw_example
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from scripts.va_sri import figstyle as fs
from scripts.va_sri.example_cache import VAR, load, pick

QUANTILES = (0.25, 0.5, 0.75)
TXT = {"en": dict(truth="ground truth", skel="skeleton $S$", va="VA-SRI", saits="SAITS-blocksup",
                  hidden="hidden block", x="time step", y="OT (standardized)", mse="MSE"),
       "cn": dict(truth="真实值", skel="骨架 $S$", va="VA-SRI", saits="SAITS-blocksup",
                  hidden="隐藏块", x="时间步", y="OT（标准化）", mse="MSE")}


def draw(lang, d, ws):
    t = TXT[lang]
    fs.setup(lang)
    fig, axes = plt.subplots(1, 3, figsize=(fs.TEXT_W, 2.35))
    steps = np.arange(d["X"].shape[1])
    for k, (ax, w) in enumerate(zip(axes, ws)):
        m = d["T"][w, :, VAR] > 0
        lo, hi = steps[m].min(), steps[m].max()
        a, b = max(lo - 14, 0), min(hi + 15, len(steps))
        seg = slice(lo - 1, hi + 2)
        X, S, V, A = (d[k_][w, :, VAR] for k_ in ("X", "S", "VA", "SA"))
        ax.axvspan(lo - 0.5, hi + 0.5, color=fs.HIDDEN, lw=0, zorder=0, label=t["hidden"])
        ax.plot(steps[a:b], X[a:b], color=fs.TRUTH, lw=1.0, label=t["truth"], zorder=4)
        ax.plot(steps[seg], S[seg], color=fs.SKEL, lw=0.9, ls=(0, (3, 1.5)), label=t["skel"], zorder=3)
        ax.plot(steps[seg], A[seg], color=fs.SAITS, lw=1.1, label=t["saits"], zorder=5)
        ax.plot(steps[seg], V[seg], color=fs.VA, lw=1.1, label=t["va"], zorder=6)
        e = lambda P: float(((P - X) ** 2)[m].mean())
        ax.set_title(f"VA-SRI {e(V):.3f} / SAITS {e(A):.3f}", fontsize=fs.FS - 1.5, color="#444444",
                     loc="center", pad=3)
        ax.set_xlim(a, b - 1); ax.set_xlabel(t["x"], labelpad=1)
        if k == 0:
            ax.set_ylabel(t["y"])
        fs.clean(ax)
        fs.panel(ax, "abc"[k], x=-0.13 if k == 0 else -0.09, y=1.0)
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(pad=0.2, w_pad=1.6, rect=(0, 0, 1, 0.89))
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, handlelength=1.6,
               columnspacing=1.0, fontsize=fs.FS - 1)
    fs.save(fig, "example", lang)


if __name__ == "__main__":
    d = load()
    ws = pick(d, QUANTILES)
    m = d["T"][:, :, VAR]; has = m.sum(1) > 0
    E = {k: (((d[k][:, :, VAR] - d["X"][:, :, VAR]) ** 2 * m).sum(1) / np.maximum(m.sum(1), 1))[has]
         for k in ("S", "VA", "SA")}
    print(f"windows {has.sum()}: VA<skeleton {np.mean(E['VA'] < E['S']):.1%}, VA<SAITS {np.mean(E['VA'] < E['SA']):.1%}")
    for lang in ("en", "cn"):
        draw(lang, d, ws)
    print("wrote", fs.OUT / "example_{en,cn}.pdf", ws)
