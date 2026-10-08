"""Shared figure style for the VA-SRI paper (Elsevier artwork rules: final size, 7-8 pt text,
~1 pt data lines, thinner axes, one font family across all figures, vector PDF).

Every method keeps the same colour in every figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = Path(__file__).resolve().parents[2] / "output/va_sri_paper/figures"
CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
TEXT_W = 390 / 72.27         # elsarticle [preprint,12pt] text width (article 12pt: 390 pt), inches

TRUTH, SKEL, VA, SAITS = "#222222", "#8a8a8a", "#2a78d6", "#eb6834"
HIDDEN = "#ececec"
AXIS = "#7a7a7a"
DATASET_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # validated categorical order
DATASET_MARKERS = ["o", "s", "^", "D", "v"]
FS = 9.0          # body text is 12 pt in elsarticle preprint; 9 pt figure text reads in proportion


def setup(lang: str):
    if lang == "cn":
        font_manager.fontManager.addfont(CJK)
        family = ["Noto Sans CJK JP"]
    else:
        family = ["Nimbus Sans", "DejaVu Sans"]
    plt.rcParams.update({
        "font.family": family, "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS - 0.5,
        "ytick.labelsize": FS - 0.5, "legend.fontsize": FS - 0.5, "mathtext.fontset": "cm",
        "pdf.fonttype": 42, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
        "axes.edgecolor": AXIS, "xtick.color": "#444444", "ytick.color": "#444444",
    })


def clean(ax, grid_y=False):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid_y:
        ax.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=0)


def panel(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontsize=FS + 0.5,
            fontweight="bold")


def save(fig, stem, lang):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}_{lang}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{stem}_{lang}.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
