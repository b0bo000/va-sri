"""Figure 1: overview of VA-SRI, drawn at the final text width.

Top row: input with skeleton -> per-variable patch tokens -> variable-axis encoder -> output.
Bottom strip: decoder -> residual added to the skeleton -> refiner. The series are a real ETTh2
test window (three of the seven variables), each hidden for 24 steps at a different time; token
shading is the true visible fraction of each patch (length 16, stride 8).

python -m scripts.va_sri.draw_architecture
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.colors import to_rgb

from scripts.va_sri import figstyle as fs
from scripts.va_sri.example_cache import load

WINDOW, VARS = 14, (0, 2, 6)                      # ETTh2 test window; HUFL, MUFL, OT
VCOL = ("#4e79a7", "#59a14f", "#b07aa1")          # variable identity colours (tokens, e_c)
INK, MUTED, LINE = "#222222", "#666666", "#9aa3ad"
ENC_BG, ENC_EC = "#f1f5fb", "#c5d3e8"

T = {
    "en": dict(h1="Input and skeleton", h2="Per-variable tokens", h3="Variable-axis encoder", h4="Output",
               patches="$P$ patches", ident="$e_c$", tatt="Temporal\nattention", vatt="Variable\nattention",
               vbias="+ visibility bias", dec="Linear decoder", ref="Refiner $G_\\phi$", layers="$\\times 3$",
               skel="skeleton $S$"),
    "cn": dict(h1="输入与骨架", h2="逐变量 token", h3="变量轴编码器", h4="输出",
               patches="$P$ 个 patch", ident="$e_c$", tatt="时间\n注意力", vatt="变量\n注意力",
               vbias="+ 可见性偏置", dec="线性解码器", ref="细化器 $G_\\phi$", layers="$\\times 3$",
               skel="骨架 $S$"),
}
W, H = 108.0, 54.0
ROWS = ((37.5, 44.5), (28.0, 35.0), (16.5, 23.5))  # y-extent of the three variable rows
DOTS_Y = 25.8


def tint(c, a):
    r = np.array(to_rgb(c))
    return tuple(1 - a * (1 - r))


def arrow(ax, p, q, lw=0.8, color=INK, ms=7):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=3))


def poly(ax, pts, lw=0.8, color=INK):
    ax.plot(*zip(*pts[:-1]), color=color, lw=lw, solid_capstyle="butt", zorder=3)
    arrow(ax, pts[-2], pts[-1], lw=lw, color=color)


def rbox(ax, x, y, w, h, fc, ec, lw=0.7, r=0.9, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                                lw=lw, zorder=z))


def inset(fig, ax, x0, y0, x1, y1):
    tr = ax.transData + fig.transFigure.inverted()
    (a, b), (c, d) = tr.transform((x0, y0)), tr.transform((x1, y1))
    a_ = fig.add_axes([a, b, c - a, d - b])
    a_.set_xticks([]); a_.set_yticks([])
    for s in a_.spines.values():
        s.set_visible(False)
    return a_


def mini(a, x, gap, skel=None, out=None):
    t = np.arange(len(x))
    gi = np.where(gap)[0]
    a.axvspan(gi.min() - 0.5, gi.max() + 0.5, color=fs.HIDDEN, lw=0, zorder=0)
    a.plot(t, np.where(gap, np.nan, x), color=INK, lw=0.8, zorder=2)
    seg = slice(gi.min() - 1, gi.max() + 2)
    if skel is not None:
        a.plot(t[seg], skel[seg], color=fs.SKEL, lw=0.9, ls=(0, (2.2, 1.4)), zorder=3)
    if out is not None:
        a.plot(t[seg], x[seg], color="#b9b9b9", lw=0.7, zorder=2)
        a.plot(t[seg], out[seg], color=fs.VA, lw=1.2, zorder=4)
    a.set_xlim(0, len(x) - 1)
    lo, hi = np.nanmin(x), np.nanmax(x)
    a.set_ylim(lo - 0.12 * (hi - lo), hi + 0.12 * (hi - lo))
    a.axhline(a.get_ylim()[0], color="#cfcfcf", lw=0.5)


def icon(ax, x, y, mode, color):
    """3 x 6 token grid; highlight one row (temporal) or one column (variable)."""
    cw, ch, gap = 1.25, 1.25, 0.3
    for r in range(3):
        for c in range(6):
            on = (mode == "row" and r == 1) or (mode == "col" and c == 3)
            ax.add_patch(Rectangle((x + c * (cw + gap), y + (2 - r) * (ch + gap)), cw, ch,
                                   fc=tint(color, 0.55) if on else "#e4e7eb", ec="none", zorder=2))
    w, h = 6 * cw + 5 * gap, 3 * ch + 2 * gap
    if mode == "row":
        ax.add_patch(FancyArrowPatch((x - 0.2, y + h / 2), (x + w + 0.2, y + h / 2), arrowstyle="<|-|>",
                                     mutation_scale=5, lw=0.7, color=color, zorder=3))
    else:
        cx = x + 3 * (cw + gap) + cw / 2
        ax.add_patch(FancyArrowPatch((cx, y - 0.4), (cx, y + h + 0.4), arrowstyle="<|-|>",
                                     mutation_scale=5, lw=0.7, color=color, zorder=3))
    return w, h


def draw(lang, d, ga=False):
    t = T[lang]
    fs.setup(lang)
    width = (13 / 2.54) if ga else fs.TEXT_W
    fig = plt.figure(figsize=(width, width * H / W))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    f = fs.FS if not ga else fs.FS - 1

    X, M, S, Y = (d[k][WINDOW][:, VARS] for k in ("X", "T", "S", "VA"))
    gaps = M > 0

    for x, head in ((11.0, t["h1"]), (36.5, t["h2"]), (65.5, t["h3"]), (96.0, t["h4"])):
        ax.text(x, 50.6, head, ha="center", va="center", fontsize=f, fontweight="bold", color=INK)

    # (1) input rows with skeleton, (4) output rows; a thin identity bar left of each input row
    for i, (y0, y1) in enumerate(ROWS):
        mini(inset(fig, ax, 2.2, y0, 20.5, y1), X[:, i], gaps[:, i], skel=S[:, i])
        ax.add_patch(Rectangle((1.0, y0 + 0.6), 0.55, y1 - y0 - 1.2, fc=VCOL[i], ec="none"))
        mini(inset(fig, ax, 86.5, y0, 106.0, y1), X[:, i], gaps[:, i], out=Y[:, i])
    for x in (11.0, 36.5, 96.0):
        ax.text(x, DOTS_Y, "$\\vdots$", ha="center", va="center", fontsize=f, color=MUTED)

    # (2) tokens: 11 patches (length 16, stride 8); fill = visible fraction
    tx, cw, gap = 28.4, 1.42, 0.24
    for i, (y0, y1) in enumerate(ROWS):
        yc = (y0 + y1) / 2
        ax.add_patch(Rectangle((25.6, yc - 1.6), 1.7, 3.2, fc=VCOL[i], ec="none", zorder=2))
        for p in range(11):
            v = 1 - gaps[p * 8:p * 8 + 16, i].mean()
            ax.add_patch(FancyBboxPatch((tx + p * (cw + gap), yc - 2.3), cw, 4.6,
                                        boxstyle="round,pad=0,rounding_size=0.35",
                                        fc=tint(VCOL[i], 0.15 + 0.55 * v), ec="none", zorder=2))
            if v < 1:
                ax.add_patch(Rectangle((tx + p * (cw + gap), yc - 2.3), cw, 4.6 * (1 - v), fc="none",
                                       ec="#ffffff", lw=0, hatch="//////", zorder=3))
    gw = 11 * cw + 10 * gap
    ax.text(26.45, ROWS[0][1] + 1.6, t["ident"], ha="center", va="bottom", fontsize=f - 0.5, color=INK)
    ax.text(tx + gw / 2, ROWS[2][0] - 1.3, t["patches"], ha="center", va="top", fontsize=f - 1, color=MUTED)
    ax.plot([tx, tx + gw], [ROWS[2][0] - 0.6] * 2, color=MUTED, lw=0.5)
    arrow(ax, (21.3, 31.5), (25.0, 31.5))

    # (3) encoder
    ex0, ex1, ey0, ey1 = 50.5, 80.5, 16.0, 46.0
    rbox(ax, ex0, ey0, ex1 - ex0, ey1 - ey0, ENC_BG, "none", r=1.4)
    ax.text(ex1 - 1.0, ey1 - 1.2, t["layers"], ha="right", va="top", fontsize=f, color=INK)
    for yb, mode, lab, sub in ((32.2, "row", t["tatt"], None), (18.2, "col", t["vatt"], t["vbias"])):
        rbox(ax, ex0 + 1.4, yb, ex1 - ex0 - 2.8, 11.6, "white", ENC_EC, lw=0.6, r=0.8, z=1.5)
        iw, ih = icon(ax, ex0 + 3.0, yb + (11.6 - 4.35) / 2 + (1.0 if sub else 0), mode, fs.VA)
        tx_ = ex0 + 3.0 + iw + 1.8
        ax.text(tx_, yb + 5.8 + (1.6 if sub else 0), lab, ha="left", va="center", fontsize=f - 0.5,
                color=INK, linespacing=1.15)
        if sub:
            ax.text(tx_, yb + 2.0, sub, ha="left", va="center", fontsize=f - 1.5, color=MUTED)
    arrow(ax, ((ex0 + ex1) / 2, 32.2), ((ex0 + ex1) / 2, 29.8))
    arrow(ax, (tx + gw + 0.6, 31.5), (ex0 - 0.1, 31.5))

    # bottom strip: decoder -> (+ S) -> refiner -> output
    py = 6.2
    rbox(ax, 57.0, 8.6, 17.0, 4.6, "white", LINE, lw=0.7, r=0.8)
    ax.text(65.5, 10.9, t["dec"], ha="center", va="center", fontsize=f - 0.5, color=INK)
    arrow(ax, (65.5, ey0), (65.5, 13.2))
    ox = 78.0
    ax.add_patch(Circle((ox, py), 1.35, fc="white", ec=INK, lw=0.8, zorder=3))
    ax.plot([ox - 0.75, ox + 0.75], [py, py], color=INK, lw=0.8, zorder=4)
    ax.plot([ox, ox], [py - 0.75, py + 0.75], color=INK, lw=0.8, zorder=4)
    poly(ax, [(65.5, 8.6), (65.5, py), (ox - 1.35, py)])
    ax.text(67.0, 7.2, "$R$", fontsize=f, ha="left", va="center")
    poly(ax, [(11.0, ROWS[2][0] - 0.8), (11.0, 2.0), (ox, 2.0), (ox, py - 1.35)])
    ax.text(40.0, 3.1, t["skel"], fontsize=f - 0.5, ha="center", va="center", color=MUTED)
    rbox(ax, 82.5, py - 2.3, 13.5, 4.6, "white", LINE, lw=0.7, r=0.8)
    ax.text(89.25, py, t["ref"], ha="center", va="center", fontsize=f - 0.5, color=INK)
    arrow(ax, (ox + 1.35, py), (82.5, py))
    ax.text(80.1, py + 1.3, "$H$", fontsize=f, ha="center", va="bottom")
    poly(ax, [(96.0, py), (101.0, py), (101.0, ROWS[2][0] - 0.6)])
    ax.text(102.0, 10.5, "$\\widehat{X}$", fontsize=f, ha="left", va="center")

    fs.save(fig, "graphical_abstract" if ga else "architecture", lang)


if __name__ == "__main__":
    d = load()
    for lang in ("en", "cn"):
        draw(lang, d)
        draw(lang, d, ga=True)
    print("wrote", fs.OUT / "architecture_{en,cn}.pdf", "window", WINDOW)
