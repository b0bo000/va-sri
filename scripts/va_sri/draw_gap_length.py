"""Figure 3: VA-SRI relative to SAITS-blocksup as a function of gap length, on the standardized (a)
and the original scale (b). Both models are retrained at every length with the settings of the
length experiment (SAITS keeps its own MAE loss), seeds 42-44 at all lengths. Negative = VA-SRI lower.

  python -m scripts.va_sri.draw_gap_length
"""
from __future__ import annotations

import statistics as st

import matplotlib.pyplot as plt

from scripts.va_sri import figstyle as fs
from scripts.va_sri.render_paper import DS, FINAL, NAME, S3, saits, va

LENGTHS = (12, 24, 48)
TXT = {"en": dict(x="Gap length (steps)", y="Relative difference (%)", a="standardized MSE", b="original-scale MSE",
                  len="{} steps"),
       "cn": dict(x="缺口长度（步）", y="相对差异（%）", a="标准化 MSE", b="原尺度 MSE",
                  len="{} 步")}


def rel(d, bl, s, metric):
    if bl == 24:
        a, b = va(FINAL, d, s), saits(f"saits_plain_final_{d}_s{s}")
    else:
        a, b = va(f"ft_{{d}}_s{{s}}_pre_p2ett_s{{s}}_beta_g_f2_b{bl}", d, s), saits(f"saits_plain_b{bl}_{d}_s{s}")
    return 100 * (a[metric] / b[metric] - 1)


def data(metric):
    return {d: [(st.mean(r), st.stdev(r)) for r in ([rel(d, bl, s, metric) for s in S3] for bl in LENGTHS)]
            for d in DS}


SHADES = ("#a9c8ee", "#5b9be0", "#1d5aa6")     # one hue, light -> dark with gap length


def draw(lang, res):
    t = TXT[lang]
    fs.setup(lang)
    fig, axes = plt.subplots(1, 2, figsize=(fs.TEXT_W, 2.3), sharey=True)
    bw = 0.26
    for k, (ax, metric) in enumerate(zip(axes, ("std_mse", "mse"))):
        for j, bl in enumerate(LENGTHS):
            xs = [i + (j - 1) * bw for i in range(len(DS))]
            m = [res[metric][d][j][0] for d in DS]; e = [res[metric][d][j][1] for d in DS]
            ax.bar(xs, m, width=bw * 0.92, color=SHADES[j], edgecolor="white", lw=0.4, zorder=2,
                   label=t["len"].format(bl))
            ax.errorbar(xs, m, yerr=e, fmt="none", ecolor="#333333", elinewidth=0.6, capsize=1.4,
                        capthick=0.6, zorder=3)
        ax.axhline(0, color="#555555", lw=0.6, zorder=2.5)
        ax.set_xticks(range(len(DS))); ax.set_xticklabels([NAME[d] for d in DS], fontsize=fs.FS - 1)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.55, len(DS) - 0.45)
        ax.set_title(t["ab"[k]], fontsize=fs.FS, pad=4)
        fs.clean(ax, grid_y=True)
        ax.spines["bottom"].set_visible(False)
        fs.panel(ax, "ab"[k], x=-0.02 if k else -0.14, y=1.0)
    axes[0].set_ylabel(t["y"])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
               handlelength=1.2, handleheight=0.8, columnspacing=1.6)
    fig.tight_layout(pad=0.2, w_pad=1.2)
    fs.save(fig, "gap_length", lang)


if __name__ == "__main__":
    res = {m: data(m) for m in ("std_mse", "mse")}
    for m in res:
        for d in DS:
            print(m, NAME[d], " ".join(f"{bl}:{a:+.1f}" for bl, (a, _) in zip(LENGTHS, res[m][d])))
    for lang in ("en", "cn"):
        draw(lang, res)
    print("wrote", fs.OUT / "gap_length_{en,cn}.pdf")
