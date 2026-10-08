"""Render the VA-SRI paper tables (EN/CN) from experiment/va_sri results.

Baseline rows are copied verbatim from the frozen V10 tables so that only the
new method row differs.  python -m scripts.va_sri.paper_tables
"""
from __future__ import annotations

import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/va_sri_v11_manuscript"
V10 = ROOT / "output/paper_argument_v10_2026-09-25/manuscript"
DS = ["etth1", "etth2", "ettm1", "ettm2", "weather"]
LABEL = {"etth1": "ETTh1", "etth2": "ETTh2", "ettm1": "ETTm1", "ettm2": "ETTm2", "weather": "Weather"}
SEED_RE = re.compile(r"_s(\d+)_")


def ours(tag: str, metric: str):
    out = {d: {} for d in DS}
    for f in (ROOT / "experiments/va_sri/runs").glob(f"ft_*{tag}*/result.json"):
        name = f.parent.name
        m = SEED_RE.search(name)
        if not m:
            continue
        ds = name.split("_")[1]
        if ds not in out:
            continue
        d = json.loads(f.read_text())
        if "test" in d:
            out[ds][int(m.group(1))] = d["test"][metric]
    return {d: (st.mean(v.values()), st.stdev(v.values()), len(v)) for d, v in out.items() if v}


def baseline_rows(table: str):
    rows = []
    for line in (V10 / table).read_text().splitlines():
        s = line.strip()
        if "&" in s and s.endswith("\\\\") and not s.startswith("\\"):
            rows.append(s)
    return [r for r in rows if not r.startswith("SRI ")]


def va_row(m, dec, bold):
    cells = []
    for d in DS:
        if d not in m:
            cells.append("--")
            continue
        v, sd, _ = m[d]
        s = f"{v:.{dec}f} $\\pm$ {sd:.{dec}f}"
        cells.append(f"\\textbf{{{s}}}" if bold else s)
    return "VA-SRI & " + " & ".join(cells) + " \\\\"


def emit(name, caption, note, rows):
    text = "\n".join([
        "\\begin{table}[!htbp]", "\\centering",
        f"\\caption{{{caption}}}", f"\\label{{tab:{name.split('_')[0]}}}",
        "\\begingroup\\fontsize{9}{11}\\selectfont",
        "\\setlength{\\tabcolsep}{3pt}\\renewcommand{\\arraystretch}{1.14}",
        "\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
        "Method & " + " & ".join(LABEL[d] for d in DS) + " \\\\", "\\midrule",
        *rows, "\\bottomrule", "\\end{tabular}", "\\par\\vspace{3pt}",
        f"\\begin{{minipage}}{{\\linewidth}}\\fontsize{{8.5}}{{10.5}}\\selectfont {note}\\end{{minipage}}",
        "\\endgroup", "\\end{table}", "",
    ])
    (OUT / name).write_text(text)
    print("wrote", name)


def main():
    specs = [("mse", 3, "Main comparison under the UFB protocol.",
              "Block-24 original-scale masked test MSE; mean $\\pm$ sample SD over seeds 42--46. Lower means are bold. "
              "The VA-SRI row is trained without pretraining; baseline rows are taken unchanged from the SRI (V10) "
              "experiment, which shares splits, masks, metrics, and seed range but was not re-run here."),
             ("mae", 3, "Original-scale MAE under the UFB protocol.",
              "Same block-24 test targets, models, and seeds as the main table; mean $\\pm$ sample SD. "
              "MAE is computed after inverse standardization."),
             ("std_mse", 4, "Standardized MSE under the UFB protocol.",
              "Same block-24 test targets, models, and seeds as the main table; mean $\\pm$ sample SD. "
              "MSE is computed before inverse standardization and is dimensionless.")]
    for metric, dec, caption, note in specs:
        m = ours("_none_beta_g_ema", metric)
        stem = {"mse": "main", "mae": "mae", "std_mse": "std_mse"}[metric]
        src = {"mse": "main_table_en.tex", "mae": "mae_table_en.tex", "std_mse": "std_mse_table_en.tex"}[metric]
        rows = [va_row(m, dec, bold=(metric == "mse")), "\\midrule"] + baseline_rows(src)
        emit(f"{stem}_table_en.tex", caption, note, rows)
        cn_note = {"mse": "块长24、原尺度掩码测试 MSE；seeds 42--46 的均值 $\\pm$ 样本 SD，均值更低者加粗。VA-SRI 行不使用预训练；基线行原样取自 SRI（V10）实验，划分、掩码、指标与种子范围相同，但本轮未重跑。",
                   "mae": "与主表相同的块长24测试目标、模型与种子；均值 $\\pm$ 样本 SD。MAE 在反标准化后计算。",
                   "std_mse": "与主表相同的块长24测试目标、模型与种子；均值 $\\pm$ 样本 SD。MSE 在反标准化之前计算，无量纲。"}[metric]
        cn_caption = {"mse": "UFB 协议下的主比较。", "mae": "UFB 协议下的原尺度 MAE。", "std_mse": "UFB 协议下的标准化 MSE。"}[metric]
        emit(f"{stem}_table_cn.tex", cn_caption, cn_note, rows)
    emit_ablation()
    emit_recipe_control()
    print()
    for d, (v, sd, n) in sorted(ours("_none_beta_g_ema", "mse").items()):
        print(f"  {LABEL[d]:8s} n={n} MSE {v:.4f} +- {sd:.4f}")




# ---------------------------------------------------------------- ablation tables
ABL = [("Full VA-SRI", "ft_*_none_beta_g_ema*/result.json"),
       ("w/o visibility-guided attention", "ft_*_pre_p2ett_s*_off_ema_abl/result.json"),
       ("w/o variable identity", "ft_*_pre_p2ett_s*_beta_g_ema_noid_abl/result.json"),
       ("with joint ETT pretraining", "ft_*_pre_p2ett_s*_beta_g_ema/result.json")]


def ablation_rows():
    import glob
    rows = []
    for label, pat in ABL:
        cells = []
        for d in DS:
            vals = []
            for f in glob.glob(str(ROOT / "experiments/va_sri/runs") + "/" + pat.replace("*", "*")):
                pass
            for f in sorted((ROOT / "experiments/va_sri/runs").glob(pat.replace("/result.json", ""))):
                name = f.name
                if name.split("_")[1] != d or not (f / "result.json").exists():
                    continue
                vals.append(json.loads((f / "result.json").read_text())["test"]["mse"])
            cells.append(f"{st.mean(vals):.3f} $\\pm$ {st.stdev(vals):.3f}" if vals else "--")
        rows.append(label + " & " + " & ".join(cells) + " \\\\")
    return rows


def emit_ablation():
    rows = ablation_rows()
    note = ("Three seeds (42--44) and all five datasets; block-24 original-scale masked test MSE, mean "
            "$\\pm$ sample SD. Each row removes or adds one component relative to the main configuration. "
            "Pretraining is absent from the main configuration, so its row adds the joint ETT-stage.")
    cn_note = ("三个种子（42--44）、全部五个数据集；块长24、原尺度掩码测试 MSE 的均值 $\\pm$ 样本 SD。"
               "每行相对主配置去掉或加入一个组件。主配置不含预训练，故最后一行是加入四个 ETT 数据集联合预训练后的结果。")
    for lang, cap, nt in (("en", "Component ablation of VA-SRI.", note),
                          ("cn", "VA-SRI 的组件消融。", cn_note)):
        text = "\n".join([
            "\\begin{table}[!htbp]", "\\centering", f"\\caption{{{cap}}}",
            "\\label{tab:ablation_va}", "\\begingroup\\fontsize{9}{11}\\selectfont",
            "\\setlength{\\tabcolsep}{3pt}\\renewcommand{\\arraystretch}{1.14}",
            "\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
            "Variant & " + " & ".join(LABEL[d] for d in DS) + " \\\\", "\\midrule",
            *rows, "\\bottomrule", "\\end{tabular}", "\\par\\vspace{3pt}",
            f"\\begin{{minipage}}{{\\linewidth}}\\fontsize{{8.5}}{{10.5}}\\selectfont {nt}\\end{{minipage}}",
            "\\endgroup", "\\end{table}", "",
        ])
        (OUT / f"ablation_table_va_{lang}.tex").write_text(text)
        print("wrote", f"ablation_table_va_{lang}.tex")


def emit_recipe_control():
    import glob
    rows = []
    v10 = {"etth2": 2.568, "ettm1": 0.693, "weather": 460.451}
    for d in ("etth2", "ettm1", "weather"):
        vals = []
        for f in sorted(glob.glob(str(ROOT / "experiments/va_sri/v10ctrl" / f"v10ema_{d}_s*" / "log"))):
            m = re.findall(r"\[Test\] masked-MSE=([0-9.]+)", Path(f).read_text())
            if m:
                vals.append(float(m[-1]))
        s = f"{st.mean(vals):.3f} $\\pm$ {st.stdev(vals):.3f}" if vals else "--"
        s += f" ({len(vals)})"
        rows.append(f"{LABEL[d]} & {v10[d]:.3f} & {s} \\\\")
    for lang, cap, hdr, nt in (
        ("en", "Training-recipe control on the joint-channel backbone.",
         "Dataset & V10 recipe & EMA, no early stopping \\\\",
         "The joint-channel SRI is retrained from its own pretrained weights with the recipe of "
         "Section~\\ref{sec:recipe}: EMA decay 0.999, early stopping disabled, and the same epoch budget "
         "as VA-SRI. Original-scale masked test MSE; number of seeds in parentheses."),
        ("cn", "联合通道主干上的训练方式对照。",
         "数据集 & V10 配方 & EMA、不早停 \\\\",
         "用 \\ref{sec:recipe} 节的训练方式（EMA 衰减 0.999、关闭早停、与 VA-SRI 相同的轮数预算）"
         "从 SRI 自身的预训练权重重新微调联合通道模型。原尺度掩码测试 MSE，括号内为种子数。")):
        text = "\n".join([
            "\\begin{table}[!htbp]", "\\centering", f"\\caption{{{cap}}}",
            "\\label{tab:recipe_control}", "\\begingroup\\fontsize{9}{11}\\selectfont",
            "\\begin{tabular}{@{}lcc@{}}", "\\toprule", hdr, "\\midrule",
            *rows, "\\bottomrule", "\\end{tabular}", "\\par\\vspace{3pt}",
            f"\\begin{{minipage}}{{\\linewidth}}\\fontsize{{8.5}}{{10.5}}\\selectfont {nt}\\end{{minipage}}",
            "\\endgroup", "\\end{table}", "",
        ])
        (OUT / f"recipe_control_table_{lang}.tex").write_text(text)
        print("wrote", f"recipe_control_table_{lang}.tex")

if __name__ == "__main__":
    main()
