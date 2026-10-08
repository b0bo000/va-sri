"""Render all tables and number macros of the VA-SRI paper from the final runs.

Primary metric: standardized masked MSE. Sources:
  VA-SRI final        experiments/va_sri/runs/*_final2           (MSE + MAE loss, lr 1e-3, EMA)
  SAITS / SAITS+fill  experiments/va_sri/final/saits_*_final_*  (equally tuned, see Section 4)
  other baselines     frozen V10 tables (own recipes, not retuned)
No number is typed by hand.        python -m scripts.va_sri.render_paper
"""
from __future__ import annotations

import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments/va_sri/runs"
FIN = ROOT / "experiments/va_sri/final"
V10 = ROOT / "output/paper_argument_v10_2026-09-25/manuscript"
OUT = ROOT / "output/va_sri_paper/tables"
DS = ["etth1", "etth2", "ettm1", "ettm2", "weather"]
NAME = dict(etth1="ETTh1", etth2="ETTh2", ettm1="ETTm1", ettm2="ETTm2", weather="Weather")
S5, S3 = (42, 43, 44, 45, 46), (42, 43, 44)
M4 = ("std_mse", "std_mae", "mse", "mae")
DEC = dict(std_mse=4, std_mae=4, mse=3, mae=3)
CELL = re.compile(r"([-0-9.]+)\s*\$\\pm\$\s*([-0-9.]+)")
SAITS_RE = re.compile(r"\[SAITS[^\]]*\] masked-MSE=([0-9.]+) \| masked-RMSE=[0-9.]+ \| masked-MAE=([0-9.]+) "
                      r"\(orig scale\)\n\[SAITS[^\]]*\] masked-STD-MSE=([0-9.]+) \| masked-STD-RMSE=[0-9.]+ "
                      r"\| masked-STD-MAE=([0-9.]+)")
V10_RE = re.compile(r"\[Test\] masked-MSE=([0-9.]+) \| masked-RMSE=[0-9.]+ \| masked-MAE=([0-9.]+) \(orig scale\)\n"
                    r"\[Test\] masked-STD-MSE=([0-9.]+) \| masked-STD-RMSE=[0-9.]+ \| masked-STD-MAE=([0-9.]+)")


# ------------------------------------------------------------------- loaders
def va(pattern, d, s):
    f = RUNS / pattern.format(d=d, s=s) / "result.json"
    return json.loads(f.read_text()).get("test") if f.exists() else None


def saits(name):
    f = FIN / name / "log"
    if not (FIN / name / "DONE").exists():
        return None
    m = SAITS_RE.findall(f.read_text())
    return dict(zip(("mse", "mae", "std_mse", "std_mae"), map(float, m[-1]))) if m else None


def backbone(model):
    """Added baselines (scripts/va_sri/run_backbone.py), lr chosen on ETTh2/ETTm1 validation."""
    import re as _re
    def get(d, s):
        f = FIN / f"{model}_{d}_s{s}_lr1e-3" / "log"
        if not (f.parent / "DONE").exists():
            return None
        m = _re.findall(r"\[(?:TimesNet|Crossformer)\] masked-MSE=([0-9.]+) \| masked-RMSE=[0-9.]+ "
                        r"\| masked-MAE=([0-9.]+) \(orig scale\)\n\[(?:TimesNet|Crossformer)\] "
                        r"masked-STD-MSE=([0-9.]+) \| masked-STD-RMSE=[0-9.]+ \| masked-STD-MAE=([0-9.]+)", f.read_text())
        return dict(zip(("mse", "mae", "std_mse", "std_mae"), map(float, m[-1]))) if m else None
    return lambda d: (lambda s: get(d, s))


G_TN, G_CF = backbone("timesnet"), backbone("crossformer")


def v10ctrl(d, s):
    f = ROOT / f"experiments/va_sri/v10ctrl/v10ema_{d}_s{s}/log"
    m = V10_RE.findall(f.read_text()) if f.exists() else []
    return dict(zip(("mse", "mae", "std_mse", "std_mae"), map(float, m[-1]))) if m else None


def agg(getter, seeds, metric):
    vals = [r[metric] for s in seeds if (r := getter(s))]
    if not vals:
        return None
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0), len(vals)


def frozen(table, label):
    for line in (V10 / table).read_text().splitlines():
        cells = [c.strip() for c in line.strip().rstrip("\\").split("&")]
        if cells and cells[0] == label and len(cells) == 6:
            return [tuple(map(float, CELL.search(c).groups())) for c in cells[1:]]
    return None


FINAL = "ft_{d}_s{s}_pre_p2ett_s{s}_beta_g_final2"
G_VA = lambda d: (lambda s: va(FINAL, d, s))
# SAITS loss chosen on ETTh2/ETTm1 validation like every other setting: MSE + MAE (the VA-SRI loss)
# was marginally better than SAITS's own MAE for both arms, so it is the tuned comparator.
G_SP = lambda d: (lambda s: saits(f"saits_plain_msemae_{d}_s{s}"))
G_SF = lambda d: (lambda s: saits(f"saits_fill_msemae_{d}_s{s}"))
G_SPA = lambda d: (lambda s: saits(f"saits_plain_final_{d}_s{s}"))       # SAITS's own MAE loss
G_SFA = lambda d: (lambda s: saits(f"saits_fill_final_{d}_s{s}"))


# ------------------------------------------------------------------- writers
def fmt(v, dec):
    sd = f"{v[1]:.{dec}f}" if v[1] >= 0.5 * 10 ** -dec else f"$<${10 ** -dec:.{dec}f}"   # SD below print precision
    return f"{v[0]:.{dec}f} $\\pm$ {sd}"


def rows_min(grid, dec, axis="col"):
    """grid: [(label, [ (mean, sd) | None ])]. Bold the lowest mean per column (or per row)."""
    out = []
    if axis == "col":
        k = len(grid[0][1])
        best = [min((r[1][j][0] for r in grid if r[1][j]), default=None) for j in range(k)]
    for lab, vals in grid:
        if axis == "row":
            b = min(v[0] for v in vals if v)
        cells = []
        for j, v in enumerate(vals):
            if v is None:
                cells.append("--")
                continue
            tgt = best[j] if axis == "col" else b
            cells.append(f"\\textbf{{{fmt(v, dec)}}}" if v[0] == tgt else fmt(v, dec))
        out.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    return out


def table(stem, cap, label, header, rows, note, colspec, size=9, colsep=4):
    for lang in ("en", "cn"):
        c, h, n, r = (cap, header, note, rows) if lang == "en" else (
            CN.get(cap, cap), tr(header), CN_NOTE.get(stem, note), [tr(x) for x in rows])
        (OUT / f"{stem}_{lang}.tex").write_text("\n".join([
            "\\begin{table}[!htbp]", "\\centering",
            f"\\begingroup\\fontsize{{{size}}}{{{size + 2.5}}}\\selectfont",
            f"\\setlength{{\\tabcolsep}}{{{colsep}pt}}\\renewcommand{{\\arraystretch}}{{1.1}}",
            "\\begin{adjustbox}{max width=\\linewidth}", "\\begin{threeparttable}", f"\\caption{{{c}}}", f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", h + " \\\\", "\\midrule", *r,
            "\\bottomrule", "\\end{tabular}",
            f"\\begin{{tablenotes}}[flushleft]\\fontsize{{8.5}}{{10.5}}\\selectfont\\item {n}\\end{{tablenotes}}",
            "\\end{threeparttable}", "\\end{adjustbox}", "\\endgroup", "\\end{table}", ""]))


ZH = {"Method": "方法", "Variant": "变体", "Dataset": "数据集", "Input": "输入", "Output": "输出",
      "Metric": "指标", "Block": "块长", "Pattern": "模式", "Zero": "零填充", "Skeleton": "骨架",
      "Absolute": "绝对值", "Residual": "残差", "Std. MSE": "标准化 MSE", "Std. MAE": "标准化 MAE",
      "Raw MSE": "原尺度 MSE", "Raw MAE": "原尺度 MAE",
      "w/o visibility bias": "去掉可见性偏置", "w/o variable identity": "去掉变量身份",
      "w/o joint pretraining": "去掉联合预训练", "VA-SRI (full)": "VA-SRI（完整）",
      "Joint-channel encoder": "联合通道编码器", "Original recipe": "原配方", "Our recipe": "本文配方", "shared gap": "共享缺口", "partial": "部分共享", "mixed": "混合", "independent gaps": "独立缺口", "12 steps": "12 步", "24 steps": "24 步", "48 steps": "48 步", "Std.": "标准化", "Orig.": "原尺度", "Params": "参数量", "Learning rate": "学习率", "Recipe": "配方", "Loss": "损失", "Selected": "选定", "Seeds": "种子", "Variable": "变量", "Var. (\\%)": "方差（\\%）", "Wins": "胜出", "Model": "模型", "original": "原配方", "ours": "本文配方", "Train (min)": "训练（分钟）", " (7 variables)": "（7 个变量）", " (21 variables)": "（21 个变量）",
      "VA-SRI, same loss": "VA-SRI·相同损失", "SAITS-blocksup, MAE loss": "SAITS-blocksup·MAE 损失", "TimesNet": "TimesNet", "Crossformer": "Crossformer",
      "Method": "方法", "Std. MSE": "标准化 MSE", "Std. MAE": "标准化 MAE", "Linear interpolation": "线性插值"}


def tr(s):
    for a, b in sorted(ZH.items(), key=lambda kv: -len(kv[0])):
        s = s.replace(a, b)
    return s


CN = {"Imputation error with 24-step gaps (masked MSE).": "缺口长度为 24 步时的插补误差（掩码 MSE）。",
      "Error at different gap lengths.": "不同缺口长度下的误差。",
      "Other missingness patterns and a wide dataset.": "其他缺失模式与高维数据集。",
      "Skeleton input and residual output.": "骨架输入与残差输出。",
      "Model size and training time.": "模型规模与训练时间。",
      "Tuning candidates and their validation error.": "调参候选及其验证误差。",
      "Per-variable error on ETTh1 and ETTm1.": "ETTh1 与 ETTm1 上的逐变量误差。",
      "Standardized masked MSE, mean $\\pm$ SD.": "标准化掩码 MSE（均值 $\\pm$ 标准差）。",
      "Original-scale masked MSE, mean $\\pm$ SD.": "原尺度掩码 MSE（均值 $\\pm$ 标准差）。",
      "Original-scale masked MAE, mean $\\pm$ SD.": "原尺度掩码 MAE（均值 $\\pm$ 标准差）。",
      "Component ablation, mean $\\pm$ SD.": "组件消融（均值 $\\pm$ 标准差）。",
      "Skeleton input and residual output, mean $\\pm$ SD.": "骨架输入与残差输出（均值 $\\pm$ 标准差）。",
      "Gap lengths 12 and 48, mean $\\pm$ SD.": "缺口长度 12 与 48（均值 $\\pm$ 标准差）。",
      "Missingness patterns, mean $\\pm$ SD.": "缺失模式（均值 $\\pm$ 标准差）。",
      "Electricity, mean $\\pm$ SD.": "Electricity（均值 $\\pm$ 标准差）。",
      "Standardized masked MSE (primary metric).": "标准化掩码 MSE（主指标）。",
      "Original-scale masked MAE.": "原尺度掩码 MAE。",
      "Original-scale masked MSE.": "原尺度掩码 MSE。",
      "Skeleton input applied to SAITS.": "将骨架输入用于 SAITS。",
      "Encoder versus training recipe: joint-channel variant and VA-SRI.": "编码器与训练方式的归因：联合通道变体与 VA-SRI。",
      "Component ablation.": "组件消融。",
      "Skeleton input $\\times$ residual output.": "骨架输入 $\\times$ 残差输出。",
      "Block lengths 12 and 48.": "块长 12 与 48。", "Missing patterns.": "缺失模式。"}
CN_NOTE = {}


# ------------------------------------------------------------------- tables
def _csdi():
    """Per-seed CSDI-blocksup test rows of the CCFA v1 study (seeds 42-44, same masks, 50 samples)."""
    out = {}
    for line in (ROOT / "experiments/ccfa_v1/results.md").read_text().splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 14 and c[0] == "main" and c[2] == "block" and c[3] == "24" and c[5] == "CSDI-blocksup":
            out[(c[1].lower(), int(c[4]))] = dict(mse=float(c[7]), mae=float(c[9]), std_mse=float(c[10]), std_mae=float(c[12]))
    return out


CSDI = _csdi()
G_CS = lambda d: (lambda s: CSDI.get((d, s)))
LIN = json.loads((FIN / "linear_baseline.json").read_text())   # scripts/va_sri/linear_baseline.py
OTHERS = ["ImputeFormer", "BRITS-strict", "MRNN-strict", "GPVAE-strict"]
MAC = {}


def main_tables():
    hdr = "Method & " + " & ".join(NAME[d] for d in DS)
    note_tail = ("VA-SRI, SAITS-blocksup and the \\emph{MAE loss} row share one tuning budget on the ETTh2/ETTm1 "
                 "validation splits (Section~\\ref{sec:tuning}), which for SAITS also chose between its own loss and "
                 "ours. TimesNet and Crossformer are trained with our protocol and loss and a tuned learning rate; ImputeFormer "
                 "and the \\emph{-strict} baselines use fixed settings (\\ref{app:settings}). Linear interpolation is "
                 "the skeleton $S$, with no training.")
    for metric, stem, cap, src in (("std_mse", "mainsd", "Standardized masked MSE, mean $\\pm$ SD.", "std_mse_table_en.tex"),
                                   ("mae", "mae", "Original-scale masked MAE, mean $\\pm$ SD.", "mae_table_en.tex"),
                                   ("mse", "rawmse", "Original-scale masked MSE, mean $\\pm$ SD.", "main_table_en.tex")):
        grid = [("VA-SRI", [(a[0], a[1]) if (a := agg(G_VA(d), S5, metric)) else None for d in DS]),
                ("SAITS-blocksup", [(a[0], a[1]) if (a := agg(G_SP(d), S5, metric)) else None for d in DS])]
        grid.append(("SAITS-blocksup, MAE loss", [(a[0], a[1]) if (a := agg(G_SPA(d), S5, metric)) else None for d in DS]))
        grid += [(m, frozen(src, m)) for m in OTHERS]
        grid += [("TimesNet", [(a[0], a[1]) if (a := agg(G_TN(d), S5, metric)) else None for d in DS]),
                 ("Crossformer", [(a[0], a[1]) if (a := agg(G_CF(d), S5, metric)) else None for d in DS])]
        grid.append(("Linear interpolation", [(a[0], a[1]) if (a := agg(lambda s: LIN[NAME[d]][str(s)], S5, metric)) else None
                                              for d in DS]))
        rows = rows_min(grid, DEC[metric])
        rows.insert(3, "\\midrule")
        rows.insert(len(rows) - 1, "\\midrule")
        note = (f"Block length 24, test split, mean $\\pm$ sample SD over seeds 42--46; lowest mean in bold. {note_tail}")
        CN_NOTE[stem] = ("块长 24、测试集，seeds 42--46 的均值 $\\pm$ 样本标准差，均值最低者加粗。VA-SRI、SAITS-blocksup 与 "
                         "\\emph{MAE 损失} 一行共用同一份调参预算（第~\\ref{sec:tuning} 节），SAITS 的这份预算还包括在自身损失与本文损失之间选择。TimesNet 与 Crossformer 采用本文的协议与损失，并调过学习率；ImputeFormer 与 \\emph{-strict} 基线使用固定设置（\\ref{app:settings}）。线性插值即骨架 $S$，没有训练参数。")
        table(stem, cap, f"tab:{stem}", hdr, rows, note, "@{}lccccc@{}")
        for d in DS:
            a, b = agg(G_VA(d), S5, metric), agg(G_SP(d), S5, metric)
            pair = [100 * (G_VA(d)(s)[metric] / G_SP(d)(s)[metric] - 1) for s in S5]
            MAC[f"{metric}VsSaits{NAME[d]}"] = f"{st.mean(pair):+.1f}"
            MAC[f"{metric}WinsSaits{NAME[d]}"] = str(sum(p < 0 for p in pair))
            MAC[f"va{metric}{NAME[d]}"] = f"{a[0]:.{DEC[metric]}f}"
            pa = [100 * (G_VA(d)(s)[metric] / G_SPA(d)(s)[metric] - 1) for s in S5]
            MAC[f"{metric}VsSaitsMae{NAME[d]}"] = f"{st.mean(pa):+.1f}"
            MAC[f"{metric}WinsSaitsMae{NAME[d]}"] = str(sum(p < 0 for p in pa))
            lf = [100 * (G_SP(d)(s)[metric] / G_SPA(d)(s)[metric] - 1) for s in S5]
            MAC[f"{metric}LossFx{NAME[d]}"] = f"{st.mean(lf):+.1f}"


def transfer_table():
    hdr = "Dataset & Metric & VA-SRI & SAITS-blocksup & SAITS + fill"
    rows = []
    for d in DS:
        for i, metric in enumerate(M4):
            vals = [(a[0], a[1]) if (a := agg(g(d), S5, metric)) else None for g in (G_VA, G_SP, G_SF)]
            r = rows_min([(None, vals)], DEC[metric], axis="row")[0].split(" & ", 1)[1]
            lab = {"std_mse": "Std. MSE", "std_mae": "Std. MAE", "mse": "Raw MSE", "mae": "Raw MAE"}[metric]
            rows.append(f"{NAME[d] if i == 0 else ''} & {lab} & {r}")
            pair = [100 * (G_VA(d)(s)[metric] / G_SF(d)(s)[metric] - 1) for s in S5]
            fp = [100 * (G_SF(d)(s)[metric] / G_SP(d)(s)[metric] - 1) for s in S5]
            MAC[f"{metric}FillGain{NAME[d]}"] = f"{st.mean(fp):+.1f}"
            MAC[f"{metric}FillGainWins{NAME[d]}"] = str(sum(p < 0 for p in fp))
            MAC[f"{metric}VsFill{NAME[d]}"] = f"{st.mean(pair):+.1f}"
            MAC[f"{metric}WinsFill{NAME[d]}"] = str(sum(p < 0 for p in pair))
        rows.append("\\midrule")
    rows = rows[:-1]
    note = ("Test split, seeds 42--46. \\emph{+ fill} feeds SAITS the same linear skeleton as VA-SRI instead of zeros; "
            "both SAITS variants are tuned exactly as in Table~\\ref{tab:main} and use the MSE + MAE loss. Lowest mean per row in bold.")
    CN_NOTE["transfer"] = ("测试集，seeds 42--46。\\emph{+ fill} 表示给 SAITS 输入与 VA-SRI 相同的线性骨架而非零填充；"
                           "两种 SAITS 的调参方式与表~\\ref{tab:main} 完全相同，均使用 MSE + MAE 损失。每行均值最低者加粗。")
    table("transfer", "Skeleton input applied to SAITS.", "tab:transfer", hdr, rows, note, "@{}llccc@{}", size=8.5)


def attribution_table():
    v10 = {d: frozen("std_mse_table_en.tex", "SRI")[DS.index(d)] for d in DS}
    v10raw = {d: frozen("main_table_en.tex", "SRI")[DS.index(d)] for d in DS}
    same_loss = "ft_{d}_s{s}_pre_p2ett_s{s}_beta_g_ema"     # VA-SRI with the joint-channel model's weighted loss
    rows = []
    for d in ("etth2", "ettm1", "weather"):
        for metric, lab, ref in (("std_mse", "Std. MSE", v10), ("mse", "Raw MSE", v10raw)):
            rec = agg(lambda s: v10ctrl(d, s), S3, metric)
            ours = agg(lambda s: va(same_loss, d, s), S3, metric)
            vals = [ref[d][0], rec[0], ours[0]]
            dec = 4 if metric == "std_mse" else 3
            r = " & ".join(f"\\textbf{{{num(v, dec)}}}" if v == min(vals) else num(v, dec) for v in vals) + " \\\\"
            rows.append(f"{NAME[d] if metric == 'std_mse' else ''} & {lab} & {r}")
            if metric == "std_mse":
                MAC[f"recipeGain{NAME[d]}"] = f"{100 * (1 - rec[0] / ref[d][0]):.1f}"
                MAC[f"archGain{NAME[d]}"] = f"{100 * (1 - ours[0] / rec[0]):.1f}"
    hdr = ("& & \\multicolumn{2}{c}{Joint-channel encoder} & VA-SRI \\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-5}\n"
           "Dataset & Metric & Original recipe & Our recipe & Our recipe")
    note = ("Test split, mean; lowest error per row in bold. The joint-channel variant embeds all variables of a patch into a single token and is otherwise "
            "a skeleton-residual model; it starts from its own pretrained weights. \\emph{Original recipe}: early stopping, "
            "no weight averaging, seeds 42--46; \\emph{our recipe}: the fixed budget and weight averaging of VA-SRI, "
            "seeds 42--44. All three columns use the same variance-weighted MSE loss, so only the encoder and the recipe differ.")
    CN_NOTE["attribution"] = ("测试集，均值；每行误差最低者加粗。联合通道变体把同一 patch 内的所有变量嵌入为一个 token，其余部分同为骨架残差模型，从其自身的预训练权重出发。"
                              "\\emph{原配方}：早停、不做权重平均，seeds 42--46；\\emph{本文配方}：VA-SRI 的固定轮数与权重平均，seeds 42--44。"
                              "三列使用相同的方差加权 MSE 损失，只有编码器与训练方式不同。")
    table("attribution", "Encoder versus training recipe: joint-channel variant and VA-SRI.", "tab:attribution", hdr, rows, note, "@{}llccc@{}", size=10)


ABL = [("VA-SRI (full)", FINAL), ("w/o visibility bias", "ft_{d}_s{s}_pre_p2ett_s{s}_off_f2_novga"),
       ("w/o variable identity", "ft_{d}_s{s}_pre_p2ett_s{s}_beta_g_f2_noid"),
       ("w/o joint pretraining", "ft_{d}_s{s}_none_beta_g_f2_nopre")]


def ablation_table():
    grid = [(lab, [(a[0], a[1]) if (a := agg(lambda s: va(p, d, s), S3, "std_mse")) else None for d in DS])
            for lab, p in ABL]
    rows = rows_min(grid, 4)
    for lab, p in ABL[1:]:
        key = {"w/o visibility bias": "Vga", "w/o variable identity": "Id", "w/o joint pretraining": "Pre"}[lab]
        for metric in ("std_mse", "std_mae"):
            ch = [100 * (va(p, d, s)[metric] / va(FINAL, d, s)[metric] - 1) for d in DS for s in S3]
            MAC[f"abl{key}{metric}"] = f"{st.mean(ch):+.1f}"
            MAC[f"abl{key}{metric}Worse"] = f"{sum(c > 0 for c in ch)}/{len(ch)}"
            for d in DS:
                cd = [100 * (va(p, d, s)[metric] / va(FINAL, d, s)[metric] - 1) for s in S3]
                MAC[f"abl{key}{metric}{NAME[d]}"] = f"{st.mean(cd):+.1f}"
                MAC[f"abl{key}{metric}Worse{NAME[d]}"] = f"{sum(c > 0 for c in cd)}/{len(cd)}"
    note = "Standardized test MSE, seeds 42--44, mean $\\pm$ sample SD. Each row removes one component from the full model."
    CN_NOTE["ablationsd"] = "标准化测试 MSE，seeds 42--44，均值 $\\pm$ 样本标准差。每行从完整模型中去掉一个组件。"
    table("ablationsd", "Component ablation, mean $\\pm$ SD.", "tab:ablationsd", "Variant & " + " & ".join(NAME[d] for d in DS),
          rows, note, "@{}lccccc@{}", size=8.5)


FACT = [("Zero & Absolute", "ft_{d}_s{s}_none_beta_g_f2_e00", "Neither"),
        ("Skeleton & Absolute", "ft_{d}_s{s}_none_beta_g_f2_e10", "InputOnly"),
        ("Zero & Residual", "ft_{d}_s{s}_none_beta_g_f2_e01", "ResidualOnly"),
        ("Skeleton & Residual", "ft_{d}_s{s}_none_beta_g_f2_nopre", "Both")]


def factorial_table():
    grid = [(lab, [(a[0], a[1]) if (a := agg(lambda s: va(p, d, s), S3, "std_mse")) else None for d in DS])
            for lab, p, _ in FACT]
    rows = rows_min(grid, 4)
    e11 = FACT[3][1]
    for lab, p, k in FACT[:3]:
        for metric in ("std_mse", "std_mae"):
            ch = [100 * (va(e11, d, s)[metric] / va(p, d, s)[metric] - 1) for d in DS for s in S3]
            MAC[f"fact{k}{metric}"] = f"{st.mean(ch):+.1f}"
            MAC[f"fact{k}{metric}Wins"] = f"{sum(c < 0 for c in ch)}/{len(ch)}"
    note = ("All arms use the VA-SRI encoder, the final loss and recipe, and no pretraining; standardized test MSE, "
            "seeds 42--44. Absolute-output arms predict every entry and are also trained on observed entries "
            "(weight 0.02); residual-output arms copy observations.")
    CN_NOTE["factorialsd"] = ("四组均使用 VA-SRI 编码器、最终损失与训练方式，不预训练；标准化测试 MSE，seeds 42--44。"
                            "绝对值输出组预测全部条目，并额外在观测条目上训练（权重 0.02）；残差输出组直接复制观测值。")
    table("factorialsd", "Skeleton input and residual output, mean $\\pm$ SD.", "tab:factorialsd",
          "Input & Output & " + " & ".join(NAME[d] for d in DS), rows, note, "@{}llccccc@{}", size=8.5)


def electricity_table():
    """Breadth check on a wide dataset (321 variables). Standardized metrics only: on Electricity
    about 95% of the original-scale error comes from a single channel, so original-scale numbers
    do not measure overall imputation quality (this is how the earlier study reported it too)."""
    grid = [(lab, [agg(g("electricity"), S3, metric) and tuple(agg(g("electricity"), S3, metric)[:2])
                   for metric in ("std_mse", "std_mae")])
            for lab, g in (("VA-SRI", G_VA), ("SAITS-blocksup", G_SP), ("SAITS-blocksup, MAE loss", G_SPA))]
    rows = rows_min(grid, 4)
    note = ("Electricity has 321 variables; seeds 42--44. About 95\% of its original-scale error comes "
            "from one channel, so only standardized metrics are reported and no claim about the model "
            "is drawn from this dataset.")
    CN_NOTE["electricity"] = ("Electricity 有 321 个变量；seeds 42--44。该数据集原尺度误差约 95\% 来自单一通道，"
                              "因此只报告标准化指标。")
    table("electricity", "Electricity, mean $\\pm$ SD.", "tab:electricity",
          "Method & Std. MSE & Std. MAE", rows, note, "@{}lcc@{}", size=9)


def electricity_macros():
    for name, g in (("Saits", G_SP), ("SaitsMae", G_SPA)):
        for metric in ("std_mse", "std_mae"):
            rel = [100 * (G_VA("electricity")(s)[metric] / g("electricity")(s)[metric] - 1) for s in S3]
            MAC[f"elec{metric.replace('std_', '').title()}Gain{name}"] = f"{st.mean(rel):+.1f}"


def length_table():
    rows = []
    for d in DS:
        for bl in (12, 48):
            vals = []
            for metric in ("std_mse", "mse"):
                a = agg(lambda s: va("ft_{d}_s{s}_pre_p2ett_s{s}_beta_g_f2_b" + str(bl), d, s), S3, metric)
                b = agg(lambda s: saits(f"saits_plain_b{bl}_{d}_s{s}"), S3, metric)
                vals += [(a[0], a[1]) if a else None, (b[0], b[1]) if b else None]
            cells = []
            for j in (0, 2):
                pair = vals[j:j + 2]
                best = min((v[0] for v in pair if v), default=None)
                dec = 4 if j == 0 else 3
                cells += ["--" if v is None else (f"\\textbf{{{fmt(v, dec)}}}" if v[0] == best else fmt(v, dec))
                          for v in pair]
            rows.append(f"{NAME[d]} & {bl} & " + " & ".join(cells) + " \\\\")
    for bl in (12, 48):
        g = [100 * (1 - va(f"ft_{{d}}_s{{s}}_pre_p2ett_s{{s}}_beta_g_f2_b{bl}", d, s)["std_mse"]
                     / saits(f"saits_plain_b{bl}_{d}_s{s}")["std_mse"]) for d in DS for s in S3]
        per = [st.mean(g[i:i + 3]) for i in range(0, len(g), 3)]
        tag = "BShort" if bl == 12 else "BLong"      # digits are translated in macro names
        MAC[f"{tag}GainMin"], MAC[f"{tag}GainMax"] = f"{min(per):.0f}", f"{max(per):.0f}"
        MAC[f"{tag}WinSets"] = f"{sum(x > 0 for x in g)}/{len(g)}"
        raww = [va(f"ft_{{d}}_s{{s}}_pre_p2ett_s{{s}}_beta_g_f2_b{bl}", d, s)["mse"]
                < float(open(f"experiments/va_sri/final/saits_plain_b{bl}_{d}_s{s}/log").read().split("masked-MSE=")[-1].split(" ")[0])
                for d in DS for s in S3]
        MAC[f"{tag}RawWins"] = f"{sum(raww)}/{len(raww)}"
    hdr = "Dataset & Block & VA-SRI & SAITS-blocksup & VA-SRI & SAITS-blocksup"
    hdr = "& & \\multicolumn{2}{c}{Std. MSE} & \\multicolumn{2}{c}{Raw MSE} \\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n" + hdr
    note = "Each model is trained and tested at the stated block length with its tuned setting (SAITS with its own MAE loss); test split, seeds 42--44."
    CN_NOTE["length"] = "各模型以其调参后的设置在所示块长下训练与测试（SAITS 使用其自身的 MAE 损失）；测试集，seeds 42--44。"
    table("length", "Gap lengths 12 and 48, mean $\\pm$ SD.", "tab:length", hdr, rows, note, "@{}lccccc@{}", size=8.5)


def pattern_table():
    rows = []
    for d in ("etth2", "weather"):
        for mode in ("block_shared", "partial", "mixed"):
            grid = []
            for lab, g in (("VA-SRI", lambda s: va("ft_{d}_s{s}_pre_p2ett_s{s}_beta_g_f2_" + mode, d, s)),
                           ("SAITS-blocksup", lambda s: saits(f"saits_plain_{mode}_{d}_s{s}"))):
                grid.append((lab, [(a[0], a[1]) if (a := agg(g, S3, m)) else None for m in ("std_mse", "std_mae", "mse")]))
            body = rows_min_mixed(grid)
            for i, r in enumerate(body):
                rows.append((f"{NAME[d]} & {mode.replace('_', chr(92) + '_')} & " if i == 0 else " & & ") + r)
            rows.append("\\midrule")
    rel_mean = [st.mean(100 * (1 - va("ft_{d}_s{s}_pre_p2ett_s{s}_beta_g_f2_" + mode, d, s)["std_mse"]
                                / saits(f"saits_plain_{mode}_{d}_s{s}")["std_mse"]) for s in S3)
                for d in ("etth2", "weather") for mode in ("block_shared", "partial", "mixed")]
    MAC["patGainMin"], MAC["patGainMax"] = f"{min(rel_mean):.1f}", f"{max(rel_mean):.1f}"
    rows = rows[:-1]
    hdr = "Dataset & Pattern & Method & Std. MSE & Std. MAE & Raw MSE"
    note = "Block length 24, test split, seeds 42--44; both methods use their tuned settings (SAITS with its own MAE loss)."
    CN_NOTE["pattern"] = "块长 24、测试集，seeds 42--44；两种方法均使用调参后的设置（SAITS 使用其自身的 MAE 损失）。"
    table("pattern", "Missingness patterns, mean $\\pm$ SD.", "tab:pattern", hdr, rows, note, "@{}lllccc@{}", size=8.5)


def rows_min_mixed(grid):
    decs = (4, 4, 3)
    best = [min((r[1][j][0] for r in grid if r[1][j]), default=None) for j in range(3)]
    out = []
    for lab, vals in grid:
        cells = ["--" if v is None else (f"\\textbf{{{fmt(v, decs[j])}}}" if v[0] == best[j] else fmt(v, decs[j]))
                 for j, v in enumerate(vals)]
        out.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    return out


def backbone_macros():
    for name, g in (("TimesNet", G_TN), ("Crossformer", G_CF)):
        for metric in ("std_mse", "mse"):
            rel = [100 * (G_VA(d)(s)[metric] / g(d)(s)[metric] - 1) for d in DS for s in S5
                   if g(d)(s) and G_VA(d)(s)]
            MAC[f"{metric}Vs{name}"] = f"{st.mean(rel):+.1f}"
            MAC[f"{metric}Wins{name}"] = f"{sum(r < 0 for r in rel)}/{len(rel)}"
            for d in DS:
                cd = [100 * (G_VA(d)(s)[metric] / g(d)(s)[metric] - 1) for s in S5 if g(d)(s)]
                MAC[f"{metric}Vs{name}{NAME[d]}"] = f"{st.mean(cd):+.1f}"


def csdi_facts():
    per = {NAME[d]: sum(1 for s in S3 if G_CS(d)(s) and G_VA(d)(s)["std_mse"] < G_CS(d)(s)["std_mse"]) for d in DS}
    MAC["csdiVaWinsPerDataset"] = " ".join(f"{k}~{v}/3" for k, v in per.items())


def csdi_macros():
    for d in DS:
        for metric in ("std_mse", "mse"):
            rel = [100 * (G_VA(d)(s)[metric] / G_CS(d)(s)[metric] - 1) for s in S3]
            MAC[f"{metric}VsCsdi{NAME[d]}"] = f"{st.mean(rel):+.1f}"
            MAC[f"{metric}WinsCsdi{NAME[d]}"] = str(sum(r < 0 for r in rel))


def significance_macros():
    from scripts.va_sri.significance import t_pvalue
    for base, g, tag in (("SAITS-blocksup", G_SP, "Saits"), ("SAITS + fill", G_SF, "Fill")):
        for metric in ("std_mse", "mse"):
            for d in DS:
                rel = [100 * (G_VA(d)(s)[metric] / g(d)(s)[metric] - 1) for s in S5]
                p = t_pvalue(rel)
                MAC[f"p{metric}{tag}{NAME[d]}"] = "$p<0.001$" if p < 0.001 else f"$p={p:.3f}$"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (main_tables, backbone_macros, significance_macros, transfer_table, attribution_table, ablation_table, factorial_table, length_table, pattern_table, electricity_table, electricity_macros, *COMPACT):
        try:
            fn()
        except (TypeError, ZeroDivisionError) as e:      # robustness runs may still be pending
            print(f"skipped {fn.__name__}: {e}")
    for k, v in list(MAC.items()):
        if isinstance(v, str) and v[:1] in "+-" and v[1:].replace(".", "").isdigit():
            MAC[k + "Abs"] = v[1:]
    digits = str.maketrans({"1": "One", "2": "Two", "_": ""})
    lines = [f"\\newcommand{{\\{k.translate(digits)}}}{{{v}}}" for k, v in sorted(MAC.items())]
    (OUT / "macros.tex").write_text("% generated by scripts/va_sri/render_paper.py\n" + "\n".join(lines) + "\n")
    print(f"{len(lines)} macros")



# ------------------------------------------------------------------- compact main-text tables
# Means only (SDs are in the appendix tables); best per column in bold, second best underlined.
def num(v, dec):
    if dec == "auto":                                  # original-scale values span 0.5 to 1300
        dec = 1 if abs(v) >= 100 else 2 if abs(v) >= 10 else 3
    return f"{v:.{dec}f}"


def ranked(cols, decs, rank_rows=None):
    """cols: list of columns, each a list of means (None allowed). Returns formatted columns."""
    out = []
    for col, dec in zip(cols, decs):
        cand = sorted({round(float(num(v, dec)), 6) for i, v in enumerate(col)
                       if v is not None and (rank_rows is None or i in rank_rows)})
        cells = []
        for i, v in enumerate(col):
            if v is None:
                cells.append("--"); continue
            s, r = num(v, dec), round(float(num(v, dec)), 6)
            ok = rank_rows is None or i in rank_rows
            cells.append(f"\\textbf{{{s}}}" if ok and r == cand[0] else
                         f"\\underline{{{s}}}" if ok and len(cand) > 1 and r == cand[1] else s)
        out.append(cells)
    return out


def ctable(stem, cap, label, header, rows, note, colspec, size=10, colsep=4):
    table(stem, cap, label, header, rows, note, colspec, size=size, colsep=colsep)


def grouped_header(first, groups, subs):
    n = len(subs)
    top = first + " & " + " & ".join(f"\\multicolumn{{{n}}}{{c}}{{{g}}}" for g in groups) + " \\\\\n"
    k = first.count("&") + 2
    rules = "".join(f"\\cmidrule(lr){{{k + i * n}-{k + i * n + n - 1}}}" for i in range(len(groups)))
    sub = "&" * first.count("&") + " & " + " & ".join(subs * len(groups))
    return top + rules + "\n" + sub


def main_compact():
    methods = [("VA-SRI", G_VA), ("SAITS-blocksup", G_SP), ("SAITS-blocksup, MAE loss", G_SPA)]
    tail = [("TimesNet", G_TN), ("Crossformer", G_CF)]
    labels, cols = [], []
    def series(metric, src):
        out = [[(a[0] if (a := agg(g(d), S5, metric)) else None) for d in DS] for _, g in methods]
        out += [[c[0] for c in frozen(src, m)] for m in OTHERS]
        out += [[(a[0] if (a := agg(g(d), S5, metric)) else None) for d in DS] for _, g in tail]
        out.append([(a[0] if (a := agg(lambda s: LIN[NAME[d]][str(s)], S5, metric)) else None) for d in DS])
        return out
    labels = ["VA-SRI", "SAITS-blocksup", "SAITS-blocksup$^{\\dagger}$"] + OTHERS + [m for m, _ in tail] + ["Linear interpolation"]
    std, raw = series("std_mse", "std_mse_table_en.tex"), series("mse", "main_table_en.tex")
    cols = []
    for j in range(len(DS)):
        cols += [[r[j] for r in std], [r[j] for r in raw]]
    fmtc = ranked(cols, [3, "auto"] * len(DS))
    rows = [f"{lab} & " + " & ".join(c[i] for c in fmtc) + " \\\\" for i, lab in enumerate(labels)]
    rows.insert(3, "\\midrule"); rows.insert(len(rows) - 1, "\\midrule")
    hdr = grouped_header("Method", [NAME[d] for d in DS], ["Std.", "Orig."])
    note = ("Masked MSE, block length 24, test split, mean over seeds 42--46 (SDs in Tables~\\ref{tab:mainsd} "
            "and~\\ref{tab:rawmse}). \\emph{Std.}: standardized scale (primary metric); \\emph{Orig.}: original scale. "
            "Best in bold, second best underlined. $^{\\dagger}$Trained with its own MAE loss. VA-SRI and both SAITS rows share one tuning "
            "budget (Section~\\ref{sec:tuning}); TimesNet and Crossformer use our protocol and loss with a tuned learning "
            "rate; ImputeFormer and the \\emph{-strict} baselines use fixed settings (\\ref{app:settings}). Linear "
            "interpolation is the skeleton $S$.")
    CN_NOTE["main"] = ("掩码 MSE，块长 24，测试集，seeds 42--46 的均值（标准差见表~\\ref{tab:mainsd} 与表~\\ref{tab:rawmse}）。"
                       "\\emph{标准化}：标准化尺度（主指标）；\\emph{原尺度}：原始量纲。最优加粗，次优加下划线。$^{\\dagger}$使用其自身的 MAE 损失训练。VA-SRI 与两行 SAITS"
                       "共用同一份调参预算（第~\\ref{sec:tuning} 节）；TimesNet 与 Crossformer 采用本文协议与损失并调过学习率；"
                       "ImputeFormer 与 \\emph{-strict} 基线使用固定设置（\\ref{app:settings}）。线性插值即骨架 $S$。")
    ctable("main", "Imputation error with 24-step gaps (masked MSE).", "tab:main", hdr, rows, note,
           "@{}l" + "cc" * len(DS) + "@{}", size=9, colsep=3)


def length_compact():
    rows = []
    cols_all = []
    for d in DS:
        line = []
        for bl in (12, 24, 48):
            if bl == 24:
                a, b = agg(G_VA(d), S3, "std_mse"), agg(G_SPA(d), S3, "std_mse")
            else:
                a = agg(lambda s: va("ft_{d}_s{s}_pre_p2ett_s{s}_beta_g_f2_b" + str(bl), d, s), S3, "std_mse")
                b = agg(lambda s: saits(f"saits_plain_b{bl}_{d}_s{s}"), S3, "std_mse")
            for x, y in ((a, b),):
                line += [x[0], y[0]]
        cells = []
        for k in range(0, 6, 2):
            va_s, sa_s = num(line[k], 4), num(line[k + 1], 4)
            better = line[k] < line[k + 1]
            cells += [f"\\textbf{{{va_s}}}" if better else va_s, sa_s if better else f"\\textbf{{{sa_s}}}"]
        rows.append(f"{NAME[d]} & " + " & ".join(cells) + " \\\\")
    hdr = grouped_header("Dataset", ["12 steps", "24 steps", "48 steps"], ["VA-SRI", "SAITS"])
    note = ("Standardized masked MSE, test split, mean over seeds 42--44; the better of the two models in bold. Both "
            "models are retrained for each gap length; SAITS uses its own MAE loss here (Section~\\ref{sec:length}).")
    CN_NOTE["lengthc"] = ("标准化掩码 MSE，测试集，seeds 42--44 的均值；两种模型中较优者加粗。两种模型在每个缺口长度下分别重新训练；"
                          "此处 SAITS 使用其自身的 MAE 损失（第~\\ref{sec:length} 节）。")
    ctable("lengthc", "Error at different gap lengths.", "tab:lengthc", hdr, rows, note, "@{}lcccccc@{}", size=10)


def pattern_compact():
    rows = []
    for d in ("etth2", "weather"):
        for i, mode in enumerate(("block_shared", "partial", "mixed")):
            gv = lambda s: va("ft_{d}_s{s}_pre_p2ett_s{s}_beta_g_f2_" + mode, d, s)
            gs = lambda s: saits(f"saits_plain_{mode}_{d}_s{s}")
            cells = []
            for metric, dec in (("std_mse", 4), ("std_mae", 4)):
                a, b = agg(gv, S3, metric)[0], agg(gs, S3, metric)[0]
                cells += [f"\\textbf{{{num(a, dec)}}}" if a < b else num(a, dec),
                          num(b, dec) if a < b else f"\\textbf{{{num(b, dec)}}}"]
            name = {"block_shared": "shared gap", "partial": "partial", "mixed": "mixed"}[mode]
            rows.append(f"{NAME[d] if i == 0 else ''} & {name} & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    cells = []
    for metric in ("std_mse", "std_mae"):
        a, b = agg(G_VA("electricity"), S3, metric)[0], agg(G_SP("electricity"), S3, metric)[0]
        cells += [f"\\textbf{{{num(a, 4)}}}" if a < b else num(a, 4), num(b, 4) if a < b else f"\\textbf{{{num(b, 4)}}}"]
    rows.append("Electricity & independent gaps & " + " & ".join(cells) + " \\\\")
    hdr = grouped_header("Dataset & Pattern", ["Std. MSE", "Std. MAE"], ["VA-SRI", "SAITS"])
    note = ("Test split, mean over seeds 42--44; the better model in bold. \\emph{Shared gap}: all variables are hidden "
            "over the same 24 steps; \\emph{partial}: two to four variables share a gap; \\emph{mixed}: points, single-"
            "variable gaps and partial gaps. SAITS uses its own MAE loss for the patterns and the tuned loss on "
            "Electricity (321 variables). SDs and original-scale errors are in Tables~\\ref{tab:pattern} and~\\ref{tab:electricity}.")
    CN_NOTE["patternc"] = ("测试集，seeds 42--44 的均值；较优者加粗。\\emph{共享缺口}：所有变量在同一 24 步内被隐藏；\\emph{部分共享}：两至四个变量共享缺口；"
                           "\\emph{混合}：点缺失、单变量缺口与部分共享缺口的混合。缺失模式实验中 SAITS 使用其自身的 MAE 损失，"
                           "Electricity（321 个变量）上使用调参后的损失。标准差与原尺度误差见表~\\ref{tab:pattern} 与表~\\ref{tab:electricity}。")
    ctable("patternc", "Other missingness patterns and a wide dataset.", "tab:patternc", hdr, rows, note,
           "@{}llcccc@{}", size=10)


def factorial_compact():
    cols = [[agg(lambda s: va(p, d, s), S3, "std_mse")[0] for _, p, _ in FACT] for d in DS]
    full = cols
    rel = []
    for _, p, _ in FACT:
        ch = [100 * (va(p, d, s)["std_mse"] / va(FACT[3][1], d, s)["std_mse"] - 1) for d in DS for s in S3]
        rel.append(st.mean(ch))
    fc = ranked(cols, [4] * len(DS))
    rows = []
    for i, (lab, _, _) in enumerate(FACT):
        r = "--" if i == 3 else f"$+${rel[i]:.1f}"
        rows.append(f"{lab} & " + " & ".join(c[i] for c in fc) + f" & {r} \\\\")
    hdr = "Input & Output & " + " & ".join(NAME[d] for d in DS) + " & $\\Delta$ (\\%)"
    note = ("Standardized test MSE, mean over seeds 42--44. All arms use the VA-SRI encoder, loss and recipe without "
            "pretraining. $\\Delta$: mean relative increase over the full design (last row). Absolute-output arms "
            "predict every entry and are also trained on observed entries (weight 0.02); residual-output arms copy "
            "the observations.")
    CN_NOTE["factorial"] = ("标准化测试 MSE，seeds 42--44 的均值。四组均使用 VA-SRI 的编码器、损失与训练配方，不预训练。$\\Delta$："
                            "相对完整设计（最后一行）的平均误差增幅。绝对值输出组预测全部条目，并额外在观测条目上训练（权重 0.02）；"
                            "残差输出组直接复制观测值。")
    ctable("factorial", "Skeleton input and residual output.", "tab:factorial", hdr, rows, note,
           "@{}ll" + "c" * len(DS) + "c@{}", size=10)


def ablation_compact():
    cols = [[agg(lambda s: va(p, d, s), S3, "std_mse")[0] for _, p in ABL] for d in DS]
    fc = ranked(cols, [4] * len(DS))
    rows = []
    for i, (lab, p) in enumerate(ABL):
        if i == 0:
            r = "--"
        else:
            ch = st.mean(100 * (va(p, d, s)["std_mse"] / va(FINAL, d, s)["std_mse"] - 1) for d in DS for s in S3)
            r = f"${'+' if ch >= 0 else '-'}${abs(ch):.1f}"
        rows.append(f"{lab} & " + " & ".join(c[i] for c in fc) + f" & {r} \\\\")
    rows.insert(1, "\\midrule")
    hdr = "Variant & " + " & ".join(NAME[d] for d in DS) + " & $\\Delta$ (\\%)"
    note = ("Standardized test MSE, mean over seeds 42--44. Each row removes one component from the full model; "
            "$\\Delta$: mean relative change of the error over the 15 dataset--seed pairs.")
    CN_NOTE["ablation"] = "标准化测试 MSE，seeds 42--44 的均值。每行从完整模型中去掉一个组件；$\\Delta$：15 组数据集—种子组合上误差的平均相对变化。"
    ctable("ablation", "Component ablation.", "tab:ablation", hdr, rows, note, "@{}l" + "c" * len(DS) + "c@{}", size=10)


def _logged(name, key):
    f = FIN / name / "log"
    m = re.findall(key + r"=([0-9.]+)", f.read_text()) if f.exists() else []
    return float(m[-1]) if m else None


def cost_table():
    rows = []
    for lab, src in (("VA-SRI", "va"), ("SAITS-blocksup", "saits_plain_msemae_{d}_s42"),
                     ("TimesNet", "timesnet_{d}_s42_lr1e-3"), ("Crossformer", "crossformer_{d}_s42_lr1e-3")):
        cells = []
        for d in ("etth2", "weather"):
            if src == "va":
                r = json.loads((RUNS / FINAL.format(d=d, s=42) / "result.json").read_text())
                p, t = r["params"], r["seconds"]
            else:
                p, t = _logged(src.format(d=d), r"\[Params\] total"), _logged(src.format(d=d), "train_wall_seconds")
            cells += [f"{p / 1e6:.2f}M" if p >= 1e6 else f"{p / 1e3:.0f}k", f"{t / 60:.1f}"]
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    hdr = grouped_header("Method", ["ETTh2 (7 variables)", "Weather (21 variables)"], ["Params", "Train (min)"])
    note = ("Seed 42. Training time is the wall-clock time of the full fine-tuning run (ETTh2 200 epochs, Weather 100 "
            "epochs for VA-SRI and SAITS; TimesNet and Crossformer use the same protocol) on an RTX 5090 shared with other "
            "jobs, so the times are indicative only. The pretraining of VA-SRI is run once per seed for all datasets "
            "(15.4 min). Every model imputes with one forward pass.")
    CN_NOTE["cost"] = ("seed 42。训练时间为完整微调过程的墙钟时间（VA-SRI 与 SAITS 在 ETTh2 上 200 轮、Weather 上 100 轮；TimesNet 与 Crossformer "
                       "采用相同协议），所用 RTX 5090 与其他任务共享，时间仅供参考。VA-SRI 的预训练每个种子只执行一次，供全部数据集共用（15.4 分钟）。"
                       "所有模型推理均只需一次前向计算。")
    ctable("cost", "Model size and training time.", "tab:cost", hdr, rows, note, "@{}lcccc@{}", size=10)


# ------------------------------------------------------------------- tuning (validation only)
def _bestval(path):
    f = Path(path) / "log"
    m = re.findall(r"\[Best\] epoch=\d+ \| val_mse=([0-9.]+)", f.read_text()) if f.exists() else []   # std MSE (--select_std)
    return float(m[-1]) if m else None


def hyper_table():
    """Validation standardized MSE of every tuning candidate (ETTh2/ETTm1, seeds 42/43); no test data."""
    TS = (42, 43)
    def va_val(tag):
        return [st.mean(json.loads((RUNS / f"ft_{d}_s{s}_pre_p2ett_s{s}_beta_g_ema_uw{tag}" / "result.json")
                                   .read_text())["val"]["std_mse"] for s in TS) for d in ("etth2", "ettm1")]
    rows_va = [("$3\\times10^{-4}$", "0", ""), ("$1\\times10^{-4}$", "0", "_lr1e-4"), ("$1\\times10^{-3}$", "0", "_lr1e-3"),
               ("$2\\times10^{-3}$", "0", "_lr2e-3"), ("$1\\times10^{-3}$", "0.5", "_lr1e-3_mae0.5"),
               ("$1\\times10^{-3}$", "1", "_lr1e-3_mae1")]
    vals = [(lr, w, va_val(tag)) for lr, w, tag in rows_va]
    out = []
    best = min(range(len(vals)), key=lambda i: sum(vals[i][2]))
    for i, (lr, w, v) in enumerate(vals):
        mark = "\\checkmark" if i == best else ""
        out.append(f"VA-SRI & {lr} & ours & MSE + {w}$\\times$MAE & 42--43 & " + " & ".join(f"{x:.4f}" for x in v) + f" & {mark} \\\\")
    out.append("\\midrule")
    sa = []
    for lr in ("3e-4", "1e-3", "2e-3"):
        for recipe, rname in (("orig", "original"), ("ema", "ours")):
            v = [st.mean(_bestval(FIN.parent / "tune" / f"saits_plain_{d}_s{s}_lr{lr}_{recipe}") for s in TS)
                 for d in ("etth2", "ettm1")]
            sa.append((lr, rname, "MAE", v))
    bs = min(range(len(sa)), key=lambda i: sum(sa[i][3]))
    texlr = {"3e-4": "$3\\times10^{-4}$", "1e-3": "$1\\times10^{-3}$", "2e-3": "$2\\times10^{-3}$"}
    for i, (lr, rec, loss, v) in enumerate(sa):
        out.append(f"SAITS-blocksup & {texlr[lr]} & {rec} & {loss} & 42--43 & " + " & ".join(f"{x:.4f}" for x in v)
                   + (" & ($\\checkmark$)" if i == bs else " & ") + " \\\\")
    # loss choice for the selected setting: all five seeds
    out.append("\\cmidrule(l){2-8}")
    S5v = {}
    for loss, name in (("MAE", "saits_plain_final"), ("MSE + MAE", "saits_plain_msemae")):
        S5v[loss] = [st.mean(_bestval(FIN / f"{name}_{d}_s{s}") for s in S5) for d in ("etth2", "ettm1")]
    win = min(S5v, key=lambda k: sum(S5v[k]))
    for loss, v in S5v.items():
        out.append(f"SAITS-blocksup & $1\\times10^{{-3}}$ & ours & {loss} & 42--46 & " + " & ".join(f"{x:.4f}" for x in v)
                   + (" & \\checkmark" if loss == win else " & ") + " \\\\")
    hdr = "Model & Learning rate & Recipe & Loss & Seeds & ETTh2 & ETTm1 & Selected"
    note = ("Standardized validation MSE, mean over the stated seeds; the test splits were not used. The selected "
            "setting has the lowest sum over the two datasets. For SAITS, the learning rate and recipe were chosen "
            "with seeds 42--43 and its own loss (parenthesised tick), and the loss was then chosen for that setting "
            "with all five seeds. \\emph{Original} recipe: 100 epochs with early stopping "
            "(patience 20); \\emph{ours}: fixed budget with weight averaging. The SAITS loss is its two-term objective "
            "with the stated error. VA-SRI was tuned one factor at a time (learning rate, then the weight of the "
            "absolute term).")
    CN_NOTE["hyper"] = ("标准化验证 MSE，所列种子上的均值；未使用测试集。选定设置为两个数据集之和最低者。SAITS 先以 seeds 42--43 和其自身损失选定学习率与配方（括号中的勾），再在该设置下以全部五个种子选择损失。\\emph{原配方}：100 轮并早停"
                        "（patience 20）；\\emph{本文配方}：固定轮数加权重平均。SAITS 的损失为其两项目标，所用误差如表所示。"
                        "VA-SRI 采用逐因素调参（先学习率，再绝对值项权重）。")
    table("hyper", "Tuning candidates and their validation error.", "tab:hyper", hdr, out, note,
          "@{}lllllccc@{}", size=10)


ETT_VARS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")


def pervar_table():
    """Per-variable original-scale MAE, VA-SRI vs SAITS-blocksup with its own loss (scripts/va_sri/per_variable.py)."""
    pv = json.loads((FIN / "per_variable_mae.json").read_text())
    by = {d: {r["var"]: r for r in pv[d]} for d in ("etth1", "ettm1")}
    order = sorted(range(7), key=lambda v: -by["etth1"][v]["variance_share"])
    rows = []
    for v in order:
        cells = []
        for d in ("etth1", "ettm1"):
            r = by[d][v]
            a, b = r["va_mae"], r["saits_mae"]
            cells += [f"{100 * r['variance_share']:.1f}", f"\\textbf{{{a:.3f}}}" if a < b else f"{a:.3f}",
                      f"\\textbf{{{b:.3f}}}" if b < a else f"{b:.3f}", f"${'+' if r['rel'] >= 0 else '-'}${abs(r['rel']):.1f}",
                      f"{r['wins']}/5"]
        rows.append(f"{ETT_VARS[v]} & " + " & ".join(cells) + " \\\\")
    hdr = grouped_header("Variable", ["ETTh1", "ETTm1"], ["Var. (\\%)", "VA-SRI", "SAITS", "$\\Delta$ (\\%)", "Wins"])
    note = ("Original-scale test MAE per variable, 24-step gaps, mean over seeds 42--46, against SAITS-blocksup with "
            "its own MAE loss (the only per-variable error recorded for SAITS). \\emph{Var.}: share of the total "
            "training variance; $\\Delta$: relative difference of VA-SRI; \\emph{Wins}: seeds in which VA-SRI is lower. "
            "Variables are sorted by their variance share on ETTh1; the lower error is in bold.")
    CN_NOTE["pervar"] = ("逐变量原尺度测试 MAE，24 步缺口，seeds 42--46 的均值，对比使用其自身 MAE 损失的 SAITS-blocksup（SAITS 只记录了逐变量的 MAE）。"
                         "\\emph{方差}：该变量在训练段总方差中的占比；$\\Delta$：VA-SRI 的相对差异；\\emph{胜出}：VA-SRI 误差更低的种子数。"
                         "变量按其在 ETTh1 上的方差占比排序，较低误差加粗。")
    table("pervar", "Per-variable error on ETTh1 and ETTm1.", "tab:pervar", hdr, rows, note, "@{}l" + "ccccc" * 2 + "@{}", size=9.5, colsep=3.5)


COMPACT = (hyper_table, pervar_table, main_compact, length_compact, pattern_compact, factorial_compact, ablation_compact, cost_table)


if __name__ == "__main__":
    main()
