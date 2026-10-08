"""Summarise VA-SRI validation results against reference arms.

References (same protocol, from earlier experiments):
  v10      : V10 joint-channel SRI with legacy SSL (ccfa_v2 fresh_full logs)
  none     : variable-axis + identity, no pretraining (09-27/09-28 runs)
  legacy   : variable-axis + identity + legacy contrastive SSL
  python -m scripts.va_sri.collect
"""
from __future__ import annotations

import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 43, 44)


def reference():
    ref = defaultdict(dict)
    for ds in ("ETTh2", "ETTm1"):
        for s in SEEDS:
            log = ROOT / f"experiments/ccfa_v2_submission_2026-09-22/training/logs/fresh_full_{ds.lower()}_b24_seed{s}.log"
            m = re.findall(r"best_val_mse=([0-9.]+)", log.read_text())
            ref[("v10", ds)][s] = {"mse": float(m[-1])}
    for exp in ("sri_variable_axis_study_2026-09-27", "sri_variable_axis_multiseed_2026-09-28"):
        for f in (ROOT / "experiments" / exp).glob("**/result.json"):
            d = json.loads(f.read_text())
            if d.get("identity") is not True or d.get("purpose") != "validation_development":
                continue
            arm = "legacy" if d["ssl"] else "none"
            ref[(arm, d["dataset"])][d["seed"]] = d["validation"]
    return ref


def ours():
    res = defaultdict(dict)
    for f in (ROOT / "experiments/va_sri/runs").glob("ft_*/result.json"):
        d = json.loads(f.read_text())
        init = d["init"]
        pre = "none" if not init else re.sub(r"_s\d+$", "", Path(init).parent.name).replace("pre_", "")
        pre = re.sub(r"_(etth2|ettm1)$", "", pre)
        recipe = ("+cos" if d.get("schedule") == "cosine" else "") + ("+ema" if d.get("ema") else "")
        res[(f"{pre}/{d['vga']}{recipe}", d["dataset"])][d["seed"]] = d["val"]
    return res


def fmt(rows, base="none"):
    lines = []
    for ds in ("ETTh2", "ETTm1"):
        lines.append(f"\n### {ds} (validation, seeds {SEEDS})\n")
        lines.append("| arm | n | MSE mean ± sd | MAE mean | STD-MSE mean | ΔMSE vs none | wins vs none | ΔMSE vs V10 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        b = rows.get((base, ds), {})
        v10 = rows.get(("v10", ds), {})
        for (arm, d), per in sorted(rows.items()):
            if d != ds:
                continue
            ms = [per[s]["mse"] for s in SEEDS if s in per]
            if not ms:
                continue
            mean = st.mean(ms)
            sd = st.stdev(ms) if len(ms) > 1 else float("nan")
            mae = st.mean(per[s]["mae"] for s in SEEDS if s in per) if "mae" in per[next(iter(per))] else float("nan")
            sm = st.mean(per[s]["std_mse"] for s in SEEDS if s in per) if "std_mse" in per[next(iter(per))] else float("nan")
            common = [s for s in SEEDS if s in per and s in b]
            dpct = 100 * (mean / st.mean(b[s]["mse"] for s in b) - 1) if b else float("nan")
            wins = f"{sum(per[s]['mse'] < b[s]['mse'] for s in common)}/{len(common)}" if common else "-"
            dv = 100 * (mean / st.mean(v10[s]["mse"] for s in v10) - 1) if v10 else float("nan")
            lines.append(f"| {arm} | {len(ms)} | {mean:.4f} ± {sd:.4f} | {mae:.4f} | {sm:.5f} | {dpct:+.2f}% | {wins} | {dv:+.2f}% |")
    return "\n".join(lines)


if __name__ == "__main__":
    rows = {**reference(), **ours()}
    text = "# VA-SRI phase summary\n" + fmt(rows)
    out = ROOT / "experiments/va_sri/summary.md"
    out.write_text(text + "\n")
    print(text)
