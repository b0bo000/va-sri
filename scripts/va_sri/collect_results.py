"""Render RESULTS.md from all VA-SRI runs (final config = no pretraining + VGA).

python -m scripts.va_sri.collect_results
"""
from __future__ import annotations

import glob
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = ["etth1", "etth2", "ettm1", "ettm2", "weather"]
LBL = {"etth1": "ETTh1", "etth2": "ETTh2", "ettm1": "ETTm1", "ettm2": "ETTm2", "weather": "Weather"}
V10 = {"etth1": 1.695, "etth2": 2.568, "ettm1": 0.693, "ettm2": 0.848, "weather": 460.451}
SAITS = {"etth1": 1.832, "etth2": 2.923, "ettm1": 0.649, "ettm2": 0.865, "weather": 495.630}
SAITS_FILL = {"etth1": 1.666, "etth2": 2.684, "ettm1": 0.640, "ettm2": 0.761, "weather": 458.853}
IMPF = {"etth1": 2.632, "etth2": 3.785, "ettm1": 0.925, "ettm2": 1.099, "weather": 501.049}


def runs(pattern, metric="mse"):
    out = {}
    for f in glob.glob(str(ROOT / "experiments/va_sri/runs") + "/" + pattern + "/result.json"):
        name = Path(f).parent.name
        ds = name.split("_")[1]
        seed = int(re.search(r"_s(\d+)_", name).group(1))
        d = json.loads(Path(f).read_text())
        if "test" in d:
            out.setdefault(ds, {})[seed] = d["test"][metric]
    return out


def main():
    final = runs("ft_*_none_beta_g_ema*")
    lines = ["# VA-SRI results", ""]
    for d in DS:
        v = final.get(d, {})
        if not v:
            lines.append(f"- {LBL[d]}: {len(v)} seeds (incomplete)")
            continue
        m, s, n = st.mean(v.values()), (st.stdev(v.values()) if len(v) > 1 else 0), len(v)
        lines.append(f"- {LBL[d]}: n={n} test MSE {m:.4f} +- {s:.4f} | "
                     f"vs V10 {100*(m/V10[d]-1):+.1f}% | vs SAITS {100*(m/SAITS[d]-1):+.1f}% | "
                     f"vs SAITS+fill {100*(m/SAITS_FILL[d]-1):+.1f}% | vs ImputeFormer {100*(m/IMPF[d]-1):+.1f}%")
    text = "\n".join(lines) + "\n"
    print(text)
    (ROOT / "experiments/va_sri/summary_final.md").write_text(text)

if __name__ == "__main__":
    main()
