"""Write job lists for the VA-SRI phases.  python -m scripts.va_sri.make_jobs phase1"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PY = "python"   # interpreter used for every job
EXP = "experiments/va_sri"
SEEDS = (42, 43, 44)
DEV = ("ETTh2", "ETTm1")
P2_UPDATES = 40000
PRETRAIN = {
    "p1": None,                                              # target dataset only, 60 epochs
    "p2ett": ["ETTh1", "ETTh2", "ETTm1", "ETTm2"],
    "p2all": ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"],
}


def pretrain_job(kind, seed, dataset=None):
    name = f"pre_{kind}_{dataset.lower()}_s{seed}" if kind == "p1" else f"pre_{kind}_s{seed}"
    out = f"{EXP}/pretrain/{name}"
    if kind == "p1":
        args = ["--datasets", dataset, "--epochs-of", dataset]
    elif kind == "p2ettctr":
        args = ["--datasets", *PRETRAIN["p2ett"], "--updates", str(P2_UPDATES), "--contrastive", "0.2"]
    else:
        args = ["--datasets", *PRETRAIN[kind], "--updates", str(P2_UPDATES)]
    return {"name": name, "done": f"{out}/shared.pt",
            "cmd": [PY, "-m", "scripts.va_sri.pretrain", *args, "--seed", str(seed), "--out", out]}


def ft_job(dataset, seed, pre, vga="off", test=False, extra=(), tag=""):
    name = f"ft_{dataset.lower()}_s{seed}_{pre['name'] if pre else 'none'}_{vga}{tag}"
    out = f"{EXP}/runs/{name}"
    cmd = [PY, "-m", "scripts.va_sri.train", "--dataset", dataset, "--seed", str(seed),
           "--vga", vga, "--out", out, *extra]
    if pre:
        cmd += ["--init", pre["done"]]
    if test:
        cmd.append("--eval-test")
    return {"name": name, "done": f"{out}/result.json", "needs": [pre["done"]] if pre else [], "cmd": cmd}


def phase1():
    pre_jobs, ft_jobs = [], []
    for seed in SEEDS:
        shared = {k: pretrain_job(k, seed) for k in ("p2ett", "p2all")}
        pre_jobs += shared.values()
        for ds in DEV:
            p1 = pretrain_job("p1", seed, ds)
            pre_jobs.append(p1)
            for pre in (p1, shared["p2ett"], shared["p2all"]):
                ft_jobs.append(ft_job(ds, seed, pre))
    return pre_jobs + ft_jobs


# Low-noise recipe: constant LR, EMA-evaluated weights, no early stopping.
# ETTh2 (~190 updates/epoch) was still improving at epoch 100, so it gets 200.
# (A cosine-LR variant undertrained ETTh2: 2.83 vs 2.68 at epoch 100, seed42.)
RECIPE = ("--ema", "0.999", "--patience", "0")
EPOCHS = {"ETTh2": 200, "ETTm1": 100, "ETTh1": 200, "ETTm2": 100, "Weather": 100}
MAIN_DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
MAIN_SEEDS = (42, 43, 44, 45, 46)


def phase1b():
    jobs = []
    for seed in SEEDS:
        for ds in DEV:
            for pre in (None, pretrain_job("p1", seed, ds), pretrain_job("p2ett", seed)):
                jobs.append(ft_job(ds, seed, pre, extra=(*RECIPE, "--epochs", str(EPOCHS[ds])), tag="_ema"))
    return jobs


def phase2():
    """VGA on the P2-ETT trunk, plus P2-ETT with a light contrastive term."""
    pre_jobs, ft_jobs = [], []
    for seed in SEEDS:
        p2 = pretrain_job("p2ett", seed)
        ctr = pretrain_job("p2ettctr", seed)
        pre_jobs.append(ctr)
        for ds in DEV:
            extra = (*RECIPE, "--epochs", str(EPOCHS[ds]))
            ft_jobs.append(ft_job(ds, seed, ctr, extra=extra, tag="_ema"))
            for vga in ("beta", "beta_g"):
                ft_jobs.append(ft_job(ds, seed, p2, vga=vga, extra=extra, tag="_ema"))
    # Interleave so a finetune can run while pretraining occupies the other GPU.
    return pre_jobs[:1] + ft_jobs + pre_jobs[1:]


def phase3():
    """Frozen config (P2-ETT trunk + VGA beta+G + low-noise recipe) on the paper's
    five datasets and five seeds, with test evaluation enabled."""
    jobs = []
    for seed in MAIN_SEEDS:
        pre = pretrain_job("p2ett", seed)
        if seed not in SEEDS:
            jobs.append(pre)
    for seed in MAIN_SEEDS:
        pre = pretrain_job("p2ett", seed)
        for ds in MAIN_DATASETS:
            jobs.append(ft_job(ds, seed, pre, vga="beta_g", test=True,
                               extra=("--ema", "0.999", "--patience", "0", "--epochs", str(EPOCHS[ds])),
                               tag="_ema"))
    return jobs


def phase4():
    """Ablations on the frozen config: no VGA, no variable identity, no pretraining."""
    jobs = []
    for seed in SEEDS:
        pre = pretrain_job("p2ett", seed)
        for ds in MAIN_DATASETS:
            extra = ("--ema", "0.999", "--patience", "0", "--epochs", str(EPOCHS[ds]), "--eval-test")
            # Distinct "_abl" names so earlier validation-only runs are never mistaken for these.
            jobs.append(ft_job(ds, seed, pre, vga="off", extra=extra, tag="_ema_abl"))          # no VGA
            jobs.append(ft_job(ds, seed, pre, vga="beta_g", extra=(*extra, "--no-identity"),
                               tag="_ema_noid_abl"))                                            # no identity
            jobs.append(ft_job(ds, seed, None, vga="beta_g", extra=extra, tag="_ema_abl"))       # no pretraining
    return jobs


def phase6():
    """Final configuration: no pretraining, VGA on, same recipe.
    Seeds 42-44 already exist as the phase-4 "noPre" ablation runs."""
    jobs = []
    for seed in MAIN_SEEDS:
        if seed in SEEDS:
            continue
        for ds in MAIN_DATASETS:
            jobs.append(ft_job(ds, seed, None, vga="beta_g", test=True,
                               extra=("--ema", "0.999", "--patience", "0", "--epochs", str(EPOCHS[ds])),
                               tag="_ema"))
    return jobs


RECIPE_FT = ("--ema", "0.999", "--patience", "0")


def phase7():
    """Robustness for the main config (P2-ETT trunk + VGA): block length 12/48 on all
    datasets x 5 seeds, and missing patterns on ETTh2 (42-44) / Weather (42-46), as in V10."""
    jobs = []
    for seed in MAIN_SEEDS:
        pre = pretrain_job("p2ett", seed)
        for bl in (12, 48):
            for ds in MAIN_DATASETS:
                jobs.append(ft_job(ds, seed, pre, vga="beta_g", test=True, tag=f"_ema_b{bl}",
                                   extra=(*RECIPE_FT, "--epochs", str(EPOCHS[ds]), "--block-len", str(bl))))
    for mode in ("block_shared", "partial", "mixed"):
        for ds, seeds in (("ETTh2", SEEDS), ("Weather", MAIN_SEEDS)):
            for seed in seeds:
                pre = pretrain_job("p2ett", seed)
                jobs.append(ft_job(ds, seed, pre, vga="beta_g", test=True, tag=f"_ema_{mode}",
                                   extra=(*RECIPE_FT, "--epochs", str(EPOCHS[ds]), "--mask-mode", mode)))
    return jobs


def phase8():
    """2x2 factorial of skeleton input x residual output, no pretraining, seeds 42-44.
    E11 (skeleton/residual) already exists as the phase-4 no-pretraining runs."""
    jobs = []
    arms = {"e00": ("zero", "absolute"), "e10": ("skeleton", "absolute"), "e01": ("zero", "residual")}
    for seed in SEEDS:
        for ds in MAIN_DATASETS:
            for tag, (fill, out) in arms.items():
                jobs.append(ft_job(ds, seed, None, vga="beta_g", test=True, tag=f"_ema_{tag}",
                                   extra=(*RECIPE_FT, "--epochs", str(EPOCHS[ds]),
                                          "--input-fill", fill, "--output", out)))
    return jobs


def phase78():
    """Block length first, then factorial, then patterns; one queue."""
    p7, p8 = phase7(), phase8()
    blocks = [j for j in p7 if "_b12" in j["name"] or "_b48" in j["name"]]
    patterns = [j for j in p7 if j not in blocks]
    return blocks + p8 + patterns


def phase9():
    """Loss check for standardized-metric evaluation: unweighted vs variance-weighted MSE.
    Validation only (no --eval-test); weighted arm = existing phase-2 beta_g runs."""
    jobs = []
    for seed in SEEDS:
        pre = pretrain_job("p2ett", seed)
        for ds in DEV:
            jobs.append(ft_job(ds, seed, pre, vga="beta_g", tag="_ema_uw",
                               extra=(*RECIPE_FT, "--epochs", str(EPOCHS[ds]), "--unweighted")))
    return jobs


SAITS_BASE = ["--root_path", "./data", "--seq_len", "96", "--block_len", "24", "--batch_size", "64",
              "--device", "cuda", "--reproducible_epoch_mask", "--mask_mode", "block", "--mask_ratio", "0.4",
              "--p_point", "0.3", "--p_block", "0.3", "--p_partial", "0.4", "--point_ratio", "0.03",
              "--partial_min_vars", "2", "--partial_max_vars", "4", "--skeleton_fill_input"]
TUNE_SEEDS = (42, 43)


def phase10():
    """Equal-budget tuning on ETTh2/ETTm1 validation, selected by standardized MSE; no test.
    VA-SRI (unweighted loss, P2-ETT trunk): lr in {1e-4, 1e-3}; lr 3e-4 = existing phase-9 runs.
    SAITS + fill: lr in {3e-4, 1e-3} x {original recipe: 100 ep / patience 20, EMA + fixed budget}."""
    jobs = []
    for seed in TUNE_SEEDS:
        pre = pretrain_job("p2ett", seed)
        for ds in DEV:
            for lr in ("1e-4", "1e-3"):
                jobs.append(ft_job(ds, seed, pre, vga="beta_g", tag=f"_ema_uw_lr{lr}",
                                   extra=(*RECIPE_FT, "--epochs", str(EPOCHS[ds]), "--unweighted",
                                          "--select", "std_mse", "--lr", lr)))
    for seed in TUNE_SEEDS:
        for ds in DEV:
            for lr in ("3e-4", "1e-3"):
                for recipe in ("orig", "ema"):
                    name = f"saits_fill_{ds.lower()}_s{seed}_lr{lr}_{recipe}"
                    out = f"{EXP}/tune/{name}"
                    if recipe == "orig":
                        train = ["--epochs", "100", "--patience", "20"]
                    else:
                        train = ["--epochs", str(EPOCHS[ds]), "--patience", str(EPOCHS[ds]), "--ema_decay", "0.999"]
                    args = [PY, "-u", "run_saits_block_supervised.py", *SAITS_BASE, "--data_path", f"{ds}.csv",
                            "--seed", str(seed), "--lr", lr, "--select_std", "--skip_test", *train]
                    jobs.append({"name": name, "done": f"{out}/DONE",
                                 "cmd": ["bash", "-c", f"mkdir -p {out} && " + " ".join(args) + f" > {out}/log 2>&1 && touch {out}/DONE"]})
    return jobs


FINAL_VA = ("--ema", "0.999", "--patience", "0", "--unweighted", "--select", "std_mse", "--lr", "1e-3")


def phase11():
    """Final test runs with the frozen, equally tuned configurations (5 datasets x seeds 42-46).
    VA-SRI: unweighted loss, lr 1e-3, EMA, select by std-MSE, P2-ETT trunk, VGA.
    SAITS + fill: lr 1e-3, EMA 0.999, fixed budget, select by std-MSE (best of 6 tuned settings)."""
    jobs = []
    for seed in MAIN_SEEDS:
        pre = pretrain_job("p2ett", seed)
        for ds in MAIN_DATASETS:
            jobs.append(ft_job(ds, seed, pre, vga="beta_g", test=True, tag="_final",
                               extra=(*FINAL_VA, "--epochs", str(EPOCHS[ds]))))
    for seed in MAIN_SEEDS:
        for ds in MAIN_DATASETS:
            name = f"saits_fill_final_{ds.lower()}_s{seed}"
            out = f"{EXP}/final/{name}"
            fname = "weather.csv" if ds == "Weather" else f"{ds}.csv"
            args = [PY, "-u", "run_saits_block_supervised.py", *SAITS_BASE, "--data_path", fname,
                    "--seed", str(seed), "--lr", "1e-3", "--select_std",
                    "--epochs", str(EPOCHS[ds]), "--patience", str(EPOCHS[ds]), "--ema_decay", "0.999"]
            jobs.append({"name": name, "done": f"{out}/DONE",
                         "cmd": ["bash", "-c", f"mkdir -p {out} && " + " ".join(args) + f" > {out}/log 2>&1 && touch {out}/DONE"]})
    return jobs


def phase12():
    """Equal tuning for plain SAITS-blocksup (zero-filled input), same grid as SAITS + fill.
    Validation only, selected by standardized MSE."""
    base = [a for a in SAITS_BASE if a != "--skeleton_fill_input"]
    jobs = []
    for seed in TUNE_SEEDS:
        for ds in DEV:
            for lr in ("3e-4", "1e-3", "2e-3"):
                for recipe in ("orig", "ema"):
                    name = f"saits_plain_{ds.lower()}_s{seed}_lr{lr}_{recipe}"
                    out = f"{EXP}/tune/{name}"
                    train = (["--epochs", "100", "--patience", "20"] if recipe == "orig" else
                             ["--epochs", str(EPOCHS[ds]), "--patience", str(EPOCHS[ds]), "--ema_decay", "0.999"])
                    args = [PY, "-u", "run_saits_block_supervised.py", *base, "--data_path", f"{ds}.csv",
                            "--seed", str(seed), "--lr", lr, "--select_std", "--skip_test", *train]
                    jobs.append({"name": name, "done": f"{out}/DONE",
                                 "cmd": ["bash", "-c", f"mkdir -p {out} && " + " ".join(args) + f" > {out}/log 2>&1 && touch {out}/DONE"]})
    return jobs


def phase13():
    """Combined MSE + MAE loss for VA-SRI, validation only (ETTh2/ETTm1, seeds 42/43).
    Reference arm = phase-10 runs with lr 1e-3 (same recipe, mae weight 0)."""
    jobs = []
    for seed in TUNE_SEEDS:
        pre = pretrain_job("p2ett", seed)
        for ds in DEV:
            for w in ("0.5", "1"):
                jobs.append(ft_job(ds, seed, pre, vga="beta_g", tag=f"_ema_uw_lr1e-3_mae{w}",
                                   extra=(*RECIPE_FT, "--epochs", str(EPOCHS[ds]), "--unweighted",
                                          "--select", "std_mse", "--lr", "1e-3", "--mae-weight", w)))
    return jobs


def phase14():
    """Final test runs for plain SAITS-blocksup (zero-filled input) with its tuned setting:
    lr 1e-3, EMA 0.999, fixed budget, select by std-MSE (best of 6 on ETTh2/ETTm1 validation)."""
    base = [a for a in SAITS_BASE if a != "--skeleton_fill_input"]
    jobs = []
    for seed in MAIN_SEEDS:
        for ds in MAIN_DATASETS:
            name = f"saits_plain_final_{ds.lower()}_s{seed}"
            out = f"{EXP}/final/{name}"
            fname = "weather.csv" if ds == "Weather" else f"{ds}.csv"
            args = [PY, "-u", "run_saits_block_supervised.py", *base, "--data_path", fname,
                    "--seed", str(seed), "--lr", "1e-3", "--select_std",
                    "--epochs", str(EPOCHS[ds]), "--patience", str(EPOCHS[ds]), "--ema_decay", "0.999"]
            jobs.append({"name": name, "done": f"{out}/DONE",
                         "cmd": ["bash", "-c", f"mkdir -p {out} && " + " ".join(args) + f" > {out}/log 2>&1 && touch {out}/DONE"]})
    return jobs


FINAL_VA2 = (*FINAL_VA, "--mae-weight", "1")


def phase15():
    """Final VA-SRI test runs with the combined loss (MSE + 1.0 x MAE), chosen on
    ETTh2/ETTm1 validation (4/4 pairs better on all four metrics); otherwise as phase 11."""
    jobs = []
    for seed in MAIN_SEEDS:
        pre = pretrain_job("p2ett", seed)
        for ds in MAIN_DATASETS:
            jobs.append(ft_job(ds, seed, pre, vga="beta_g", test=True, tag="_final2",
                               extra=(*FINAL_VA2, "--epochs", str(EPOCHS[ds]))))
    return jobs


def phase16():
    """Re-run the supporting experiments with the final configuration (FINAL_VA2), seeds 42-44.
    Order: ablation, factorial, block length, missing pattern."""
    jobs = []
    fin = lambda ds: (*FINAL_VA2, "--epochs", str(EPOCHS[ds]))
    for seed in SEEDS:
        pre = pretrain_job("p2ett", seed)
        for ds in MAIN_DATASETS:
            jobs.append(ft_job(ds, seed, pre, vga="off", test=True, tag="_f2_novga", extra=fin(ds)))
            jobs.append(ft_job(ds, seed, pre, vga="beta_g", test=True, tag="_f2_noid", extra=(*fin(ds), "--no-identity")))
            jobs.append(ft_job(ds, seed, None, vga="beta_g", test=True, tag="_f2_nopre", extra=fin(ds)))
    arms = {"e00": ("zero", "absolute"), "e10": ("skeleton", "absolute"), "e01": ("zero", "residual")}
    for seed in SEEDS:
        for ds in MAIN_DATASETS:
            for tag, (fill, out) in arms.items():
                jobs.append(ft_job(ds, seed, None, vga="beta_g", test=True, tag=f"_f2_{tag}",
                                   extra=(*fin(ds), "--input-fill", fill, "--output", out)))
    for seed in SEEDS:
        pre = pretrain_job("p2ett", seed)
        for bl in (12, 48):
            for ds in MAIN_DATASETS:
                jobs.append(ft_job(ds, seed, pre, vga="beta_g", test=True, tag=f"_f2_b{bl}",
                                   extra=(*fin(ds), "--block-len", str(bl))))
        for mode in ("block_shared", "partial", "mixed"):
            for ds in ("ETTh2", "Weather"):
                jobs.append(ft_job(ds, seed, pre, vga="beta_g", test=True, tag=f"_f2_{mode}",
                                   extra=(*fin(ds), "--mask-mode", mode)))
    return jobs


def phase17():
    """Tuned plain SAITS (lr 1e-3, EMA, select std-MSE) for block length 12/48 and missing patterns,
    seeds 42-44, so the robustness tables use equally tuned baselines."""
    base = [a for a in SAITS_BASE if a != "--skeleton_fill_input"]
    def job(ds, seed, tag, extra):
        name = f"saits_plain_{tag}_{ds.lower()}_s{seed}"
        out = f"{EXP}/final/{name}"
        fname = "weather.csv" if ds == "Weather" else f"{ds}.csv"
        args = [PY, "-u", "run_saits_block_supervised.py", *base, "--data_path", fname, "--seed", str(seed),
                "--lr", "1e-3", "--select_std", "--epochs", str(EPOCHS[ds]), "--patience", str(EPOCHS[ds]),
                "--ema_decay", "0.999", *extra]
        if "--block_len_override" in extra:
            i = args.index("--block_len"); args[i + 1] = extra[extra.index("--block_len_override") + 1]
            args = [a for a in args if a not in ("--block_len_override", extra[extra.index("--block_len_override") + 1])] if False else args[:len(args) - len(extra)]
        elif "--mask_mode_override" in extra:
            i = args.index("--mask_mode"); args[i + 1] = extra[extra.index("--mask_mode_override") + 1]
            args = args[:len(args) - len(extra)]
        return {"name": name, "done": f"{out}/DONE",
                "cmd": ["bash", "-c", f"mkdir -p {out} && " + " ".join(args) + f" > {out}/log 2>&1 && touch {out}/DONE"]}
    jobs = []
    for seed in SEEDS:
        for bl in (12, 48):
            for ds in MAIN_DATASETS:
                jobs.append(job(ds, seed, f"b{bl}", ["--block_len_override", str(bl)]))
        for mode in ("block_shared", "partial", "mixed"):
            for ds in ("ETTh2", "Weather"):
                extra = ["--block_shared"] if mode == "block_shared" else ["--mask_mode_override", mode]
                jobs.append(job(ds, seed, mode, extra))
    return jobs


def phase18():
    """Electricity (C=321) breadth check with the frozen configurations, seeds 42-44, 100 epochs.
    No tuning on Electricity; VA-SRI uses the ETT-pretrained trunk as on every other dataset."""
    jobs = []
    for seed in SEEDS:
        pre = pretrain_job("p2ett", seed)
        jobs.append(ft_job("Electricity", seed, pre, vga="beta_g", test=True, tag="_final2",
                           extra=(*FINAL_VA2, "--epochs", "100", "--micro-batch", "16")))
    base = [a for a in SAITS_BASE if a != "--skeleton_fill_input"]
    for seed in SEEDS:
        name = f"saits_plain_final_electricity_s{seed}"
        out = f"{EXP}/final/{name}"
        args = [PY, "-u", "run_saits_block_supervised.py", *base, "--data_path", "electricity.csv",
                "--seed", str(seed), "--lr", "1e-3", "--select_std",
                "--epochs", "100", "--patience", "100", "--ema_decay", "0.999"]
        jobs.append({"name": name, "done": f"{out}/DONE",
                     "cmd": ["bash", "-c", f"mkdir -p {out} && " + " ".join(args) + f" > {out}/log 2>&1 && touch {out}/DONE"]})
    return jobs


def saits_job(name, ds, seed, fill, epochs, extra=()):
    base = SAITS_BASE if fill else [a for a in SAITS_BASE if a != "--skeleton_fill_input"]
    out = f"{EXP}/final/{name}"
    fname = {"Weather": "weather.csv", "Electricity": "electricity.csv"}.get(ds, f"{ds}.csv")
    args = [PY, "-u", "run_saits_block_supervised.py", *base, "--data_path", fname, "--seed", str(seed),
            "--lr", "1e-3", "--select_std", "--epochs", str(epochs), "--patience", str(epochs),
            "--ema_decay", "0.999", *extra]
    return {"name": name, "done": f"{out}/DONE",
            "cmd": ["bash", "-c", f"mkdir -p {out} && " + " ".join(args) + f" > {out}/log 2>&1 && touch {out}/DONE"]}


def phase19():
    """Equal-loss control: SAITS and SAITS + fill trained with VA-SRI's loss (masked MSE + MAE),
    otherwise their tuned setting (lr 1e-3, EMA, fixed budget, std-MSE selection). Test, seeds 42-46."""
    return [saits_job(f"saits_{arm}_msemae_{ds.lower()}_s{seed}", ds, seed, arm == "fill", EPOCHS[ds],
                      ("--mse_mae_loss",))
            for seed in MAIN_SEEDS for ds in MAIN_DATASETS for arm in ("plain", "fill")]


BACKBONES = ("timesnet", "crossformer")


def backbone_job(model, ds, seed, lr, test):
    name = f"{model}_{ds.lower()}_s{seed}_lr{lr}" + ("" if test else "_tune")
    out = f"{EXP}/final/{name}"
    fname = {"Weather": "weather.csv", "Electricity": "electricity.csv"}.get(ds, f"{ds}.csv")
    args = [PY, "-u", "-m", "scripts.va_sri.run_backbone", "--model", model, "--data_path", fname,
            "--seed", str(seed), "--lr", lr, "--epochs", str(EPOCHS[ds]), *(() if test else ("--skip_test",))]
    return {"name": name, "done": f"{out}/DONE",
            "cmd": ["bash", "-c", f"mkdir -p {out} && " + " ".join(args) + f" > {out}/log 2>&1 && touch {out}/DONE"]}


def phase20():
    """Tuning of the added baselines: lr in {3e-4, 1e-3, 2e-3}, ETTh2/ETTm1 validation, seed 42
    only (GPU time on the shared server), no test."""
    return [backbone_job(m, ds, 42, lr, False) for m in BACKBONES for ds in ("ETTh2", "ETTm1")
            for lr in ("3e-4", "1e-3", "2e-3")]


def pick_backbone_lr(model):
    """Validation-only choice: lr whose std-MSE, relative to the best lr of each tuning dataset,
    is lowest on average over ETTh2/ETTm1 (seed 42). Returns None until all tuning runs are done."""
    import re
    lrs, score = ("3e-4", "1e-3", "2e-3"), {}
    for ds in ("ETTh2", "ETTm1"):
        vals = {}
        for lr in lrs:
            d = Path(f"{EXP}/final/{model}_{ds.lower()}_s42_lr{lr}_tune")
            if not (d / "DONE").exists():
                return None
            vals[lr] = float(re.search(r"\[Best\] epoch=\d+ \| val_std_mse=([0-9.]+)", (d / "log").read_text()).group(1))
        for lr in lrs:
            score[lr] = score.get(lr, 0) + vals[lr] / min(vals.values()) / 2
    return min(score, key=score.get), score


def phase21():
    """Test runs of the added baselines with their validation-chosen lr, seeds 42-46."""
    jobs = []
    for m in BACKBONES:
        pick = pick_backbone_lr(m)
        if pick is None:
            print(f"{m}: tuning not finished, skipped")
            continue
        print(m, pick)
        jobs += [backbone_job(m, ds, s, pick[0], True) for s in MAIN_SEEDS for ds in MAIN_DATASETS]
    return jobs


def phase22():
    """Electricity breadth runs with the final recipe: VA-SRI is already in phase18; this adds
    SAITS with the equal (MSE + MAE) loss so the breadth table uses the same comparator."""
    return [saits_job(f"saits_plain_msemae_electricity_s{seed}", "Electricity", seed, False, 100, ("--mse_mae_loss",))
            for seed in SEEDS]


def phase23():
    """Validation-only: is VA-SRI under-trained? Best epochs sit at the end of the budget on ETTm1.
    Double the budget (ETTm1 200, ETTh2 400), constant vs cosine lr, seeds 42/43, no test."""
    jobs = []
    for seed in TUNE_SEEDS:
        pre = pretrain_job("p2ett", seed)
        for ds, ep in (("ETTm1", 200), ("ETTh2", 400)):
            for sched in ("const", "cosine"):
                extra = [a for a in FINAL_VA2]
                jobs.append(ft_job(ds, seed, pre, vga="beta_g", test=False, tag=f"_ep{ep}{sched}",
                                   extra=(*extra, "--epochs", str(ep), "--schedule", sched)))
    return jobs


POINT_RATIOS = (0.125, 0.25, 0.375, 0.5)


def phase24():
    """Random point missingness (the Time-Series-Library imputation protocol): each variable has
    r*96 randomly chosen hidden steps, r in {12.5, 25, 37.5, 50}%. Frozen settings, no tuning:
    VA-SRI final config with the ETT-pretrained trunk, tuned SAITS (MSE + MAE loss). Test, seeds 42-44."""
    jobs = []
    for seed in SEEDS:
        pre = pretrain_job("p2ett", seed)
        for r in POINT_RATIOS:
            tag = f"pt{int(r * 1000)}"
            for ds in MAIN_DATASETS:
                jobs.append(ft_job(ds, seed, pre, vga="beta_g", test=True, tag=f"_f2_{tag}",
                                   extra=(*FINAL_VA2, "--epochs", str(EPOCHS[ds]), "--mask-mode", "point",
                                          "--point-ratio", str(r))))
                jobs.append(saits_job(f"saits_plain_msemae_{tag}_{ds.lower()}_s{seed}", ds, seed, False, EPOCHS[ds],
                                      ("--mse_mae_loss", "--mask_mode", "point", "--point_ratio", str(r))))
    return jobs


def phase25():
    """Factorial (input zero/skeleton x output absolute/residual) at gap lengths 12 and 48, so the
    claim that the residual formulation matters more for longer gaps is tested directly. Same arms
    and settings as the 24-step factorial (no pretraining), test, seeds 42-44."""
    arms = {"e00": ("zero", "absolute"), "e10": ("skeleton", "absolute"), "e01": ("zero", "residual"),
            "e11": ("skeleton", "residual")}
    jobs = []
    for seed in SEEDS:
        for bl in (12, 48):
            for ds in MAIN_DATASETS:
                for tag, (fill, out) in arms.items():
                    jobs.append(ft_job(ds, seed, None, vga="beta_g", test=True, tag=f"_f2_{tag}_b{bl}",
                                       extra=(*FINAL_VA2, "--epochs", str(EPOCHS[ds]), "--block-len", str(bl),
                                              "--input-fill", fill, "--output", out)))
    return jobs


def phase2425():
    return phase24() + phase25()


if __name__ == "__main__":
    phase = sys.argv[1]
    jobs = {"phase1": phase1, "phase1b": phase1b, "phase2": phase2, "phase3": phase3, "phase4": phase4, "phase6": phase6,
            "phase7": phase7, "phase8": phase8, "phase78": phase78, "phase9": phase9, "phase10": phase10, "phase11": phase11, "phase12": phase12, "phase13": phase13, "phase14": phase14, "phase15": phase15, "phase16": phase16, "phase17": phase17, "phase18": phase18, "phase19": phase19, "phase20": phase20, "phase21": phase21, "phase22": phase22, "phase23": phase23, "phase24": phase24, "phase25": phase25, "phase2425": phase2425}[phase]()
    path = Path(f"{EXP}/{phase}_jobs.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jobs, indent=2))
    print(f"{len(jobs)} jobs -> {path}")
