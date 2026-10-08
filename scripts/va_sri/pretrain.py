"""Gap-structured masked pretraining (GSM) of the shared VA-SRI trunk.

P1: one dataset.  P2: several datasets jointly (train splits only).
Variable-axis tokens make the trunk independent of the number of variables,
so a single trunk is trained across datasets; identity, VGA and refiner are
finetune-only. Each batch comes from one dataset, sampled with probability
proportional to sqrt(#train windows).

Example:
  python -m scripts.va_sri.pretrain --datasets ETTh2 --seed 42 \
      --epochs-of ETTh2 --out experiments/va_sri/pretrain/p1_etth2_s42
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from engine.ssl import augment_xcat
from losses.losses import nt_xent, weighted_masked_mse_loss
from scripts.va_sri.data import load_splits, variance_weights
from scripts.va_sri.masks import gsm_mask
from scripts.va_sri.model import VASRI, shared_state
from utils.seed import seed_everything

SEQ_LEN, BATCH = 96, 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seed", type=int, required=True)
    budget = ap.add_mutually_exclusive_group(required=True)
    budget.add_argument("--updates", type=int)
    budget.add_argument("--epochs-of", help="60 passes over this dataset's train windows")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--contrastive", type=float, default=0.0,
                    help="weight of window-level NT-Xent (legacy SRI augmentations); 0 = off")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if (args.out / "shared.pt").exists():
        raise SystemExit(f"{args.out} already completed")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))
    torch.set_num_threads(2)
    device = torch.device(args.device)

    sets = []
    for name in args.datasets:
        train, _, _, scaler = load_splits(name)
        obs = ~np.isnan(train)
        sets.append({"name": name, "x": torch.from_numpy(np.nan_to_num(train, nan=0.0)).float(),
                     "obs": torch.from_numpy(obs.astype(np.float32)),
                     "w": torch.as_tensor(variance_weights(scaler), device=device),
                     "windows": len(train) - SEQ_LEN + 1})
    if args.updates:
        total = args.updates
    else:
        ref = next(s for s in sets if s["name"] == args.epochs_of)
        total = args.epochs * math.ceil(ref["windows"] / BATCH)
    probs = np.sqrt([s["windows"] for s in sets]); probs = probs / probs.sum()

    seed_everything(args.seed)
    model = VASRI(sets[0]["x"].shape[1], seed=args.seed, identity=False, vga="off").to(device)
    params = [p for n, p in model.named_parameters()
              if not n.startswith(("base.temporal_refine.", "base.refine_scale"))
              and (args.contrastive > 0 or not n.startswith("base.proj."))]
    opt = torch.optim.Adam(params, lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    gen = torch.Generator().manual_seed(50_000_000 + args.seed)
    rng = np.random.default_rng(50_000_000 + args.seed)
    offsets = torch.arange(SEQ_LEN)

    log = (args.out / "log.jsonl").open("w")
    t0, running = time.time(), {s["name"]: [] for s in sets}
    model.train()
    for step in range(1, total + 1):
        s = sets[rng.choice(len(sets), p=probs)]
        starts = torch.randint(s["windows"], (BATCH,), generator=gen)
        idx = starts[:, None] + offsets[None]
        x, obs = s["x"][idx].to(device), s["obs"][idx].to(device)
        target = gsm_mask(obs, gen)
        visible = obs * (1 - target)
        model.base.out_channels = x.shape[-1]
        stage1 = model.compose_stage1(torch.cat([x * visible, visible], dim=-1))
        loss = weighted_masked_mse_loss(stage1, x, target, s["w"])
        if args.contrastive > 0:
            x_obs = torch.cat([x * obs, obs], dim=-1)
            z1 = model.base.encode_repr(augment_xcat(x_obs, 0.02, 0.1, 0.1))
            z2 = model.base.encode_repr(augment_xcat(x_obs, 0.02, 0.1, 0.1))
            loss = loss + args.contrastive * nt_xent(z1, z2, tau=0.2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        running[s["name"]].append(float(loss.detach()))
        if step % 500 == 0 or step == total:
            row = {"step": step, "of": total, "seconds": round(time.time() - t0, 1),
                   "loss": {k: float(np.mean(v)) for k, v in running.items() if v}}
            log.write(json.dumps(row) + "\n"); log.flush()
            print(json.dumps(row), flush=True)
            running = {k: [] for k in running}
    torch.save({"shared": {k: v.cpu() for k, v in shared_state(model).items()},
                "datasets": args.datasets, "updates": total, "seed": args.seed},
               args.out / "shared.pt")
    print(f"[done] {total} updates in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
