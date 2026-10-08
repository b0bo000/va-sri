"""Finetune VA-SRI on one dataset (optionally from a pretrained trunk).

The loop reproduces scripts/sri_variable_axis_training/train.py (seeding,
loaders, loss, selection), so --vga off --no-pretrain matches the 09-27 runs.

Example:
  python -m scripts.va_sri.train --dataset ETTh2 --seed 42 --vga off \
      --out experiments/va_sri/runs/etth2_s42_none_off
"""
from __future__ import annotations

import argparse
import json
import os
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.datasets import ImputationDataset
from losses.losses import weighted_masked_mse_loss
from scripts.va_sri.data import load_splits, variance_weights
from scripts.va_sri.model import VASRI, load_shared
from utils.seed import seed_everything


NUM_WORKERS = int(os.environ.get("VA_SRI_WORKERS", "0"))


def make_loader(ds, shuffle, seed, batch_size=64):
    # Masks are seeded per (seed, epoch, window), so worker processes do not change
    # the data; workers are not persistent so that dataset.set_epoch() reaches them.
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=NUM_WORKERS,
                      drop_last=False, generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def evaluate(model, loader, std, device, channels, check_observed=True):
    model.eval()
    totals = torch.zeros(5, dtype=torch.float64, device=device)
    for x, truth, target, _ in loader:
        x, truth, target = [v.to(device).float() for v in (x, truth, target)]
        _, pred = model.forward_recon_refine(x)
        if check_observed:
            visible = x[..., channels:].bool()
            assert torch.equal(pred[visible], x[..., :channels][visible])
        err = pred.double() - truth.double()
        m = target.double()
        totals += torch.stack(((err * std).square().mul(m).sum(), (err * std).abs().mul(m).sum(),
                               err.square().mul(m).sum(), err.abs().mul(m).sum(), m.sum()))
    mse, mae, std_mse, std_mae, n = totals.cpu().tolist()
    return {"mse": mse / n, "mae": mae / n, "std_mse": std_mse / n, "std_mae": std_mae / n, "targets": int(n)}


def mask_kwargs(args):
    """Same mask settings as the V10 protocol (run.py defaults used by its experiments)."""
    mode = getattr(args, "mask_mode", "block")
    kw = dict(seq_len=96, fixed_seed=args.seed, block_len=getattr(args, "block_len", 24),
              mask_mode="block" if mode == "block_shared" else mode, block_shared=mode == "block_shared",
              mask_ratio=0.4, p_point=0.3, p_block=0.3, p_partial=0.4, point_ratio=getattr(args, "point_ratio", 0.03),
              partial_min_vars=2, partial_max_vars=4)
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--vga", default="off", choices=("off", "beta", "beta_g"))
    ap.add_argument("--no-identity", action="store_true")
    ap.add_argument("--init", type=Path, help="pretrained trunk (pretrain.py output)")
    ap.add_argument("--eval-test", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--schedule", default="const", choices=("const", "cosine"))
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of evaluated weights; 0 = off")
    ap.add_argument("--block-len", type=int, default=24)
    ap.add_argument("--mask-mode", default="block", choices=("block", "block_shared", "partial", "mixed", "point"))
    ap.add_argument("--point-ratio", type=float, default=0.03, help="point mode: fraction of steps hidden per variable")
    ap.add_argument("--input-fill", default="skeleton", choices=("skeleton", "zero"))
    ap.add_argument("--unweighted", action="store_true", help="plain MSE on standardized values (w_c = 1)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--micro-batch", type=int, default=0,
                    help="split each batch into chunks of this size with exact gradient accumulation (memory only)")
    ap.add_argument("--mae-weight", type=float, default=0.0,
                    help="add this weight x masked MAE (same stage weights) to the squared-error loss")
    ap.add_argument("--select", default="mse", choices=("mse", "std_mse"),
                    help="validation metric used to pick the reported epoch")
    ap.add_argument("--output", default="residual", choices=("residual", "absolute"))
    ap.add_argument("--max-train-windows", type=int)  # smoke tests only
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if (args.out / "result.json").exists():
        raise SystemExit(f"{args.out} already completed")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))

    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)

    train, val, test, scaler = load_splits(args.dataset)
    channels = train.shape[1]
    kw = mask_kwargs(args)
    train_ds = ImputationDataset(train, mode="train", reproducible_epoch_mask=True, **kw)
    val_ds = ImputationDataset(val, mode="val", **kw)
    if args.max_train_windows:
        idx = np.linspace(0, len(train_ds) - 1, args.max_train_windows, dtype=np.int64)
        train_set = torch.utils.data.Subset(train_ds, idx.tolist())
        val_set = torch.utils.data.Subset(val_ds, list(range(min(len(val_ds), args.max_train_windows))))
    else:
        train_set, val_set = train_ds, val_ds
    ft_seed = 10_000_000 + args.seed
    train_loader = make_loader(train_set, True, ft_seed)
    val_loader = make_loader(val_set, False, ft_seed + 1)

    seed_everything(args.seed)
    model = VASRI(channels, seed=args.seed, identity=not args.no_identity, vga=args.vga,
                  input_fill=args.input_fill, output=args.output)
    absolute = args.output == "absolute"
    if args.init:
        loaded = load_shared(model, torch.load(args.init, map_location="cpu", weights_only=True)["shared"])
        print(f"[init] loaded {len(loaded)} shared tensors from {args.init}", flush=True)
    model.to(device)
    n_params = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("base.proj."))
    print(f"[model] {args.dataset} C={channels} vga={args.vga} params={n_params}", flush=True)

    seed_everything(ft_seed)
    std = torch.as_tensor(scaler.std, dtype=torch.float64, device=device)
    weights = torch.as_tensor(variance_weights(scaler), device=device)
    if args.unweighted:
        weights = torch.ones_like(weights)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader))
             if args.schedule == "cosine" else None)
    ema = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema)) if args.ema > 0 else None
    scored = ema.module if ema is not None else model   # weights that are evaluated/selected
    best, best_epoch, bad = float("inf"), 0, 0
    t0 = time.time()
    log = (args.out / "epochs.jsonl").open("w")
    for epoch in range(args.epochs):
        train_ds.set_epoch(epoch)
        model.train()
        total, nb = 0.0, 0
        for x, truth, target, _ in train_loader:
            x, truth, target = [v.to(device).float() for v in (x, truth, target)]
            if args.micro_batch and not absolute:
                # Exact gradient accumulation: every term shares the denominator
                # sum(target * w) over the whole batch, so each chunk contributes
                # numerator / D_total and the summed gradient equals the batch-64 one.
                opt.zero_grad(set_to_none=True)
                wv = weights.view(1, 1, -1)
                denom = (target * wv).sum() + 1e-8
                loss = 0.0
                for i in range(0, x.shape[0], args.micro_batch):
                    xs, ts, ms = x[i:i + args.micro_batch], truth[i:i + args.micro_batch], target[i:i + args.micro_batch]
                    s1, pr = model.forward_recon_refine(xs)
                    tm = ms * wv
                    part = (0.3 * ((s1 - ts) ** 2 * tm).sum() + ((pr - ts) ** 2 * tm).sum()) / denom
                    if args.mae_weight > 0:
                        part = part + args.mae_weight * (0.3 * ((s1 - ts).abs() * tm).sum()
                                                         + ((pr - ts).abs() * tm).sum()) / denom
                    part.backward()
                    loss = loss + part.detach()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if sched is not None:
                    sched.step()
                if ema is not None:
                    ema.update_parameters(model)
                total += float(loss); nb += 1
                continue
            stage1, pred = model.forward_recon_refine(x)
            loss = 0.3 * weighted_masked_mse_loss(stage1, truth, target, weights) \
                + weighted_masked_mse_loss(pred, truth, target, weights)
            if args.mae_weight > 0:
                tm = target * weights.view(1, 1, -1)
                mae = lambda z: ((z - truth).abs() * tm).sum() / (tm.sum() + 1e-8)
                loss = loss + args.mae_weight * (0.3 * mae(stage1) + mae(pred))
            if absolute:   # absolute outputs must also learn observed entries (V10 factorial)
                loss = loss + 0.02 * weighted_masked_mse_loss(pred, truth, x[..., channels:], weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched is not None:
                sched.step()
            if ema is not None:
                ema.update_parameters(model)
            total += float(loss.detach()); nb += 1
        metrics = evaluate(scored, val_loader, std, device, channels, not absolute)
        improved = metrics[args.select] < best
        if improved:
            best, best_epoch, bad = metrics[args.select], epoch + 1, 0
            torch.save({"model": scored.state_dict()}, args.out / "best.pt")
        else:
            bad += 1
        row = {"epoch": epoch + 1, "loss": total / nb, "val": metrics, "best": improved,
               "seconds": round(time.time() - t0, 1)}
        log.write(json.dumps(row) + "\n"); log.flush()
        print(json.dumps(row), flush=True)
        if args.patience > 0 and bad >= args.patience:
            break
    log.close()

    model.load_state_dict(torch.load(args.out / "best.pt", map_location=device, weights_only=True)["model"])
    result = {"dataset": args.dataset, "seed": args.seed, "vga": args.vga,
              "schedule": args.schedule, "ema": args.ema, "patience": args.patience,
              "identity": not args.no_identity, "init": str(args.init) if args.init else None,
              "params": n_params, "best_epoch": best_epoch, "epochs_run": epoch + 1,
              "val": evaluate(model, val_loader, std, device, channels, not absolute),
              "block_len": args.block_len, "mask_mode": args.mask_mode, "point_ratio": args.point_ratio,
              "input_fill": args.input_fill, "output": args.output, "unweighted": args.unweighted, "select": args.select, "lr": args.lr, "mae_weight": args.mae_weight, "micro_batch": args.micro_batch,
              "seconds": round(time.time() - t0, 1)}
    if args.vga != "off":
        result["vga_beta"] = model.vga_beta.detach().cpu().tolist()
    if args.eval_test:
        test_ds = ImputationDataset(test, mode="test", **kw)
        result["test"] = evaluate(model, make_loader(test_ds, False, 0), std, device, channels, not absolute)
    (args.out / "result.json").write_text(json.dumps(result, indent=2))
    print("[done] " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
