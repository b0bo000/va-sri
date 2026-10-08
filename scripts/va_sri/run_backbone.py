"""Additional baselines (PyPOTS 0.4.1 cores: TimesNet, PatchTST, Crossformer) under the exact
protocol of run_saits_block_supervised.py: same windows, masks, metrics and checkpoint selection.

Every model is trained with the loss VA-SRI uses (unweighted masked MSE + MAE on standardized
values, supervised target entries only), Adam, EMA 0.999 of the weights, a fixed epoch budget and
selection of the epoch with the lowest standardized validation MSE.  Only the network differs.

  python -u -m scripts.va_sri.run_backbone --model timesnet --data_path ETTh2.csv --seed 42 --lr 1e-3
"""
from __future__ import annotations

import argparse
import copy
import time

import torch
from torch.utils.data import DataLoader

from data.datasets import ImputationDataset
from data.loaders import load_csv_multivariate, split_and_scale
from run_saits_block_supervised import evaluate, materialize_windows
from utils.seed import seed_everything

TAG = dict(timesnet="TimesNet", patchtst="PatchTST", crossformer="Crossformer")


def build(name, c, seq_len):
    if name == "timesnet":       # TSLib imputation configuration (ETT), d_model 64
        from pypots.imputation.timesnet.core import _TimesNet
        return _TimesNet(n_layers=2, n_steps=seq_len, n_features=c, top_k=3, d_model=64, d_ffn=64,
                         n_kernels=6, dropout=0.1, apply_nonstationary_norm=True)
    if name == "patchtst":       # same patching as VA-SRI (16 / 8), width 64
        from pypots.imputation.patchtst.core import _PatchTST
        return _PatchTST(n_steps=seq_len, n_features=c, n_layers=3, n_heads=4, d_model=64, d_ffn=128,
                         d_k=16, d_v=16, patch_len=16, stride=8, dropout=0.1, attn_dropout=0.0)
    if name == "crossformer":    # two-stage (time + variable) attention
        from pypots.imputation.crossformer.core import _Crossformer
        return _Crossformer(n_steps=seq_len, n_features=c, n_layers=2, n_heads=4, d_model=64, d_ffn=128,
                            factor=5, seg_len=12, win_size=2, dropout=0.1)
    raise ValueError(name)


def forward(model, x_cat, c):
    """Imputation with observations copied (PyPOTS TimesNet rebuilds observed entries from its
    non-stationary-normalized input, so the copy is done here for every model)."""
    x, m = x_cat[..., :c], x_cat[..., c:]
    pred = model({"X": x, "missing_mask": m}, training=False)["imputed_data"]
    return m * x + (1 - m) * pred


@torch.no_grad()
def impute(model, loader, device, c):
    model.eval()
    return torch.cat([forward(model, x_cat.to(device).float(), c).cpu() for x_cat, *_ in loader]).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(TAG), required=True)
    p.add_argument("--root_path", default="./data")
    p.add_argument("--data_path", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--block_len", type=int, default=24)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--skip_test", action="store_true")
    args = p.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, val, test, scaler = split_and_scale(load_csv_multivariate(args.root_path, args.data_path))
    c = train.shape[1]
    kw = dict(seq_len=args.seq_len, fixed_seed=args.seed, mask_mode="block", mask_ratio=0.4, block_shared=False,
              p_point=0.3, p_block=0.3, p_partial=0.4, point_ratio=0.03, partial_min_vars=2, partial_max_vars=4)
    train_ds = ImputationDataset(train, mode="train", reproducible_epoch_mask=True, block_len=args.block_len,
                                 block_lens=None, block_len_curriculum=False, curriculum_epochs=args.epochs, **kw)
    val_ds = ImputationDataset(val, mode="val", block_len=args.block_len, **kw)
    test_ds = ImputationDataset(test, mode="test", block_len=args.block_len, **kw)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_in, val_tg, val_true = materialize_windows(val_ds, c)
    test_in, test_tg, test_true = materialize_windows(test_ds, c)
    print(f"[Data] C={c} | train={len(train)} val={len(val)} test={len(test)} | windows val={len(val_ds)} test={len(test_ds)}")

    model = build(args.model, c, args.seq_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema_decay))
    print(f"[Params] total={sum(q.numel() for q in model.parameters())}")
    best, best_ep, best_state = float("inf"), 0, copy.deepcopy(ema.module.state_dict())
    t0 = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        train_ds.set_epoch(ep - 1)
        model.train()
        tot, nb = 0.0, 0
        for x_cat, x_true, tm, _ in train_loader:
            x_cat, x_true, tm = x_cat.to(device).float(), x_true.to(device).float(), tm.to(device).float()
            pred = forward(model, x_cat, c)
            d, n = pred - x_true, tm.sum() + 1e-8
            loss = ((d ** 2) * tm).sum() / n + (d.abs() * tm).sum() / n
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            ema.update_parameters(model)
            tot, nb = tot + float(loss), nb + 1
        evaluate("Val", impute(ema.module, val_loader, device, c), val_true, val_in, val_tg, scaler, c)
        v = evaluate.last_std_mse
        print(f"[Train] Epoch {ep:03d} | loss={tot / max(nb, 1):.6f} | val_std_mse={v:.6f}", flush=True)
        if v < best:
            best, best_ep, best_state = v, ep, copy.deepcopy(ema.module.state_dict())
    ema.module.load_state_dict(best_state)
    print(f"[Timing] train_wall_seconds={time.perf_counter() - t0:.3f}")
    print(f"[Best] epoch={best_ep} | val_std_mse={best:.6f}")
    evaluate("BestVal", impute(ema.module, val_loader, device, c), val_true, val_in, val_tg, scaler, c)
    if args.skip_test:
        print("[SkipTest] test split not evaluated")
        return
    t1 = time.perf_counter()
    evaluate(TAG[args.model], impute(ema.module, test_loader, device, c), test_true, test_in, test_tg, scaler, c)
    print(f"[Timing] test_wall_seconds={time.perf_counter() - t1:.3f}")


if __name__ == "__main__":
    main()
