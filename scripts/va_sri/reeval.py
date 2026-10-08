"""Recompute all metrics (incl. standardized MAE) from saved best.pt, no training.

Validation is always recomputed; test only for runs that already report a test
result, so this never opens a test split that was not already evaluated.
Results are stored as "val_full" / "test_full" next to the original fields.

  python -m scripts.va_sri.reeval experiments/va_sri/runs/<run> [...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from data.datasets import ImputationDataset
from scripts.va_sri.data import load_splits
from scripts.va_sri.model import VASRI
from scripts.va_sri.train import evaluate, make_loader, mask_kwargs


def main(runs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    for run in map(Path, runs):
        res = json.loads((run / "result.json").read_text())
        if "val_full" in res and ("test" not in res or "test_full" in res):
            continue
        cfg = json.loads((run / "config.json").read_text())
        _, val, test, scaler = load_splits(cfg["dataset"])
        channels = val.shape[1]
        model = VASRI(channels, seed=cfg["seed"], identity=not cfg["no_identity"], vga=cfg["vga"],
                      input_fill=cfg.get("input_fill", "skeleton"), output=cfg.get("output", "residual"))
        model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True)["model"])
        model.to(device)
        kw = mask_kwargs(argparse.Namespace(**cfg))
        std = torch.as_tensor(scaler.std, dtype=torch.float64, device=device)
        check = cfg.get("output", "residual") == "residual"
        ft_seed = 10_000_000 + cfg["seed"]
        res["val_full"] = evaluate(model, make_loader(ImputationDataset(val, mode="val", **kw), False, ft_seed + 1),
                                   std, device, channels, check)
        assert abs(res["val_full"]["mse"] - res["val"]["mse"]) <= 1e-3 * res["val"]["mse"], "val mismatch (beyond CPU/GPU rounding)"
        if "test" in res:
            res["test_full"] = evaluate(model, make_loader(ImputationDataset(test, mode="test", **kw), False, 0),
                                        std, device, channels, check)
        (run / "result.json").write_text(json.dumps(res, indent=2))
        print(f"{run.name}: val std_mae {res['val_full']['std_mae']:.4f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
