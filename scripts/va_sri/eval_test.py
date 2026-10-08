"""Add test-set metrics to a finished run from its saved best.pt (no training).

  python -m scripts.va_sri.eval_test experiments/va_sri/runs/<run> [...]
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
        if "test" in res:
            print(f"{run.name}: already has test"); continue
        cfg = json.loads((run / "config.json").read_text())
        _, _, test, scaler = load_splits(cfg["dataset"])
        channels = test.shape[1]
        model = VASRI(channels, seed=cfg["seed"], identity=not cfg["no_identity"], vga=cfg["vga"],
                      input_fill=cfg.get("input_fill", "skeleton"), output=cfg.get("output", "residual"))
        model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True)["model"])
        model.to(device)
        kw = mask_kwargs(argparse.Namespace(**cfg))
        std = torch.as_tensor(scaler.std, dtype=torch.float64, device=device)
        loader = make_loader(ImputationDataset(test, mode="test", **kw), False, 0)
        res["test"] = evaluate(model, loader, std, device, channels, cfg.get("output", "residual") == "residual")
        res["test_added_by"] = "eval_test.py"
        (run / "result.json").write_text(json.dumps(res, indent=2))
        print(f"{run.name}: test mse {res['test']['mse']:.4f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
