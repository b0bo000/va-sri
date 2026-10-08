"""Dataset loading with the V10 protocol: 7:1:2 split, train-only scaler."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from data.loaders import load_csv_multivariate, split_and_scale

ROOT = Path(__file__).resolve().parents[2] / "data"
FILES = {"ETTh1": "ETTh1.csv", "ETTh2": "ETTh2.csv", "ETTm1": "ETTm1.csv",
         "ETTm2": "ETTm2.csv", "Weather": "weather.csv", "Electricity": "electricity.csv"}


def load_splits(name: str):
    """Return train, val, test (standardised, NaN = originally missing) and scaler."""
    raw = load_csv_multivariate(str(ROOT), FILES[name])
    return split_and_scale(raw)


def variance_weights(scaler) -> np.ndarray:
    w = scaler.std.astype(np.float64) ** 2
    return (w / (w.mean() + 1e-8)).astype(np.float32)
