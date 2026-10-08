# -*- coding: utf-8 -*-
from typing import Optional, Dict
import numpy as np

class StandardScaler:
    """Per-channel standard scaler: (x-mean)/std. Fit on train only."""
    def __init__(self):
        self.mean: Optional[np.ndarray] = None  # [C]
        self.std: Optional[np.ndarray] = None   # [C]

    def fit(self, data: np.ndarray) -> None:
        self.mean = np.nanmean(data, axis=0)
        self.std = np.nanstd(data, axis=0)
        self.std[self.std == 0] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, sd: Dict[str, np.ndarray]) -> None:
        self.mean = sd["mean"]
        self.std = sd["std"]
