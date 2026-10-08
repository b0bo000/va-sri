# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from typing import Tuple

from .scaler import StandardScaler

def load_csv_multivariate(root_path: str, data_path: str) -> np.ndarray:
    fp = os.path.join(root_path, data_path)
    df = pd.read_csv(fp)
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    data = df.values.astype(np.float32)  # [T,C]
    # Some public benchmark CSVs encode missing sensor readings as -9999.
    # The rest of the pipeline already treats NaN as original missingness.
    data[data == -9999.0] = np.nan
    return data

def split_and_scale(data: np.ndarray, train_ratio: float = 0.7, val_ratio: float = 0.1
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    T = len(data)
    n_train = int(T * train_ratio)
    n_val = int(T * val_ratio)
    train_raw = data[:n_train]
    val_raw = data[n_train:n_train + n_val]
    test_raw = data[n_train + n_val:]

    scaler = StandardScaler()
    scaler.fit(train_raw)
    train = scaler.transform(train_raw)
    val = scaler.transform(val_raw)
    test = scaler.transform(test_raw)
    return train, val, test, scaler
