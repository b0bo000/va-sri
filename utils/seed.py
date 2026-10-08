# -*- coding: utf-8 -*-
import random
import numpy as np
import torch

def seed_everything(seed: int = 42) -> None:
    """Fix random seeds for reproducibility (best-effort)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
