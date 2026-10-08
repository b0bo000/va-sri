# -*- coding: utf-8 -*-
import numpy as np
from torch.utils.data import Dataset

class WindowDataset(Dataset):
    """SSL pretrain dataset: returns (filled window, original observed mask)."""
    def __init__(self, data: np.ndarray, seq_len: int):
        self.data = data
        self.seq_len = seq_len
        self.n = len(self.data) - self.seq_len + 1

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        x = self.data[idx:idx + self.seq_len].astype(np.float32)
        obs_mask = (~np.isnan(x)).astype(np.float32)
        x_filled = np.nan_to_num(x, nan=0.0)
        return x_filled, obs_mask

class ImputationDataset(Dataset):
    """
    Finetune/Val/Test dataset: create artificial missing positions inside each window.

    Returns:
      x_cat: [L,2C]  = concat([x_in, input_mask])
      x_true:[L,C]   (normalized; NaN->0)
      target_mask:[L,C] (1 where we supervise/evaluate = artificially removed)
      obs_mask:[L,C] (1 where original is observed; 0 where original is NaN)
    """
    def __init__(self, data: np.ndarray, seq_len: int, mode: str, fixed_seed: int,
                 mask_mode: str = "block", mask_ratio: float = 0.4,
                 block_len: int = 24, block_shared: bool = False,
                 block_lens=None,
                 p_point: float = 0.3, p_block: float = 0.3, p_partial: float = 0.4,
                 point_ratio: float = 0.15,
                 partial_min_vars: int = 2, partial_max_vars: int = -1,
                 last_mask_type=None,
                 deterministic_train_mask: bool = False,
                 reproducible_epoch_mask: bool = False,
                 block_len_curriculum: str = "none",
                 curriculum_epochs: int = 100):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.mode = mode
        self.fixed_seed = fixed_seed
        self.mask_mode = mask_mode
        self.mask_ratio = mask_ratio
        self.block_len = block_len
        self.block_lens = None if block_lens is None else [int(x) for x in block_lens]
        self.block_shared = block_shared

        self.p_point = p_point
        self.p_block = p_block
        self.p_partial = p_partial
        self.point_ratio = point_ratio
        self.partial_min_vars = partial_min_vars
        self.partial_max_vars = partial_max_vars
        self.last_mask_type = last_mask_type
        self.deterministic_train_mask = deterministic_train_mask
        self.reproducible_epoch_mask = reproducible_epoch_mask
        self.current_epoch = 0
        self.block_len_curriculum = block_len_curriculum
        self.curriculum_epochs = max(1, int(curriculum_epochs))

        self.n = len(self.data) - self.seq_len + 1

    def __len__(self) -> int:
        return self.n

    def _sample_block_len(self, rng: np.random.RandomState) -> int:
        if self.block_lens:
            if self.mode == "train" and self.block_len_curriculum != "none":
                probs = self._block_len_curriculum_probs()
                return int(rng.choice(self.block_lens, p=probs))
            return int(rng.choice(self.block_lens))
        return int(self.block_len)

    def _block_len_curriculum_probs(self) -> np.ndarray:
        lens = np.asarray(self.block_lens, dtype=np.float64)
        n = len(lens)
        if n == 0:
            return np.asarray([], dtype=np.float64)
        if n == 1:
            return np.ones(1, dtype=np.float64)

        order = np.argsort(lens)
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.linspace(0.0, 1.0, n)
        if self.curriculum_epochs <= 1:
            progress = 1.0
        else:
            progress = min(1.0, max(0.0, float(self.current_epoch) / float(self.curriculum_epochs - 1)))

        if self.block_len_curriculum == "short_to_long":
            # Smoothly shifts from short-block-heavy to long-block-heavy.
            strength = 2.0
            short_bias = np.exp(-strength * ranks)
            long_bias = np.exp(strength * ranks)
            probs = (1.0 - progress) * short_bias + progress * long_bias
        elif self.block_len_curriculum == "three_phase":
            # Early local reconstruction, middle uniform regularization, late target-long adaptation.
            uniform = np.ones(n, dtype=np.float64)
            short_bias = np.exp(-2.0 * ranks)
            long_bias = np.exp(2.0 * ranks)
            if progress < 0.33:
                mix = progress / 0.33
                probs = (1.0 - mix) * short_bias + mix * uniform
            elif progress < 0.66:
                probs = uniform
            else:
                mix = (progress - 0.66) / 0.34
                probs = (1.0 - mix) * uniform + mix * long_bias
        else:
            probs = np.ones(n, dtype=np.float64)

        probs = probs / probs.sum()
        return probs.astype(np.float64)

    def _rng(self, idx: int) -> np.random.RandomState:
        # train:
        #   1) non-deterministic augmentation by default
        #   2) deterministic per-window masks for controlled sweeps
        #   3) reproducible epoch-wise resampling: new masks every epoch, but
        #      fully determined by (seed, epoch, idx) for academic reruns
        if self.mode == "train":
            if self.deterministic_train_mask:
                return np.random.RandomState(self.fixed_seed + idx)
            if self.reproducible_epoch_mask:
                epoch_seed = self.fixed_seed + 1000003 * int(self.current_epoch)
                return np.random.RandomState(epoch_seed + idx)
            return np.random.RandomState(None)
        return np.random.RandomState(self.fixed_seed + idx)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _make_point_mask(self, obs_mask: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        L, C = obs_mask.shape
        target = np.zeros((L, C), dtype=np.float32)

        for c in range(C):
            idxs = np.where(obs_mask[:, c] > 0.5)[0]
            if len(idxs) == 0:
                continue
            k = max(1, int(self.point_ratio * L))
            choose = rng.choice(idxs, size=min(k, len(idxs)), replace=False)
            target[choose, c] = 1.0

        return target

    def _make_single_block_mask(self, obs_mask: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        L, C = obs_mask.shape
        target = np.zeros((L, C), dtype=np.float32)

        blen = min(self._sample_block_len(rng), L)
        if C == 0 or L == 0:
            return target

        c = rng.randint(0, C)
        s = rng.randint(0, L - blen + 1)

        target[s:s + blen, c] = 1.0
        target[:, c] *= obs_mask[:, c]
        return target

    def _make_ratio_block_mask(self, obs_mask: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        L, C = obs_mask.shape
        target = np.zeros((L, C), dtype=np.float32)
        if C == 0 or L == 0:
            return target

        ratio = min(1.0, max(0.0, float(self.mask_ratio)))
        blen = min(max(1, int(round(ratio * L))), L)
        for c in range(C):
            s = rng.randint(0, L - blen + 1)
            target[s:s + blen, c] = 1.0
            target[:, c] *= obs_mask[:, c]
        return target

    def _make_partial_blackout_mask(self, obs_mask: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        L, C = obs_mask.shape
        target = np.zeros((L, C), dtype=np.float32)

        blen = min(self._sample_block_len(rng), L)
        if C == 0 or L == 0:
            return target

        min_vars = max(1, self.partial_min_vars)
        if self.partial_max_vars < 0:
            max_vars = max(min_vars, C // 2)
        else:
            max_vars = min(C, max(min_vars, self.partial_max_vars))

        min_vars = min(min_vars, C)
        max_vars = min(max_vars, C)
        if max_vars < min_vars:
            max_vars = min_vars

        n_vars = rng.randint(min_vars, max_vars + 1)
        chosen_vars = rng.choice(C, size=n_vars, replace=False)

        s = rng.randint(0, L - blen + 1)
        target[s:s + blen, chosen_vars] = 1.0
        target = target * obs_mask
        return target

    def _make_target_mask(self, obs_mask: np.ndarray, idx: int) -> np.ndarray:
        L, C = obs_mask.shape
        rng = self._rng(idx)

        if self.mask_mode == "point":
            self.last_mask_type = "point"
            return self._make_point_mask(obs_mask, rng)

        elif self.mask_mode == "block":
            self.last_mask_type = "block"
            target = np.zeros((L, C), dtype=np.float32)
            blen = min(self._sample_block_len(rng), L)

            if self.block_shared:
                s = rng.randint(0, L - blen + 1)
                target[s:s + blen, :] = 1.0
                return target * obs_mask
            else:
                for c in range(C):
                    s = rng.randint(0, L - blen + 1)
                    target[s:s + blen, c] = 1.0
                    target[:, c] *= obs_mask[:, c]
                return target

        elif self.mask_mode == "block_ratio":
            self.last_mask_type = "block_ratio"
            return self._make_ratio_block_mask(obs_mask, rng)

        elif self.mask_mode == "single_block":
            self.last_mask_type = "single_block"
            return self._make_single_block_mask(obs_mask, rng)

        elif self.mask_mode == "partial":
            self.last_mask_type = "partial"
            return self._make_partial_blackout_mask(obs_mask, rng)
        
        elif self.mask_mode == "mixed":
            s = self.p_point + self.p_block + self.p_partial
            p_point = self.p_point / s
            p_block = self.p_block / s
            p_partial = self.p_partial / s

            r = rng.rand()
            if r < p_point:
                self.last_mask_type = "point"
                return self._make_point_mask(obs_mask, rng)
            elif r < p_point + p_block:
                self.last_mask_type = "single_block"
                return self._make_single_block_mask(obs_mask, rng)
            else:
                self.last_mask_type = "partial"
                return self._make_partial_blackout_mask(obs_mask, rng)

        else:
            raise ValueError(f"Unknown mask_mode: {self.mask_mode}")

    def __getitem__(self, idx: int):
        L = self.seq_len
        x_gt = self.data[idx: idx + L].astype(np.float32)  # [L,C]

        obs_mask = (~np.isnan(x_gt)).astype(np.float32)    # [L,C]
        x_filled = np.nan_to_num(x_gt, nan=0.0)            # NaN->0

        target_mask = self._make_target_mask(obs_mask, idx)  # [L,C] supervise positions

        # input missing = original missing OR artificially removed
        missing_in_input = (obs_mask < 0.5) | (target_mask > 0.5)
        input_mask = (~missing_in_input).astype(np.float32)   # 1=observed in input

        x_in = x_filled.copy()
        x_in[missing_in_input] = 0.0

        x_cat = np.concatenate([x_in, input_mask], axis=1)    # [L,2C]
        x_true = x_filled                                     # [L,C]
        return x_cat, x_true, target_mask, obs_mask
