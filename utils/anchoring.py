# -*- coding: utf-8 -*-
import numpy as np

def anchor_fill_from_raw_pred(true_inv: np.ndarray,
                              pred_inv_raw: np.ndarray,
                              input_mask: np.ndarray,
                              anchor: bool = True) -> np.ndarray:
    """
    true_inv: [L,C] ground truth (orig scale) for observed points.
    pred_inv_raw: [L,C] model raw prediction (orig scale) on full window.
    input_mask: [L,C] 1=observed in input, 0=missing in input.
    Returns filled series [L,C]:
      - observed points = true_inv (keep)
      - missing points = pred_inv_raw (+ optional anchor offset)
    Anchor logic:
      For each contiguous missing segment, estimate boundary deltas using raw pred at boundary,
      then add linearly interpolated delta inside segment.
    """
    out = true_inv.copy()
    miss = input_mask < 0.5
    out[miss] = pred_inv_raw[miss]

    if not anchor:
        return out

    L, C = true_inv.shape
    for c in range(C):
        m = miss[:, c]
        if not m.any():
            continue

        i = 0
        while i < L:
            if not m[i]:
                i += 1
                continue
            s = i
            while i < L and m[i]:
                i += 1
            e = i - 1  # [s,e] missing segment

            left_ok = (s - 1) >= 0 and (not m[s - 1])
            right_ok = (e + 1) < L and (not m[e + 1])

            if left_ok and right_ok:
                delta_L = true_inv[s - 1, c] - pred_inv_raw[s - 1, c]
                delta_R = true_inv[e + 1, c] - pred_inv_raw[e + 1, c]
                seg_len = (e - s + 1)
                for t in range(seg_len):
                    alpha = (t + 1) / (seg_len + 1.0)
                    delta = (1.0 - alpha) * delta_L + alpha * delta_R
                    out[s + t, c] = pred_inv_raw[s + t, c] + delta
            elif left_ok:
                delta_L = true_inv[s - 1, c] - pred_inv_raw[s - 1, c]
                out[s:e + 1, c] = pred_inv_raw[s:e + 1, c] + delta_L
            elif right_ok:
                delta_R = true_inv[e + 1, c] - pred_inv_raw[e + 1, c]
                out[s:e + 1, c] = pred_inv_raw[s:e + 1, c] + delta_R
            else:
                pass

    return out
