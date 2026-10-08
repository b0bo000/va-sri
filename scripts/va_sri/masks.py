"""Gap-structured masks (GSM) for pretraining.

Every variable independently receives 1-2 contiguous gaps whose lengths are
drawn from GAP_LENGTHS; with probability WHOLE_VAR_PROB one variable of the
sample is hidden for the entire window. Gaps are restricted to originally
observed entries and every sample keeps at least one visible value.
"""
from __future__ import annotations

import torch

GAP_LENGTHS = (8, 16, 24, 32, 48)
MAX_GAPS = 2
WHOLE_VAR_PROB = 0.15


def gsm_mask(obs: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """obs [B,L,C] (1=observed) -> target mask [B,L,C] (1=hidden & supervised)."""
    bsz, length, c = obs.shape
    lens_table = torch.tensor(GAP_LENGTHS)
    lens = lens_table[torch.randint(len(GAP_LENGTHS), (bsz, c, MAX_GAPS), generator=gen)]
    lens = lens.clamp(max=length)
    starts = (torch.rand((bsz, c, MAX_GAPS), generator=gen) * (length - lens + 1)).floor().long()
    active = torch.ones(bsz, c, MAX_GAPS, dtype=torch.bool)
    active[..., 1:] = torch.rand((bsz, c, MAX_GAPS - 1), generator=gen) < 0.5
    t = torch.arange(length).view(1, 1, 1, length)
    seg = (t >= starts[..., None]) & (t < (starts + lens)[..., None]) & active[..., None]
    mask = seg.any(dim=2)                                   # [B,C,L]
    whole = torch.rand(bsz, generator=gen) < WHOLE_VAR_PROB
    if c > 1:
        var = torch.randint(c, (bsz,), generator=gen)
        mask[whole, var[whole], :] = True
    mask = mask.transpose(1, 2).to(obs.device).float() * obs
    # Guarantee each sample keeps at least one visible observation.
    visible = (obs * (1 - mask)).flatten(1).sum(1)
    if bool((visible == 0).any()):
        mask[visible == 0] = 0
    return mask
