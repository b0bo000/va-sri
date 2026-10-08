"""CPU property checks for VA-SRI.  python -m scripts.va_sri.checks"""
from __future__ import annotations

import torch

from scripts.va_sri.masks import GAP_LENGTHS, gsm_mask
from scripts.va_sri.model import VASRI, load_shared, shared_state

torch.set_num_threads(2)
failures = []


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        failures.append(name)


def batch(c, b=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, 96, c, generator=g)
    m = torch.ones(b, 96, c)
    for i in range(b):
        for j in range(c):
            s = int(torch.randint(0, 72, (1,), generator=g))
            m[i, s:s + 24, j] = 0
    return torch.cat([x * m, m], dim=-1), x, m


x_cat, x, m = batch(7)
off = VASRI(7, seed=42, vga="off").eval()
beta = VASRI(7, seed=42, vga="beta").eval()
full = VASRI(7, seed=42, vga="beta_g").eval()
with torch.no_grad():
    ref = off.base.forward_recon_refine(x_cat)[1]
    y_off = off(x_cat)
    y_beta, y_full = beta(x_cat), full(x_cat)
check("vga=off equals 09-27 base forward exactly", torch.equal(ref, y_off))
check("base parameters identical across VGA modes",
      all(torch.equal(a, b) for a, b in zip(off.base.state_dict().values(), full.base.state_dict().values())))
check("zero-init beta reproduces base", torch.allclose(y_beta, y_off, atol=1e-6))
check("zero-init beta+G reproduces base", torch.allclose(y_full, y_off, atol=1e-6))
check("observed entries preserved", torch.equal(y_full[m.bool()], x_cat[..., :7][m.bool()]))

full.train()
loss = (full(x_cat) - x).pow(2).mul(1 - m).mean()
loss.backward()
check("beta receives gradient", full.vga_beta.grad is not None and full.vga_beta.grad.abs().sum() > 0)
check("G receives gradient", full.vga_g.grad is not None and full.vga_g.grad.abs().sum() > 0)
with torch.no_grad():
    full.vga_beta.fill_(1.0)
full.eval()
with torch.no_grad():
    check("nonzero beta changes output", not torch.allclose(full(x_cat), y_off, atol=1e-5))

vis = off.patch_visibility(m)
check("patch visibility in [0,1] with shape [B,C,P]",
      vis.shape == (4, 7, off.base.num_patches) and bool((vis >= 0).all() and (vis <= 1).all()))

big = VASRI(40, seed=1, vga="beta_g")
xb, tb, mb = batch(40, b=2)
big.train()
(big(xb) - tb).pow(2).mul(1 - mb).mean().backward()
check("low-rank G for C=40: V receives gradient", big.vga_v.grad.abs().sum() > 0)
check("low-rank prior is exactly zero at init",
      torch.equal(VASRI(40, seed=1, vga="beta_g").pair_prior(0), torch.zeros(4, 40, 40)))

g = torch.Generator().manual_seed(0)
obs = torch.ones(256, 96, 7); obs[:, 10:13, 2] = 0
t = gsm_mask(obs, g)
check("GSM targets only observed entries", bool((t * (1 - obs)).sum() == 0))
check("GSM keeps visible values in every sample", bool(((obs * (1 - t)).flatten(1).sum(1) > 0).all()))
per_var = t.sum(1)
check("GSM masks every variable in most samples", float((per_var > 0).float().mean()) > 0.95)
check("GSM includes whole-variable gaps", bool((per_var == 96).any()))
check("GSM shortest gap present", bool((per_var == min(GAP_LENGTHS)).any()))

pre = VASRI(7, seed=3, identity=False, vga="off")
pre.base.out_channels = 21
xc, _, _ = batch(21, b=2)
check("trunk runs on C=21 after switching out_channels", pre.compose_stage1(xc).shape == (2, 96, 21))
state = shared_state(pre)
check("shared state excludes identity/VGA/refiner/projector",
      not any(k.startswith(("base.input_dropout.embedding", "vga_", "base.temporal_refine",
                            "base.proj")) for k in state))
target = VASRI(21, seed=4, identity=True, vga="beta_g")
loaded = load_shared(target, state)
check("pretrained trunk loads into C=21 identity+VGA model",
      len(loaded) == len(state) and all(torch.equal(target.state_dict()[k], v) for k, v in state.items()))
check("identity embedding stays zero after load",
      torch.equal(target.base.input_dropout.embedding, torch.zeros_like(target.base.input_dropout.embedding)))

print(f"\n{len(failures)} failures")
raise SystemExit(1 if failures else 0)
