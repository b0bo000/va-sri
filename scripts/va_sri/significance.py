"""Paired significance tests for the main comparisons (no training).

For each dataset and metric, compares VA-SRI with tuned SAITS / SAITS + fill on the
same seeds (same masks). Reports mean paired difference, wins, a two-sided paired
t-test and an exact two-sided sign-flip permutation test (2^5 = 32 sign patterns,
so the smallest attainable p is 1/16 = 0.0625 with five seeds).

  python -m scripts.va_sri.significance
"""
from __future__ import annotations

import itertools
import math
import statistics as st

from scripts.va_sri.render_paper import DS, NAME, S5, G_VA, G_SP, G_SF


def t_pvalue(d):
    n = len(d)
    m, s = st.mean(d), st.stdev(d)
    if s == 0:
        return 0.0
    t = m / (s / math.sqrt(n))
    # two-sided p from Student t with n-1 df (regularized incomplete beta)
    df = n - 1
    x = df / (df + t * t)
    return betainc(df / 2, 0.5, x)


def betainc(a, b, x, it=200):
    """Regularized incomplete beta I_x(a, b) via continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x > (a + 1) / (a + b + 2):
        return 1 - betainc(b, a, 1 - x, it)
    f, c, d = 1.0, 1.0, 0.0
    for i in range(it):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1 + num * d
        d = 1 / d if abs(d) > 1e-30 else 1e30
        c = 1 + num / c if abs(c) > 1e-30 else 1e30
        f *= c * d
        if abs(c * d - 1) < 1e-12:
            break
    return math.exp(lbeta) * (f - 1) / a


def perm_pvalue(d):
    obs = abs(sum(d))
    count = sum(abs(sum(s * x for s, x in zip(signs, d))) >= obs - 1e-15
                for signs in itertools.product((1, -1), repeat=len(d)))
    return count / 2 ** len(d)


def main():
    rows = []
    for base, g in (("SAITS-blocksup", G_SP), ("SAITS + fill", G_SF)):
        for metric in ("std_mse", "std_mae", "mse", "mae"):
            for d in DS:
                a = [G_VA(d)(s)[metric] for s in S5]
                b = [g(d)(s)[metric] for s in S5]
                rel = [100 * (x / y - 1) for x, y in zip(a, b)]
                rows.append((base, metric, NAME[d], st.mean(rel), sum(r < 0 for r in rel),
                             t_pvalue(rel), perm_pvalue(rel)))
    print(f"{'baseline':15s} {'metric':8s} {'dataset':8s} {'mean %':>8s} wins  t-test p  perm p")
    for r in rows:
        print(f"{r[0]:15s} {r[1]:8s} {r[2]:8s} {r[3]:+8.1f} {r[4]}/5  {r[5]:8.4f}  {r[6]:.4f}")
    return rows


if __name__ == "__main__":
    main()
