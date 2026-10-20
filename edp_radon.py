#!/usr/bin/env python3
"""Section 9: exact two-round EDP of the Radon layer A = M + alpha I, for
p in {5, 7} and alpha in {2, 3}, over all inputs of weight one and all outputs
(the keyed model of Definition 9.1 and formula (14)).

For a = a_i e_i the intermediate difference is c e_i with c != 0, the
difference after the layer is c A e_i, and EDP(a,b) vanishes unless
supp(b) = supp(A e_i); then

    EDP(a,b) = sum_c DP(a_i, c) prod_{j in supp A e_i} DP(c A_{j,i}, b_j).

The value depends only on a_i and on the multiset of non-zero entries of the
column, so the maximum over b is taken exactly, one column type at a time.
Ratios are given to the bound (delta/p)^(B-1) of Theorem 9.3, to the trail
bound (delta/p)^B and to the per-pair bound (delta/p)^(s - dim U), where
s = 1 + wt(b) and dim U = 1.  B = p + 1 by Corollary 4.8.

Runtime a few seconds.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modp import radon_layer, sbox_table  # noqa: E402


def ddt(p):
    T = sbox_table(p)
    D = np.zeros((p, p))
    for a in range(p):
        for x in range(p):
            D[a, (T[(x + a) % p] - T[x]) % p] += 1
    return D / p


def max_edp_column(D, p, ai, entries):
    """max over b (support = column support) of sum_c D[ai,c] prod_j D[c e_j, b_j]."""
    # a sum over c of rank-one tensors on the coordinates of the column
    total = None
    for c in range(1, p):
        f = D[ai, c]
        if f == 0:
            continue
        t = np.array([f])
        for e in entries:
            t = np.outer(t, D[(c * e) % p, 1:]).ravel()   # b_j != 0
        total = t if total is None else total + t
    return total.max()


def main():
    print("Exact two-round EDP of A = M + alpha I, weight-one inputs, all outputs")
    for p in (5, 7):
        D = ddt(p)
        delta = int(round(p * D[1:].max()))
        for alpha in (2, 3):
            A = radon_layer(p, alpha)
            B = p + 1                     # Corollary 4.8, alpha not in {0, 1, -1}
            types = {}
            for i in range(p * p):
                col = tuple(sorted(int(v) for v in A[:, i] if v % p))
                types.setdefault(col, []).append(i)
            best = (0.0, None)
            per_pair = 0.0
            for col, cells in types.items():
                for ai in range(1, p):
                    m = max_edp_column(D, p, ai, col)
                    if m > best[0]:
                        best = (m, (col, ai, cells[0]))
                    per_pair = max(per_pair, m / (delta / p) ** len(col))
            m, (col, ai, i0) = best
            thm = (delta / p) ** (B - 1)
            trail = (delta / p) ** B
            print(f"  p={p}, alpha={alpha}: delta={delta}, B={B}, column types "
                  f"{ {k: len(v) for k, v in types.items()} }")
            print(f"     max EDP = {m:.6g} (a_i = {ai}, column entries {col}, wt(b) = {len(col)})")
            print(f"     / (delta/p)^(B-1) = {m / thm:.3f},  / (delta/p)^B = {m / trail:.3f},"
                  f"  max EDP / (delta/p)^(s-dim U) = {per_pair:.3f}")


if __name__ == "__main__":
    main()
