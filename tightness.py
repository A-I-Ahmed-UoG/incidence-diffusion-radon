#!/usr/bin/env python3
"""Theorem 9.3 and Lemma 9.2: how tight the two-round bounds are.

Exact two-round EDP on small layers against three bounds:

    trail      (delta/p)^B
    Thm 9.3    (delta/p)^(B-1)
    per pair   (delta/p)^(s - dim U), s = wt(a) + wt(b), U = U(a,b) of (12)

The full transition matrix E = T P_L T (T the n-fold tensor power of the
normalised DDT, P_L the permutation u -> L u), as in tworound.py, for every
non-zero pair (a, b), for Cauchy MDS layers of size 3 over F_7, F_11, F_13 and
F_17 and for non-MDS circulant layers.  The Fano layer (n = 7) uses the exact
row computation of tworound2.py over the inputs of weight at most 3 and all
outputs, for N + I and N - I over F_5.  The largest per-pair ratio, 0.956,
is the circulant (1,1,0) over F_17.

Runtime about a minute.
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tworound as tr  # noqa: E402
import tworound2 as tr2  # noqa: E402
from modp import rank_mod  # noqa: E402


def dimU(L, p, asupp, bsupp):
    n = L.shape[0]
    out = [j for j in range(n) if j not in bsupp]
    if not out:
        return len(asupp)
    return len(asupp) - rank_mod(L[np.ix_(out, list(asupp))] % p, p)


def full(name, L, p):
    L = np.array(L, dtype=np.int64) % p
    n = L.shape[0]
    D = tr.ddt(p, tr.inverse_sbox(p))
    delta = tr.delta_of(D, p)
    B = tr.branch_number(L.tolist(), p, n)
    T = tr.tensor(D, n)
    V = tr.vectors(p, n)
    perm = np.array([tr.index(tuple(int(x) % p for x in (L @ np.array(u)) % p), p) for u in V])
    # (T P)[a, w] = T[a, u] with L u = w  ->  columns permuted
    TP = np.zeros_like(T)
    TP[:, perm] = T
    E = TP @ T
    masks = np.array([sum(1 << j for j in range(n) if v[j]) for v in V])
    q = delta / p
    best_pp, at_pp, viol = 0.0, None, 0
    nzi = np.nonzero(masks)[0]
    Esub = E[np.ix_(nzi, nzi)]
    maxE = Esub.max()
    ma, mb = masks[nzi], masks[nzi]
    for sa in range(1, 1 << n):
        rows = np.nonzero(ma == sa)[0]
        asupp = [j for j in range(n) if sa >> j & 1]
        for sb in range(1, 1 << n):
            cols = np.nonzero(mb == sb)[0]
            m = Esub[np.ix_(rows, cols)].max()
            bsupp = [j for j in range(n) if sb >> j & 1]
            d = dimU(L, p, asupp, bsupp)
            s = len(asupp) + len(bsupp)
            if d == 0:
                viol += m > 1e-15
                continue
            r = m / q ** (s - d)
            if r > best_pp:
                best_pp, at_pp = r, (len(asupp), len(bsupp), d)
    print(f"  {name:<34} p={p:2d} n={n} B={B} delta={delta}: max EDP/(d/p)^(B-1) = {maxE / q ** (B - 1):.3f},"
          f" /(d/p)^B = {maxE / q ** B:.3f}; max per-pair ratio = {best_pp:.3f} at (wa,wb,dimU)={at_pp};"
          f" pairs with U=0 and EDP>0: {viol}")
    return maxE / q ** (B - 1), maxE / q ** B, best_pp


def fano(alpha, p, maxwt):
    L = tr2.layer(alpha, p)
    n = 7
    if tr2.det_mod(L, p) == 0:
        print(f"  Fano N+{alpha}I over F_{p}: singular, skipped")
        return (0.0, 0.0, 0.0)
    D = tr2.ddt(p)
    delta = int(round(p * D[1:].max()))
    B = tr2.branch(L, p, n)
    q = delta / p
    masks = np.array([sum(1 << j for j, ch in enumerate(np.base_repr(i, p).zfill(n)) if ch != '0')
                      for i in range(p ** n)])
    best, best_pp, at = 0.0, 0.0, None
    cache = {}
    for a in itertools.product(range(p), repeat=n):
        wa = sum(1 for c in a if c)
        if wa == 0 or wa > maxwt:
            continue
        row = tr2.row_edp(list(a), L, D, p, n)
        row[0] = 0.0
        best = max(best, row.max())
        mx = np.zeros(1 << n)
        np.maximum.at(mx, masks, row)
        asupp = tuple(i for i in range(n) if a[i])
        for sb in np.nonzero(mx)[0]:
            bsupp = [j for j in range(n) if sb >> j & 1]
            key = (asupp, int(sb))
            if key not in cache:
                cache[key] = dimU(L, p, list(asupp), bsupp)
            d = cache[key]
            s = len(asupp) + len(bsupp)
            r = mx[sb] / q ** (s - d) if d else float("inf")
            if r > best_pp:
                best_pp, at = r, (wa, len(bsupp), d)
    print(f"  Fano 7_3, N+{alpha}I (diagonal ones)  p={p} B={B} delta={delta}, wt(a)<={maxwt}, all b:"
          f" max EDP/(d/p)^(B-1) = {best / q ** (B - 1):.3f}, /(d/p)^B = {best / q ** B:.3f};"
          f" max per-pair ratio = {best_pp:.3f} at {at}")
    return best / q ** (B - 1), best / q ** B, best_pp


def main():
    res = []
    print("Exact two-round EDP, all non-zero pairs")
    res.append(full("2-cell (1 1; 1 2)", [[1, 1], [1, 2]], 5))
    res.append(full("3-cell (1 1 1; 1 2 3; 1 3 2)", [[1, 1, 1], [1, 2, 3], [1, 3, 2]], 5))
    res.append(full("3x3 Cauchy MDS", tr.cauchy_mds(7, 3), 7))
    res.append(full("3x3 Cauchy MDS", tr.cauchy_mds(11, 3), 11))
    res.append(full("3x3 Cauchy MDS", tr.cauchy_mds(13, 3), 13))
    res.append(full("circulant (1,1,0)", [[1, 1, 0], [0, 1, 1], [1, 0, 1]], 7))
    res.append(full("circulant (1,1,0)", [[1, 1, 0], [0, 1, 1], [1, 0, 1]], 11))
    res.append(full("circulant (2,1,1)", [[2, 1, 1], [1, 2, 1], [1, 1, 2]], 5))
    res.append(full("circulant (2,1,1)", [[2, 1, 1], [1, 2, 1], [1, 1, 2]], 7))
    res.append(full("circulant 0/1 4x4", [[1, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1]], 7))
    res.append(full("3x3 Cauchy MDS", tr.cauchy_mds(17, 3), 17))
    res.append(full("circulant (1,1,0)", [[1, 1, 0], [0, 1, 1], [1, 0, 1]], 17))
    print("Fano configuration, exact rows")
    res.append(fano(1, 5, 3))
    res.append(fano(4, 5, 3))
    print(f"largest ratio to Thm 9.3 over these instances: {max(r[0] for r in res):.3f};"
          f" largest ratio to the trail bound: {max(r[1] for r in res):.3f};"
          f" largest per-pair ratio: {max(r[2] for r in res):.3f}")


if __name__ == "__main__":
    main()
