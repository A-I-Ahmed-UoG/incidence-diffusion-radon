#!/usr/bin/env python3
"""Proposition 4.3 and Remark 4.4: ker M = GRM_p(p-2, 2), and the Jordan form
of M over F_p, from the ranks of its powers.

For each p in {3, 5, 7, 11}: the monomials y^i x^j with i + j <= p-2 lie in
ker M and span a space of dimension C(p,2) = dim ker M; M has p-1 nilpotent
blocks of size 3, (p-1)(p-2)/2 of size 2, none of size 1, and a single 1x1
block for the eigenvalue -1.  For the diagonal-free N of Proposition 4.10 the
same computation gives one block of size 4, p-2 of size 3 and (p-1)(p-2)/2 of
size 2.

Runtime a few seconds.
"""
import os
import sys
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import radon, rank  # noqa: E402
from diagfree import ident_N, nonsquare  # noqa: E402


def nilpotent_blocks(A, p, n):
    """Block sizes for the eigenvalue 0: with r_j = rank A^j, the number of
    blocks of size >= j is r_{j-1} - r_j."""
    r = [n]
    P = np.eye(n, dtype=np.int64)
    while True:
        P = P @ A % p
        r.append(rank(P, p))
        if r[-1] == r[-2]:
            break
    ge = [r[j - 1] - r[j] for j in range(1, len(r))]
    exact = {j: ge[j - 1] - (ge[j] if j < len(ge) else 0) for j in range(1, len(ge) + 1)}
    return r, {j: c for j, c in exact.items() if c}


def main():
    for p in [3, 5, 7, 11]:
        n = p * p
        M = radon(p)
        # the monomials y^i x^j, i + j <= p-2, as vectors on the cells (y,x)
        mons = []
        for i in range(p - 1):
            for j in range(p - 1 - i):
                mons.append([pow(y, i, p) * pow(x, j, p) % p for y in range(p) for x in range(p)])
        G = np.array(mons, dtype=np.int64)
        in_ker = not (M @ G.T % p).any()
        dimG = rank(G, p)
        dimK = n - rank(M, p)
        print(f"p={p:2d}: {len(mons)} monomials of degree <= p-2, all in ker M: {in_ker}; "
              f"dim GRM_p(p-2,2) = {dimG}, dim ker M = {dimK}, C(p,2) = {comb(p, 2)}")
        assert in_ker and dimG == dimK == comb(p, 2)

        r, blocks = nilpotent_blocks(M, p, n)
        I = np.eye(n, dtype=np.int64)
        rI = rank((M + I) % p, p)
        rI2 = rank(np.linalg.matrix_power(M + I, 2) % p, p)
        pred = {k: v for k, v in {3: p - 1, 2: (p - 1) * (p - 2) // 2}.items() if v}
        total = sum(k * v for k, v in blocks.items()) + (n - rI2)
        print(f"      M: ranks of powers {r}; nilpotent blocks {blocks} (predicted {pred}); "
              f"eigenvalue -1: rank(M+I)={rI}, rank(M+I)^2={rI2}, one 1x1 block: {n - rI == 1 and rI == rI2}; "
              f"sum of sizes {total} = p^2 {total == n}")
        assert blocks == pred and n - rI == 1 and rI == rI2 and total == n

        N = ident_N(p, nonsquare(p))
        rN, bN = nilpotent_blocks(N, p, n)
        predN = {k: v for k, v in {4: 1, 3: p - 2, 2: (p - 1) * (p - 2) // 2}.items() if v}
        print(f"      N: ranks of powers {rN}; nilpotent blocks {bN} (predicted {predN}); "
              f"sum {sum(k * v for k, v in bN.items())}")
        assert bN == predN
    print("\nALL KERNEL AND JORDAN-FORM CHECKS PASSED")


if __name__ == "__main__":
    main()
