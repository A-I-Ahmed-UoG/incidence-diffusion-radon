#!/usr/bin/env python3
"""The paragraph after Proposition 8.11: signed eigenvectors and the search
for invariant lines.

1. chi(a y + b x + c), chi the quadratic character, lies in ker M for every
   (a,b) != (0,0) and every c, at p = 5, 7; for chi(y + 2x) also
   A chi = alpha chi, sigma(chi) = chi, values in {0, +-1}, support p(p-1).
2. A vector v is signed if all its non-zero entries have the same square.
   For each tested signed v we record j(v), the largest j such that
   v, Av, ..., A^j v are all signed (capped at 6), separately for v in ker M
   and v outside it.  Supports of size 2 and 3 exhaustively (all sign
   patterns, scalar 1), and 20 000 random signed vectors (random support,
   random signs).  No v outside ker M reaches j = 3.
3. The eigenvector of M for -1 (of A for alpha - 1) is the indicator
   1[y != 0], which is signed.

Runtime a few seconds.
"""
import itertools
import math
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modp import radon, cell, rank_mod, rref_mod, sbox_table  # noqa: E402

rng = np.random.default_rng(810)
ALPHA = 2
CAP = 6


def chi_table(p):
    sq = {(x * x) % p for x in range(1, p)}
    return np.array([0] + [1 if z in sq else p - 1 for z in range(1, p)], dtype=np.int64)


def func_of_linear(p, g, a, b, c):
    v = np.zeros(p * p, dtype=np.int64)
    for y in range(p):
        for x in range(p):
            v[cell(p, y, x)] = g[(a * y + b * x + c) % p]
    return v


def part1():
    print("1. chi(ay+bx+c) as signed eigenvectors")
    for p in (5, 7):
        M = radon(p)
        A = (M + ALPHA * np.eye(p * p, dtype=np.int64)) % p
        T = sbox_table(p)
        ch = chi_table(p)
        v = func_of_linear(p, ch, 1, 2, 0)
        print(f"   p={p}: chi(y+2x): Mv=0 {not (M @ v % p).any()}, "
              f"Av=alpha v {not ((A @ v - ALPHA * v) % p).any()}, "
              f"sigma(v)=v {not ((T[v] - v) % p).any()}, "
              f"values in {{0,+-1}} {set(v.tolist()) <= {0, 1, p - 1}}, "
              f"support {np.count_nonzero(v)} = p(p-1) = {p * (p - 1)}")
        allok = True
        for a in range(p):
            for b in range(p):
                if a == 0 and b == 0:
                    continue
                for c in range(p):
                    w = func_of_linear(p, ch, a, b, c)
                    allok &= not (M @ w % p).any()
        print(f"   p={p}: chi(ay+bx+c) in ker M for all (a,b) != 0 and all c: {allok}")
        # a difference of two row or column indicators has support 2p, so chi is not one
        print(f"   p={p}: support of a row/column indicator difference is 2p = {2 * p}")
        # dim of the alpha-eigenspace of A equals dim ker M = C(p,2)
        print(f"   p={p}: dim ker M = {p * p - rank_mod(M, p)}, C(p,2) = {math.comb(p, 2)}")
    print()


def signed(v, p):
    nz = v[v % p != 0] % p
    if len(nz) == 0:
        return True
    sq = (nz * nz) % p
    return bool((sq == sq[0]).all())


def depth(v, A, p):
    j = 0
    w = v.copy()
    while j < CAP:
        w = A @ w % p
        if not signed(w, p) or not w.any():
            return j
        j += 1
    return j


def search():
    for p in (5, 7):
        n = p * p
        M = radon(p)
        A = (M + ALPHA * np.eye(n, dtype=np.int64)) % p
        stats = {"ker": Counter(), "out": Counter()}
        for size in (2, 3):
            for supp in itertools.combinations(range(n), size):
                for signs in itertools.product((1, p - 1), repeat=size - 1):
                    v = np.zeros(n, dtype=np.int64)
                    v[list(supp)] = (1,) + signs
                    key = "ker" if not (M @ v % p).any() else "out"
                    stats[key][depth(v, A, p)] += 1
        for _ in range(20000):
            size = int(rng.integers(2, n + 1))
            supp = rng.choice(n, size=size, replace=False)
            v = np.zeros(n, dtype=np.int64)
            v[supp] = rng.choice([1, p - 1], size=size)
            key = "ker" if not (M @ v % p).any() else "out"
            stats[key][depth(v, A, p)] += 1
        print(f"p={p}: depth j (v, Av, ..., A^j v all signed) for signed v outside ker M: "
              f"{dict(sorted(stats['out'].items()))}; in ker M: {dict(sorted(stats['ker'].items()))}")
        # eigenvector of M for -1
        R, piv = rref_mod((M + np.eye(n, dtype=np.int64)) % p, p)
        free = [j for j in range(n) if j not in piv]
        assert len(free) == 1
        e = np.zeros(n, dtype=np.int64)
        e[free[0]] = 1
        for i, c in enumerate(piv):
            e[c] = (-R[i, free[0]]) % p
        assert not ((M @ e + e) % p).any()
        print(f"p={p}: eigenvector of M for -1 (A for alpha-1): signed {signed(e, p)}, "
              f"support {np.count_nonzero(e)}, values {sorted(set((e % p).tolist()))}")


def main():
    part1()
    search()


if __name__ == "__main__":
    main()
