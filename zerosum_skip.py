#!/usr/bin/env python3
"""Lemma 8.7, second part, and Corollary 8.8: zero-sums with round-skipping,
that is, the linearisation of two forward rounds.

Free cells S = {(y,0) : y in Y}, Y a subset of F_p^*, at the input of round r0.
Every row of A = M + 2I meets at most one cell of S (row (m,c) meets (y,0)
through M iff c = y, through the diagonal iff (m,c) = (y,0)), and each cell of
S meets p+1 rows.  Writing u = sigma(v) + kappa_r0, the fixed cells are chosen
so that (11) holds: for every i in S and every row m with A[m,i] != 0

        sum_{j not in S} A[m,j] u_j = - A[m,i] kappa_r0(i),

which is k(p+1) linear equations in the p^2 - k values u_j.  Then the output
of round r0 in row m is A[m,i] sigma(v_i), the S-box of round r0+1 returns
A[m,i]^{-1} v_i, and two rounds are affine in the free cells.  Hence s forward
rounds have degree (p-2)^(s-2) in the free cells.

Checked, with the cubic constants of Construction 8.1 and alpha = 2:
  1. solvability of (11) for Y = {1..k}, every k <= p-1, and the exact
     affinity of two rounds on random inputs whenever it is solvable;
  2. forward zero-sums over 3 and 4 rounds, chosen against random fixed
     values (the forward rows of Table 6 marked "chosen");
  3. inside-out zero-sums set at the input of round 4, both ends (the
     inside-out rows marked "chosen");
  4. solvability for k = p-2 and k = p-1 at the primes of Table 3, and the
     six rounds within 2^256 states quoted after Corollary 8.8;
  5. the reach at p = 23 of Corollary 8.8: four rounds with k = 1, five and
     six with k = 21, seven with k = 421, and none for eight with k <= 529.

Runtime about 6 minutes.
"""
import itertools
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modp import radon_layer, constants, inv_layer, sbox_table, fwd, bwd, chunk_sum, cell, solve_mod  # noqa: E402

rng = np.random.default_rng(5)
ALPHA = 2


class Inst:
    def __init__(self, p, R=16, need_inv=True):
        self.p = p
        self.n = p * p
        self.A = radon_layer(p, ALPHA)
        self.Ai = inv_layer(p, self.A) if need_inv else None
        self.K = constants(p, R)
        self.T = sbox_table(p)


def free_cells(I, Y):
    return [cell(I.p, y, 0) for y in Y]


def linearise(I, Y, r0, rng=None):
    """A base state (fixed cells) realising the linearisation at round r0, or None."""
    p, n, A, K, T = I.p, I.n, I.A, I.K, I.T
    S = free_cells(I, Y)
    Sset = set(S)
    fixed = [j for j in range(n) if j not in Sset]
    rows, rhs = [], []
    for i in S:
        for m in np.nonzero(A[:, i])[0]:
            others = [s for s in S if s != i and A[m, s]]
            assert not others, "a row meets two free cells"
            rows.append(A[m, fixed])
            rhs.append((-A[m, i] * K[r0][i]) % p)
    u = solve_mod(np.array(rows), np.array(rhs), p, rng)
    if u is None:
        return None, len(rows)
    base = np.zeros(n, dtype=np.int64)
    base[fixed] = T[(u - K[r0][fixed]) % p]
    return base, len(rows)


def states(I, base, S, X):
    St = np.tile(base, (X.shape[0], 1))
    St[:, S] = X
    return St


def affine_check(I, base, Y, r0, trials=200):
    p, k = I.p, len(Y)
    S = free_cells(I, Y)
    f = lambda X: fwd(states(I, base, S, X), p, I.A, I.K, r0, 2)
    X1 = rng.integers(0, p, size=(trials, k))
    X2 = rng.integers(0, p, size=(trials, k))
    c = rng.integers(0, p, size=(trials, 1))
    Z = np.zeros((1, k), dtype=np.int64)
    f0 = f(Z)
    add = (f(X1) + f(X2) - f((X1 + X2) % p) - f0) % p
    sca = (f((c * X1) % p) - f0 - c * (f(X1) - f0)) % p
    return not add.any() and not sca.any()


def fsum(I, base, S, r0, s):
    return not chunk_sum(I.p, I.n, S, base, lambda St: fwd(St, I.p, I.A, I.K, r0, s)).any()


def bsum(I, base, S, r0, s):
    return not chunk_sum(I.p, I.n, S, base, lambda St: bwd(St, I.p, I.Ai, I.K, r0 - 1, s)).any()


def forward_experiment(I, s, k, r0=0, n_choices=6, n_controls=20):
    """Chosen fixed values: fraction of (Y, solution) choices giving a zero sum
    after s rounds; random fixed values on the same cells: fraction of zero sums."""
    p = I.p
    Ys = list(itertools.combinations(range(1, p), k))
    rng.shuffle(Ys)
    Ys = Ys[:n_choices]
    ok = tot = 0
    unsolv = 0
    for Y in Ys:
        for _ in range(2):
            base, _ = linearise(I, Y, r0, rng)
            if base is None:
                unsolv += 1
                break
            tot += 1
            ok += fsum(I, base, free_cells(I, Y), r0, s)
    ctrl = 0
    for t in range(n_controls):
        Y = Ys[t % len(Ys)]
        base = rng.integers(0, p, size=I.n)
        ctrl += fsum(I, base, free_cells(I, Y), r0, s)
    return ok, tot, unsolv, ctrl, n_controls


def insideout_experiment(I, rb, rf, k, r0=4, n_choices=4, n_controls=10):
    p = I.p
    Ys = list(itertools.combinations(range(1, p), k))
    rng.shuffle(Ys)
    Ys = Ys[:n_choices]
    both = tot = 0
    for Y in Ys:
        base, _ = linearise(I, Y, r0, rng)
        if base is None:
            continue
        tot += 1
        S = free_cells(I, Y)
        both += fsum(I, base, S, r0, rf) and bsum(I, base, S, r0, rb)
    ctrl = 0
    for t in range(n_controls):
        S = free_cells(I, Ys[t % len(Ys)])
        base = rng.integers(0, p, size=I.n)
        ctrl += fsum(I, base, S, r0, rf) and bsum(I, base, S, r0, rb)
    return both, tot, ctrl, n_controls


def main():
    print("=" * 78)
    print("1. Solvability of the linearisation system and affinity of two rounds")
    print("=" * 78)
    for p in (5, 7, 11, 13, 23):
        I = Inst(p, need_inv=False)
        res = []
        for k in range(1, p):
            Y = tuple(range(1, k + 1))
            base, neq = linearise(I, Y, 0, rng)
            aff = affine_check(I, base, Y, 0) if base is not None else None
            res.append((k, neq, I.n - k, base is not None, aff))
        kmax = max(k for k, _, _, s, _ in res if s)
        print(f"  p={p:2d}: largest k (Y = 1..k) with a solvable system: {kmax}; "
              f"two rounds exactly affine whenever solvable: {all(a for _, _, _, s, a in res if s)}")
        for k, neq, nunk, s, a in res:
            if k >= p - 3 or k <= 2:
                print(f"         k={k:2d}: {neq} equations, {nunk} unknowns, solvable {s}, affine {a}")
    print()

    print("=" * 78)
    print("2. Forward zero-sums from the input (r0 = 0), chosen vs random fixed values")
    print("=" * 78)
    print("   p  rounds  k  pred(k)  chosen: zero/total (unsolvable)  random: zero/total")
    for p, s, ks in ((5, 3, [1]), (7, 3, [1]), (11, 3, [1]), (13, 3, [1]), (23, 3, [1]),
                     (5, 4, [1, 2, 3]), (7, 4, [3, 4, 5])):
        I = Inst(p, need_inv=False)
        for k in ks:
            pred = next(kk for kk in range(1, 10 ** 6) if (p - 2) ** (s - 2) < kk * (p - 1))
            ok, tot, uns, c, nc = forward_experiment(I, s, k)
            print(f"  {p:2d}  {s:6d}  {k}  {pred:7d}  {ok:3d}/{tot:<3d} ({uns})"
                  f"                        {c:3d}/{nc}")
    print()

    print("=" * 78)
    print("3. Inside-out zero-sums, set at the input of round 4, both ends")
    print("=" * 78)
    print("   p  rb+rf  k  pred(k)  chosen: both zero/total  random: both zero/total")
    for p, rb, rf, ks in ((5, 1, 3, [1]), (7, 1, 3, [1]), (11, 1, 3, [1]), (23, 1, 3, [1]),
                          (5, 2, 4, [2, 3]), (7, 2, 4, [4, 5])):
        I = Inst(p)
        for k in ks:
            pred = next(kk for kk in range(1, 10 ** 6)
                        if (p - 2) ** rb < kk * (p - 1) and (p - 2) ** (rf - 2) < kk * (p - 1))
            b, t, c, nc = insideout_experiment(I, rb, rf, k)
            print(f"  {p:2d}  {rb}+{rf}    {k}  {pred:7d}  {b:3d}/{t:<3d}"
                  f"                  {c:3d}/{nc}")
    print()

    print("=" * 78)
    print("4. Six rounds (2 back, 4 forward with skipping) within 2^256 at the primes of Table 3")
    print("=" * 78)
    for p in (11, 17, 23, 29, 41, 47):
        k = next(kk for kk in range(1, 10 ** 6) if (p - 2) ** 2 < kk * (p - 1))
        I = Inst(p, need_inv=False)
        Y = tuple(range(1, k + 1))
        base, neq = linearise(I, Y, 0, rng)
        aff = affine_check(I, base, Y, 0, trials=40) if base is not None else None
        base2, neq2 = linearise(I, tuple(range(1, k + 2)), 0, rng) if k + 1 <= p - 1 else (None, 0)
        print(f"  p={p:2d}: k = {k} (= p-2: {k == p - 2}), {neq} equations in {I.n - k} unknowns, "
              f"solvable {base is not None}, two rounds affine {aff}; k+1 solvable {base2 is not None}; "
              f"data p^k = 2^{k * math.log2(p):.1f}")
    print()

    print("=" * 78)
    print("5. Reach at p = 23 (degree conditions; skipping needs k <= p-1 cells in column 0)")
    print("=" * 78)
    p = 23
    for R in range(4, 9):
        best = None
        for rb in range(0, R + 1):
            rf = R - rb
            for k in range(1, 530):
                okb = (p - 2) ** rb < k * (p - 1)
                okf_plain = (p - 2) ** (rf - 1) < k * (p - 1)
                okf_skip = rf >= 2 and k <= p - 2 and (p - 2) ** (rf - 2) < k * (p - 1)
                okf_single = rf == 3 and k >= 2  # Proposition 8.6 with a column-0 pair
                if okb and (okf_plain or okf_skip or okf_single):
                    if best is None or k < best[0]:
                        best = (k, rb, rf)
                    break
        if best:
            k, rb, rf = best
            print(f"  {R} rounds: least k = {k} at (rb, rf) = ({rb}, {rf}), data 23^{k} = 2^{k * math.log2(23):.1f}")
        else:
            print(f"  {R} rounds: not reachable with k <= 529")


if __name__ == "__main__":
    main()
