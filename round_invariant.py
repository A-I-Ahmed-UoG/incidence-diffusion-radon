#!/usr/bin/env python3
"""Propositions 8.3(iii) and 8.4, and the criterion of Beierle et al.

1. Proposition 8.3(iii): the witness of the proof, w = e_u - e_iota(u) with
   u = (0,0) and q = (2,0), has (Aw)_q = 1 and (Aw)_iota(q) = 0 for every
   alpha, at every prime p <= 31; so Aw is not iota-odd.
2. p = 7, which p = 2 (mod 3) excludes: kappa_r takes values in {0, +-1}, so
   the line it spans is invariant under rho_r.
3. dim W_A(D), D the differences of the round constants: 3 for the constants
   of Construction 8.1 at p in {5, 11, 17, 23}, 34 for random constants, and
   the threshold C(p,2) on the number of random differences at p in {5, 7, 11}.
4. Proposition 8.4: W_r = {g(z) : g odd}, z = y + t(x+1), t = r+1, at
   p in {5, 11, 17, 23} and r <= 9: the dimension, W_r in ker M,
   kappa_r in W_r, sigma(W_r) = W_r, rho_r(W_r) in W_r,
   rho_{r+1}(W_r) in W_r + alpha kappa_{r+1}, W_r cap W_{r+1} = 0 (on random
   elements, and exhaustively over W_0 at p = 5), and the dimension of the
   affine span of the image of W_r after 0..4 rounds.
5. The affine dimensions 11, 11, 11, 231, 526 at p = 23 again, from 2000
   samples at r = 0 and 5 and 1500 at r = 9.

Runtime about a minute.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modp import radon, cell, rank_mod, constants, sbox_table  # noqa: E402

rng = np.random.default_rng(20261020)
ALPHA = 2


def iota(p, y, x):
    return (-y) % p, (-x - 2) % p


def part2():
    print("1. Proposition 8.3(iii): A(e_u - e_iota(u)) is not iota-odd, u = (0,0), q = (2,0)")
    for p in (5, 7, 11, 13, 17, 19, 23, 29, 31):
        M = radon(p)
        A = (M + ALPHA * np.eye(p * p, dtype=np.int64)) % p
        w = np.zeros(p * p, dtype=np.int64)
        w[cell(p, 0, 0)] = 1
        w[cell(p, *iota(p, 0, 0))] = p - 1
        res = {}
        for al in range(p):
            Aa = (M + al * np.eye(p * p, dtype=np.int64)) % p
            z = Aa @ w % p
            q = cell(p, 2, 0)
            iq = cell(p, *iota(p, 2, 0))
            res[al] = (int(z[q]), int(z[iq]))
        ok = all(v == (1, 0) for v in res.values())
        # also: the iota-odd space is not A-invariant (direct check, alpha = 2)
        z = A @ w % p
        odd = all((z[cell(p, *iota(p, y, x))] + z[cell(p, y, x)]) % p == 0
                  for y in range(p) for x in range(p))
        print(f"   p={p:2d}: (Aw)_q, (Aw)_iota(q) = (1, 0) for every alpha: {ok};"
              f" Aw iota-odd (alpha=2): {odd}")
    print()


def W_basis(p, r):
    t = r + 1
    B = []
    for j in range(1, p - 1, 2):
        g = np.array([pow(z, j, p) for z in range(p)], dtype=np.int64)
        v = np.zeros(p * p, dtype=np.int64)
        for y in range(p):
            for x in range(p):
                v[cell(p, y, x)] = g[(y + t * (x + 1)) % p]
        B.append(v)
    return np.array(B)


def in_span(B, v, p):
    return rank_mod(np.vstack([B, v]), p) == rank_mod(B, p)


def rho(v, p, A, K, r, T):
    return ((T[v] + K[r]) % p) @ A.T % p


def affine_dim(X, p):
    D = (X[1:] - X[0]) % p
    return rank_mod(D, p)


def part3():
    print("4. Proposition 8.4: the per-round subspaces W_r")
    for p, R, nsamp in ((5, 12, 200), (11, 12, 400), (17, 12, 700), (23, 12, 800)):
        M = radon(p)
        n = p * p
        A = (M + ALPHA * np.eye(n, dtype=np.int64)) % p
        K = constants(p, R + 4)
        T = sbox_table(p)
        dims = []
        print(f"   p={p}:")
        for r in range(R - 2):
            B = W_basis(p, r)
            d = rank_mod(B, p)
            ker = not (M @ B.T % p).any()
            kin = in_span(B, K[r], p)
            X = (rng.integers(0, p, size=(nsamp, B.shape[0])) @ B) % p
            X[0] = 0
            sig = all(in_span(B, T[x], p) for x in X[:30])
            Y1 = np.array([rho(x, p, A, K, r, T) for x in X])
            r1 = all(in_span(B, y, p) for y in Y1[:30])
            Y2 = np.array([rho(y, p, A, K, r + 1, T) for y in Y1])
            shift = (Y2 - ALPHA * K[r + 1]) % p
            r2 = all(in_span(B, y, p) for y in shift[:30])
            Y3 = np.array([rho(y, p, A, K, r + 2, T) for y in Y2])
            Y4 = np.array([rho(y, p, A, K, r + 3, T) for y in Y3])
            dd = [affine_dim(Z, p) for Z in (X, Y1, Y2, Y3, Y4)]
            Bn = W_basis(p, r + 1)
            inter = d + rank_mod(Bn, p) - rank_mod(np.vstack([B, Bn]), p)
            dims.append(dd)
            if r < 2 or p == 23:
                print(f"     r={r}: dim W_r={d} ((p-1)/2={(p - 1) // 2}), W_r in ker M {ker}, "
                      f"kappa_r in W_r {kin}, sigma(W_r)=W_r {sig}, rho_r(W_r) in W_r {r1}, "
                      f"rho_(r+1)(W_r) in W_r + alpha kappa_(r+1) {r2}, "
                      f"dim(W_r cap W_(r+1))={inter}; affine dims after 0..4 rounds {dd}")
        d3 = [x[3] for x in dims]
        d4 = [x[4] for x in dims]
        print(f"     third-round affine dim over r=0..{R - 3}: min {min(d3)}, max {max(d3)};"
              f" fourth: min {min(d4)}, max {max(d4)} (samples {nsamp}, state {n})")
        if p == 5:
            # exhaustive check of sigma(W_r) = W_r and rho_r(W_r) = W_r at p = 5
            B = W_basis(p, 0)
            allv = np.array([(np.array(c) @ B) % p for c in np.ndindex(*(p,) * B.shape[0])])
            ok = all(in_span(B, T[x], p) and in_span(B, rho(x, p, A, K, 0, T), p) for x in allv)
            print(f"     exhaustive over all {len(allv)} elements of W_0: sigma and rho_0 stay in W_0: {ok}")
    print()


def part4():
    print("2. p = 7 (excluded by p = 2 mod 3): kappa_r is signed and spans a rho_r-invariant line")
    p = 7
    M = radon(p)
    n = p * p
    A = (M + ALPHA * np.eye(n, dtype=np.int64)) % p
    K = constants(p, 12)
    T = sbox_table(p)
    for r in range(3):
        vals = set(K[r].tolist())
        inv = all(in_span(K[r][None, :], rho(lam * K[r] % p, p, A, K, r, T), p) for lam in range(p))
        print(f"   r={r}: values of kappa_r {sorted(vals)}; rho_r(<kappa_r>) in <kappa_r>: {inv}")
    print()


def W_AD(A, D, p):
    """A-invariant span of D, via the Krylov vectors d, Ad, A^2 d, ... (iterated to closure)."""
    V = [d % p for d in D]
    rk = rank_mod(np.array(V), p)
    cur = list(V)
    while True:
        cur = [A @ v % p for v in cur]
        V2 = V + cur
        rk2 = rank_mod(np.array(V2), p)
        V = V2
        if rk2 == rk:
            return rk
        rk = rk2


def part5():
    print("3. Beierle-Canteaut-Leander-Rotella: dim W_A(D), D = differences of round constants")
    for p in (5, 11, 17, 23):
        M = radon(p)
        n = p * p
        A = (M + ALPHA * np.eye(n, dtype=np.int64)) % p
        R = 12
        K = constants(p, R)
        D = [(K[r] - K[0]) % p for r in range(1, R)]
        Dc = [A @ d % p for d in D]  # differences of the constants as added after the layer
        d_con = W_AD(A, D, p)
        d_con2 = W_AD(A, Dc, p)
        Kr = rng.integers(0, p, size=(R, n))
        Dr = [(Kr[r] - Kr[0]) % p for r in range(1, R)]
        d_rand = W_AD(A, Dr, p)
        print(f"   p={p}: R=12: constants of Construction 8.1: dim W_A(D) = {d_con} (with A kappa: {d_con2});"
              f" random constants {d_rand}; bounds 4(R-1) = {4 * (R - 1)}, 3(R-1)+1 = {3 * (R - 1) + 1};"
              f" C(p,2) = {math.comb(p, 2)}, n = {n}")
    for p in (5, 7, 11):
        M = radon(p)
        n = p * p
        A = (M + ALPHA * np.eye(n, dtype=np.int64)) % p
        c = math.comb(p, 2)
        for m in (c - 1, c):
            Dr = list(rng.integers(0, p, size=(m, n)))
            print(f"   p={p}: {m} random differences: dim W_A(D) = {W_AD(A, Dr, p)} of {n}")
        # minimal polynomial degree of A (cyclic spaces have dim <= that)
        v = rng.integers(0, p, size=n)
        kr = [v]
        for _ in range(6):
            kr.append(A @ kr[-1] % p)
        print(f"   p={p}: dim of A-cyclic space of a random vector = {rank_mod(np.array(kr), p)}")
    print()


def part6():
    print("5. p = 23: affine dimensions of the image of W_r with more samples")
    r6 = np.random.default_rng(7)
    p = 23
    n = p * p
    A = (radon(p) + ALPHA * np.eye(n, dtype=np.int64)) % p
    K = constants(p, 16)
    T = sbox_table(p)
    for r, ns in ((0, 2000), (5, 2000), (9, 1500)):
        B = W_basis(p, r)
        X = (r6.integers(0, p, size=(ns, B.shape[0])) @ B) % p
        Y = X
        dims = [affine_dim(X, p)]
        for j in range(4):
            Y = np.array([rho(y, p, A, K, r + j, T) for y in Y])
            dims.append(affine_dim(Y, p))
        print(f"   p=23 r={r} samples={ns}: affine dims after 0..4 rounds {dims}")


if __name__ == "__main__":
    part2()
    part4()
    part5()
    part3()
    part6()
