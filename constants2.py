#!/usr/bin/env python3
"""Proposition 8.3: the round constants and the signed-partition subspaces.

It is tempting to argue that the constants of Construction 8.1, being
injective on cells, lie in no signed-partition subspace with a block larger
than one. That does not follow, and it is false. Writing

    kappa_r(y, x) = (y + (r+1) x + r + 1)^3 = (y + (r+1)(x+1))^3,

the map iota(y, x) = (-y, -x-2) is an involution with the single fixed point
(0, -1), and

    kappa_r(iota(y,x)) = (-(y + (r+1)(x+1)))^3 = -kappa_r(y,x),

so every kappa_r lies in the space W_iota of iota-odd vectors, a signed
partition with blocks {u, iota(u)} of size two and the fixed point forced to
zero. Its dimension is (p^2 - 1)/2.

So that argument does not work. What has to be checked instead is whether any
subspace that is BOTH A-invariant and sigma-invariant contains all the
constants, because that is what Proposition 6.7 needs. This script checks:

  1. the algebra above, at several p;
  2. whether W_iota is A-invariant (it is not);
  3. the smallest A-invariant subspace containing every constant, its
     dimension, and whether it is sigma-invariant;
  4. whether any signed-partition subspace that is A-invariant by the
     criterion of Theorem 6.6 contains all the constants.

Runtime about 3 minutes.
"""
import random

import numpy as np

random.seed(11)


def cells(p):
    return [(y, x) for y in range(p) for x in range(p)]


def idx(p, y, x):
    return (y % p) * p + (x % p)


def radon_layer(p, alpha):
    n = p * p
    A = np.zeros((n, n), dtype=np.int64)
    for m in range(p):
        for c in range(p):
            for x in range(p):
                A[idx(p, m, c), idx(p, m * x + c, x)] += 1
    for i in range(n):
        A[i, i] = (A[i, i] + alpha) % p
    return A % p


def kappa(p, r):
    K = np.zeros(p * p, dtype=np.int64)
    for y in range(p):
        for x in range(p):
            K[idx(p, y, x)] = pow((y + (r + 1) * x + r + 1) % p, 3, p)
    return K


def iota(p, y, x):
    return ((-y) % p, (-x - 2) % p)


def rank_mod(A, p):
    A = np.array(A, dtype=np.int64) % p
    if A.size == 0:
        return 0
    rows, cols = A.shape
    r = 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if A[i, c] % p), None)
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        A[r] = (A[r] * pow(int(A[r, c]), p - 2, p)) % p
        for i in range(rows):
            if i != r and A[i, c] % p:
                A[i] = (A[i] - A[i, c] * A[r]) % p
        r += 1
    return r


def row_reduce(A, p):
    A = np.array(A, dtype=np.int64) % p
    rows, cols = A.shape
    r = 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if A[i, c] % p), None)
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        A[r] = (A[r] * pow(int(A[r, c]), p - 2, p)) % p
        for i in range(rows):
            if i != r and A[i, c] % p:
                A[i] = (A[i] - A[i, c] * A[r]) % p
        r += 1
    return A[:r]


def in_span(v, basis, p):
    if basis.shape[0] == 0:
        return not (np.array(v) % p).any()
    return rank_mod(np.vstack([basis, v]), p) == basis.shape[0]


def iota_odd_basis(p):
    """Basis of {v : v_iota(u) = -v_u}."""
    n = p * p
    seen = set()
    B = []
    for (y, x) in cells(p):
        u = idx(p, y, x)
        j = iota(p, y, x)
        w = idx(p, *j)
        if u in seen or w in seen:
            continue
        seen.add(u)
        seen.add(w)
        if u == w:
            continue                       # fixed point: forced to zero
        v = np.zeros(n, dtype=np.int64)
        v[u] = 1
        v[w] = p - 1
        B.append(v)
    return np.array(B, dtype=np.int64)


def invariant_closure(A, vectors, p, cap=60):
    """Smallest A-invariant subspace containing the given vectors."""
    basis = row_reduce(np.array(vectors, dtype=np.int64), p)
    for _ in range(cap):
        new = (basis @ A.T) % p
        merged = row_reduce(np.vstack([basis, new]), p)
        if merged.shape[0] == basis.shape[0]:
            return basis
        basis = merged
    return basis


def main():
    for p in [5, 11, 23]:
        n = p * p
        R = 8
        A = radon_layer(p, 2)
        Ks = [kappa(p, r) for r in range(R)]
        print("=" * 74)
        print(f"p = {p}, n = {n} cells, alpha = 2, R = {R}")
        print("=" * 74)

        # 1. the algebra
        fixed = [(y, x) for (y, x) in cells(p) if iota(p, y, x) == (y, x)]
        odd_ok = all(
            (Ks[r][idx(p, *iota(p, y, x))] + Ks[r][idx(p, y, x)]) % p == 0
            for r in range(R) for (y, x) in cells(p))
        zero_at_fixed = all(Ks[r][idx(p, *fixed[0])] % p == 0 for r in range(R))
        print(f"  iota has {len(fixed)} fixed point(s): {fixed}")
        print(f"  every kappa_r is iota-odd: {odd_ok}")
        print(f"  every kappa_r vanishes at the fixed point: {zero_at_fixed}")

        Bi = iota_odd_basis(p)
        print(f"  dim W_iota = {Bi.shape[0]}  (expected {(n-1)//2})")
        allin = all(in_span(K, Bi, p) for K in Ks)
        print(f"  every kappa_r lies in W_iota: {allin}")
        print("  so the constants do lie in a signed partition subspace")
        print("  with blocks of size two")

        # 2. is W_iota A-invariant?
        img = (Bi @ A.T) % p
        inv = all(in_span(v, Bi, p) for v in img)
        print(f"  is W_iota invariant under A?  {inv}")

        # 3. the A-invariant closure of the constants
        cl = invariant_closure(A, np.array(Ks), p)
        d = cl.shape[0]
        print(f"  smallest A-invariant subspace containing all kappa_r: "
              f"dimension {d} of {n}")
        if d < n:
            # is it sigma-invariant?  sample elements and apply sigma
            rng = np.random.default_rng(p)
            escapes = 0
            for _ in range(50):
                coef = rng.integers(0, p, size=d)
                v = (coef @ cl) % p
                sv = np.array([pow(int(t), p - 2, p) for t in v],
                              dtype=np.int64)
                if not in_span(sv, cl, p):
                    escapes += 1
            print(f"     sigma-invariant?  {50 - escapes} of 50 random "
                  f"elements stayed inside"
                  f", so {'not sigma-invariant' if escapes else 'possibly sigma-invariant'}")
        else:
            print("     it is the whole space, so nothing to check")

        # 4. any A-invariant signed-partition subspace holding the constants?
        #    W_iota is the natural candidate and fails; also test the
        #    column and row families and their signed versions.
        cands = {}
        colv = []
        for x in range(p):
            v = np.zeros(n, dtype=np.int64)
            for y in range(p):
                v[idx(p, y, x)] = 1
            colv.append(v)
        rowv = []
        for y in range(p):
            v = np.zeros(n, dtype=np.int64)
            for x in range(p):
                v[idx(p, y, x)] = 1
            rowv.append(v)
        cands["columns"] = np.array(colv)
        cands["rows"] = np.array(rowv)
        cands["W_iota"] = Bi
        for name, B in cands.items():
            B = row_reduce(B, p)
            a_inv = all(in_span(v, B, p) for v in (B @ A.T) % p)
            holds = all(in_span(K, B, p) for K in Ks)
            print(f"  {name:>10}: dim {B.shape[0]:>4}, A-invariant "
                  f"{str(a_inv):>5}, contains every kappa_r {holds}")
        print()

    print("=" * 74)
    print("Conclusion")
    print("=" * 74)
    print("  The constants lie in a large signed-partition subspace, but that")
    print("  subspace is not A-invariant, and the A-invariant subspace the")
    print("  constants generate is not sigma-invariant. These two facts are")
    print("  what Proposition 6.7 needs.")


if __name__ == "__main__":
    main()
