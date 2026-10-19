#!/usr/bin/env python3
"""Exact two-round EDP for small layers, MDS and not.

For a layer L and an S-box S acting on each cell, the expected differential
probability of the two-round differential (a,b) under independent round keys is

    EDP(a,b) = sum_u DP(a -> u) * DP(L u -> b),

which is the (a,b) entry of  T . P_L . T,  where T[a,u] = prod_i DP_S(a_i,u_i)
is the n-fold tensor power of the normalised DDT and P_L is the permutation
matrix of u -> L u.  We build that matrix exactly and take the maximum over
a, b nonzero, and also over the pairs of a given pair of weights.

Two exponents are compared:

    (delta/p)^(B-1)         Theorem 9.3, in line with the bounds for SPNs
                            of Hong et al. (FSE 2000) and its successors
    (delta/p)^ceil(B/2)     a weaker exponent, shown for comparison

The Fano incidence layer is too large for the full matrix; tworound2.py
handles it row by row.

Runtime about 3 minutes.
"""
import itertools

import numpy as np


def ddt(p, sbox):
    """Normalised DDT: D[a,u] = #{x : S(x+a) - S(x) = u} / p."""
    D = np.zeros((p, p), dtype=np.float64)
    for a in range(p):
        for x in range(p):
            u = (sbox[(x + a) % p] - sbox[x]) % p
            D[a, u] += 1
    return D / p


def inverse_sbox(p):
    return [pow(x, p - 2, p) for x in range(p)]


def tensor(D, n):
    T = D
    for _ in range(n - 1):
        T = np.kron(T, D)
    return T


def index(vec, p):
    i = 0
    for c in vec:
        i = i * p + int(c)
    return i


def vectors(p, n):
    return list(itertools.product(range(p), repeat=n))


def perm_matrix(L, p, n):
    """P[u, Lu] = 1, with states indexed in base p, most significant first."""
    V = vectors(p, n)
    P = np.zeros((p ** n, p ** n), dtype=np.float64)
    for u in V:
        w = tuple(int(x) % p for x in (np.array(L) @ np.array(u)) % p)
        P[index(u, p), index(w, p)] = 1.0
    return P


def branch_number(L, p, n):
    """min over x != 0 of wt(x) + wt(Lx)."""
    best = None
    A = np.array(L)
    for x in itertools.product(range(p), repeat=n):
        if not any(x):
            continue
        y = (A @ np.array(x)) % p
        w = sum(1 for c in x if c) + sum(1 for c in y if c)
        best = w if best is None else min(best, w)
    return best


def delta_of(D, p):
    """Differential uniformity: max over a != 0 of p * max_u D[a,u]."""
    return int(round(p * D[1:, :].max()))


def analyse(name, L, p, n, sbox=None):
    sbox = sbox or inverse_sbox(p)
    D = ddt(p, sbox)
    delta = delta_of(D, p)
    B = branch_number(L, p, n)
    T = tensor(D, n)
    P = perm_matrix(L, p, n)
    E = T @ P @ T                      # EDP[a, b]

    V = vectors(p, n)
    wt = np.array([sum(1 for c in v if c) for v in V])
    nz = wt > 0

    sub = E[np.ix_(nz, nz)]
    overall = sub.max()

    half = (delta / p) ** ((B + 1) // 2)
    strong = (delta / p) ** (B - 1)

    print(f"  {name}")
    print(f"      p = {p}, n = {n}, branch number B = {B}, delta = {delta}")
    print(f"      max EDP over all (a,b)            {overall:.6g}")
    print(f"      (delta/p)^ceil(B/2) = ^{(B+1)//2:<2d}           {half:.6g}"
          f"   {'OK' if overall <= half + 1e-12 else 'VIOLATED'}")
    print(f"      (delta/p)^(B-1)     = ^{B-1:<2d}           {strong:.6g}"
          f"   {'OK' if overall <= strong + 1e-12 else 'VIOLATED'}")

    # the worst pair, and the maximum for each pair of weights
    i, j = np.unravel_index(np.argmax(sub), sub.shape)
    idx = np.where(nz)[0]
    wa, wb = wt[idx[i]], wt[idx[j]]
    print(f"      attained at wt(a) = {wa}, wt(b) = {wb}")
    rows = {}
    for u in range(1, n + 1):
        for v in range(1, n + 1):
            m = E[np.ix_(wt == u, wt == v)]
            if m.size:
                rows[(u, v)] = m.max()
    best_bal = max((val for (u, v), val in rows.items() if abs(u - v) <= 1),
                   default=0.0)
    print("      by weight pair: " + ", ".join(
        f"({u},{v}):{val:.4g}" for (u, v), val in sorted(rows.items())
        if val > 0))
    return dict(name=name, p=p, n=n, B=B, delta=delta, overall=overall,
                half=half, strong=strong, balanced=best_bal)


def is_mds(L, p, n):
    """Every square submatrix nonsingular."""
    A = np.array(L) % p
    for k in range(1, n + 1):
        for rows in itertools.combinations(range(n), k):
            for cols in itertools.combinations(range(n), k):
                sub = A[np.ix_(rows, cols)]
                if det_mod(sub, p) == 0:
                    return False
    return True


def det_mod(A, p):
    A = A.copy() % p
    n = A.shape[0]
    det = 1
    for c in range(n):
        piv = next((r for r in range(c, n) if A[r, c] % p), None)
        if piv is None:
            return 0
        if piv != c:
            A[[c, piv]] = A[[piv, c]]
            det = -det
        det = (det * int(A[c, c])) % p
        inv = pow(int(A[c, c]), p - 2, p)
        A[c] = (A[c] * inv) % p
        for r in range(c + 1, n):
            if A[r, c] % p:
                A[r] = (A[r] - A[r, c] * A[c]) % p
    return det % p


def cauchy_mds(p, n):
    """A Cauchy matrix, which is MDS, on 2n distinct elements of F_p."""
    xs = list(range(n))
    ys = list(range(n, 2 * n))
    return [[pow((xs[i] - ys[j]) % p, p - 2, p) for j in range(n)]
            for i in range(n)]


def main():
    print("=" * 74)
    print("1.  MDS layers: is the exponent B-1 or ceil(B/2)?")
    print("=" * 74)
    res = []
    for p, n in [(11, 3), (17, 3), (7, 3)]:
        L = cauchy_mds(p, n)
        assert is_mds(L, p, n), (p, n)
        res.append(analyse(f"Cauchy MDS {n}x{n} over F_{p}", L, p, n))
        print()

    print("=" * 74)
    print("2.  Non-MDS layers with the same or smaller branch number")
    print("=" * 74)
    # circulant 0/1 layers, which have singular minors
    layers = []
    L = [[1, 1, 0], [0, 1, 1], [1, 0, 1]]           # branch number 3, not MDS
    layers.append(("circulant 0/1 3x3", L, 11, 3))
    layers.append(("circulant 0/1 3x3", L, 17, 3))
    L4 = [[1, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1]]
    layers.append(("circulant 0/1 4x4", L4, 7, 4))
    for name, L, p, n in layers:
        assert not is_mds(L, p, n)
        res.append(analyse(f"{name} over F_{p}", L, p, n))
        print()

    print("=" * 74)
    print("Summary")
    print("=" * 74)
    print(f"  {'layer':<34} {'B':>2} {'max EDP':>10} {'^ceil(B/2)':>11} "
          f"{'^(B-1)':>10}  verdict")
    for r in res:
        v = "B-1 holds" if r['overall'] <= r['strong'] + 1e-12 else "B-1 FAILS"
        print(f"  {r['name']:<34} {r['B']:>2} {r['overall']:>10.4g} "
              f"{r['half']:>11.4g} {r['strong']:>10.4g}  {v}")


if __name__ == "__main__":
    main()
