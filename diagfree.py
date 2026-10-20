#!/usr/bin/env python3
"""Proposition 4.10 and Remark 4.11: the diagonal-free identification.

With e a non-square, the cell (y,x) is identified with the line l_{x, y-e}:

    N[(y,x),(y',x')] = 1[y' = x x' + y - e].

Checked at p = 3, 5, 7, 11: N is a row permutation of M with no diagonal one
and ker N = ker M; N^2 = 1[x + x'' != 0], N^3 = -J, N^4 = 0, ranks
C(p+1,2), p, 1 and minimal polynomial x^4; the inverse of N + alpha I for every
alpha != 0, and its evaluation through column sums.  Then the branch number
p + 2 for every alpha != 0 at p = 5 and 7, by a complete search: a vector
with wt(x) + wt(Ax) <= 2w + 1 has wt(x) <= w or wt(Ax) <= w, so searching
x and A x up to weight w settles every B <= 2w + 1.

For Remark 4.11, the symmetric identification N'[(y,x),(y',x')] = 1[y + y' = x x']:
N'^2 = 1[x != x''], N'^3 = -J, N'^4 = 0, every shift invertible, p diagonal
ones (one in each column), and branch numbers 6, 6, 6, 5 at p = 5.

The p = 7 branch-number search takes about 18 minutes; --quick skips it.
"""
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import radon, rank  # noqa: E402


def ident_N(p, e):
    """N[(y,x),(y',x')] = 1[y' = x x' + y - e], cell index y*p + x."""
    N = np.zeros((p * p, p * p), dtype=np.int64)
    for y in range(p):
        for x in range(p):
            for xx in range(p):
                N[y * p + x, ((x * xx + y - e) % p) * p + xx] = 1
    return N


def ident_sym(p):
    """The symmetric identification N'[(y,x),(y',x')] = 1[y + y' = x x']."""
    N = np.zeros((p * p, p * p), dtype=np.int64)
    for y in range(p):
        for x in range(p):
            for xx in range(p):
                N[y * p + x, ((x * xx - y) % p) * p + xx] = 1
    return N


def nonsquare(p):
    sq = {x * x % p for x in range(1, p)}
    return min(a for a in range(1, p) if a not in sq)


def inv(A, p):
    """Inverse mod p, or None if A is singular."""
    n = A.shape[0]
    Aug = np.hstack([A % p, np.eye(n, dtype=np.int64)])
    r = 0
    for c in range(n):
        nz = np.nonzero(Aug[r:, c])[0]
        if len(nz) == 0:
            return None
        s = r + nz[0]
        Aug[[r, s]] = Aug[[s, r]]
        Aug[r] = Aug[r] * pow(int(Aug[r, c]), -1, p) % p
        col = Aug[:, c].copy()
        col[r] = 0
        Aug = (Aug - np.outer(col, Aug[r])) % p
        r += 1
    return Aug[:, n:]


def minpoly_deg(A, p, maxdeg=10):
    """Degree of the minimal polynomial, from the rank of the stacked powers."""
    n = A.shape[0]
    pows = [np.eye(n, dtype=np.int64)]
    for d in range(1, maxdeg + 1):
        pows.append(pows[-1] @ A % p)
        if rank(np.array([P.ravel() for P in pows]), p) < len(pows):
            return d
    return None


def branch_exhaustive(A, Ainv, p, maxw):
    """min of wt(x) + wt(Ax) over wt(x) <= maxw, and the same for Ainv (that is,
    over wt(Ax) <= maxw).  Weights are scale invariant, so the first non-zero
    coordinate of x is taken to be 1."""
    n = A.shape[0]
    best = 10 ** 9
    for B in (A, Ainv):
        Bt = B.T.copy()
        for w in range(1, maxw + 1):
            vals = np.array([(1,) + t for t in itertools.product(range(1, p), repeat=w - 1)],
                            dtype=np.int64)
            combs = np.array(list(itertools.combinations(range(n), w)), dtype=np.int64)
            for start in range(0, len(combs), 4000):
                cols = Bt[combs[start:start + 4000]]          # (C, w, n)
                img = np.einsum('vw,cwn->cvn', vals, cols) % p
                wts = np.count_nonzero(img, axis=2) + w
                best = min(best, int(wts.min()))
    return best


def check_column_sums(N, p, rng):
    """(N + alpha I)^{-1} w through N w and the column sums of w, as in the text."""
    for alpha in range(1, p):
        A = (N + alpha * np.eye(p * p, dtype=np.int64)) % p
        w = rng.integers(0, p, p * p)
        C = w.reshape(p, p).sum(axis=0) % p            # column sums W_t
        Sig = C.sum() % p
        ai = pow(alpha, -1, p)
        Nw = N @ w % p
        N2w = np.array([(Sig - C[(-x) % p]) % p for y in range(p) for x in range(p)])
        N3w = np.full(p * p, (-Sig) % p)
        assert np.array_equal(N2w, N @ Nw % p) and np.array_equal(N3w, N @ (N @ Nw % p) % p)
        v = (ai * w - ai ** 2 * Nw + ai ** 3 * N2w - ai ** 4 * N3w) % p
        assert np.array_equal(A @ v % p, w)


def algebra():
    rng = np.random.default_rng(1)
    for p in [3, 5, 7, 11]:
        e = nonsquare(p)
        M = radon(p)
        N = ident_N(p, e)
        n = p * p
        perm_ok = sorted(map(tuple, N)) == sorted(map(tuple, M))
        diag = int(np.trace(N))
        N2 = N @ N % p
        N3 = N2 @ N % p
        N4 = N3 @ N % p
        ex2 = np.array([[1 if (x + xx) % p else 0 for yy in range(p) for xx in range(p)]
                        for y in range(p) for x in range(p)])
        ok2 = np.array_equal(N2, ex2)
        ok3 = np.array_equal(N3, (-np.ones((n, n), dtype=np.int64)) % p)
        ok4 = not N4.any()
        r = (rank(N, p), rank(N2, p), rank(N3, p))
        mp = minpoly_deg(N, p)
        inv_ok = True
        for alpha in range(1, p):
            A = (N + alpha * np.eye(n, dtype=np.int64)) % p
            ai = pow(alpha, -1, p)
            S = sum(pow(-ai, k, p) * np.linalg.matrix_power(N, k) for k in range(4)) * ai % p
            inv_ok &= np.array_equal(A @ S % p, np.eye(n, dtype=np.int64))
        check_column_sums(N, p, rng)
        ker_eq = rank(np.vstack([M, N]), p) == rank(M, p)   # same row space, same kernel
        print(f"p={p:2d} e={e}: row-perm of M {perm_ok}, diagonal ones {diag}, "
              f"N^2=1[x+x''!=0] {ok2}, N^3=-J {ok3}, N^4=0 {ok4}, "
              f"ranks {r} (expect {(p * (p + 1) // 2, p, 1)}), minpoly degree {mp}, "
              f"inverse formula all alpha!=0 {inv_ok}, column-sum inverse OK, ker N = ker M {ker_eq}")
        assert perm_ok and diag == 0 and ok2 and ok3 and ok4 and inv_ok and ker_eq
        assert r == (p * (p + 1) // 2, p, 1) and mp == 4

        # Remark 4.11: the symmetric identification
        Ns = ident_sym(p)
        S2 = Ns @ Ns % p
        S3 = S2 @ Ns % p
        S4 = S3 @ Ns % p
        ex2 = np.array([[1 if x != xx else 0 for yy in range(p) for xx in range(p)]
                        for y in range(p) for x in range(p)])
        pw = np.array_equal(S2, ex2) and np.array_equal(S3, np.full((n, n), p - 1)) and not S4.any()
        sym = np.array_equal(Ns, Ns.T)
        ds = int(np.trace(Ns))
        diag_cols = sorted({x for y in range(p) for x in range(p) if Ns[y * p + x, y * p + x]})
        invs = all(rank((Ns + a * np.eye(n, dtype=np.int64)) % p, p) == n for a in range(1, p))
        print(f"        y+y'=xx': symmetric {sym}, N'^2 = 1[x != x''], N'^3 = -J, N'^4 = 0: {pw}; "
              f"diagonal ones {ds}, one in each column: {diag_cols == list(range(p))}; "
              f"all alpha!=0 invertible {invs}")
        assert sym and pw and ds == p and diag_cols == list(range(p)) and invs


def branch_numbers(primes):
    print("\nBranch numbers (complete: a vector with wt(x)+wt(Ax) <= 2w+1 has wt(x)<=w or wt(Ax)<=w)")
    for p, w in [(5, 3), (7, 4)]:
        if p not in primes:
            continue
        N = ident_N(p, nonsquare(p))
        n = p * p
        for alpha in range(1, p):
            A = (N + alpha * np.eye(n, dtype=np.int64)) % p
            Ai = inv(A, p)
            t = time.time()
            bd = branch_exhaustive(A, Ai, p, w)
            bl = branch_exhaustive(A.T.copy(), Ai.T.copy(), p, w)
            print(f"  p={p} alpha={alpha}: min over searched = Bd {bd}, Bl {bl}  "
                  f"(proves B >= {min(bd, bl, 2 * w + 2)}; p+2 = {p + 2})  [{time.time() - t:.0f}s]")
            assert bd == bl == p + 2
    # Remark 4.11: the symmetric identification at p = 5
    p = 5
    Ns = ident_sym(p)
    n = p * p
    got = []
    for alpha in range(1, p):
        A = (Ns + alpha * np.eye(n, dtype=np.int64)) % p
        Ai = inv(A, p)
        bd = branch_exhaustive(A, Ai, p, 3)
        bl = branch_exhaustive(A.T.copy(), Ai.T.copy(), p, 3)
        print(f"  y+y'=xx' at p=5, alpha={alpha}: Bd {bd}, Bl {bl}")
        got.append((bd, bl))
    assert got == [(6, 6), (6, 6), (6, 6), (5, 5)]


def main():
    quick = "--quick" in sys.argv[1:]
    algebra()
    branch_numbers([5] if quick else [5, 7])
    if quick:
        print("\n(p = 7 branch-number search skipped)")
    print("\nALL DIAGONAL-FREE CHECKS PASSED")


if __name__ == "__main__":
    main()
