#!/usr/bin/env python3
"""Two-round EDP for incidence layers, and a test of the exponent B-1.

The full transition matrix is too large for n = 7 cells, so for each input
difference a we build the row EDP(a, .) directly: the sum over u supported on
supp(a) of a product distribution, pushed through L and expanded as a sum of
rank-one tensors.  That is exact and costs (p-1)^wt(a) * p^n per row.

Checked: the Fano configuration 7_3 over F_5 and F_7 (an incidence layer of
the kind the paper is about, small enough to be exhaustive), against

    (delta/p)^(B-1)       the bound of Theorem 9.3
    (delta/p)^ceil(B/2)   a weaker exponent, shown for comparison

and against the per-pair bound  EDP <= (delta/p)^(s-d)  of the same theorem,
which comes from the Singleton bound on dim U (Lemma 9.2).

Runtime about 5 minutes.
"""
import itertools

import numpy as np


def ddt(p):
    S = [pow(x, p - 2, p) for x in range(p)]
    D = np.zeros((p, p))
    for a in range(p):
        for x in range(p):
            D[a, (S[(x + a) % p] - S[x]) % p] += 1
    return D / p


def fano():
    return [frozenset(((i + j) % 7 for j in (0, 1, 3))) for i in range(7)]


def layer(alpha, p):
    F = fano()
    return np.array([[ (1 if q in F[r] else 0) + (alpha if r == q else 0)
                       for q in range(7)] for r in range(7)]) % p


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


def branch(L, p, n):
    best = None
    for x in itertools.product(range(p), repeat=n):
        if not any(x):
            continue
        y = (L @ np.array(x)) % p
        w = sum(1 for c in x if c) + sum(1 for c in y if c)
        best = w if best is None else min(best, w)
    return best


def rank_mod(A, p):
    A = A.copy() % p
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


def row_edp(a, L, D, p, n):
    """The exact vector b -> EDP(a,b), indexed by b in base p."""
    A = [i for i in range(n) if a[i]]
    out = np.zeros(p ** n)
    for vals in itertools.product(range(1, p), repeat=len(A)):
        u = np.zeros(n, dtype=np.int64)
        f = 1.0
        for i, val in zip(A, vals):
            u[i] = val
            f *= D[a[i], val]
        if f == 0.0:
            continue
        v = (L @ u) % p
        # rank-one tensor  prod_j D[v_j, b_j]
        t = np.array([1.0])
        for j in range(n):
            t = np.outer(t, D[v[j]]).ravel()
        out += f * t
    return out


def dim_U(a_supp, b_supp, L, p, n):
    """dim of {u : supp(u) in A, supp(Lu) in B}."""
    outside = [j for j in range(n) if j not in b_supp]
    if not outside:
        return len(a_supp)
    Msub = L[np.ix_(outside, list(a_supp))] % p
    return len(a_supp) - rank_mod(Msub, p)


def analyse(name, L, p, n, max_wt=None):
    D = ddt(p)
    delta = int(round(p * D[1:, :].max()))
    B = branch(L, p, n)
    half = (delta / p) ** ((B + 1) // 2)
    strong = (delta / p) ** (B - 1)
    print(f"  {name}: B = {B}, delta = {delta}")

    wt_b = np.array([sum(1 for c in np.base_repr(i, p).zfill(n) if c != '0')
                     for i in range(p ** n)])
    best = 0.0
    best_at = None
    worst_ratio = 0.0
    tested = 0
    for a in itertools.product(range(p), repeat=n):
        wa = sum(1 for c in a if c)
        if wa == 0 or (max_wt and wa > max_wt):
            continue
        tested += 1
        row = row_edp(list(a), L, D, p, n)
        row[0] = 0.0
        k = int(np.argmax(row))
        if row[k] > best:
            best = row[k]
            best_at = (a, k, wa, int(wt_b[k]))
        # the per-pair Singleton bound, on the arg-max b
        for k2 in np.argsort(row)[-3:]:
            if row[k2] <= 0:
                continue
            bs = [j for j in range(n)
                  if int(np.base_repr(k2, p).zfill(n)[j]) != 0]
            asupp = [i for i in range(n) if a[i]]
            d = dim_U(asupp, bs, L, p, n)
            s = len(asupp) + len(bs)
            bound = (delta / p) ** (s - d)
            worst_ratio = max(worst_ratio, row[k2] / bound)
    a, k, wa, wb = best_at
    print(f"      {tested} input differences tested")
    print(f"      max EDP {best:.6g}  at wt(a) = {wa}, wt(b) = {wb}")
    print(f"      (delta/p)^ceil(B/2) = {half:.6g}"
          f"   {'OK' if best <= half + 1e-12 else 'VIOLATED'}")
    print(f"      (delta/p)^(B-1)     = {strong:.6g}"
          f"   {'OK' if best <= strong + 1e-12 else 'VIOLATED'}")
    print(f"      max ratio EDP / (delta/p)^(s-d)  = {worst_ratio:.4f}"
          f"   {'OK' if worst_ratio <= 1 + 1e-9 else 'VIOLATED'}")


def main():
    print("=" * 74)
    print("Incidence layers: the Fano configuration 7_3, N + alpha I")
    print("=" * 74)
    for p in [5, 7]:
        for alpha in range(1, p):
            L = layer(alpha, p)
            if det_mod(L, p) == 0:
                continue
            analyse(f"Fano, alpha = {alpha} over F_{p}", L, p, 7,
                    max_wt=2 if p == 7 else 3)
            print()
            break

    print("=" * 74)
    print("The Singleton bound on dim U, over many (a,b) and layers")
    print("=" * 74)
    rng = np.random.default_rng(11)
    for p, n in [(5, 4), (7, 4), (11, 3)]:
        for trial in range(3):
            L = rng.integers(0, p, size=(n, n))
            if det_mod(L, p) == 0:
                continue
            B = branch(L, p, n)
            bad = 0
            checked = 0
            for a in itertools.product(range(p), repeat=n):
                if not any(a):
                    continue
                asupp = [i for i in range(n) if a[i]]
                for bsupp in itertools.chain.from_iterable(
                        itertools.combinations(range(n), r)
                        for r in range(1, n + 1)):
                    d = dim_U(asupp, list(bsupp), L, p, n)
                    s = len(asupp) + len(bsupp)
                    checked += 1
                    if d > s - B + 1:
                        bad += 1
            print(f"    p = {p}, n = {n}, B = {B}: "
                  f"dim U <= wa + wb - B + 1 on {checked} supports, "
                  f"{bad} violations")
    print()
    print("ALL TWO-ROUND CHECKS DONE")


if __name__ == "__main__":
    main()
