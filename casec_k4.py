#!/usr/bin/env python3
"""The sharpened counting step for case (c): k >= 4 instead of k >= 7.

The idea.  In the rigid case of Lemma 3.5 there are exactly
k tangent lines, one per point of S, and the row phi(t) of a tangent t is
non-zero whenever phi(t) is not in S.  So if wt(A x) <= 1 then at most one
tangent row lies outside S, which means at least k-1 of the k elements of S
are taken up by tangent rows.  At most one element of S is therefore left to
be a secant row, so at most one secant is discarded, and at most one more for
the single non-zero coordinate that is allowed:

    |E(G)| >= C(k,2) - 1 .

Mantel then forces a triangle as soon as C(k,2) - 1 > floor(k^2/4), which is
k >= 4, not k >= 7.

This script checks the combinatorial step directly, and then the conclusion
itself at p = 5 (k = 5) for many diagonal-free bijections, a case the k >= 7
argument of casec_general.py does not reach.

Runtime about 3 minutes.
"""
import itertools
import random
from math import comb

import numpy as np

random.seed(4)


def cells(p):
    return [(a, b) for a in range(p) for b in range(p)]


def lines(p):
    return {(m, c): frozenset(((m * x + c) % p, x) for x in range(p))
            for m in range(p) for c in range(p)}


def random_nodiag_bijection(p, L, rng):
    """A random bijection phi: lines -> points with P not on phi^{-1}(P)."""
    pts = cells(p)
    idx = {q: i for i, q in enumerate(pts)}
    order = list(range(len(pts)))
    rng.shuffle(order)
    adj = []
    for u in pts:
        cand = [idx[q] for q in pts if q not in L[u]]
        rng.shuffle(cand)
        adj.append(cand)
    match = [-1] * len(pts)

    def augment(u, seen):
        for v in adj[u]:
            if v in seen:
                continue
            seen.add(v)
            if match[v] == -1 or augment(match[v], seen):
                match[v] = u
                return True
        return False

    for u in order:
        if not augment(u, set()):
            return None
    phi = {pts[u]: pts[v] for v, u in enumerate(match)}
    assert all(q not in L[u] for u, q in phi.items())
    return phi


def exceptional_sets(p):
    """Supports with n_1 = k: one point per column, no three collinear."""
    slope = {(x, y): pow(y - x, p - 2, p)
             for x in range(p) for y in range(x + 1, p)}

    def collinear(f, x, y, z):
        return ((f[y] - f[x]) * slope[(x, y)]
                - (f[z] - f[x]) * slope[(x, z)]) % p == 0

    out = []

    def extend(f):
        x = len(f)
        if x == p:
            out.append(frozenset((f[j], j) for j in range(p)))
            return
        for a in range(p):
            f.append(a)
            if all(not collinear(f, i, j, x)
                   for i, j in itertools.combinations(range(x), 2)):
                extend(f)
            f.pop()

    extend([])
    return out


def counting_step(p, L, phi, sets):
    """Check the two branches of the sharpened argument.

    For each rigid support S, either at least two tangent rows fall outside S,
    in which case wt(A x) >= 2 at once and there is nothing more to prove, or
    at most one does and then the secant graph keeps at least C(k,2) - 1 edges.
    """
    k = p
    settled_directly = 0
    worst = comb(k, 2)
    for S in sets:
        tangent_rows, secant_pairs = [], []
        for u, line in L.items():
            met = line & S
            if len(met) == 1:
                tangent_rows.append(phi[u])
            elif len(met) == 2:
                secant_pairs.append((tuple(sorted(met)), phi[u]))
        assert len(tangent_rows) == k
        outside = [Q for Q in tangent_rows if Q not in S]
        if len(outside) >= 2:
            settled_directly += 1
            continue
        # at most one tangent row outside S, so at least k-1 of the k elements
        # of S are tangent rows, leaving at most one to be a secant row
        free = set(S) - set(tangent_rows)
        assert len(free) <= 1, (len(free), len(outside))
        in_S = [pr for pr, R in secant_pairs if R in S]
        assert len(in_S) <= 1, (len(in_S), len(free))
        surviving = comb(k, 2) - len(in_S) - (0 if outside else 1)
        worst = min(worst, surviving)
    return worst, settled_directly


def has_triangle(S, edges):
    E = set(edges)
    return any(tuple(sorted((a, b))) in E and tuple(sorted((b, c))) in E
               and tuple(sorted((a, c))) in E
               for a, b, c in itertools.combinations(sorted(S), 3))


def matrix(p, L, phi, alpha):
    pts = cells(p)
    idx = {q: i for i, q in enumerate(pts)}
    A = np.zeros((p * p, p * p), dtype=np.int64)
    for u, line in L.items():
        for P in line:
            A[idx[phi[u]], idx[P]] = 1
    for i in range(p * p):
        A[i, i] = (A[i, i] + alpha) % p
    return A, idx


def is_invertible(A, p):
    A = A.copy() % p
    n = A.shape[0]
    for c in range(n):
        piv = next((r for r in range(c, n) if A[r, c] % p), None)
        if piv is None:
            return 0
        A[[c, piv]] = A[[piv, c]]
        A[c] = (A[c] * pow(int(A[c, c]), p - 2, p)) % p
        for r in range(c + 1, n):
            if A[r, c] % p:
                A[r] = (A[r] - A[r, c] * A[c]) % p
    return 1


def weight_k_check(p, L, phi, alpha, sets):
    """min wt(A x) over x of weight k supported on an exceptional set."""
    A, idx = matrix(p, L, phi, alpha)
    best = None
    for S in sets:
        cols = [idx[P] for P in sorted(S)]
        tails = np.array(list(itertools.product(range(1, p), repeat=p - 1)),
                         dtype=np.int64)
        V = np.hstack([np.ones((tails.shape[0], 1), dtype=np.int64), tails])
        W = (V @ A[:, cols].T) % p
        m = int((W != 0).sum(axis=1).min())
        best = m if best is None else min(best, m)
    return best


def main():
    print("=" * 72)
    print("1.  Mantel's threshold with the sharpened count C(k,2) - 1")
    print("=" * 72)
    print(f"    {'k':>3} {'C(k,2)-1':>9} {'floor(k^2/4)':>13}  triangle forced?")
    for k in range(3, 12):
        lhs, rhs = comb(k, 2) - 1, k * k // 4
        print(f"    {k:>3} {lhs:>9} {rhs:>13}  {'yes' if lhs > rhs else 'no'}")
    for k in range(3, 400):
        assert (comb(k, 2) - 1 > k * k // 4) == (k >= 4)
    print("    => the triangle is forced exactly for k >= 4")

    print()
    print("=" * 72)
    print("2.  The counting step on the real configuration")
    print("=" * 72)
    for p in [5, 7]:
        L = lines(p)
        sets = exceptional_sets(p)
        rng = random.Random(p)
        phi = random_nodiag_bijection(p, L, rng)
        w, direct = counting_step(p, L, phi, sets)
        rest = len(sets) - direct
        print(f"    p = {p}: {len(sets)} rigid supports; {direct} have two or "
              f"more tangent rows outside S (done at once),")
        if rest:
            print(f"             the other {rest} leave at least {w} pairs, "
                  f"and C(k,2) - 1 = {comb(p,2)-1}")
            assert w >= comb(p, 2) - 1
        else:
            print(f"             none of them reach the graph argument at all")

    print()
    print("=" * 72)
    print("3.  The conclusion at p = 5 (k = 5), which k >= 7 did not cover")
    print("=" * 72)
    p = 5
    L = lines(p)
    sets = exceptional_sets(p)
    rng = random.Random(99)
    ok = 0
    for trial in range(6):
        phi = random_nodiag_bijection(p, L, rng)
        for alpha in range(1, p):
            A, _ = matrix(p, L, phi, alpha)
            if not is_invertible(A, p):
                print(f"    bijection {trial}, alpha = {alpha}: SINGULAR "
                      f"(so no branch number to speak of)")
                continue
            m = weight_k_check(p, L, phi, alpha, sets)
            status = "OK" if m >= 2 else "FAILS"
            print(f"    bijection {trial}, alpha = {alpha}: "
                  f"min wt(A x) at weight k is {m:>2}  ({status}, need >= 2)")
            assert m >= 2
            ok += 1
    print(f"    {ok} invertible cases, all with wt(A x) >= 2 at weight k")
    print()
    print("ALL k >= 4 CHECKS PASSED")


if __name__ == "__main__":
    main()
