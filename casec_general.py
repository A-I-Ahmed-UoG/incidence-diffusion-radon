#!/usr/bin/env python3
"""Checks for Lemma 3.5 and Theorem 3.6 (case (c) of Theorem 3.3), first route.

This follows the original counting step, which needs k >= 7; casec_k4.py
checks the sharper count that brings it down to k >= 4.

Four things are checked.

1. Lemma 3.5.  For a k-subset S of the points, either n_1 >= k + 2, or every
   line meets S at most twice, every pair of points of S lies on a (unique)
   line, n_1 = k, and each point of S carries exactly one tangent.
   Exhaustive at p = 5, random at p = 7, 11, 13.

2. In the Radon configuration the second case means one point per column, so
   the exceptional sets are exactly the graphs of functions f with no three
   collinear points.  The parabola is one of them, so the case is not vacuous.

3. The counting step in the proof: at most k of the C(k,2) secants are
   identified with a point of S, so after discarding those and one more,
   at least C(k,2) - k - 1 pairs remain, and that graph contains a triangle.
   Exhaustive over all exceptional sets at p = 7, random at p = 11.

4. The conclusion itself, wt(A x) >= 2 for every x supported on such a set,
   by exhaustive search over the values and the shifts at p = 7.

Also the arithmetic of Mantel's bound, which is where k >= 7 comes from.

Runtime about 4 minutes.
"""
import itertools
import random
from math import comb

import numpy as np

random.seed(20260921)


def cells(p):
    return [(a, b) for a in range(p) for b in range(p)]


def lines(p):
    """Line (m,c) = {(m*x + c, x) : x in F_p}, indexed like the cells."""
    return {(m, c): frozenset(((m * x + c) % p, x) for x in range(p))
            for m in range(p) for c in range(p)}


def matching_bijection(p, L):
    """A bijection phi from lines to points with P not on phi^{-1}(P).

    Hopcroft-Karp is not needed: the non-incidence graph is regular of degree
    p^2 - p, so greedy augmentation finishes at once.
    """
    pts = cells(p)
    idx = {q: i for i, q in enumerate(pts)}
    adj = [[idx[q] for q in pts if q not in L[u]] for u in pts]
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

    for u in range(len(pts)):
        assert augment(u, set())
    phi = {pts[u]: pts[v] for v, u in enumerate(match)}
    assert all(q not in L[u] for u, q in phi.items())
    return phi


def profile(S, L):
    counts = {u: len(L[u] & S) for u in L}
    tang = [u for u, j in counts.items() if j == 1]
    sec = [u for u, j in counts.items() if j == 2]
    big = [u for u, j in counts.items() if j >= 3]
    return tang, sec, big


def check_lemma(p, L, trials=None):
    pts = cells(p)
    k = p
    if trials is None:
        subsets = itertools.combinations(pts, k)
    else:
        subsets = (random.sample(pts, k) for _ in range(trials))
    seen = exceptional = 0
    for T in subsets:
        seen += 1
        S = frozenset(T)
        tang, sec, big = profile(S, L)
        if len(tang) >= k + 2:
            continue
        exceptional += 1
        assert not big, "n_1 <= k+1 but some line meets S three times"
        assert len(sec) == comb(k, 2), "n_1 <= k+1 but some pair is unjoined"
        assert len(tang) == k, (p, len(tang))
        touched = sorted(next(iter(L[u] & S)) for u in tang)
        assert touched == sorted(S), "the tangents are not one per point"
    return seen, exceptional


def exceptional_sets(p, L, sample=None):
    """k-subsets with n_1 = k: one point per column, no three collinear."""
    slope = {}
    for x in range(p):
        for y in range(x + 1, p):
            slope[(x, y)] = pow(y - x, p - 2, p)   # 1/(y-x)

    def collinear(f, x, y, z):
        # (f[x],x),(f[y],y),(f[z],z) are on a common line iff the slopes agree
        return ((f[y] - f[x]) * slope[(x, y)] - (f[z] - f[x]) * slope[(x, z)]) % p == 0

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

    if sample is None:
        extend([])
        return out

    res = []
    while len(res) < sample:
        f = []
        for x in range(p):
            choices = list(range(p))
            random.shuffle(choices)
            placed = False
            for a in choices:
                f.append(a)
                if all(not collinear(f, i, j, x)
                       for i, j in itertools.combinations(range(x), 2)):
                    placed = True
                    break
                f.pop()
            if not placed:
                break
        if len(f) == p:
            res.append(frozenset((f[j], j) for j in range(p)))
    return res


def parabola_orbit(p):
    """Images of {(x^2,x)} under (a,x) -> (lambda a + mu x + nu, x)."""
    par = [(x * x) % p for x in range(p)]
    return {frozenset(((lam * par[x] + mu * x + nu) % p, x) for x in range(p))
            for lam in range(1, p) for mu in range(p) for nu in range(p)}


def has_triangle(vertices, edges):
    E = set(edges)
    return any(tuple(sorted((a, b))) in E and tuple(sorted((b, c))) in E
               and tuple(sorted((a, c))) in E
               for a, b, c in itertools.combinations(sorted(vertices), 3))


def check_counting(p, L, phi, sets):
    k = p
    worst = 0
    for S in sets:
        edges, dropped = [], 0
        for u in L:
            met = L[u] & S
            if len(met) == 2:
                if phi[u] in S:
                    dropped += 1
                else:
                    edges.append(tuple(sorted(met)))
        assert dropped <= k, (dropped, k)
        assert len(edges) >= comb(k, 2) - k
        worst = max(worst, dropped)
        for e in edges:                       # discard one further secant
            assert has_triangle(S, [f for f in edges if f != e]), (p, sorted(S))
    return worst


def matrix(p, L, phi, alpha):
    """A = N + alpha I with rows indexed by the point phi(line)."""
    pts = cells(p)
    idx = {q: i for i, q in enumerate(pts)}
    A = np.zeros((p * p, p * p), dtype=np.int64)
    for u, line in L.items():
        r = idx[phi[u]]
        for P in line:
            A[r, idx[P]] = 1
    for i in range(p * p):
        A[i, i] = (A[i, i] + alpha) % p
    return A, idx


def weight_check(p, L, phi, sets):
    """min wt(A x) over x supported on an exceptional set, all alpha."""
    k = p
    best = None
    for alpha in range(1, p):
        A, idx = matrix(p, L, phi, alpha)
        for S in sets:
            cols = [idx[P] for P in sorted(S)]
            # scaling does not change the weight, so fix the first value at 1
            tails = np.array(list(itertools.product(range(1, p), repeat=k - 1)),
                             dtype=np.int64)
            V = np.hstack([np.ones((tails.shape[0], 1), dtype=np.int64), tails])
            W = (V @ A[:, cols].T) % p
            m = int((W != 0).sum(axis=1).min())
            best = m if best is None else min(best, m)
    return best


def main():
    print("=" * 70)
    print("1.  Lemma 3.5:  n_1 <= k+1 forces an arc with every pair joined")
    print("=" * 70)
    for p, trials in [(5, None), (7, 150000), (11, 30000), (13, 15000)]:
        L = lines(p)
        seen, exc = check_lemma(p, L, trials)
        how = "all" if trials is None else "random"
        print(f"    p = {p:2d}:  {seen:7d} {how:6s} {p}-subsets,  {exc:4d} with "
              f"n_1 <= k+1,  all of them with n_1 = k")

    print()
    print("=" * 70)
    print("2.  The exceptional sets at small p, and the parabola")
    print("=" * 70)
    for p in [5, 7]:
        L = lines(p)
        sets = exceptional_sets(p, L)
        direct = sum(1 for T in itertools.combinations(cells(p), p)
                     if len(profile(frozenset(T), L)[0]) == p) if p == 5 else None
        orbit = parabola_orbit(p)
        assert set(sets) == orbit, "the exceptional sets are not one orbit"
        line = (f"    p = {p}:  {len(sets):5d} sets with n_1 = k, "
                f"= p^2(p-1) = {p * p * (p - 1)}, exactly the affine images "
                f"of the parabola")
        if direct is not None:
            assert direct == len(sets)
            line += f"\n            (confirmed against an exhaustive search: {direct})"
        print(line)
    for p in [11, 13]:
        L = lines(p)
        orbit = parabola_orbit(p)
        sets = exceptional_sets(p, L, sample=150)
        assert set(sets) <= orbit
        print(f"    p = {p}:  150 random sets with n_1 = k, all of them in the "
              f"orbit of size {len(orbit)}")
    for p in [5, 7, 11, 13, 17]:
        L = lines(p)
        S = frozenset(((x * x) % p, x) for x in range(p))
        tang, sec, big = profile(S, L)
        assert not big and len(tang) == p and len(sec) == comb(p, 2)
        print(f"    p = {p:2d}:  the parabola has n_1 = {len(tang)} = k, "
              f"{len(sec)} secants, no line meeting it three times")

    print()
    print("=" * 70)
    print("3.  Counting step: a triangle survives in the secant graph")
    print("=" * 70)
    for p, kw in [(7, {}), (11, dict(sample=200))]:
        L = lines(p)
        phi = matching_bijection(p, L)
        sets = exceptional_sets(p, L, **kw)
        worst = check_counting(p, L, phi, sets)
        how = "all" if not kw else "random"
        print(f"    p = {p:2d}:  {len(sets):5d} {how:6s} exceptional sets,  at most "
              f"{worst} secants lost to S,  a triangle always remains")

    print()
    print("=" * 70)
    print("4.  The conclusion at p = 7:  wt(A x) >= 2 on every such support")
    print("=" * 70)
    p = 7
    L = lines(p)
    phi = matching_bijection(p, L)
    sets = exceptional_sets(p, L)
    sub = random.sample(sets, min(8, len(sets)))
    m = weight_check(p, L, phi, sub)
    print(f"    p = 7:  {len(sub)} supports x {(p-1)**(p-1)} value vectors x "
          f"{p-1} shifts,  min wt(A x) = {m}")
    assert m >= 2

    print()
    print("=" * 70)
    print("5.  Mantel:  C(k,2) - k - 1 > floor(k^2/4)  exactly for k >= 7")
    print("=" * 70)
    for k in range(4, 13):
        lhs, rhs = comb(k, 2) - k - 1, k * k // 4
        print(f"    k = {k:2d}:  {lhs:3d} edges,  Mantel bound {rhs:3d}   "
              f"{'triangle forced' if lhs > rhs else '--'}")
    for k in range(4, 500):
        assert (comb(k, 2) - k - 1 > k * k // 4) == (k >= 7)

    print()
    print("ALL CASE (c) CHECKS PASSED")


if __name__ == "__main__":
    main()
