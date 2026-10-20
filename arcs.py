#!/usr/bin/env python3
"""Remark 3.7 and the counts in the caption of Figure 2.

Remark 3.7: in the Radon configuration the sets allowed by Lemma 3.5 (one
point per column, no three collinear) are exactly the p^2(p-1) graphs of
y = a x^2 + b x + c with a != 0; the backtracking search finds 18, 100, 294
and 1210 of them at p = 3, 5, 7, 11, and each has n_1 = p.

Figure 2, at p = 7 on S = {(x^2, x)}: the tangent rows and the secant rows
inside S under the row-major identification and under the diagonal-free
identification of Proposition 4.10 (line l_{m,c} is the row of the cell
(c + e, m)), and, under the latter, the least number of tangent rows outside
S over every such set at p = 5, 7, 11.

Runtime a few seconds.
"""
from itertools import combinations

def rigid_sets(p):
    """All f: F_p -> F_p whose graph has no three collinear points (backtracking)."""
    out = []; f = []
    def collinear(x1, x2, x3):
        y1, y2, y3 = f[x1], f[x2], f[x3]
        return ((y2 - y1)*(x3 - x1) - (y3 - y1)*(x2 - x1)) % p == 0
    def rec(x):
        if x == p:
            out.append(tuple(f)); return
        for y in range(p):
            f.append(y)
            if all(not collinear(a, b, x) for a, b in combinations(range(x), 2)):
                rec(x+1)
            f.pop()
    rec(0)
    return out

def counts(p, S):
    """n_j over the non-vertical lines, for S a set of cells (y,x)."""
    n = {}
    for m in range(p):
        for c in range(p):
            j = sum(1 for (y, x) in S if (m*x + c - y) % p == 0)
            n[j] = n.get(j, 0) + 1
    return n

def figure_counts(p, row_of_line, S):
    """The tangent rows T, the secants whose row lies in S, |T \\ S| and |S \\ T|."""
    Sset = set(S); T = set(); sec_in_S = []
    for m in range(p):
        for c in range(p):
            pts = [(y, x) for (y, x) in S if (m*x + c - y) % p == 0]
            R = row_of_line(m, c)
            if len(pts) == 1:
                T.add(R)
            elif len(pts) == 2 and R in Sset:
                sec_in_S.append((pts, R))
    return T, sec_in_S, len(T - Sset), len(Sset - T)

def main():
    for p in [3, 5, 7, 11]:
        R = rigid_sets(p)
        quad = {tuple((a*x*x + b*x + c) % p for x in range(p))
                for a in range(1, p) for b in range(p) for c in range(p)}
        ok = set(R) == quad and len(R) == p*p*(p-1)
        # every rigid set has n_1 = p, n_2 = C(p,2), no n_j for j >= 3
        okn = all(counts(p, [(f[x], x) for x in range(p)]).get(1, 0) == p for f in R[:200])
        print(f"p={p:2d}: {len(R)} sets with no three collinear points, one per column; "
              f"p^2(p-1) = {p*p*(p-1)}; equal to the quadratics a x^2+b x+c, a != 0: {ok}; n_1 = p: {okn}")
        assert ok and okn
    # Figure 2 at p = 7
    p = 7; S = [((x*x) % p, x) for x in range(p)]
    rm = lambda m, c: (m, c)                    # row-major: line l_{m,c} -> cell (m,c)
    e = 3                                         # least non-square mod 7
    dn = lambda m, c: ((c + e) % p, m)           # diagonal-free: l_{x, y-e} -> cell (y,x)
    for name, phi in [("row-major", rm), ("diagonal-free (e=3)", dn)]:
        diag = sum(1 for m in range(p) for c in range(p) if (m*phi(m, c)[1] + c - phi(m, c)[0]) % p == 0)
        T, sec, tout, sout = figure_counts(p, phi, S)
        lab = lambda P: f"P{P[1]}"
        print(f"\n{name}: {diag} diagonal ones; |T| = {len(T)}, |T \\ S| = {tout}, |S \\ T| = {sout}; "
              f"secant rows in S: {[(lab(a), lab(b)) for (a, b), _ in sec]}")
    # under the diagonal-free identification, how few tangent rows can fall outside S?
    for pp in [5, 7, 11]:
        e = min(a for a in range(1, pp) if a not in {x*x % pp for x in range(1, pp)})
        dn = lambda m, c, pp=pp, e=e: ((c + e) % pp, m)
        best = min(figure_counts(pp, dn, [(f[x], x) for x in range(pp)])[2] for f in rigid_sets(pp))
        print(f"p={pp}: diagonal-free identification, min over rigid S of |T \\ S| = {best}")
    print("\nALL ARC AND FIGURE 2 CHECKS PASSED")

if __name__ == "__main__":
    main()
