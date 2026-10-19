"""Corollary 3.9 (case (c) of Theorem 3.3): a bijection with no diagonal one, at p = 5.
Finds a perfect matching in the non-incidence bipartite graph, then computes
the exact branch numbers of N + aI by exhaustive search over low weights."""
import itertools, numpy as np
from lib import radon, rank

p = 5; n = p*p
M = radon(p)                     # rows = lines (m,c), cols = points (y,x)

# perfect matching in the non-incidence graph (Hopcroft-Karp is overkill: greedy + augment)
adj = [[q for q in range(n) if M[l, q] == 0] for l in range(n)]
match_line = [-1]*n; match_pt = [-1]*n
def try_aug(l, seen):
    for q in adj[l]:
        if seen[q]: continue
        seen[q] = True
        if match_pt[q] == -1 or try_aug(match_pt[q], seen):
            match_line[l] = q; match_pt[q] = l; return True
    return False
for l in range(n):
    try_aug(l, [False]*n)
assert all(q != -1 for q in match_line), "no perfect matching"
# N[phi(line), point]: row index = matched point
N = np.zeros((n, n), dtype=np.int64)
for l in range(n):
    N[match_line[l]] = M[l]
assert all(N[i, i] == 0 for i in range(n)), "diagonal not free of ones"
print("bijection with no diagonal one: OK")

def min_w_plus_wt(A, p, maxw):
    best = 10**9
    per_w = []
    for w in range(1, maxw+1):
        vals = np.array(list(itertools.product(range(1, p), repeat=w)), dtype=np.int64)
        b = 10**9
        for S in itertools.combinations(range(A.shape[0]), w):
            b = min(b, w + int(np.count_nonzero((A[:, S] @ vals.T) % p, axis=0).min()))
        per_w.append(b); best = min(best, b)
    return best, per_w

I = np.eye(n, dtype=np.int64)
for a in range(p):
    A = (N + a*I) % p
    inv = rank(A, p) == n
    if not inv:
        print(f"a={a}: singular"); continue
    d, per_w = min_w_plus_wt(A, p, 5)
    l, _ = min_w_plus_wt(A.T.copy(), p, 5)
    print(f"a={a}: invertible, min over w<=5: differential {d}, linear {l}, per weight {per_w}")
