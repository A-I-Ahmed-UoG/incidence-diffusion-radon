#!/usr/bin/env python3
"""Theorem 3.6 in characteristic two.

Random diagonal-free bijections for PG(2,3) (v = 13, k = 4) and for AG(2,4)
without a parallel class (v = 16, k = 4), over F_2 (alpha = 1, exhaustive
over all 2^v inputs) and over F_4 (every alpha != 0, complete low-weight
search on x and on A x).  Every invertible shift has branch number k + 2 = 6.
As a control on the search, a bijection with diagonal ones over F_4 gives
k + 1 = 5, as Theorem 3.3(a) predicts.

Runtime about 10 seconds.
"""
import itertools
import numpy as np

# ---- GF(4) = F_2[w]/(w^2+w+1), elements 0,1,2=w,3=w+1 -------------------
MUL4 = np.array([[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]], dtype=np.int64)
INV4 = [None, 1, 3, 2]

class GF:
    def __init__(self, q):
        self.q = q
    def add(self, a, b):
        return a ^ b
    def mul(self, a, b):
        return (a & b) if self.q == 2 else MUL4[a, b]
    def inv(self, a):
        return 1 if self.q == 2 else INV4[a]

def matvec(F, A, X):
    """A (n x n), X (n x K) over F; returns A X."""
    n, K = A.shape[0], X.shape[1]
    out = np.zeros((n, K), dtype=np.int64)
    for j in range(A.shape[1]):
        col = A[:, j][:, None]
        out ^= F.mul(np.broadcast_to(col, (n, K)), np.broadcast_to(X[j][None, :], (n, K)))
    return out

def inverse(F, A):
    n = A.shape[0]
    Aug = np.hstack([A.copy(), np.eye(n, dtype=np.int64)])
    r = 0
    for c in range(n):
        nz = np.nonzero(Aug[r:, c])[0]
        if len(nz) == 0:
            return None
        s = r + nz[0]; Aug[[r, s]] = Aug[[s, r]]
        Aug[r] = F.mul(Aug[r], F.inv(int(Aug[r, c])))
        for i in range(n):
            if i != r and Aug[i, c]:
                Aug[i] ^= F.mul(Aug[r], int(Aug[i, c]))
        r += 1
    return Aug[:, n:]

def lowweight_min(F, B, maxw):
    n = B.shape[0]; best = 10**9
    for w in range(1, maxw+1):
        vals = [(1,)+t for t in itertools.product(range(1, F.q), repeat=w-1)]
        for S in itertools.combinations(range(n), w):
            X = np.zeros((n, len(vals)), dtype=np.int64)
            for k, v in enumerate(vals):
                X[list(S), k] = v
            Y = matvec(F, B, X)
            best = min(best, w + int(np.count_nonzero(Y, axis=0).min()))
    return best

def branch(F, A):
    Ai = inverse(F, A)
    if Ai is None:
        return None
    n = A.shape[0]
    if F.q == 2:
        X = np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.int64).T[:, 1:]
        bd = int((X.sum(0) + (A @ X % 2).astype(bool).sum(0)).min())
        bl = int((X.sum(0) + (A.T @ X % 2).astype(bool).sum(0)).min())
        return bd, bl
    # complete for B <= 2*maxw+1 ; maxw = 3 -> certifies up to 7 > k+2 = 6
    m = 3
    bd = min(lowweight_min(F, A, m), lowweight_min(F, Ai, m), 2*m+2)
    bl = min(lowweight_min(F, A.T.copy(), m), lowweight_min(F, Ai.T.copy(), m), 2*m+2)
    return bd, bl

def pg23():
    pts = []
    for v in itertools.product(range(3), repeat=3):
        if any(v):
            first = next(c for c in v if c)
            if first == 1:
                pts.append(v)
    lines = [frozenset(i for i, P in enumerate(pts) if sum(a*b for a, b in zip(L, P)) % 3 == 0) for L in pts]
    return len(pts), lines

def ag4():
    # points (y,x) in F_4^2, lines y = m x + c
    pts = [(y, x) for y in range(4) for x in range(4)]
    idx = {P: i for i, P in enumerate(pts)}
    lines = [frozenset(idx[(MUL4[m, x] ^ c, x)] for x in range(4)) for m in range(4) for c in range(4)]
    return 16, lines

def incidence(v, lines, phi):
    """phi[line index] = point; row Q records line phi^{-1}(Q)."""
    N = np.zeros((v, v), dtype=np.int64)
    for li, Q in enumerate(phi):
        for P in lines[li]:
            N[Q, P] = 1
    return N

def main():
    rng = np.random.default_rng(2026)
    for name, (v, lines) in [("PG(2,3)", pg23()), ("AG(2,4) minus a parallel class", ag4())]:
        k = len(lines[0])
        assert all(len(L) == k for L in lines)
        for q in (2, 4):
            F = GF(q); tried = 0; res = []
            while len(res) < (25 if q == 2 else 8) and tried < 5000:
                phi = rng.permutation(v)
                if any(phi[li] in lines[li] for li in range(v)):
                    continue
                N = incidence(v, lines, phi)
                for alpha in range(1, q):
                    tried += 1
                    A = N.copy()
                    for i in range(v):
                        A[i, i] ^= alpha
                    b = branch(F, A)
                    if b is not None:
                        res.append((alpha, b))
            print(f"{name}, k={k}, over F_{q}: {tried} diagonal-free shifts sampled, "
                  f"{len(res)} invertible, (alpha, (Bd, Bl)) = {res}")
            assert all(b == (k+2, k+2) for _, b in res)
        # control on the search: a bijection with diagonal ones over F_4,
        # alpha not in {0, 1}, where Theorem 3.3(a) gives k+1 = 5
        F = GF(4)
        while True:
            phi = rng.permutation(v)
            if any(phi[li] in lines[li] for li in range(v)):
                N = incidence(v, lines, phi); A = N.copy()
                for i in range(v):
                    A[i, i] ^= 2
                b = branch(F, A)
                if b is not None:
                    break
        print(f"  control: bijection with diagonal ones over F_4, alpha = w: (Bd, Bl) = {b}")
        assert b == (k+1, k+1)
    print("\nALL CHARACTERISTIC-2 CHECKS PASSED (every invertible diagonal-free shift has B = k+2 = 6)")

if __name__ == "__main__":
    main()
