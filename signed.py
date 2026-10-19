#!/usr/bin/env python3
"""Signed partition subspaces, and the generalised invariance criterion.

The S-box sigma(x) = x^(p-2) is odd: sigma(-x) = -sigma(x), and sigma(0) = 0.
So for a partition Q of the cells and a sign vector eps in {+1,-1}^cells, the
space

    W(Q, eps) = { v : v_u = eps_u h(B_u) for some h }

is mapped to itself by the S-box layer, not only by the unsigned version.
That puts a family of shared subspaces outside Theorem 6.3.  The criterion
generalises (Theorem 6.6): W(Q,eps) is
invariant under M + alpha I if and only if

    eps_u * sum_{P in l_u cap B} eps_P  ==  eps_u' * sum_{P in l_u' cap B} eps_P   (mod p)

for every block B and every u, u' in a common block.  Setting eps = 1 recovers
the unsigned criterion.

This script checks
  1. that the S-box layer really does preserve W(Q,eps);
  2. the criterion against a direct invariance test, on random signed
     partitions at p = 5 and p = 7;
  3. that signed partitions genuinely add subspaces the unsigned criterion
     does not see, by exhibiting invariant W(Q,eps) whose underlying Q is not
     invariant;
  4. the simplest example: the line spanned by the difference of two column
     indicators is invariant under the unkeyed round.

Runtime about 2 minutes.
"""
import itertools
import random

import numpy as np

random.seed(6)


def cells(p):
    return [(a, b) for a in range(p) for b in range(p)]


def radon(p):
    """M with rows indexed by (m,c) and columns by cells, row-major phi."""
    pts = cells(p)
    idx = {q: i for i, q in enumerate(pts)}
    M = np.zeros((p * p, p * p), dtype=np.int64)
    for m in range(p):
        for c in range(p):
            for x in range(p):
                M[idx[(m, c)], idx[((m * x + c) % p, x)]] = 1
    return M, idx


def line_of(p, m, c):
    return frozenset(((m * x + c) % p, x) for x in range(p))


def random_signed_partition(p, rng, nblocks=None):
    pts = cells(p)
    nb = nblocks or rng.randint(2, min(6, len(pts)))
    block = {q: rng.randrange(nb) for q in pts}
    eps = {q: rng.choice([1, p - 1]) for q in pts}    # +1 and -1 in F_p
    return block, eps, nb


def basis(p, block, eps, nb, idx, zero_block=None):
    """Basis of W(Q,eps): one vector per free block."""
    B = []
    for b in range(nb):
        if b == zero_block:
            continue
        v = np.zeros(p * p, dtype=np.int64)
        for q, bq in block.items():
            if bq == b:
                v[idx[q]] = eps[q] % p
        if v.any():
            B.append(v)
    return np.array(B) if B else np.zeros((0, p * p), dtype=np.int64)


def in_span(v, B, p):
    """Is v in the row space of B over F_p?  B has disjoint supports."""
    A = np.vstack([B, v]) % p
    return rank_mod(A, p) == rank_mod(B, p)


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


def criterion(p, block, eps, nb, zero_block=None):
    """The signed congruence.

    Block index `zero_block` (if given) is the distinguished block whose cells
    are forced to zero; the others carry a free value each.  Setting
    zero_block = None and eps = 1 recovers the unsigned criterion of
    Theorem 6.3.
    """
    members = {b: [q for q, bq in block.items() if bq == b] for b in range(nb)}
    free = [b for b in range(nb) if b != zero_block]
    raw = {}
    for u in cells(p):
        line = line_of(p, u[0], u[1])
        raw[u] = [sum(eps[P] for P in line if block[P] == b) % p for b in free]
    # cells of the zero block must see every free block an even (zero) number
    # of times, with signs
    if zero_block is not None:
        for u in members.get(zero_block, []):
            if any(t % p for t in raw[u]):
                return False
    for b in free:
        us = members[b]
        for u1, u2 in itertools.combinations(us, 2):
            r1 = [(eps[u1] * t) % p for t in raw[u1]]
            r2 = [(eps[u2] * t) % p for t in raw[u2]]
            if r1 != r2:
                return False
    return True


def direct_invariance(p, M, alpha, block, eps, nb, idx, zero_block=None):
    B = basis(p, block, eps, nb, idx, zero_block)
    if B.shape[0] == 0:
        return True
    A = (M + alpha * np.eye(p * p, dtype=np.int64)) % p
    for v in B:
        w = (A @ v) % p
        if not in_span(w, B, p):
            return False
    return True


def main():
    p = 5
    M, idx = radon(p)
    print("=" * 72)
    print("1.  The S-box layer preserves W(Q, eps)")
    print("=" * 72)
    rng = random.Random(1)
    ok = 0
    for _ in range(200):
        block, eps, nb = random_signed_partition(p, rng)
        B = basis(p, block, eps, nb, idx)
        if B.shape[0] == 0:
            continue
        # a random element of W(Q,eps)
        h = [rng.randrange(p) for _ in range(nb)]
        v = np.zeros(p * p, dtype=np.int64)
        for q, bq in block.items():
            v[idx[q]] = (eps[q] * h[bq]) % p
        sv = np.array([pow(int(t), p - 2, p) for t in v])
        assert in_span(sv, B, p), "sigma(W) not inside W"
        ok += 1
    print(f"    {ok} random signed partitions at p = {p}: sigma(v) stays in "
          f"W(Q,eps) every time")

    print()
    print("=" * 72)
    print("2.  The signed criterion against a direct invariance check")
    print("=" * 72)
    for p in [5, 7]:
        M, idx = radon(p)
        rng = random.Random(p)
        agree = disagree = inv = 0
        trials = 300 if p == 5 else 150
        for _ in range(trials):
            block, eps, nb = random_signed_partition(p, rng)
            zb = rng.choice([None, 0])
            for alpha in [0, 2]:
                c = criterion(p, block, eps, nb, zb)
                d = direct_invariance(p, M, alpha, block, eps, nb, idx, zb)
                if c == d:
                    agree += 1
                else:
                    disagree += 1
                inv += int(d)
        print(f"    p = {p}: {agree} agreements, {disagree} disagreements "
              f"({inv} of them invariant)")
        assert disagree == 0

    print()
    print("=" * 72)
    print("3.  Signed partitions that the unsigned criterion does not see")
    print("=" * 72)
    for p in [5, 7]:
        M, idx = radon(p)
        # blocks: one free block = columns x0 and x1 with opposite signs,
        # everything else forced to zero
        block, eps = {}, {}
        for q in cells(p):
            a, x = q
            if x in (0, 1):
                block[q] = 1
                eps[q] = 1 if x == 0 else p - 1
            else:
                block[q] = 0
                eps[q] = 1
        c = criterion(p, block, eps, 2, zero_block=0)
        d = all(direct_invariance(p, M, al, block, eps, 2, idx, 0)
                for al in range(0, p))
        ones = {q: 1 for q in block}
        c_unsigned = criterion(p, block, ones, 2, zero_block=0)
        print(f"    p = {p}: the two-column block with opposite signs: "
              f"criterion {c}, direct {d}, unsigned criterion {c_unsigned}")
        assert c and d and not c_unsigned

    print()
    print("=" * 72)
    print("4.  The line spanned by a difference of two column indicators")
    print("=" * 72)
    for p in [5, 7, 11]:
        M, idx = radon(p)
        v = np.zeros(p * p, dtype=np.int64)
        for a in range(p):
            v[idx[(a, 0)]] = 1
            v[idx[(a, 1)]] = p - 1
        assert not (M @ v % p).any(), "M v is not zero"
        for alpha in range(1, p):
            A = (M + alpha * np.eye(p * p, dtype=np.int64)) % p
            w = (A @ v) % p
            assert np.array_equal(w, (alpha * v) % p)
        # sigma maps the line to itself
        for t in range(1, p):
            tv = (t * v) % p
            stv = np.array([pow(int(x), p - 2, p) for x in tv])
            assert np.array_equal(stv, (pow(t, p - 2, p) * v) % p)
        print(f"    p = {p:2d}: M v = 0, (M + alpha I) v = alpha v, and "
              f"sigma(t v) = t^(-1) v, so the line is shared")

    print()
    print("ALL SIGNED-SUBSPACE CHECKS PASSED")


if __name__ == "__main__":
    main()
