"""Linear algebra mod p, and the permutation of zerosum.py, for the scripts
of Sections 8 and 9.

Cells are indexed y*p + x, and row (m,c) of M is identified with cell (m,c),
the row-major identification, as in zerosum.py.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zerosum import radon_layer, constants, inv_layer, sbox_table, fwd, bwd, chunk_sum  # noqa: E402,F401


def radon(p):
    return radon_layer(p, 0)


def cell(p, y, x):
    return (y % p) * p + (x % p)


def rref_mod(A, p):
    """Reduced row echelon form mod p; returns (R, pivot columns)."""
    A = np.array(A, dtype=np.int64) % p
    rows, cols = A.shape
    r = 0
    piv = []
    for c in range(cols):
        if r == rows:
            break
        nz = np.nonzero(A[r:, c])[0]
        if len(nz) == 0:
            continue
        s = r + nz[0]
        if s != r:
            A[[r, s]] = A[[s, r]]
        A[r] = (A[r] * pow(int(A[r, c]), p - 2, p)) % p
        col = A[:, c].copy()
        col[r] = 0
        nzr = np.nonzero(col)[0]
        if len(nzr):
            A[nzr] = (A[nzr] - np.outer(col[nzr], A[r])) % p
        piv.append(c)
        r += 1
    return A, piv


def rank_mod(A, p):
    """Rank mod p (forward elimination only)."""
    A = np.array(A, dtype=np.int64) % p
    rows, cols = A.shape
    r = 0
    for c in range(cols):
        if r == rows:
            break
        nz = np.nonzero(A[r:, c])[0]
        if len(nz) == 0:
            continue
        s = r + nz[0]
        if s != r:
            A[[r, s]] = A[[s, r]]
        A[r] = (A[r] * pow(int(A[r, c]), p - 2, p)) % p
        below = A[r + 1:, c]
        nzr = np.nonzero(below)[0]
        if len(nzr):
            idx = r + 1 + nzr
            A[idx] = (A[idx] - np.outer(A[idx, c], A[r])) % p
        r += 1
    return r


def solve_mod(C, d, p, rng=None):
    """One solution of C u = d mod p (random in the free variables), or None."""
    C = np.array(C, dtype=np.int64) % p
    d = np.array(d, dtype=np.int64).reshape(-1, 1) % p
    R, piv = rref_mod(np.hstack([C, d]), p)
    n = C.shape[1]
    if n in piv:
        return None
    u = np.zeros(n, dtype=np.int64)
    free = [j for j in range(n) if j not in set(piv)]
    if rng is not None and free:
        u[free] = rng.integers(0, p, size=len(free))
    for i, c in enumerate(piv):
        u[c] = (R[i, n] - R[i, :n] @ u + u[c]) % p   # R[i,c] = 1
    assert not ((C @ u - d.ravel()) % p).any()
    return u
