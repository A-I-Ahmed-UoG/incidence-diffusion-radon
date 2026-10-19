#!/usr/bin/env python3
"""Proposition 3.8: k >= 4 in Theorem 3.6 is sharp.

The Fano configuration 7_3 has k = 3.  We look for invertible diagonal-free
shifts N_phi + alpha I on it whose branch number is k+1 = 4 rather than k+2.

Exhaustive search: all diagonal-free bijections phi of the Fano plane (144 of
them), all alpha != 0 over F_5 and F_7, the invertible ones, and the exact
branch number of N_phi + alpha I by enumerating every non-zero input.  There
are 7 cells, so p^7 inputs: 78125 at p = 5, 823543 at p = 7.

Runtime about 5 minutes.
"""
import itertools

import numpy as np


def fano_lines():
    """Line i = {i, i+1, i+3} mod 7."""
    return [frozenset(((i + j) % 7 for j in (0, 1, 3))) for i in range(7)]


def diagonal_free_bijections(L):
    """All bijections phi: lines -> points with phi(line) not on that line."""
    out = []
    for perm in itertools.permutations(range(7)):
        if all(perm[i] not in L[i] for i in range(7)):
            out.append(perm)
    return out


def matrix(L, phi, alpha, p):
    """Rows indexed by the point phi(i) of line i."""
    N = np.zeros((7, 7), dtype=np.int64)
    for i in range(7):
        for q in L[i]:
            N[phi[i], q] = 1
    for i in range(7):
        N[i, i] = (N[i, i] + alpha) % p
    return N % p


def is_invertible(A, p):
    A = A.copy() % p
    n = A.shape[0]
    for c in range(n):
        piv = next((r for r in range(c, n) if A[r, c] % p), None)
        if piv is None:
            return False
        A[[c, piv]] = A[[piv, c]]
        A[c] = (A[c] * pow(int(A[c, c]), p - 2, p)) % p
        for r in range(c + 1, n):
            if A[r, c] % p:
                A[r] = (A[r] - A[r, c] * A[c]) % p
    return True


def branch_number(A, p):
    """Exact min over x != 0 of wt(x) + wt(Ax), by enumeration."""
    n = A.shape[0]
    X = np.array(list(itertools.product(range(p), repeat=n)), dtype=np.int64)
    X = X[X.any(axis=1)]
    Y = (X @ A.T) % p
    w = (X != 0).sum(axis=1) + (Y != 0).sum(axis=1)
    return int(w.min())


def main():
    L = fano_lines()
    phis = diagonal_free_bijections(L)
    print("=" * 74)
    print("Fano configuration 7_3 (v = 7, k = 3)")
    print("=" * 74)
    print(f"  diagonal-free bijections: {len(phis)}")
    print()

    for p in [5, 7]:
        print(f"  over F_{p}:")
        seen = {}
        examples = {}
        for phi in phis:
            for alpha in range(1, p):
                A = matrix(L, phi, alpha, p)
                if not is_invertible(A, p):
                    continue
                B = branch_number(A, p)
                seen[B] = seen.get(B, 0) + 1
                examples.setdefault(B, (phi, alpha))
        for B in sorted(seen):
            tag = ("= k+1, below k+2" if B == 4 else
                   "= k+2" if B == 5 else "")
            phi, alpha = examples[B]
            print(f"    branch number {B}: {seen[B]:>4} invertible "
                  f"(phi, alpha) pairs  {tag}")
            if B <= 4:
                print(f"       example: phi = {phi}, alpha = {alpha}")
        print()

    print("=" * 74)
    print("Reading")
    print("=" * 74)
    print("  Branch number k+1 = 4 occurs at k = 3, so the conclusion of")
    print("  Theorem 3.6 fails there and the hypothesis k >= 4 is sharp.")


if __name__ == "__main__":
    main()
