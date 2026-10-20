#!/usr/bin/env python3
"""Checks Algorithms 1-2 and the operation counts of Proposition 4.7.

The forward layer is evaluated line by line.  The inverse is evaluated
through the closed forms of Proposition 4.1: M^2 v and M^3 v are constant on
each row m and cost O(p^2), so the only expensive step is the single forward
transform M w.

Both routines carry counters for field additions (a subtraction counts as an
addition) and field multiplications (a scalar inversion is counted as a
multiplication and listed separately).  The outputs are compared against
dense linear algebra over F_p, and the counts against the formulas

    forward:   p^2 (p-1) + p^2  additions,   p^2      multiplications
    inverse:   p^2 (p-1) + O(p^2) additions, 2p^2 + O(p) multiplications

The inverse here skips the shift by zero inside Layer(w, 0), as Proposition
4.7 says an implementation does; the exact counts of Algorithms 1 and 2 as
printed are checked in cost_literal.py.

Runtime about 1 minute.
"""
import numpy as np


class Count:
    def __init__(self):
        self.add = 0
        self.mul = 0
        self.inv = 0


def forward(v, alpha, p, cnt):
    """(M + alpha I) v, with v indexed v[y][x] (value y, column x)."""
    out = np.zeros((p, p), dtype=np.int64)
    for m in range(p):
        for c in range(p):
            s = 0
            for x in range(p):
                s += int(v[(m * x + c) % p, x])
                if x:                      # the first term is a copy
                    cnt.add += 1
            out[m, c] = s % p
    for m in range(p):
        for c in range(p):
            out[m, c] = (out[m, c] + alpha * int(v[m, c])) % p
            cnt.mul += 1
            cnt.add += 1
    return out


def inverse(w, alpha, p, cnt):
    """(M + alpha I)^{-1} w through the inverse formula of Corollary 4.6."""
    ai = pow(alpha, p - 2, p)
    am1 = pow((alpha - 1) % p, p - 2, p)
    cnt.inv += 2
    c0 = ai
    c1 = (-ai * ai) % p
    c2 = (ai * ai * ai) % p
    c3 = (-c2 * am1) % p
    cnt.mul += 5

    # (i) one forward Radon transform, M w
    Mw = np.zeros((p, p), dtype=np.int64)
    for m in range(p):
        for c in range(p):
            s = 0
            for x in range(p):
                s += int(w[(m * x + c) % p, x])
                if x:
                    cnt.add += 1
            Mw[m, c] = s % p

    # (ii) column sums, their total, and M^2 w, M^3 w, all O(p^2)
    C = np.zeros(p, dtype=np.int64)
    for t in range(p):
        s = 0
        for y in range(p):
            s += int(w[y, t])
            if y:
                cnt.add += 1
        C[t] = s % p
    Sigma = 0
    for t in range(p):
        Sigma += int(C[t])
        if t:
            cnt.add += 1
    Sigma %= p

    M2 = np.zeros(p, dtype=np.int64)       # constant on each row m
    M3 = np.zeros(p, dtype=np.int64)
    s3 = (Sigma - int(C[0])) % p
    cnt.add += 1
    for m in range(p):
        if m == 0:
            M2[m] = Sigma
            M3[m] = 0
        else:
            M2[m] = (Sigma - int(C[(-pow(m, p - 2, p)) % p])) % p
            cnt.inv += 1
            cnt.add += 1
            M3[m] = (-s3) % p

    # (iii) the row-constant part of the answer, one value per row
    tail = np.zeros(p, dtype=np.int64)
    for m in range(p):
        tail[m] = (c2 * int(M2[m]) + c3 * int(M3[m])) % p
        cnt.mul += 2
        cnt.add += 1

    out = np.zeros((p, p), dtype=np.int64)
    for m in range(p):
        for c in range(p):
            out[m, c] = (c0 * int(w[m, c]) + c1 * int(Mw[m, c]) + int(tail[m])) % p
            cnt.mul += 2
            cnt.add += 2
    return out


def dense(p, alpha):
    A = np.zeros((p * p, p * p), dtype=np.int64)
    idx = lambda y, x: y * p + x
    for m in range(p):
        for c in range(p):
            for x in range(p):
                A[idx(m, c), idx((m * x + c) % p, x)] += 1
    for i in range(p * p):
        A[i, i] = (A[i, i] + alpha) % p
    return A % p


def inv_mod(A, p):
    n = A.shape[0]
    Aug = np.hstack([A % p, np.eye(n, dtype=np.int64)]) % p
    r = 0
    for c in range(n):
        piv = next((i for i in range(r, n) if Aug[i, c] % p), None)
        assert piv is not None
        Aug[[r, piv]] = Aug[[piv, r]]
        Aug[r] = (Aug[r] * pow(int(Aug[r, c]), p - 2, p)) % p
        for i in range(n):
            if i != r and Aug[i, c] % p:
                Aug[i] = (Aug[i] - Aug[i, c] * Aug[r]) % p
        r += 1
    return Aug[:, n:] % p


def main():
    rng = np.random.default_rng(20260921)
    print("=" * 72)
    print("Algorithms 1 and 2 against dense linear algebra over F_p")
    print("=" * 72)
    for p in [5, 7, 11, 13, 17]:
        for alpha in [2, 3, p - 1]:
            if alpha in (0, 1):
                continue
            v = rng.integers(0, p, size=(p, p)).astype(np.int64)
            A = dense(p, alpha)
            ref = (A @ v.reshape(-1)) % p
            cf = Count()
            got = forward(v, alpha, p, cf)
            assert np.array_equal(got.reshape(-1), ref), (p, alpha)

            Ainv = inv_mod(A, p)
            ref2 = (Ainv @ v.reshape(-1)) % p
            ci = Count()
            got2 = inverse(v, alpha, p, ci)
            assert np.array_equal(got2.reshape(-1), ref2), (p, alpha)
            # and the round trip
            assert np.array_equal(
                forward(got2, alpha, p, Count()).reshape(-1), v.reshape(-1))
        print(f"  p = {p:2d}: forward and inverse agree with the dense matrix "
              f"for every alpha tested, and compose to the identity")

    print()
    print("=" * 72)
    print("Operation counts")
    print("=" * 72)
    print(f"  {'p':>3}  {'fwd add':>9} {'p^3':>9}  {'fwd mul':>8} {'p^2':>7}   "
          f"{'inv add':>9} {'ratio':>6}  {'inv mul':>8} {'2p^2':>7}")
    for p in [5, 7, 11, 13, 17, 19]:
        v = rng.integers(0, p, size=(p, p)).astype(np.int64)
        cf, ci = Count(), Count()
        forward(v, 2, p, cf)
        inverse(v, 2, p, ci)
        assert cf.add == p * p * (p - 1) + p * p == p ** 3
        assert cf.mul == p * p
        assert ci.add <= p * p * (p - 1) + 6 * p * p
        assert ci.mul <= 2 * p * p + 4 * p
        print(f"  {p:3d}  {cf.add:9d} {p**3:9d}  {cf.mul:8d} {p*p:7d}   "
              f"{ci.add:9d} {ci.add/cf.add:6.2f}  {ci.mul:8d} {2*p*p:7d}")

    print()
    print("  forward:  p^2(p-1) additions for M v, then p^2 multiplications")
    print("            and p^2 additions for the shift; total p^3 additions.")
    print("  inverse:  the same p^2(p-1) additions for M w, plus O(p^2);")
    print("            the ratio above tends to 1 as p grows.")
    print()
    print("ALL COST CHECKS PASSED")


if __name__ == "__main__":
    main()
