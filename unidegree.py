#!/usr/bin/env python3
"""The algebraic degree of a CICO equation in one unknown cell.

Fix every input cell except one, u, and follow one output cell o through R
rounds.  Over the algebraic closure, with the S-box read as x -> 1/x, the
output is a rational function F_R(u), and the CICO equation F_R(u) = t has
deg F_R solutions, where deg is the larger of the degrees of the reduced
numerator and denominator.  This is the degree D of the ideal of the
one-variable CICO system, and a lower bound on the cost of any method that
produces its solutions from a Groebner basis.

Two facts are checked here.

  1. deg F_R <= (N^R)_{o,u}, the number of walks of length R from u to o in
     the support N of A (degrees add under sums and are kept by 1/x);
     at R = 2 this is W(o,u) of Section 8.4.
  2. The exact degree, computed with reduced rational functions over F_p,
     and its growth per round once the state is fully diffused.

Runtime a few minutes.
"""
import os

import numpy as np

from zerosum import radon_layer, constants, sbox_table

rng = np.random.default_rng(99)


def trim(a):
    nz = np.nonzero(a)[0]
    return a[: nz[-1] + 1] if nz.size else np.zeros(1, dtype=np.int64)


def pmul(a, b, p):
    return trim(np.convolve(a, b) % p)


def padd(a, b, p):
    if len(a) < len(b):
        a, b = b, a
    c = a.copy()
    c[: len(b)] = (c[: len(b)] + b) % p
    return trim(c)


def pmod(a, b, p):
    a = trim(a.copy())
    b = trim(b)
    inv = pow(int(b[-1]), p - 2, p)
    while len(a) >= len(b) and a.any():
        k = len(a) - len(b)
        f = (a[-1] * inv) % p
        a[k:] = (a[k:] - f * b) % p
        a = trim(a)
    return a


def pgcd(a, b, p):
    a, b = trim(a), trim(b)
    while b.any():
        a, b = b, pmod(a, b, p)
    return (a * pow(int(a[-1]), p - 2, p)) % p


def pdiv(a, b, p):
    a = trim(a.copy())
    b = trim(b)
    inv = pow(int(b[-1]), p - 2, p)
    q = np.zeros(max(1, len(a) - len(b) + 1), dtype=np.int64)
    while len(a) >= len(b) and a.any():
        k = len(a) - len(b)
        f = (a[-1] * inv) % p
        q[k] = f
        a[k:] = (a[k:] - f * b) % p
        a = trim(a)
    return trim(q)


def reduce(num, den, p):
    g = pgcd(num, den, p)
    if len(g) > 1:
        num, den = pdiv(num, g, p), pdiv(den, g, p)
    return num, den


def const(c):
    return (np.array([c], dtype=np.int64), np.array([1], dtype=np.int64))


def rdeg(r):
    return max(len(trim(r[0])), len(trim(r[1]))) - 1


def run(p, Rmax, dmax=6000):
    n = p * p
    A = radon_layer(p, 2)
    K = constants(p, Rmax)
    s = sbox_table(p)
    N = (A != 0).astype(np.int64)
    u = int(rng.integers(n))
    x = rng.integers(0, p, size=n)
    y = [const(int((s[x[k]] + K[0][k]) % p)) for k in range(n)]
    y[u] = (np.array([0, 1], dtype=np.int64), np.array([1], dtype=np.int64))
    walk = np.zeros(n, dtype=object)
    walk[u] = 1
    out = []
    for R in range(1, Rmax + 1):
        # v = A y  (y already includes the constant of the previous round)
        v = []
        for k in range(n):
            num, den = np.zeros(1, dtype=np.int64), np.ones(1, dtype=np.int64)
            for j in np.nonzero(A[k])[0]:
                a = int(A[k, j])
                yn, yd = y[j]
                num = padd(pmul(num, yd, p), pmul((a * yn) % p, den, p), p)
                den = pmul(den, yd, p)
            v.append(reduce(num, den, p))
        walk = N.dot(walk)
        degs = np.array([rdeg(r) for r in v])
        out.append((R, int(np.median(degs)), int(degs.max()),
                    float(np.median([w for w in walk])),
                    bool(all(degs[k] <= walk[k] for k in range(n)))))
        if degs.max() > dmax or R == Rmax:
            break
        # next round: S-box as 1/x, then the constant
        y = []
        for k in range(n):
            vn, vd = v[k]
            yn, yd = vd, vn
            if len(trim(vn)) == 1 and trim(vn)[0] == 0:
                yn, yd = np.zeros(1, dtype=np.int64), np.ones(1, dtype=np.int64)
            c = int(K[R][k])
            y.append((padd(yn, (c * yd) % p, p), yd))
    return out


def main():
    print("=" * 78)
    print("Degree of the one-cell CICO equation after R rounds (median over")
    print("output cells), against the walk-count bound (N^R)_{o,u}")
    print("=" * 78)
    for p, Rmax in [(5, 7), (7, 6), (11, 5)]:
        res = run(p, Rmax)
        with open(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figdata"), f"degree_p{p}.dat"), "w") as f:
            f.write("R deg walk\n")
            for R, med, mx, wmed, ok in res:
                if med:
                    f.write(f"{R} {med} {wmed:.0f}\n")
        print(f"  p = {p}")
        prev = None
        for R, med, mx, wmed, ok in res:
            growth = f"{med/prev:.2f}" if prev else "--"
            print(f"    R = {R}: median degree {med:>6}, max {mx:>6}, "
                  f"median walk bound {wmed:>9.0f}, bound holds {ok}, "
                  f"growth {growth}")
            prev = med if med else None
        last = res[-1]
        print(f"    degree / walk bound at R = {last[0]}: {last[1]/last[3]:.3f}")


if __name__ == "__main__":
    main()
