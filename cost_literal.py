#!/usr/bin/env python3
"""Proposition 4.7: Algorithms 1 and 2 instrumented exactly as printed.

A subtraction counts as an addition, negating a constant is free, and index
arithmetic (m x + c, -m^{-1} as an index) is not counted, except that the
inversions m^{-1} are listed with the inversions, as in the proof.  The
constants are c0 = a^{-1}, c1 = -(c0*c0), c2 = c0*(c0*c0) and
c3 = -(c2 * (a-1)^{-1}): three multiplications, two inversions, one addition.

Both routines are checked against dense linear algebra and against each
other for every alpha not in {0, 1} and p <= 19.  The counts are compared
with p^3 additions and p^2 multiplications forward, and p^3 + 3p^2 + 3p - 3
additions, 3p^2 + 2p + 2 multiplications and p + 1 inversions for the
inverse; without the shift by zero inside Layer(w, 0), p^2 additions and p^2
multiplications fewer.  cost.py checks an implementation that leaves that
shift out.

Runtime a few seconds.
"""
import numpy as np

class C:
    def __init__(self):
        self.add = self.mul = self.inv = 0

def layer(v, a, p, k):
    w = np.zeros((p, p), dtype=np.int64)
    for m in range(p):
        for c in range(p):
            s = int(v[c, 0])                               # line 3: a copy
            for x in range(1, p):
                s = (s + int(v[(m*x + c) % p, x])) % p; k.add += 1
            t = a * int(v[m, c]) % p; k.mul += 1           # line 6
            w[m, c] = (s + t) % p; k.add += 1
    return w

def layer_inv(w, a, p, k):
    c0 = pow(a, -1, p); k.inv += 1
    am1 = (a - 1) % p; k.add += 1
    i1 = pow(am1, -1, p); k.inv += 1
    sq = c0 * c0 % p; k.mul += 1
    c1 = (-sq) % p
    c2 = c0 * sq % p; k.mul += 1
    c3 = (-(c2 * i1)) % p; k.mul += 1
    u = layer(w, 0, p, k)                                   # Layer(w, 0), as written
    V = [0]*p
    for t in range(p):
        s = int(w[0, t])
        for y in range(1, p):
            s = (s + int(w[y, t])) % p; k.add += 1
        V[t] = s
    Sig = V[0]
    for t in range(1, p):
        Sig = (Sig + V[t]) % p; k.add += 1
    d = [0]*p
    for m in range(p):
        if m == 0:
            d[m] = c2 * Sig % p; k.mul += 1
        else:
            mi = pow(m, -1, p); k.inv += 1
            t1 = (Sig - V[(-mi) % p]) % p; k.add += 1
            t2 = (Sig - V[0]) % p; k.add += 1            # recomputed for every m, as written
            d[m] = (c2 * t1 - c3 * t2) % p; k.mul += 2; k.add += 1
    v = np.zeros((p, p), dtype=np.int64)
    for m in range(p):
        for c in range(p):
            v[m, c] = (c0 * int(w[m, c]) + c1 * int(u[m, c]) + d[m]) % p
            k.mul += 2; k.add += 2
    return v

def dense(p, a):
    A = np.zeros((p*p, p*p), dtype=np.int64)
    for m in range(p):
        for c in range(p):
            for x in range(p):
                A[m*p + c, ((m*x + c) % p)*p + x] += 1
    return (A + a*np.eye(p*p, dtype=np.int64)) % p

def main():
    rng = np.random.default_rng(7)
    print(f"{'p':>3} {'fwd add':>8} {'fwd mul':>8} | {'inv add':>8} {'formula':>8} {'inv mul':>8} {'formula':>8} {'inv':>4} | without the zero shift: add   mul")
    for p in [3, 5, 7, 11, 13, 17, 19]:
        for a in range(2, p):
            v = rng.integers(0, p, (p, p))
            kf, ki = C(), C()
            w = layer(v, a, p, kf)
            assert np.array_equal(w.reshape(-1), dense(p, a) @ v.reshape(-1) % p)
            v2 = layer_inv(w, a, p, ki)
            assert np.array_equal(v2, v)
            assert (kf.add, kf.mul) == (p**3, p**2)
            assert ki.add == p**3 + 3*p*p + 3*p - 3, ki.add
            assert ki.mul == 3*p*p + 2*p + 2, ki.mul
            assert ki.inv == p + 1
            # Layer(w, 0) costs p^3 additions and p^2 multiplications, and its
            # last line, the shift by zero, accounts for p^2 of each
            kz = C()
            layer(w, 0, p, kz)
            assert (kz.add, kz.mul) == (p**3, p*p)
            assert ki.add - p*p == p**3 + 2*p*p + 3*p - 3 and ki.mul - p*p == 2*p*p + 2*p + 2
        print(f"{p:3d} {kf.add:8d} {kf.mul:8d} | {ki.add:8d} {p**3+3*p*p+3*p-3:8d} {ki.mul:8d} {3*p*p+2*p+2:8d} {ki.inv:4d} |"
              f" {ki.add - p*p:28d} {ki.mul - p*p:5d}")
    print("\nAs written: forward p^3 adds, p^2 mults; inverse p^3+3p^2+3p-3 adds,")
    print("3p^2+2p+2 mults, p+1 inversions.  The call Layer(w,0) contributes p^2 mults")
    print("by zero and p^2 additions of zero; without them: p^3+2p^2+3p-3 adds, 2p^2+2p+2 mults.")
    print("ALL LITERAL COST CHECKS PASSED")

if __name__ == "__main__":
    main()
