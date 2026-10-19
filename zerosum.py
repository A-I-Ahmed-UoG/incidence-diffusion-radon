#!/usr/bin/env python3
"""How far do integral and zero-sum distinguishers reach, as p grows?

integral.py finds integral distinguishers on r rounds needing k active
cells, with k far below the (p-2)^r degree bound.  The pattern there looks
like k = 2^(r-1) - 1, independent of p.  If that is right it decides the
margin of the instance, so here it is measured at three primes with k pushed
as far as the running time allows.

Sums over p^k states are accumulated in chunks, so k is limited by time rather
than memory.

For each p and r we report the smallest k for which a cell-aligned subspace of
dimension k sums to zero in every output coordinate, testing a few random
choices of active cells and base point.  The attacker only needs one good
choice, so a "no" here is the stronger statement and a "yes" is the weaker one;
we report both the all-trials and the any-trial version.

Runtime about 25 minutes.
"""
import numpy as np

CHUNK = 1 << 17


def radon_layer(p, alpha):
    n = p * p
    A = np.zeros((n, n), dtype=np.int64)
    for m in range(p):
        for c in range(p):
            for x in range(p):
                A[m * p + c, ((m * x + c) % p) * p + x] += 1
    for i in range(n):
        A[i, i] = (A[i, i] + alpha) % p
    return A % p


def constants(p, R):
    K = np.zeros((R, p * p), dtype=np.int64)
    for r in range(R):
        for y in range(p):
            for x in range(p):
                K[r, y * p + x] = pow((y + (r + 1) * x + r + 1) % p, 3, p)
    return K


def inv_layer(p, A):
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


def sbox_table(p):
    return np.array([0] + [pow(x, p - 2, p) for x in range(1, p)],
                    dtype=np.int64)


def fwd(v, p, A, K, first, count):
    T = sbox_table(p)                 # a table: x^(p-2) overflows int64
    for r in range(first, first + count):
        v = T[v % p]
        v = (v + K[r]) % p
        v = (v @ A.T) % p
    return v


def bwd(v, p, Ainv, K, last, count):
    for r in range(last, last - count, -1):
        v = (v @ Ainv.T) % p
        v = (v - K[r]) % p
        v = sbox_table(p)[v]
    return v


def chunk_sum(p, n, active, base, apply_fn):
    """sum over the p^k cell-aligned subspace, in chunks."""
    k = len(active)
    total = np.zeros(n, dtype=np.int64)
    N = p ** k
    done = 0
    digits = np.empty((min(CHUNK, N), k), dtype=np.int64)
    while done < N:
        m = min(CHUNK, N - done)
        idx = np.arange(done, done + m)
        for j in range(k):
            digits[:m, j] = (idx // (p ** (k - 1 - j))) % p
        S = np.tile(base, (m, 1))
        for j, cell in enumerate(active):
            S[:, cell] = digits[:m, j]
        out = apply_fn(S)
        total = (total + out.sum(axis=0)) % p
        done += m
    return total


def smallest_k(p, n, A, Ainv, K, r, kmax, rng, trials=3, direction="f"):
    """Smallest k where the sum vanishes; returns (all-trials k, any-trial k)."""
    k_all = k_any = None
    for k in range(1, kmax + 1):
        zeros = 0
        for _ in range(trials):
            active = list(rng.choice(n, size=k, replace=False))
            base = rng.integers(0, p, size=n)
            if direction == "f":
                f = lambda S: fwd(S, p, A, K, 0, r)
            else:
                f = lambda S: bwd(S, p, Ainv, K, r - 1, r)
            if not (chunk_sum(p, n, active, base, f) % p).any():
                zeros += 1
        if zeros and k_any is None:
            k_any = k
        if zeros == trials:
            k_all = k
            break
    return k_all, k_any


def main():
    rng = np.random.default_rng(31337)
    print("=" * 78)
    print("Smallest number of active cells for an integral distinguisher")
    print("=" * 78)
    print(f"  {'p':>3} {'rounds':>7} {'k (all trials)':>16} "
          f"{'k (some trial)':>16} {'2^(r-1)-1':>11}  data p^k")
    law = {}
    for p, kmax, rmax in [(5, 9, 5), (7, 8, 5), (11, 7, 4)]:
        n = p * p
        A = radon_layer(p, 2)
        Ainv = inv_layer(p, A)
        K = constants(p, 12)
        for r in range(2, rmax + 1):
            ka, kn = smallest_k(p, n, A, Ainv, K, r, kmax, rng)
            pred = 2 ** (r - 1) - 1
            data = f"{p}^{ka}" if ka else "--"
            print(f"  {p:>3} {r:>7} {str(ka) if ka else '> '+str(kmax):>16} "
                  f"{str(kn) if kn else '> '+str(kmax):>16} {pred:>11}  {data}")
            law[(p, r)] = ka
        print()

    print("=" * 78)
    print("Does k = 2^(r-1) - 1 hold?")
    print("=" * 78)
    agree = total = 0
    for (p, r), ka in sorted(law.items()):
        if ka is None:
            continue
        total += 1
        agree += int(ka == 2 ** (r - 1) - 1)
    print(f"  {agree} of {total} measured values match the law exactly")
    print()
    print("  If it holds, then r rounds need k = 2^(r-1) - 1 active cells and")
    print("  p^k data, independent of p.  At p = 23 that reads:")
    print(f"  {'rounds':>7} {'k':>7} {'log2 data':>11}  within 2^256?")
    for r in range(2, 10):
        k = 2 ** (r - 1) - 1
        lg = k * np.log2(23)
        ok = "yes" if lg <= 256 and k <= 529 else "no"
        print(f"  {r:>7} {k:>7} {lg:>11.1f}  {ok}")


if __name__ == "__main__":
    main()
