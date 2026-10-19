#!/usr/bin/env python3
"""Algebraic degree growth, integral distinguishers and zero-sums.

The permutation of Construction 8.1 is run at p = 5 and p = 7, where the whole
state is small enough that sums over affine subspaces can be taken exactly.

The relevant bound.  A function F : F_p^n -> F_p of total degree d satisfies

    sum_{x in V} F(x) = 0   whenever   dim(V) * (p-1) > d,

for V an affine subspace, because a monomial of degree < k(p-1) cannot survive
a k-fold derivative.  One round has degree p-2 (the S-box; the linear layer and
the constant do not raise it), so r rounds have degree at most (p-2)^r, capped
at n(p-1).  An integral distinguisher over r rounds therefore needs

    k  >  (p-2)^r / (p-1)

active cells, and exists only while that number is at most n.

What is measured here:

  1. the smallest k for which summing over every tested k-dimensional
     cell-aligned subspace gives zero in every output coordinate, per round
     count, against the bound above;
  2. the same from the inverse direction, since sigma is an involution and the
     inverse round has the same degree;
  3. inside-out zero-sums: the number of rounds r + s for which a subspace can
     be placed in the middle so that both ends sum to zero;
  4. what this says about the 12-round instance at p = 23, and about the sponge,
     where the attacker controls only the 415 rate cells.

Runtime about 12 minutes.
"""
import itertools

import numpy as np


# ----------------------------------------------------------------- the cipher
def radon_layer(p, alpha):
    """A = M + alpha I on p^2 cells, cells indexed (y, x) row-major."""
    n = p * p
    idx = lambda y, x: y * p + x
    A = np.zeros((n, n), dtype=np.int64)
    for m in range(p):
        for c in range(p):
            for x in range(p):
                A[idx(m, c), idx((m * x + c) % p, x)] += 1
    for i in range(n):
        A[i, i] = (A[i, i] + alpha) % p
    return A % p


def constants(p, R):
    """kappa_r(y,x) = (y + (r+1) x + r + 1)^3."""
    K = np.zeros((R, p * p), dtype=np.int64)
    for r in range(R):
        for y in range(p):
            for x in range(p):
                K[r, y * p + x] = pow((y + (r + 1) * x + r + 1) % p, 3, p)
    return K


def sbox(v, p):
    return np.power(v, p - 2, dtype=np.int64) % p


def permute(states, p, A, K, rounds):
    """Applies `rounds` rounds to each row of the (N, n) array `states`."""
    v = states % p
    for r in range(rounds):
        v = sbox(v, p)
        v = (v + K[r]) % p
        v = (v @ A.T) % p
    return v


def inv_layer(p, A):
    n = A.shape[0]
    Aug = np.hstack([A % p, np.eye(n, dtype=np.int64)]) % p
    r = 0
    for c in range(n):
        piv = next((i for i in range(r, n) if Aug[i, c] % p), None)
        assert piv is not None, "layer is singular"
        Aug[[r, piv]] = Aug[[piv, r]]
        Aug[r] = (Aug[r] * pow(int(Aug[r, c]), p - 2, p)) % p
        for i in range(n):
            if i != r and Aug[i, c] % p:
                Aug[i] = (Aug[i] - Aug[i, c] * Aug[r]) % p
        r += 1
    return Aug[:, n:] % p


def unpermute(states, p, Ainv, K, rounds, total):
    """Inverse of the last `rounds` rounds of a `total`-round permutation."""
    v = states % p
    for r in range(total - 1, total - 1 - rounds, -1):
        v = (v @ Ainv.T) % p
        v = (v - K[r]) % p
        v = sbox(v, p)
    return v


# ------------------------------------------------------------------ the tests
def subspace(p, n, active, base):
    """All p^|active| states agreeing with `base` off `active`."""
    k = len(active)
    N = p ** k
    S = np.tile(base, (N, 1))
    for j, cell in enumerate(active):
        rep = p ** (k - 1 - j)
        col = np.repeat(np.arange(p), rep)
        S[:, cell] = np.tile(col, N // (p * rep))
    return S


def integral_test(p, A, K, rounds, k, trials, rng, inverse=False, Ainv=None,
                  total=None):
    """Is the sum over every tested k-dim cell-aligned subspace zero?"""
    n = p * p
    worst = 0
    for _ in range(trials):
        active = list(rng.choice(n, size=k, replace=False))
        base = rng.integers(0, p, size=n)
        S = subspace(p, n, active, base)
        if inverse:
            out = unpermute(S, p, Ainv, K, rounds, total)
        else:
            out = permute(S, p, A, K, rounds)
        tot = out.sum(axis=0) % p
        worst = max(worst, int((tot != 0).sum()))
        if worst:
            break
    return worst


def degree_bound(p, r, n):
    return min((p - 2) ** r, n * (p - 1))


def needed_k(p, r, n):
    """Smallest k with k(p-1) > degree bound."""
    d = degree_bound(p, r, n)
    k = d // (p - 1) + 1
    return k if k <= n else None


def main():
    rng = np.random.default_rng(2026)

    for p, kmax in [(5, 8), (7, 6)]:
        n = p * p
        alpha = 2
        A = radon_layer(p, alpha)
        Ainv = inv_layer(p, A)
        R = 10
        K = constants(p, R)
        print("=" * 76)
        print(f"p = {p},  n = {n} cells,  alpha = {alpha},  "
              f"S-box degree {p-2},  max degree {n*(p-1)}")
        print("=" * 76)
        print(f"  {'rounds':>7} {'deg bound':>11} {'k needed':>9} "
              f"{'k observed':>11}  verdict")
        for r in range(1, 6):
            d = degree_bound(p, r, n)
            kb = needed_k(p, r, n)
            observed = None
            for k in range(1, kmax + 1):
                bad = integral_test(p, A, K, r, k, 3, rng)
                if bad == 0:
                    observed = k
                    break
            kb_s = str(kb) if kb else "> n"
            obs_s = str(observed) if observed else f"> {kmax}"
            if observed is not None and kb is not None:
                verdict = ("matches bound" if observed == kb else
                           f"beats bound by {kb - observed}" if observed < kb
                           else "worse than bound")
            else:
                verdict = "not reached in range tested"
            print(f"  {r:>7} {d:>11} {kb_s:>9} {obs_s:>11}  {verdict}")

        # the inverse direction
        print()
        print(f"  inverse direction (sigma is an involution, same degree):")
        print(f"  {'rounds':>7} {'k observed':>11}")
        for r in range(1, 4):
            observed = None
            for k in range(1, kmax + 1):
                bad = integral_test(p, A, K, r, k, 3, rng, inverse=True,
                                    Ainv=Ainv, total=R)
                if bad == 0:
                    observed = k
                    break
            print(f"  {r:>7} {str(observed) if observed else '> '+str(kmax):>11}")

        # inside-out zero sums, on a genuine contiguous permutation
        print()
        print("  inside-out zero-sums on a contiguous round sequence:")
        print("    (subspace placed before round m; m rounds inverted"
              " backwards, s applied forwards)")
        best = None
        for total in range(2, 9):
            hit = None
            for m in range(1, total):
                s_fwd = total - m
                for k in range(1, kmax + 1):
                    ok = True
                    for _ in range(3):
                        active = list(rng.choice(n, size=k, replace=False))
                        base = rng.integers(0, p, size=n)
                        S = subspace(p, n, active, base)
                        # backward over rounds m-1, ..., 0
                        bwd = unpermute(S, p, Ainv, K, m, m)
                        # forward over rounds m, ..., m+s_fwd-1
                        fwd = S % p
                        for r in range(m, m + s_fwd):
                            fwd = sbox(fwd, p)
                            fwd = (fwd + K[r]) % p
                            fwd = (fwd @ A.T) % p
                        if (bwd.sum(axis=0) % p != 0).any() or \
                           (fwd.sum(axis=0) % p != 0).any():
                            ok = False
                            break
                    if ok:
                        hit = (m, s_fwd, k)
                        break
                if hit:
                    break
            if hit:
                best = (total,) + hit
                print(f"    {total} rounds: m = {hit[0]} backward, "
                      f"{hit[1]} forward, {hit[2]} active cells, "
                      f"data p^{hit[2]} = {p**hit[2]}")
            else:
                print(f"    {total} rounds: none with k <= {kmax}")
                break
        if best:
            print(f"    longest confirmed: {best[0]} rounds with "
                  f"{best[3]} active cells")
        print()

    # ------------------------------------------------------ the real instance
    print("=" * 76)
    print("Extrapolation to the instance: p = 23, n = 529 cells, R = 12")
    print("=" * 76)
    p, n = 23, 529
    print(f"  maximum degree of any function on the state: n(p-1) = {n*(p-1)}")
    print(f"  {'rounds':>7} {'deg bound':>12} {'active cells needed':>21} "
          f"{'data':>12}  reachable?")
    for r in range(1, 7):
        d = degree_bound(p, r, n)
        kb = needed_k(p, r, n)
        if kb is None:
            print(f"  {r:>7} {d:>12} {'> n, none':>21} {'--':>12}  no")
        else:
            data = f"23^{kb}"
            reach = ("yes" if kb <= n else "no")
            note = "" if kb <= 415 else "  (exceeds the 415 rate cells)"
            print(f"  {r:>7} {d:>12} {kb:>21} {data:>12}  {reach}{note}")
    print()
    print("  So the degree bound alone gives an integral distinguisher on at")
    print("  most 3 rounds of the 12-round permutation, and the subspace it")
    print("  needs is larger than the sponge's rate, so it is not reachable")
    print("  through the hash at all.")


if __name__ == "__main__":
    main()
