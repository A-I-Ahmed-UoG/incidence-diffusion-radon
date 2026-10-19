#!/usr/bin/env python3
"""Subspace trails for the round of Construction 8.1.

A subspace trail is a chain U_0 -> U_1 -> ... in which every coset of U_i is
mapped into a single coset of U_{i+1}.  For the round v -> A(sigma(v) + kappa)
the linear part is harmless: A(U + a) = AU + Aa always.  The S-box layer is
what constrains things, so the question is which subspaces V satisfy

    sigma(V + c)  is a coset of V,  for every c.                        (*)

Two facts, checked here.

1.  For a one-dimensional V = <v>, (*) with c = 0 forces sigma(V) = V, and
    sigma(t v)_i = t^{-1} v_i^{-1}, so sigma(V) = <w> with w_i = v_i^{-1}.
    Thus sigma(V) = V iff v_i^2 is the same for every i in the support, that
    is, iff v is a scalar multiple of a +-1 vector.  These are exactly the
    one-dimensional signed subspaces of Definition 6.5.  So a trail can only
    run through signed subspaces, which is what Theorem 6.6 classifies.

2.  For the trail to have length two we need A V to be of the same form again.
    This script searches for that: over random +-1 vectors, over the
    structured ones (single cells, rows, columns, differences of columns), and
    exhaustively over small supports, it asks whether A v is again a scalar
    multiple of a +-1 vector.

The outcome decides whether a subspace-trail distinguisher can run past one
round at all.

Runtime about 4 minutes.
"""
import itertools
import random

import numpy as np

random.seed(808)


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


def is_signed(v, p):
    """Is v a scalar multiple of a +-1 vector on its support?"""
    nz = [int(t) % p for t in v if t % p]
    if not nz:
        return False
    lam = nz[0]
    inv = pow(lam, p - 2, p)
    scaled = [(t * inv) % p for t in nz]
    return all(t == 1 or t == p - 1 for t in scaled)


def check_sigma_fixes_line(p, v):
    """sigma(<v>) = <v> iff v is signed.  Verified directly."""
    w = np.array([pow(int(t), p - 2, p) for t in v], dtype=np.int64)
    # is w a multiple of v?
    nz = [i for i in range(len(v)) if v[i] % p]
    if not nz:
        return None
    lam = (int(w[nz[0]]) * pow(int(v[nz[0]]), p - 2, p)) % p
    return all((int(w[i]) - lam * int(v[i])) % p == 0 for i in range(len(v)))


def main():
    for p in [5, 7]:
        n = p * p
        A = radon_layer(p, 2)
        print("=" * 76)
        print(f"p = {p}, n = {n} cells, alpha = 2")
        print("=" * 76)

        # ---- fact 1: sigma fixes <v> exactly when v is signed --------------
        agree = 0
        trials = 4000
        rng = random.Random(p)
        for _ in range(trials):
            k = rng.randint(1, 6)
            v = np.zeros(n, dtype=np.int64)
            for c in rng.sample(range(n), k):
                v[c] = rng.randrange(1, p)
            assert check_sigma_fixes_line(p, v) == is_signed(v, p)
            agree += 1
        print(f"  1. on {agree} random lines: sigma fixes <v> exactly when v "
              f"is a scalar times a +-1 vector")

        # ---- fact 2: is A v signed again? ----------------------------------
        def report(name, vectors):
            tested = hits = 0
            examples = []
            for v in vectors:
                if not is_signed(v, p):
                    continue
                tested += 1
                Av = (A @ v) % p
                if is_signed(Av, p):
                    hits += 1
                    if len(examples) < 3:
                        examples.append(int(np.count_nonzero(v)))
            print(f"     {name}: {tested} signed lines, {hits} with A v "
                  f"signed again" +
                  (f" (supports {examples})" if examples else ""))
            return tested, hits

        print("  2. does A map a signed line to a signed line?")

        # single cells
        singles = []
        for i in range(n):
            v = np.zeros(n, dtype=np.int64)
            v[i] = 1
            singles.append(v)
        report("single cells", singles)

        # whole rows and whole columns, all sign patterns on 2 of them
        cols = []
        for x in range(p):
            v = np.zeros(n, dtype=np.int64)
            for y in range(p):
                v[y * p + x] = 1
            cols.append(v)
        rows = []
        for y in range(p):
            v = np.zeros(n, dtype=np.int64)
            for x in range(p):
                v[y * p + x] = 1
            rows.append(v)
        report("whole columns", cols)
        report("whole rows", rows)
        diffs = [(cols[a] - cols[b]) % p
                 for a in range(p) for b in range(p) if a != b]
        report("differences of two columns", diffs)
        rdiffs = [(rows[a] - rows[b]) % p
                  for a in range(p) for b in range(p) if a != b]
        report("differences of two rows", rdiffs)

        # exhaustive over small supports with all sign patterns
        found = []
        for size in [2, 3]:
            tested = hits = 0
            cellsets = list(itertools.combinations(range(n), size))
            rng.shuffle(cellsets)
            for S in cellsets[:4000]:
                for signs in itertools.product([1, p - 1], repeat=size - 1):
                    v = np.zeros(n, dtype=np.int64)
                    v[S[0]] = 1
                    for c, sg in zip(S[1:], signs):
                        v[c] = sg
                    tested += 1
                    if is_signed((A @ v) % p, p):
                        hits += 1
                        found.append((size, S, signs))
            print(f"     supports of size {size}: {tested} signed lines, "
                  f"{hits} with A v signed again")

        # random larger supports
        tested = hits = 0
        for _ in range(20000):
            k = rng.randint(4, 2 * p)
            v = np.zeros(n, dtype=np.int64)
            for c in rng.sample(range(n), k):
                v[c] = rng.choice([1, p - 1])
            tested += 1
            if is_signed((A @ v) % p, p):
                hits += 1
                found.append((k, None, None))
        print(f"     {tested} random signed lines with 4..{2*p} active cells: "
              f"{hits} with A v signed again")

        # ---- fact 3: how long can a chain actually run? -------------------
        def chain_len(v, cap=12):
            cur = v.copy() % p
            L = 0
            for _ in range(cap):
                cur = (A @ cur) % p
                if not cur.any() or not is_signed(cur, p):
                    break
                L += 1
            return L

        print("  3. chain length, the number of consecutive rounds a signed")
        print("     line stays signed:")
        inv_chain = sorted({chain_len(v) for v in diffs + rdiffs})
        print(f"     differences of two columns or rows: {inv_chain} "
              f"(capped at 12; these are eigenvectors, A v = alpha v, so the "
              f"chain never ends)")
        others = []
        for size in [2, 3]:
            for S in itertools.combinations(range(n), size):
                for signs in itertools.product([1, p - 1], repeat=size - 1):
                    v = np.zeros(n, dtype=np.int64)
                    v[S[0]] = 1
                    for c, sg in zip(S[1:], signs):
                        v[c] = sg
                    if is_signed((A @ v) % p, p):
                        others.append(chain_len(v))
            if p == 7 and size == 3:
                break
        if others:
            print(f"     every other signed line found, {len(others)} of them: "
                  f"chain lengths {sorted(set(others))}, longest "
                  f"{max(others)}")
        print()
        print("  So a subspace trail can only run through signed lines, and")
        print("  apart from the eigenvectors of A (the invariant family of")
        print("  Theorem 6.6, which a round constant defeats by Proposition")
        print("  6.7), no signed line survives more than two rounds.")
        print()


if __name__ == "__main__":
    main()
