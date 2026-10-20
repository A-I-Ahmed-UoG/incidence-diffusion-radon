#!/usr/bin/env python3
"""Three-round integrals through single walks, and the reach of zero-sums.

Let N be the 0/1 support of A = M + alpha I and W = N^2 over Z, so that W(m,i)
counts the cells j with A[m,j] A[j,i] != 0, the two-step walks from cell i to
cell m.  Section 8.4 proves:

  (P) Proposition 8.6: if S is a set of cells such that every cell m has
      some i in S with W(m,i) <= 1, then three rounds of the permutation sum
      to zero over the p^|S| inputs that run through all values on S,
      whatever the S-boxes (as long as they are permutations of F_p, one per
      cell and round), the round constants and the values outside S;

  (B) W(m,i) >= 2 only if m lies in
          B(i) = L(i) u {i} u E(i),
      with L(i) the lines through i (as cells), and E(i) = {(-1/b, a/b)} for
      i = (a,b), b != 0, and E(i) empty for b = 0;

  (C) in particular S = {(a,0), (a',0)} with a, a' distinct and non-zero
      satisfies (P), so three rounds are broken with p^2 inputs for every p.

  (D) Lemma 8.7, first part, the degree asymmetry: from a cell-aligned cube
      of dimension k at the input of a round, r forward rounds sum to zero if
      (p-2)^(r-1) < k(p-1), and r backward rounds if (p-2)^r < k(p-1);
      A^{-1} has almost no zero entries, so (P) has no backward analogue.

This script checks (B) exactly, checks (P) on every pair at p = 5, 7 and on
samples at p = 11, 13 with random per-cell per-round S-boxes and constants,
checks (C), measures the inside-out zero-sums that (D) predicts, and prints
the reach of Corollary 8.8 without round-skipping, which gives the column L of
Table 3.  The second part of Lemma 8.7 (round-skipping), the rows of Table 6
marked "chosen" and the six-round figure within 2^256 states are checked in
zerosum_skip.py.

Runtime about 3 minutes.
"""
import itertools
import math

import numpy as np

from zerosum import radon_layer, constants, inv_layer, fwd, bwd, chunk_sum

rng = np.random.default_rng(2026)


def cell(p, y, x):
    return (y % p) * p + (x % p)


def walks(p, A):
    N = (A % p != 0).astype(np.int64)
    return N @ N


def closed_bad(p, i):
    a, b = divmod(i, p)
    S = {cell(p, m, a - m * b) for m in range(p)}      # lines through i
    S.add(i)
    if b:
        inv = pow(b, p - 2, p)
        S.add(cell(p, (-inv) % p, a * inv % p))
    return S


def random_round_maps(p, n, R):
    """Independent random permutations of F_p for each cell and round."""
    return np.array([[rng.permutation(p) for _ in range(n)] for _ in range(R)],
                    dtype=np.int64)


def fwd_generic(S, A, K, T, rounds, p):
    cols = np.arange(S.shape[1])
    for r in range(rounds):
        S = T[r][cols, S]                  # per-cell permutation
        S = (S + K[r]) % p
        S = (S @ A.T) % p
    return S


def cube_sum(p, active, base, fn):
    return chunk_sum(p, p * p, active, base, fn)


def check_bad_sets(p):
    A = radon_layer(p, 2)
    W = walks(p, A)
    n = p * p
    exact = sub = 0
    for i in range(n):
        bad = {m for m in range(n) if W[m, i] >= 2}
        cf = closed_bad(p, i)
        sub += bad <= cf
        exact += bad == cf
    return sub, exact, n


def check_pairs(p, sample=None):
    n = p * p
    A = radon_layer(p, 2)
    W = walks(p, A)
    good_pairs, bad_pairs = [], []
    for i, j in itertools.combinations(range(n), 2):
        if all(min(W[m, i], W[m, j]) <= 1 for m in range(n)):
            good_pairs.append((i, j))
        else:
            bad_pairs.append((i, j))
    col_good = sum(1 for (i, j) in good_pairs if i % p == j % p)
    col_all = p * math.comb(p, 2)
    stats = dict(good=len(good_pairs), total=math.comb(n, 2),
                 col_good=col_good, col_all=col_all)

    def test(pairs, generic):
        zero = 0
        for (i, j) in pairs:
            base = rng.integers(0, p, size=n)
            if generic:
                T = random_round_maps(p, n, 3)
                K = rng.integers(0, p, size=(3, n))
                fn = lambda S: fwd_generic(S, A, K, T, 3, p)
            else:
                K = constants(p, 3)
                fn = lambda S: fwd(S, p, A, K, 0, 3)
            zero += not cube_sum(p, [i, j], base, fn).any()
        return zero

    if sample:
        gp = [good_pairs[t] for t in rng.choice(len(good_pairs), sample, False)]
        bp = [bad_pairs[t] for t in rng.choice(len(bad_pairs), sample, False)]
    else:
        gp, bp = good_pairs, bad_pairs
    stats["good_tested"] = len(gp)
    stats["good_zero_instance"] = test(gp, False)
    stats["good_zero_generic"] = test(gp, True)
    stats["bad_tested"] = len(bp)
    stats["bad_zero_instance"] = test(bp, False)
    return stats


def check_column0(p):
    n = p * p
    A = radon_layer(p, 2)
    W = walks(p, A)
    ok = True
    for a, a2 in itertools.combinations(range(1, p), 2):
        i, j = cell(p, a, 0), cell(p, a2, 0)
        ok &= all(min(W[m, i], W[m, j]) <= 1 for m in range(n))
    return ok


def density_inverse(p):
    A = radon_layer(p, 2)
    Ai = inv_layer(p, A)
    return int((Ai == 0).sum()), Ai.size, int((A == 0).sum())


def inside_out(p, kb, kf_rounds, active, first=4):
    """Zero-sum at both ends: cube at the input of round `first`."""
    n = p * p
    A = radon_layer(p, 2)
    Ai = inv_layer(p, A)
    K = constants(p, first + kf_rounds + 1)
    base = rng.integers(0, p, size=n)
    sf = cube_sum(p, active, base, lambda S: fwd(S, p, A, K, first, kf_rounds))
    sb = cube_sum(p, active, base, lambda S: bwd(S, p, Ai, K, first - 1, kb))
    return not sf.any(), not sb.any()


def r_fwd(p, k, single):
    r = 1
    while (p - 2) ** r < k * (p - 1):
        r += 1
    return max(r, 3 if single and k >= 2 else 0)


def r_bwd(p, k):
    r = 0
    while (p - 2) ** (r + 1) < k * (p - 1):
        r += 1
    return r


def reach(p, kmax):
    best = (0, None)
    for k in range(1, kmax + 1):
        L = r_fwd(p, k, True) + r_bwd(p, k)
        if L > best[0]:
            best = (L, k)
    return best


def main():
    print("=" * 78)
    print("(B) the cells m with W(m,i) >= 2, against the closed form")
    print("=" * 78)
    for p in [5, 7, 11, 13]:
        sub, exact, n = check_bad_sets(p)
        print(f"  p = {p:>2}: contained in B(i) for {sub}/{n} cells i, "
              f"equal to B(i) for {exact}/{n}")

    print()
    print("=" * 78)
    print("(P) pairs of active cells, three rounds")
    print("=" * 78)
    for p, sample in [(5, None), (7, None), (11, 150), (13, 100)]:
        s = check_pairs(p, sample)
        print(f"  p = {p:>2}: {s['good']} of {s['total']} pairs satisfy the "
              f"criterion ({100*s['good']/s['total']:.1f}%); same-column "
              f"pairs {s['col_good']} of {s['col_all']} "
              f"({100*s['col_good']/s['col_all']:.1f}%)")
        print(f"          criterion pairs tested {s['good_tested']}: zero sum "
              f"for the instance {s['good_zero_instance']}, for random "
              f"S-boxes and constants {s['good_zero_generic']}")
        print(f"          other pairs tested {s['bad_tested']}: zero sum for "
              f"the instance {s['bad_zero_instance']}")

    print()
    print("  one active cell, three rounds, instance, every cell:")
    for p in [5, 7, 11]:
        n = p * p
        A = radon_layer(p, 2)
        K = constants(p, 3)
        z = sum(not cube_sum(p, [i], rng.integers(0, p, size=n),
                             lambda S: fwd(S, p, A, K, 0, 3)).any()
                for i in range(n))
        print(f"  p = {p:>2}: zero sum for {z} of {n} cells")

    print()
    print("(C) column-0 pairs {(a,0),(a',0)}, a != a' non-zero, satisfy (P):")
    for p in [5, 7, 11, 13, 17, 23]:
        print(f"  p = {p:>2}: {check_column0(p)}")

    print()
    print("Proposition 8.9: cells left inactive after two rounds from one cell")
    for p in [5, 7, 11, 13, 23]:
        A = radon_layer(p, 2)
        W = walks(p, A)
        n = p * p
        z = [int((W[:, i] == 0).sum()) for i in range(n)]
        i0 = cell(p, 1, pow(2, p - 2, p))
        three = ((W @ (A % p != 0).astype(np.int64)) > 0).all()
        print(f"  p = {p:>2}: values {sorted(set(z))}, at (1, 1/2): {z[i0]}, "
              f"b = 0 cells all 0: {all(z[cell(p, a, 0)] == 0 for a in range(p))}, "
              f"three rounds reach every cell: {bool(three)}")

    print()
    print("=" * 78)
    print("(D) density of A and A^{-1}")
    print("=" * 78)
    for p in [5, 7, 11, 13]:
        z, tot, zA = density_inverse(p)
        print(f"  p = {p:>2}: A has {zA} zero entries of {tot}, "
              f"A^-1 has {z} ({100*z/tot:.2f}%)")

    print()
    print("=" * 78)
    print("Inside-out zero-sums predicted by (C) and (D), measured")
    print("=" * 78)
    cases = [
        (5, 3, 4, 7, "7 rounds, k = 7"),
        (5, 3, 4, 6, "7 rounds, k = 6 (one below)"),
        (7, 2, 3, 5, "5 rounds, k = 5"),
        (7, 2, 3, 4, "5 rounds, k = 4 (one below)"),
    ]
    for p, kb, kf, k, label in cases:
        n = p * p
        res = []
        for _ in range(3):
            active = list(rng.choice(n, size=k, replace=False))
            res.append(inside_out(p, kb, kf, active))
        print(f"  p = {p}, {label}: {kb} back + {kf} forward, "
              f"(forward zero, backward zero) over 3 trials: {res}")
    for p in [5, 7, 11, 13, 23]:
        active = [cell(p, 1, 0), cell(p, 2, 0)]
        res = [inside_out(p, 1, 3, active) for _ in range(3)]
        print(f"  p = {p:>2}, 4 rounds with k = 2 (column-0 pair), 1 back + 3 "
              f"forward: {res}")

    print()
    print("=" * 78)
    print("Reach of the zero-sums (Corollary 8.8, without round-skipping)")
    print("=" * 78)
    print(f"  {'p':>3} {'cells':>6} {'L any data':>11} {'k':>6} {'log2 data':>10}"
          f" {'L <= 2^256':>11} {'k':>4} {'log2 data':>10} {'k for L+1':>10}")
    for p in [5, 7, 11, 17, 23, 29, 41, 47]:
        n = p * p
        La, ka = reach(p, n)
        kb = min(n, int(256 / math.log2(p)))
        Lb, kbb = reach(p, kb)
        # smallest k that would give one more round than L any data
        knext = None
        for k in range(1, 10 ** 7):
            if r_fwd(p, k, True) + r_bwd(p, k) > La:
                knext = k
                break
        print(f"  {p:>3} {n:>6} {La:>11} {ka:>6} {ka*math.log2(p):>10.1f}"
              f" {Lb:>11} {kbb:>4} {kbb*math.log2(p):>10.1f} {knext:>10}")


if __name__ == "__main__":
    main()
