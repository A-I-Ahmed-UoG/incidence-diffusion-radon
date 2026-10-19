#!/usr/bin/env python3
"""A necessary condition for a rebound attack, and the algebraic system size.

This is a counting argument, not an attack. A rebound distinguisher splits the
permutation into an inbound phase of two rounds, where the attacker solves for
values with the difference distribution table, and an outbound phase over the
remaining rounds, which has to hold probabilistically. The attack can only work
while

    (number of starting points the inbound can supply)
        * (probability of the outbound)   >=   1.

The outbound over R-2 rounds contains at least floor((R-2)/2) * B active
S-boxes by the wide-trail argument, each contributing at most delta/p. The
starting points are bounded by the degrees of freedom: p^n for a distinguisher
on the permutation, p^rate for an attack that has to go through the sponge.

What the condition gives is the round count at which a rebound attack is
excluded by counting alone. It does NOT say an attack exists below that: the
attacker still needs a truncated pattern whose inbound is solvable, and we have
not searched for one. The number below is therefore an upper bound on what
counting can rule out, and the gap between it and R is the part of the analysis
that is missing.

Also reported: the size of the algebraic system for a Groebner attack, using
the cubic relations of Proposition 7.3.

Runtime a few seconds.
"""
import math


def sbox_delta(p):
    return 2 if p % 3 == 2 else 4


def rounds_by_trail(p):
    """The rule Construction 8.1 uses: least even R with the trail bound
    past 2^-256."""
    import cmath
    s = [0] + [pow(x, -1, p) for x in range(1, p)]
    w = cmath.exp(2j * math.pi / p)
    d = max(sum(1 for x in range(p) if (s[(x + a) % p] - s[x]) % p == b)
            for a in range(1, p) for b in range(p))
    L = max(abs(sum(w ** ((a * x + b * s[x]) % p) for x in range(p)))
            for a in range(1, p) for b in range(1, p))
    ldp, llp = math.log2(d / p), math.log2(L * L / p / p)
    B, R = p + 1, 2
    while (R // 2) * B * max(ldp, llp) > -256:
        R += 2
    return R, ldp, llp


def main():
    print("=" * 78)
    print("Rebound: the round count that counting alone excludes")
    print("=" * 78)
    print(f"  {'p':>3} {'n':>6} {'rate':>6} {'B':>4} {'R (adopted)':>15} "
          f"{'R excluded from':>17} {'gap':>6}")
    for p in [11, 17, 23, 29, 41, 47]:
        n = p * p
        B = p + 1
        delta = sbox_delta(p)
        lg = math.log2(delta / p)
        R_trail, _, _ = rounds_by_trail(p)
        R_trail = max(R_trail, 12)        # the adopted round count
        # attacker degrees of freedom through the sponge: the rate
        # (rate chosen as in Section 8: capacity c with p^c > 2^512)
        c = math.ceil(512 / math.log2(p))
        rate = n - c
        if rate <= 0:
            print(f"  {p:>3} {n:>6} {'--':>6} {B:>4} {R_trail:>15} "
                  f"{'n/a':>17} {'':>6}"
                  f"   (capacity {c} exceeds the state: no 256-bit sponge)")
            continue
        dof = rate * math.log2(p)
        # need floor((R-2)/2) * B * |lg| > dof
        R = 4
        while math.floor((R - 2) / 2) * B * (-lg) <= dof:
            R += 2
            if R > 400:
                break
        print(f"  {p:>3} {n:>6} {rate:>6} {B:>4} {R_trail:>15} {R:>17} "
              f"{R - R_trail:>6}")

    print()
    print("  Reading: at every size the counting bound sits far above the")
    print("  round count the trail rule picks, so counting does not exclude a")
    print("  rebound attack on the instance. That is not evidence that one")
    print("  exists; it is evidence that the question is open.")

    print()
    print("=" * 78)
    print("Algebraic system size, using the cubic relations of Proposition 7.3")
    print("=" * 78)
    print(f"  {'p':>3} {'cells':>7} {'R':>4} {'variables':>11} "
          f"{'cubic equations':>17}")
    for p in [11, 17, 23]:
        n = p * p
        R, _, _ = rounds_by_trail(p)
        R = max(R, 12)                    # the adopted round count
        print(f"  {p:>3} {n:>7} {R:>4} {n*R:>11} {2*n*R:>17}")
    print()
    print("  Proposition 7.3 says the S-box graph satisfies no quadratic")
    print("  relation for p >= 11, and exactly two independent cubics, so a")
    print("  degree-2 modelling is unavailable and the lowest-degree one is")
    print("  cubic. We did not run a Groebner computation and give no")
    print("  estimate of the degree of regularity.")


if __name__ == "__main__":
    main()
