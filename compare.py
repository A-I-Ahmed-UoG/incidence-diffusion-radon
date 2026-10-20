#!/usr/bin/env python3
"""Table 4: an incidence layer against an AES-like layout, and against
layouts over F_{p^2}; the F_{p^2} rows of Table 1.

At a state of n cells over F_p the standard wide-trail alternative is not one
large MDS matrix but a layout of t-cell MDS blocks with a ShiftRows-style
permutation between them, which over four rounds guarantees (t+1)^2 active
S-boxes.  The incidence layer is a single global map and guarantees
2(p+1) over four rounds.  This script puts the two side by side on

  * active S-boxes over 4 and 8 rounds,
  * the resulting trail bounds,
  * the two-round bound of Theorem 9.3,
  * cost per round, counted as field multiplications and additions.

Ball's theorem caps an MDS block over F_p at t <= (p+1)/2 cells, which is what
limits the AES-like column height.

Costs.  A t x t MDS block costs t^2 multiplications and t(t-1) additions;
n/t blocks per round give n*t multiplications and n*(t-1) additions.  The
incidence layer costs p^3 additions and p^2 multiplications for the whole
state of p^2 cells (Proposition 4.7), so p additions and 1 multiplication per
cell.  A ShiftRows-style permutation is free.

The second part prints Table 4 in its long form, one row per p and layout,
every column in cells of F_p:

  inc   M + alpha I on p^2 cells.
  aesp  AES-like, t x t MDS blocks over F_p, t = (p+1)/2, on t*floor(p^2/t)
        cells (the layout above).
  mds2  one n x n MDS matrix over F_{p^2}, n = (p^2-1)/2 cells of F_{p^2}
        (n <= (p^2+1)/2, so a Reed-Solomon or Cauchy matrix exists), read as
        an F_p-linear map on 2n = p^2 - 1 cells of F_p.
  aes2  AES-like with b MDS blocks of t cells over F_{p^2}, t the largest
        with t^2 <= (p^2-1)/2 and b = floor(((p^2-1)/2)/t) >= t, so that the
        cell permutation can send the t cells of a block to t distinct
        blocks; 2tb cells of F_p.

An active cell of F_{p^2} holds one or two active cells of F_p, so an
F_{p^2}-MDS block of t cells has branch number between t+1 and 2t+1 in cells
of F_p (the upper end is the Singleton bound on 2t cells), and only t+1 is
guaranteed.  The linear branch number has the same bound: the F_p-transpose
is, up to a change of basis inside each cell, the F_p-image of the
F_{p^2}-transpose, which is again MDS.  The S-boxes act on cells of F_p and
keep a cell of F_{p^2} active exactly when it was, so the four-round argument
runs in cells of F_{p^2}: (t+1)^2 active S-boxes for aes2, 2(n+1) for mds2.

Costs over F_{p^2}: a product by a fixed c = a + b w, w^2 = d, is
(ax + (bd)y) + (bx + ay)w, that is 4 multiplications and 2 additions (bd
precomputed), and an addition is 2 additions.  A t x t block then costs
4t^2 multiplications and 2t(2t-1) additions, the count of a dense 2t x 2t
matrix over F_p.  The hull column is (B_l - 1) * 2 log2(L/p) with the
guaranteed B_l.

Runtime a few seconds.
"""
import math


def sbox_delta(p):
    return 2 if p % 3 == 2 else 4


def lin_exact(p):
    """The exact linearity L_sigma, as Table 3 uses (not the Weil bound)."""
    import cmath
    s = [0] + [pow(x, -1, p) for x in range(1, p)]
    w = cmath.exp(2j * math.pi / p)
    return max(abs(sum(w ** ((a * x + b * s[x]) % p) for x in range(p)))
               for a in range(1, p) for b in range(1, p))


def row(p):
    n = p * p                       # state, in cells
    t = (p + 1) // 2                # largest MDS block over F_p (Ball)
    B_inc = p + 1                   # branch number of M + alpha I
    B_mds = t + 1                   # branch number of one MDS block
    delta = sbox_delta(p)
    ldp = math.log2(delta / p)
    llp = 2 * math.log2(lin_exact(p) / p)

    # active S-boxes
    act4_inc = 2 * B_inc            # two disjoint round pairs
    act4_mds = B_mds ** 2           # the AES four-round argument
    act8_inc = 4 * B_inc
    act8_mds = 2 * B_mds ** 2

    # the AES-like layout cannot tile n = p^2 cells with t-cell blocks when
    # p^2 is a prime square, so it runs on the largest multiple of t below n
    blocks = n // t
    n_mds = blocks * t

    # cost per round, whole state
    mul_inc, add_inc = n, p * n     # p^2 multiplications, p^3 additions
    mul_mds, add_mds = n_mds * t, n_mds * (t - 1)

    return dict(p=p, n=n, t=t, blocks=blocks, n_mds=n_mds,
                B_inc=B_inc, B_mds=B_mds,
                act4_inc=act4_inc, act4_mds=act4_mds,
                act8_inc=act8_inc, act8_mds=act8_mds,
                dt4_inc=act4_inc * ldp, dt4_mds=act4_mds * ldp,
                lt8_inc=act8_inc * llp, lt8_mds=act8_mds * llp,
                hull_inc=(B_inc - 1) * llp, hull_mds=(B_mds - 1) * llp,
                mul_inc=mul_inc, add_inc=add_inc,
                mul_mds=mul_mds, add_mds=add_mds)


def rows_long(p):
    """The four layouts of the long form of Table 4."""
    llp = 2 * math.log2(lin_exact(p) / p)
    n = p * p
    out = [dict(lay='inc', cells=n, block=n, B=p + 1, act4=2 * (p + 1),
                mul=n, add=p * n, hull=p * llp)]
    t = (p + 1) // 2
    nm = t * (n // t)
    out.append(dict(lay='aesp', cells=nm, block=t, B=t + 1, act4=(t + 1) ** 2,
                    mul=nm * t, add=nm * (t - 1), hull=t * llp))
    m = (p * p - 1) // 2
    assert m <= (p * p + 1) // 2
    out.append(dict(lay='mds2', cells=2 * m, block=2 * m, B=m + 1, Bmax=2 * m + 1,
                    act4=2 * (m + 1), mul=4 * m * m, add=2 * m * (2 * m - 1),
                    hull=m * llp))
    t2 = math.isqrt(m)
    b = m // t2
    assert b >= t2
    out.append(dict(lay='aes2', cells=2 * t2 * b, block=2 * t2, B=t2 + 1,
                    Bmax=2 * t2 + 1, act4=(t2 + 1) ** 2, mul=b * 4 * t2 * t2,
                    add=b * 2 * t2 * (2 * t2 - 1), hull=t2 * llp, t2=t2, b=b))
    return out


def long_form(primes):
    print("=" * 96)
    print("Table 4, long form: the incidence layer against three layouts, in cells of F_p")
    print("=" * 96)
    ok = True
    for p in primes:
        r = row(p)
        inc, aesp = rows_long(p)[:2]
        chk = [(r['act4_inc'], inc['act4']), (r['act4_mds'], aesp['act4']),
               (r['mul_inc'], inc['mul']), (r['mul_mds'], aesp['mul']),
               (r['add_inc'], inc['add']), (r['add_mds'], aesp['add']),
               (round(r['hull_inc'], 1), round(inc['hull'], 1)),
               (round(r['hull_mds'], 1), round(aesp['hull'], 1))]
        ok &= all(a == b for a, b in chk)
    print(f"  inc and aesp rows agree with the tables above at every p: {ok}")
    assert ok
    print()
    print(f"{'p':>3} {'layout':>6} {'cells':>6} {'block':>6} {'B>=':>5} {'Bmax':>5} "
          f"{'act4>=':>7} {'mult':>9} {'add':>9} {'lg hull<=':>10}")
    for p in primes:
        for r in rows_long(p):
            print(f"{p:>3} {r['lay']:>6} {r['cells']:>6} {r['block']:>6} {r['B']:>5} "
                  f"{r.get('Bmax', r['B']):>5} {r['act4']:>7} {r['mul']:>9} {r['add']:>9} "
                  f"{r['hull']:>10.1f}"
                  + (f"   (t={r['t2']} F_p2 cells, b={r['b']} blocks)" if 't2' in r else ''))
        print()
    print("Table 1 rows (F_p cells, N = state size):")
    print("  F_p2-MDS used F_p-linearly: N even, N <= p^2+1, N/2+1 <= B <= N+1, row weight N/2..N")
    print("  AES-like F_p2 blocks of 2t F_p cells: t <= (p^2+1)/2, per-layer t+1 <= B <= 2t+1")
    print()
    r = {x['lay']: x for x in rows_long(23)}
    for k in ['aesp', 'mds2', 'aes2']:
        print(f"p=23 {k}: mult ratio {r[k]['mul']/r['inc']['mul']:.1f}, add ratio inc/{k} "
              f"{r['inc']['add']/r[k]['add']:.2f}, act4 ratio {r[k]['act4']/r['inc']['act4']:.2f}")


def main():
    print("=" * 96)
    print("Incidence layer on p^2 cells vs an AES-like layout of "
          "(p+1)/2-cell MDS blocks on the same state")
    print("=" * 96)
    print(f"{'p':>4}{'cells':>7}{'t':>4}{'B inc':>7}{'B mds':>7}"
          f"{'act4 inc':>10}{'act4 mds':>10}"
          f"{'lgDT4 inc':>11}{'lgDT4 mds':>11}"
          f"{'mul/round':>11}{'mul/round':>11}")
    print(f"{'':>4}{'':>7}{'':>4}{'':>7}{'':>7}{'':>10}{'':>10}"
          f"{'':>11}{'':>11}{'incidence':>11}{'AES-like':>11}")
    rows = [row(p) for p in [11, 17, 23, 29, 41, 47]]
    for r in rows:
        print(f"{r['p']:>4}{r['n']:>7}{r['t']:>4}{r['B_inc']:>7}{r['B_mds']:>7}"
              f"{r['act4_inc']:>10}{r['act4_mds']:>10}"
              f"{r['dt4_inc']:>11.1f}{r['dt4_mds']:>11.1f}"
              f"{r['mul_inc']:>11}{r['mul_mds']:>11}")

    print()
    print("Additions per round, and the two-round bound of Theorem 9.3:")
    print(f"{'p':>4}{'add inc':>10}{'add mds':>10}"
          f"{'lg hull inc':>13}{'lg hull mds':>13}")
    for r in rows:
        print(f"{r['p']:>4}{r['add_inc']:>10}{r['add_mds']:>10}"
              f"{r['hull_inc']:>13.1f}{r['hull_mds']:>13.1f}")

    print()
    print("Reading, at p = 23:")
    r = rows[2]
    print(f"  incidence layer: {r['n']} cells; AES-like: {r['blocks']} "
          f"blocks of t = {r['t']} (Ball's cap) = {r['n_mds']} cells, since "
          f"{r['n']} = {int(r['n']**0.5)}^2 does not tile")
    print(f"  four rounds: {r['act4_mds']} active S-boxes for the AES-like "
          f"layout against {r['act4_inc']} for the incidence layer, "
          f"a factor {r['act4_mds']/r['act4_inc']:.1f}")
    print(f"  cost per round: {r['mul_inc']} multiplications and "
          f"{r['add_inc']} additions for the incidence layer,")
    print(f"                  {r['mul_mds']} multiplications and "
          f"{r['add_mds']} additions for the AES-like layout")
    print(f"  so the incidence layer is cheaper in multiplications by "
          f"{r['mul_mds']/r['mul_inc']:.0f}x and dearer in additions by "
          f"{r['add_inc']/r['add_mds']:.1f}x")
    print()
    print("  The AES-like layout wins on active S-boxes at every size, and the")
    print("  incidence layer's case rests on the multiplication count, on the")
    print("  branch number following from a theorem rather than a search, and")
    print("  on there being one global map instead of blocks plus a shuffle.")
    print()
    long_form([11, 17, 23, 29, 41, 47])


if __name__ == "__main__":
    main()
