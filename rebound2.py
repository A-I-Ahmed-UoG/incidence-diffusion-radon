#!/usr/bin/env python3
"""A five-round rebound distinguisher on the permutation of Construction 8.1.

Trail (rounds 0..4, each round S-box layer, constant, then A):

  round 0   input difference supported on a set X0 of a cells
            (the S-box layer keeps the support), A spreads it
  round 1   INBOUND: the S-box layer of round 1 is matched through the
            difference table; its output difference is lambda * A^{-1} e_j,
            so that after A only the cell j is active
  round 2   one active cell j  ->  its p+1 neighbours          (probability 1)
  round 3   p+1 cells          ->  Z(j), all but p-1 cells     (probability 1)
  round 4   S-box keeps Z(j), A maps span(Z(j)) to a subspace of
            codimension p-1                                    (probability 1)

So the output difference lies in A span(Z(j)), of dimension b = n-(p-1),
and the input difference in span(X0), of dimension a.  For an ideal
permutation, finding such a pair is the limited-birthday problem, whose
complexity is max{ p^((n-max(a,b))/2), p^(n-a-b) } (Gilbert-Peyrin, FSE 2010;
lower bound by Iwamoto-Peyrin-Sasaki, ASIACRYPT 2013).  With a = b this is
p^((p-1)/2), which is 2^49.8 at p = 23.

The inbound.  The output difference of the S-box layer of round 1 is fixed,
beta = A^{-1} e_j, with support S_j (N cells).  The input difference must be
A y with y supported on X0, and must have support exactly S_j, which puts
n - N linear conditions on y; what is left is a subspace of dimension d of
candidate input differences g on S_j.  A cell i is solvable when the
difference-table entry DT(g_i, beta_i) is non-zero.  We fix g on an
information set of d coordinates, choosing values in the solvable sets, and
the other N - d coordinates are then determined; each is solvable with
probability about 1/2.  The expected number of trials is therefore about
(p / #solvable)^(N-d), with N - d = n - a when the conditions are
independent.

The script builds the inbound, runs the search, turns the solution into an
actual pair of inputs, and checks the pair through five rounds of the real
permutation.  It does this at p = 11, 13 and 23.

Runtime about 1 minute.
"""
import math
import os
import time

import numpy as np

from zerosum import radon_layer, inv_layer, constants, sbox_table
from singlepath import walks, cell

rng = np.random.default_rng(1729)


def rref(M, p):
    """Row-reduce M over F_p; return (R, pivot columns)."""
    M = np.array(M, dtype=np.int64) % p
    rows, cols = M.shape
    piv = []
    r = 0
    for c in range(cols):
        if r == rows:
            break
        nz = np.nonzero(M[r:, c])[0]
        if nz.size == 0:
            continue
        k = r + nz[0]
        M[[r, k]] = M[[k, r]]
        M[r] = (M[r] * pow(int(M[r, c]), p - 2, p)) % p
        col = M[:, c].copy()
        col[r] = 0
        nzr = np.nonzero(col)[0]
        if nzr.size:
            M[nzr] = (M[nzr] - np.outer(col[nzr], M[r])) % p
        piv.append(c)
        r += 1
    return M[:r], piv


def nullspace(M, p):
    R, piv = rref(M, p)
    cols = M.shape[1]
    free = [c for c in range(cols) if c not in piv]
    B = np.zeros((len(free), cols), dtype=np.int64)
    for t, f in enumerate(free):
        B[t, f] = 1
        for i, c in enumerate(piv):
            B[t, c] = (-R[i, f]) % p
    return B


def ddt_table(p):
    s = sbox_table(p)
    D = np.zeros((p, p), dtype=np.int64)
    for a in range(p):
        for x in range(p):
            D[a, (s[(x + a) % p] - s[x]) % p] += 1
    return D


def run(p, a_cells=None, verbose=True):
    n = p * p
    A = radon_layer(p, 2)
    Ai = inv_layer(p, A)
    W = walks(p, A)
    D = ddt_table(p)
    s = sbox_table(p)

    # source j: p-1 cells unreachable in two rounds, fewest non-zero in A^-1 e_j
    cands = [j for j in range(n) if int((W[:, j] == 0).sum()) == p - 1]
    j = min(cands, key=lambda j: int((Ai[:, j] != 0).sum()))
    beta = Ai[:, j] % p
    Sj = np.nonzero(beta)[0]
    N = len(Sj)
    Z = np.nonzero(W[:, j] > 0)[0]            # active cells after two rounds
    b_dim = len(Z)

    # input set X0 of a cells: the default a = b
    a = b_dim if a_cells is None else a_cells
    X0 = np.sort(rng.choice(n, size=a, replace=False))
    notS = np.setdiff1d(np.arange(n), Sj)

    # y on X0 with (A y) zero off S_j
    Nb = nullspace(A[np.ix_(notS, X0)], p)    # rows: basis of y-space
    d = Nb.shape[0]
    G = (Nb @ A[np.ix_(Sj, X0)].T) % p        # d x N, rows span the g's
    R, piv = rref(G, p)                       # systematic: coords piv are free
    assert len(piv) == d
    rest = [c for c in range(N) if c not in piv]
    # g_rest = g_piv @ R[:, rest]
    P = R[:, rest] % p

    # solvable sets
    allowed = [np.nonzero(D[:, beta[Sj[c]]])[0] for c in range(N)]
    frac = np.mean([len(al) / p for al in allowed])
    ok_rest = np.zeros((len(rest), p), dtype=bool)
    for t, c in enumerate(rest):
        ok_rest[t, allowed[c]] = True

    expected = 1.0
    for c in rest:
        expected *= p / len(allowed[c])

    # search: fix g on the pivots from the solvable sets, vary 40 of them
    vary = min(40, d)
    base = np.array([rng.choice(allowed[c]) for c in piv], dtype=np.int64)
    Pv = P[:vary]                              # vary the first `vary` pivots
    const = (base[vary:] @ P[vary:]) % p
    trials = 0
    t0 = time.time()
    found = None
    batch = 1 << 16
    while found is None:
        vals = np.stack([rng.choice(allowed[piv[t]], size=batch)
                         for t in range(vary)], axis=1)
        gr = (const + vals @ Pv) % p
        good = ok_rest[np.arange(len(rest)), gr].all(axis=1)
        idx = np.nonzero(good)[0]
        if idx.size:
            k = idx[0]
            trials += k + 1
            gp = base.copy()
            gp[:vary] = vals[k]
            found = (gp, gr[k])
        else:
            trials += batch
    elapsed = time.time() - t0

    g = np.zeros(N, dtype=np.int64)
    g[piv] = found[0]
    g[rest] = found[1]
    dX1 = np.zeros(n, dtype=np.int64)
    dX1[Sj] = g

    # values at the input of the S-box layer of round 1
    v = rng.integers(0, p, size=n)
    for c, i in enumerate(Sj):
        xs = [x for x in range(p)
              if (s[(x + g[c]) % p] - s[x]) % p == beta[i]]
        v[i] = xs[0]
    v2 = (v + dX1) % p

    K = constants(p, 5)
    def back0(w):                              # invert round 0
        return s[((Ai @ w) - K[0]) % p]
    def fwd(x, rounds):
        for r in range(rounds):
            x = (A @ ((s[x] + K[r]) % p)) % p
        return x

    x1, x2 = back0(v), back0(v2)
    din = (x2 - x1) % p
    in_ok = not din[np.setdiff1d(np.arange(n), X0)].any()
    y1, y2 = fwd(x1, 5), fwd(x2, 5)
    after1 = (fwd(x2, 2) - fwd(x1, 2)) % p
    one_cell = set(np.nonzero(after1)[0].tolist()) == {j}
    dout = (y2 - y1) % p
    pre = (Ai @ dout) % p                     # difference before the last A
    out_ok = not pre[np.setdiff1d(np.arange(n), Z)].any()

    # supports of the difference at the input of each S-box layer, rounds 0-4
    sup = []
    z1, z2 = x1.copy(), x2.copy()
    for r in range(5):
        sup.append(np.nonzero((z2 - z1) % p)[0])
        z1 = (A @ ((s[z1] + K[r]) % p)) % p
        z2 = (A @ ((s[z2] + K[r]) % p)) % p

    lg = math.log2(p)
    generic = max((n - max(a, b_dim)) / 2, n - a - b_dim) * lg
    if verbose:
        print(f"p = {p}: n = {n}, source j = {divmod(j, p)}, N = |S_j| = {N} "
              f"({100*N/n:.1f}% of cells)")
        print(f"   a = |X0| = {a}, b = dim of output space = {b_dim}, "
              f"d = {d}, N - d = {N - d} (n - a = {n - a})")
        print(f"   solvable fraction per cell {frac:.3f}; expected trials "
              f"2^{math.log2(expected):.1f}, used 2^{math.log2(trials):.1f} "
              f"in {elapsed:.1f} s")
        print(f"   pair checked through 5 rounds: input difference in "
              f"span(X0) {in_ok}, one active cell after round 1 {one_cell}, "
              f"output difference in A span(Z(j)) {out_ok}")
        print(f"   active cells at the S-box layers of rounds 0-4: "
              f"{[len(x) for x in sup]}")
        print(f"   limited-birthday bound for an ideal permutation: "
              f"2^{generic:.1f}")
    return dict(p=p, N=N, a=a, b=b_dim, d=d, expected=math.log2(expected),
                trials=math.log2(trials), generic=generic,
                ok=in_ok and one_cell and out_ok, supports=sup, j=j)


def figdata(p=7):
    res = run(p, verbose=False)
    for r, sp in enumerate(res["supports"]):
        with open(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figdata"), f"rebound_r{r}.dat"), "w") as f:
            f.write("x y\n")
            for c in sp:
                y, x = divmod(int(c), p)
                f.write(f"{x} {y}\n")
    print(f"  figure data at p = {p}: active cells per round",
          [len(sp) for sp in res["supports"]], "source", divmod(res['j'], p))


def backward_extension(p):
    """Six rounds by one more round backwards: the input space shrinks from
    X0 to X_{-1} (a' cells), which costs the rank of A restricted to the
    rows outside X0 and the columns X_{-1}.  We compute that rank for the
    configurations that the cost inequality leaves open."""
    n = p * p
    A = radon_layer(p, 2)
    full = True
    for miss_in in [p - 1, 2 * (p - 1), 60 if p == 23 else 3 * p]:
        for miss_prev in [p - 1, 2 * (p - 1), 70 if p == 23 else 3 * p]:
            for _ in range(5):
                K = rng.choice(n, size=miss_in, replace=False)
                Xm1 = np.setdiff1d(np.arange(n),
                                   rng.choice(n, size=miss_prev, replace=False))
                rk = len(rref(A[np.ix_(K, Xm1)], p)[1])
                if rk < miss_in:
                    full = False
                    print(f"   rank deficiency found: {miss_in} rows, rank {rk}")
    if full:
        print(f"   p = {p}: rows outside X0 restricted to X_(-1) had full rank "
              f"in every sample")


def main():
    print("=" * 78)
    print("Five-round rebound distinguisher, a = b")
    print("=" * 78)
    for p in [11, 13, 23]:
        run(p)
    print()
    print("=" * 78)
    print("Six rounds, one more round backwards: the rank condition")
    print("=" * 78)
    for p in [11, 23]:
        backward_extension(p)
    print()
    figdata(7)


if __name__ == "__main__":
    main()
