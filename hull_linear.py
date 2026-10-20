#!/usr/bin/env python3
"""Theorem 9.3, linear half, and formula (15).

Exact ELP for small layers, averaged over the middle key (|c_kappa| does not
depend on the first key), compared with
  (a) formula (15),
      ELP(a,b) = sum_d prod_i LP(a_i, (A^T d)_i) prod_j LP(d_j, b_j);
  (b) the per-pair bound (L^2/p^2)^(s - dim U^l) with the space (13),
      U^l(a,b) = {d : supp d in supp b, supp(A^T d) in supp a};
  (c) the same bound with a and b not exchanged, that is (12) with A^T in
      place of A, {c : supp c in supp a, supp(A^T c) in supp b}, which fails
      on the 3-cell layers and shows that the exchange is needed;
  (d) the global bound (L^2/p^2)^(B_l - 1);
and the dimension bound dim U^l <= s - B_l + 1 of the proof.

Runtime under 30 seconds.
"""
import itertools
import numpy as np

def sbox(p):
    return np.array([pow(x, p-2, p) for x in range(p)])

def lp_table(p):
    s = sbox(p); z = np.exp(2j*np.pi/p); x = np.arange(p)
    # LP[u, v] = |p^-1 sum_x zeta^{v s(x) - u x}|^2
    C = np.array([[np.sum(z**((v*s - u*x) % p)) / p for v in range(p)] for u in range(p)])
    return np.abs(C)**2

def all_vectors(p, n):
    return np.array(list(itertools.product(range(p), repeat=n)), dtype=np.int64)

def exact_elp(A, p):
    n = A.shape[0]; X = all_vectors(p, n); s = sbox(p); z = np.exp(2j*np.pi/p)
    Z = z ** ((X @ X.T) % p)                     # Z[a, x] = zeta^{<a,x>}
    E = np.zeros((p**n, p**n))
    for k1 in X:                                  # kappa_0 = 0 w.l.o.g.
        T = s[(s[X] @ A.T + k1) % p]              # T(x) for all x
        idxT = T @ (p ** np.arange(n-1, -1, -1))
        Cm = (Z[:, idxT] @ np.conj(Z).T) / p**n   # C[b, a] = p^-n sum_x zeta^{<b,T x> - <a,x>}
        E += np.abs(Cm.T)**2                     # E[a, b]
    return E / p**n, X

def linearity(p):
    s = sbox(p); z = np.exp(2j*np.pi/p); x = np.arange(p)
    return max(abs(np.sum(z**((a*x + b*s) % p))) for a in range(p) for b in range(1, p))

def branch(A, p):
    n = A.shape[0]; X = all_vectors(p, n)[1:]
    return int((np.count_nonzero(X, 1) + np.count_nonzero(X @ A.T % p, 1)).min())

def check(name, A, p):
    n = A.shape[0]; At = A.T % p
    E, X = exact_elp(A, p)
    LP = lp_table(p); L2 = linearity(p)**2 / p**2
    Bl = branch(At, p)
    nz = lambda v: frozenset(np.nonzero(v)[0])
    supp = [nz(v) for v in X]
    AtX = X @ A % p                             # A^T d for each d (row form: d^T A)
    supp_At = [nz(v) for v in AtX]
    # sanity: row sums of LP are 1
    assert np.allclose(LP.sum(0), 1) and np.allclose(LP.sum(1), 1)
    bits = lambda V: (V != 0) @ (1 << np.arange(n))
    sm = bits(X); sAt = bits(AtX)
    # Markov formula as a matrix product
    F1 = np.ones((p**n, p**n)); F2 = np.ones((p**n, p**n))
    for i in range(n):
        F1 *= LP[X[:, i][:, None], AtX[:, i][None, :]]     # [a, d]
        F2 *= LP[X[:, i][:, None], X[:, i][None, :]]       # [d, b]
    Mk = F1 @ F2
    markov_err = float(np.abs(Mk - E)[1:, 1:].max())
    # dimensions of U^l (new) and of the unexchanged space (old), per support pair
    cnt_new = np.zeros((1 << n, 1 << n), dtype=np.int64); cnt_old = np.zeros_like(cnt_new)
    for SA in range(1 << n):
        for SB in range(1 << n):
            cnt_new[SA, SB] = np.sum(((sm & ~SB) == 0) & ((sAt & ~SA) == 0))
            cnt_old[SA, SB] = np.sum(((sm & ~SA) == 0) & ((sAt & ~SB) == 0))
    dimf = lambda c: np.rint(np.log(c) / np.log(p)).astype(int)
    wts = np.array([bin(m).count("1") for m in range(1 << n)])
    SA = sm[:, None]; SB = sm[None, :]
    s_ab = wts[SA] + wts[SB]
    dn = dimf(cnt_new[SA, SB]); do = dimf(cnt_old[SA, SB])
    bnew = np.where(dn > 0, L2 ** (s_ab - dn), 0.0)
    bold = np.where(do > 0, L2 ** (s_ab - do), 0.0)
    E1 = E[1:, 1:]; bnew = bnew[1:, 1:]; bold = bold[1:, 1:]; do1 = do[1:, 1:]
    tol = lambda b: b * (1 + 1e-9) + 1e-12
    viol_new = int(np.sum(E1 > tol(bnew))); viol_old = int(np.sum(E1 > tol(bold)))
    zero_old_pos = int(np.sum((do1 == 0) & (E1 > 1e-12)))
    worst_new = float(np.max(np.where(bnew > 0, E1 / np.where(bnew > 0, bnew, 1), 0)))
    worst_glob = float(E1.max() / L2 ** (Bl - 1))
    # the bound of Lemma 9.2 for U^l, with B_l
    dn1 = dn[1:, 1:]; s1 = s_ab[1:, 1:]
    lemma_ok = bool(np.all((dn1 == 0) | (dn1 <= s1 - Bl + 1)))
    print(f"{name} over F_{p}: B_l = {Bl}; max |ELP - Markov| = {markov_err:.1e}; "
          f"U^l of (13): {viol_new} violations (max ratio {worst_new:.3f}); "
          f"a, b not exchanged: {viol_old} violations, {zero_old_pos} pairs with U = 0 but ELP > 0; "
          f"global bound: max ratio {worst_glob:.3f}; dim U^l <= s - B_l + 1: {lemma_ok}")
    assert viol_new == 0 and lemma_ok and markov_err < 1e-12 and worst_glob <= 1 + 1e-9
    return viol_old

def main():
    tot = 0
    tot += check("3-cell (1 1 1; 1 2 3; 1 3 2)", np.array([[1,1,1],[1,2,3],[1,3,2]]), 5)
    tot += check("circulant (1,1,0)", np.array([[1,1,0],[0,1,1],[1,0,1]]), 7)
    tot += check("2-cell (1 1; 1 2)", np.array([[1,1],[1,2]]), 5)
    assert tot > 0
    print("\nLINEAR HULL CHECKS PASSED: the bound holds with U^l, and fails without the exchange of a and b")

if __name__ == "__main__":
    main()
