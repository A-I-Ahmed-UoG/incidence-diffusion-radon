"""Checks the computational statements of Sections 3-8 (Appendix A).
Run: python3 verify_all.py   (needs numpy; ~5 min)"""
import itertools, math, cmath, numpy as np
from lib import radon, rank

def ok(name, cond): print(("PASS " if cond else "FAIL ") + name); assert cond

# --- Prop 4.1: explicit M^2, M^3, M^4 = -M^3 over F_p
for p in [3,5,7,11,13,17,31]:
    M = radon(p); n = p*p
    m_of = np.arange(n)//p; x_of = np.arange(n)%p
    M2 = M@M % p; M3 = M2@M % p; M4 = M3@M % p
    ok(f"p={p}: M^2[(m,c),(y,t)] = 1[mt != -1]", (M2 == ((np.outer(m_of, x_of) % p) != p-1)).all())
    ok(f"p={p}: M^3 = -f g^T", (M3 == (-np.outer(m_of != 0, x_of != 0).astype(np.int64)) % p).all())
    ok(f"p={p}: M^4 = -M^3", (M4 == (-M3) % p).all())

# --- Thm 4.2: ranks
for p in [3,5,7,11,13,17,19,23,29,31]:
    M = radon(p)
    ok(f"p={p}: rank_p M = p(p+1)/2", rank(M, p) == p*(p+1)//2)
for p in [5,7,11]:
    M = radon(p)
    ok(f"p={p}: rank M^2 = p, rank M^3 = 1",
       rank(M@M % p, p) == p and rank(M@M@M % p, p) == 1)

# --- Cor 4.4: invertibility of M + aI and explicit inverse
for p in [5,7,11,13]:
    M = radon(p); I = np.eye(p*p, dtype=np.int64)
    inv_alphas = [a for a in range(p) if rank((M + a*I) % p, p) == p*p]
    ok(f"p={p}: M+aI invertible iff a not in {{0,1}}", inv_alphas == list(range(2, p)))
for p in [5,7,11,31]:
    M = radon(p); I = np.eye(p*p, dtype=np.int64)
    for a in range(2, min(p, 6)):
        iv = lambda z: pow(z, -1, p)
        Ai = (iv(a)*I - iv(a)**2*M + iv(a)**3*(M@M) - iv(a)**3*iv(a-1)*(M@M@M)) % p
        ok(f"p={p}, a={a}: explicit inverse", (((M + a*I) @ Ai) % p == I).all())

# --- Thm 3.3 / Cor 4.6: branch numbers, exhaustive over low weights
def min_w_plus_wt(A, p, maxw):
    n = A.shape[0]; best = 10**9
    for w in range(1, maxw+1):
        vals = np.array(list(itertools.product(range(1, p), repeat=w)), dtype=np.int64)
        for S in itertools.combinations(range(n), w):
            best = min(best, w + np.count_nonzero((A[:, S] @ vals.T) % p, axis=0).min())
    return best
p = 5; M = radon(p); I = np.eye(25, dtype=np.int64)
for a, expect in [(0, 6), (2, 6), (3, 6), (4, 5)]:
    A = (M + a*I) % p
    # weights <= 4 exhaustive; weight >= 5 gives >= 6 automatically when A invertible
    # (for a = 0 the proof bound w(k-w+2) covers w = 5)
    d = min_w_plus_wt(A, p, 4); l = min_w_plus_wt(A.T.copy(), p, 4)
    ok(f"p=5, a={a}: differential & linear branch number = {expect}", d == expect and l == expect)
p = 7; M = radon(p); I = np.eye(49, dtype=np.int64)
for a, expect in [(2, 8), (3, 8), (6, 7)]:
    A = (M + a*I) % p
    ok(f"p=7, a={a}: min over w<=2 equals {expect}",
       min_w_plus_wt(A, p, 2) == expect and min_w_plus_wt(A.T.copy(), p, 2) == expect)

# --- Cor 6.4: invariant partition subspaces (rows / columns)
for p in [5,7,11]:
    A = (radon(p) + 2*np.eye(p*p, dtype=np.int64)) % p
    rng = np.random.default_rng(0)
    h = rng.integers(0, p, p)
    col = np.array([h[i % p] for i in range(p*p)]); row = np.array([h[i//p] for i in range(p*p)])
    oc = A@col % p; orow = A@row % p
    ok(f"p={p}: column-constant subspace invariant", all(len(set(oc[i::p])) == 1 for i in range(p)))
    ok(f"p={p}: row-constant subspace invariant", all(len(set(orow[i*p:(i+1)*p])) == 1 for i in range(p)))

# --- Prop 6.1: full diffusion after exactly 3 layers (pattern level)
for p in [5,7,11,31]:
    for a in [0, 2]:
        B = (radon(p) + a*np.eye(p*p, dtype=np.int64)) > 0
        B2 = (B.astype(np.int64)@B) > 0; B3 = (B2.astype(np.int64)@B) > 0
        ok(f"p={p}, a={a}: pattern full at 3 layers, not at 2", B3.all() and not B2.all())

# --- Table 2: S-box constants of x -> x^{-1} on F_p
def inv(p): return [0] + [pow(x, -1, p) for x in range(1, p)]
for p in [5,7,11,13,17,19,23,29,31]:
    s = inv(p); w = cmath.exp(2j*math.pi/p)
    d = max(sum((s[(x+a) % p]-s[x]) % p == b for x in range(p)) for a in range(1, p) for b in range(p))
    be = max(sum((s[(s[(x+a) % p]+b) % p]-s[(s[x]+b) % p]) % p == a for x in range(p))
             for a in range(1, p) for b in range(1, p))
    L = max(abs(sum(w**((a*x+b*s[x]) % p) for x in range(p))) for a in range(1, p) for b in range(1, p))
    ok(f"p={p}: delta={d} (expected {2 if p%3==2 else 4}), beta={be}, L={L:.3f} <= 2sqrt(p)+1",
       d == (2 if p % 3 == 2 else 4) and L <= 2*math.sqrt(p) + 1)

# ---------------------------------------------------------------------------
# Further statements from Sections 4-8
# ---------------------------------------------------------------------------
# minimal polynomial x^3 (x + 1): M^3 != 0, M^2 (M + I) != 0, M^3 (M + I) = 0
for p in [3,5,7,11,13]:
    M = radon(p); I = np.eye(p*p, dtype=np.int64)
    M2 = M@M % p; M3 = M2@M % p
    ok(f"p={p}: minimal polynomial x^3(x+1)",
       M3.any() and ((M2@(M+I)) % p).any() and not ((M3@(M+I)) % p).any())
# Prop 4.3: rank p^2 - p + 1 in characteristic != p
M = radon(31)
for q in [61, 1117]:
    ok(f"p=31: rank over F_{q} = 931", rank(M, q) == 931)
# Prop 6.1: (1,1) cannot reach (-1,0) in two steps
for p in [5,7,11,13,31]:
    B = ((radon(p) + 2*np.eye(p*p, dtype=np.int64)) > 0).astype(np.int64)
    ok(f"p={p}: (1,1) -> (-1,0) not reachable in 2 steps", not ((B@B) > 0)[(p-1)*p, p+1])
# Prop 7.3: graph relations of x^{p-2}
def relation_dims(p, dmax=3):
    s = [0] + [pow(x, -1, p) for x in range(1, p)]
    out = []
    for d in range(1, dmax+1):
        mons = [(a, t-a) for t in range(d+1) for a in range(t+1)]
        E = np.array([[pow(x, a, p)*pow(s[x], b, p) % p for x in range(p)] for a, b in mons], dtype=np.int64)
        out.append(len(mons) - rank(E, p))
    return out
for p in [11, 13, 23, 31]:
    ok(f"p={p}: graph relations of degree <=1,2,3 have dims 0,0,2", relation_dims(p) == [0, 0, 2])
# Section 8: numbers for p = 23, R = 12
s = [0] + [pow(x, -1, 23) for x in range(1, 23)]
w = cmath.exp(2j*math.pi/23)
L23 = max(abs(sum(w**((a*x+b*s[x]) % 23) for x in range(23))) for a in range(1, 23) for b in range(1, 23))
ok("p=23, R=8 would suffice for the trail bound: 2^-261.1", abs(96*math.log2(L23**2/23**2) + 261.1) < 0.05)
ok("p=23, R=12: 144 active S-boxes", (12//2)*24 == 144)
ok("p=23, R=12: differential trail bound 2^-507.4", abs(144*math.log2(2/23) + 507.4) < 0.05)
ok("p=23, R=12: linear trail bound 2^-391.7", abs(144*math.log2(L23**2/23**2) + 391.7) < 0.05)
ok("p=23: two-round differential bound 2^-81.0", abs(23*math.log2(2/23) + 81.0) < 0.05)
ok("p=23, R=12: algebraic system 6348 vars, 12696 cubics, ~407 bits",
   529*12 == 6348 and 2*529*12 == 12696 and round(6348*math.log2(23/22)) == 407)
# Corollary 8.7: zero-sum reach at p = 23
ok("p=23: 4 rounds with k=2", 21 < 2*22)
ok("p=23: 5 rounds with k=21", 21**2 < 21*22 and not 21**2 < 20*22)
ok("p=23: 7 rounds with k=421", 21**3 < 421*22 and not 21**3 < 420*22)
ok("p=23: 8 rounds need k >= 8841 > 529", 21**4 < 8841*22 and not 21**4 < 8840*22)
# Proposition 8.9: limited-birthday bound for a = b = n - (p - 1)
ok("p=23: limited birthday 23^11 = 2^49.8", abs(11*math.log2(23) - 49.8) < 0.05)
ok("p=11, 13: limited birthday 2^17.3, 2^22.2", abs(5*math.log2(11) - 17.3) < 0.05 and abs(6*math.log2(13) - 22.2) < 0.05)
ok("p=23: 57 unknown cells give >= 2^256 candidates", 57*math.log2(23) >= 256 > 56*math.log2(23))
ok("capacity 114 cells > 2^515", 114*math.log2(23) > 515)
ok("output distance 2^256/23^85 < 2^-128", 256 - 85*math.log2(23) < -128)
print("ALL MANUSCRIPT CHECKS PASSED")
