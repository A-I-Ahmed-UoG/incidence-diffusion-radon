"""Numerical check of Theorem 9.3 (the two-round bound) on a toy SPN:
state F_5^2, A = [[1,1],[1,2]] (branch number 3), S-box x -> x^{p-2}."""
import itertools, numpy as np
p = 5
S = np.array([0] + [pow(x, -1, p) for x in range(1, p)], dtype=np.int64)
A = np.array([[1, 1], [1, 2]], dtype=np.int64)
states = [np.array(v, dtype=np.int64) for v in itertools.product(range(p), repeat=2)]

def branch(A):
    best = 10**9
    for x in states:
        if not x.any(): continue
        best = min(best, np.count_nonzero(x) + np.count_nonzero(A@x % p))
    return best
B = branch(A); Bl = branch(A.T.copy())
delta = max(sum(1 for x in range(p) if (S[(x+a) % p] - S[x]) % p == b)
            for a in range(1, p) for b in range(p))

def enc(x, k0, k1):
    y = (S[(x + k0) % p]) % p
    y = (A @ y) % p
    y = (y + k1) % p
    return S[y % p]

worst = 0
for a in states:
    if not a.any(): continue
    for b in states:
        if not b.any(): continue
        tot = 0
        for k0 in states:
            for k1 in states:
                cnt = sum(1 for x in states
                          if ((enc((x + a) % p, k0, k1) - enc(x, k0, k1)) % p == b).all())
                tot += cnt / len(states)
        edp = tot / (len(states) ** 2)
        worst = max(worst, edp)
bound = (delta / p) ** (B - 1)
trail = (delta / p) ** B
print(f"p={p}, branch numbers B_d={B}, B_l={Bl}, delta={delta}")
print(f"max EDP over 2 rounds = {worst:.5f}")
print(f"Theorem 9.3 bound (delta/p)^(B-1) = {bound:.5f}  -> holds: {worst <= bound + 1e-12}")
print(f"single-trail bound (delta/p)^B    = {trail:.5f}  -> exceeded by hulls: {worst > trail}")
