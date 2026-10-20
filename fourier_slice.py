#!/usr/bin/env python3
"""The Fourier-slice identity of Section 4.3 (before Algorithm 1): M v through
an auxiliary prime q with q = 1 (mod p) and q > p(p-1).  The integer sums
(M v)_{(m,c)} lie in [0, p(p-1)], so they are recovered exactly from their
residues mod q and then reduced mod p.  With omega of order p in F_q and
    vhat(a, b) = sum_{y,x} v(y,x) omega^{a y + b x},
the projection P_m(c) = sum_x v(m x + c, x) satisfies
    sum_c P_m(c) omega^{k c} = vhat(k, -m k),
so M v = p inverse length-p DFTs of p slices of one p x p two-dimensional DFT.
With fast length-p DFTs (Rader or Bluestein) this is O(p^2 log p) operations
in F_q; here the DFTs are evaluated naively, to check the identity at
p = 5, 7, 11, 13, 23.

Runtime a few seconds.
"""
import numpy as np

def isprime(n):
    return n > 1 and all(n % d for d in range(2, int(n**0.5) + 1))

def setup(p):
    q = next(q for q in range(p*(p-1) + 1, 10**7) if q % p == 1 and isprime(q))
    g = next(g for g in range(2, q) if all(pow(g, (q-1)//r, q) != 1
             for r in {r for r in range(2, q) if (q-1) % r == 0 and isprime(r)}))
    return q, pow(g, (q-1)//p, q)

def radon_direct(v, p):
    return np.array([[sum(int(v[(m*x + c) % p, x]) for x in range(p)) for c in range(p)] for m in range(p)])

def radon_slice(v, p, q, om):
    W = np.array([[pow(om, (i*j) % p, q) for j in range(p)] for i in range(p)], dtype=object)
    vhat = (W @ np.array(v, dtype=object) @ W) % q          # vhat[a,b] = sum v[y,x] om^{a y + b x}
    pinv = pow(p, -1, q)
    Winv = np.array([[pow(om, (-i*j) % p, q) for j in range(p)] for i in range(p)], dtype=object)
    out = np.zeros((p, p), dtype=object)
    for m in range(p):
        sl = np.array([vhat[k % p, (-m*k) % p] for k in range(p)], dtype=object)
        out[m] = (Winv @ sl) * pinv % q                       # inverse DFT over c
    return out.astype(np.int64)

def main():
    rng = np.random.default_rng(3)
    for p in [5, 7, 11, 13, 23]:
        q, om = setup(p)
        for _ in range(3):
            v = rng.integers(0, p, (p, p))
            d = radon_direct(v, p); s = radon_slice(v, p, q, om)
            assert np.array_equal(d, s) and np.array_equal(d % p, s % p)
        print(f"p={p:2d}: auxiliary prime q={q} (q = 1 mod p, q > p(p-1)); slice route = direct Radon transform")
    print("FOURIER-SLICE CHECK PASSED")

if __name__ == "__main__":
    main()
