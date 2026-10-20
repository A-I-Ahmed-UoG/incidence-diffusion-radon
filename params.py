"""Table of design parameters for the instances of Section 8.

R_tr  : least even R with the trail bounds of Theorem 8.5 past 2^-256;
R_cnt : least even R at which counting excludes a rebound on the permutation
        (see rebound.py);
L     : the longest zero-sum at any data complexity (Corollary 8.8,
        see singlepath.py);
R     : the adopted round count, max(R_tr, 12).
DT, LT are the trail bounds at the adopted R; hull is the two-round bound of
Theorem 9.3, which does not depend on R.
"""
import math, cmath
from singlepath import reach


def stats(p):
    s=[0]+[pow(x,-1,p) for x in range(1,p)]
    w=cmath.exp(2j*math.pi/p)
    d=max(sum(1 for x in range(p) if (s[(x+a)%p]-s[x])%p==b) for a in range(1,p) for b in range(p))
    be=max(sum(1 for x in range(p) if (s[(s[(x+a)%p]+b)%p]-s[(s[x]+b)%p])%p==a) for a in range(1,p) for b in range(1,p))
    L=max(abs(sum(w**((a*x+b*s[x])%p) for x in range(p))) for a in range(1,p) for b in range(1,p))
    return d,be,L


def r_cnt(p, B, ldp):
    R = 4
    while math.floor((R - 2) / 2) * B * (-ldp) <= p * p * math.log2(p):
        R += 2
    return R


if __name__ == "__main__":
    print(f"{'p':>4}{'cells':>7}{'bits':>7}{'B':>5}{'delta':>7}{'beta':>6}"
          f"{'R_tr':>6}{'R_cnt':>6}{'L':>4}{'R':>4}"
          f"{'lg DT':>9}{'lg LT':>9}{'lg hull':>9}")
    for p in [11,17,23,29,41,47]:
        d,be,L=stats(p)
        ldp=math.log2(d/p); llp=math.log2(L*L/p/p); B=p+1
        Rtr=2
        while (Rtr//2)*B*max(ldp,llp) > -256: Rtr+=2
        R = max(Rtr, 12)
        Lz, _ = reach(p, p * p)
        print(f"{p:>4}{p*p:>7}{round(p*p*math.log2(p)):>7}{B:>5}{d:>7}{be:>6}"
              f"{Rtr:>6}{r_cnt(p,B,ldp):>6}{Lz:>4}{R:>4}"
              f"{(R//2)*B*ldp:>9.1f}{(R//2)*B*llp:>9.1f}{(B-1)*llp:>9.1f}")
