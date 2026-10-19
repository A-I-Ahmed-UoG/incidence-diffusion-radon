"""Theorem 5.1 for PG(2,q), continued: symmetry, absolute points, N^2 = J,
the explicit inverse of N + aI, and branch numbers at q = 3, 5."""
import itertools, numpy as np
from pg import pg_incidence
for q in [3,5,7,11]:
    N=pg_incidence(q); n=N.shape[0]
    print(f"q={q}: symmetric: {(N==N.T).all()}, diagonal ones: {int(np.trace(N))} (q+1={q+1})")
    J=np.ones((n,n),dtype=np.int64)
    print(f"   N^2 = J over F_q: {((N@N)%q==J%q).all()}")
    I=np.eye(n,dtype=np.int64)
    for a in [1,2]:
        if a>=q: continue
        A=(N+a*I)%q
        iv=lambda z: pow(z,-1,q)
        Ainv=(iv(a)*I - iv(a)**2*N + iv(a)**2*iv(a+1)*(N@N%q))%q if (a+1)%q else None
        if Ainv is not None:
            print(f"   a={a}: explicit inverse ok: {((A@Ainv)%q==I).all()}")
def minw(A,q,maxw):
    best=10**9
    for w in range(1,maxw+1):
        vals=np.array(list(itertools.product(range(1,q),repeat=w)),dtype=np.int64)
        for S in itertools.combinations(range(A.shape[0]),w):
            best=min(best,w+int(np.count_nonzero((A[:,S]@vals.T)%q,axis=0).min()))
    return best
for q in [3,5]:
    N=pg_incidence(q); n=N.shape[0]; I=np.eye(n,dtype=np.int64)
    for a in range(1,q-1):
        A=(N+a*I)%q
        mw = 3 if q==5 else 4
        print(f"q={q}, a={a}: min over w<={mw}: {minw(A,q,mw)}  (expect q+2 = {q+2})")
