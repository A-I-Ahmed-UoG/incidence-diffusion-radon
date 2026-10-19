"""Theorem 5.1 for PG(2,q): N N^T = J, the rank of N, its minimal polynomial
and the invertible shifts N + aI, for q = 3, 5, 7, 11."""
import numpy as np
from lib import rank

def pg_incidence(q):
    """Point-line incidence matrix of PG(2,q) for q prime, points = 1-dim subspaces."""
    pts=[]
    for x in range(q):
        for y in range(q): pts.append((1,x,y))
    for y in range(q): pts.append((0,1,y))
    pts.append((0,0,1))
    n=len(pts)
    N=np.zeros((n,n),dtype=np.int64)
    for i,L in enumerate(pts):          # lines indexed by the same coordinates (duality)
        for j,P in enumerate(pts):
            if (L[0]*P[0]+L[1]*P[1]+L[2]*P[2])%q==0: N[i,j]=1
    return N

def minpoly_deg(N,p,maxd=8):
    n=N.shape[0]; pows=[np.eye(n,dtype=np.int64)]
    for d in range(1,maxd+1):
        pows.append(pows[-1]@N%p)
        B=np.array([P.ravel() for P in pows]).T % p
        if rank(B,p)<len(pows):
            return d, pows
    return None, pows

def main():
  for q in [3,5,7,11]:
    N=pg_incidence(q); n=N.shape[0]
    assert (N.sum(1)==q+1).all() and (N.sum(0)==q+1).all()
    NNT=(N@N.T)%q
    J=np.ones((n,n),dtype=np.int64)
    print(f"q={q}: n={n}, N N^T = J over F_q: {(NNT==J).all()}, rank_q N = {rank(N,q)}, (q^2+q+2)/2 = {(q*q+q+2)//2}")
    d,pows=minpoly_deg(N,q)
    # express N^d in terms of lower powers
    # solve B c = vec(N^d) over F_q by gaussian elimination
    B=np.array([P.ravel() for P in pows[:d]]).T % q
    A=np.concatenate([B, pows[d].ravel().reshape(-1,1)],axis=1)%q
    r=0; piv=[]
    for c in range(A.shape[1]):
        nz=[i for i in range(r,A.shape[0]) if A[i,c]%q]
        if not nz: continue
        A[[r,nz[0]]]=A[[nz[0],r]]; A[r]=A[r]*pow(int(A[r,c]),-1,q)%q
        for i in range(A.shape[0]):
            if i!=r and A[i,c]%q: A[i]=(A[i]-A[i,c]*A[r])%q
        piv.append(c); r+=1
    coef=[0]*d
    for i,c in enumerate(piv):
        if c<d: coef[c]=int(A[i,d])
    print(f"   minimal polynomial degree {d}: N^{d} = " + " + ".join(f"{coef[i]}N^{i}" for i in range(d) if coef[i]))
    inv=[a for a in range(q) if rank((N+a*np.eye(n,dtype=np.int64))%q,q)==n]
    print(f"   N + aI invertible for a in {inv}")

if __name__ == "__main__":
    main()
