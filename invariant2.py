"""Checks for the classification of invariant partition subspaces."""
import itertools, numpy as np
from lib import radon

def line_points(p,m,c): return {((m*x+c)%p, x) for x in range(p)}

def criterion(p,Q):
    prof={}
    for blk in Q:
        for u in blk:
            L=line_points(p,*u)
            prof[u]=tuple(len(L & b)%p for b in Q)
    return all(all(prof[u]==prof[next(iter(blk))] for u in blk) for blk in Q)

def direct(p,Q,alpha):
    """W_Q invariant under M + alpha I, checked on a basis."""
    M=radon(p); A=(M+alpha*np.eye(p*p,dtype=np.int64))%p
    idx=lambda u: u[0]*p+u[1]
    for blk in Q:
        v=np.zeros(p*p,dtype=np.int64)
        for u in blk: v[idx(u)]=1
        w=(A@v)%p
        for b in Q:
            vals={int(w[idx(u)]) for u in b}
            if len(vals)>1: return False
    return True

rng=np.random.default_rng(0)
for p in [5,7]:
    cells=[(m,c) for m in range(p) for c in range(p)]
    agree=0; tested=0; inv=0
    for t in range(400):
        k=int(rng.integers(2,6))
        lab=rng.integers(0,k,len(cells))
        Q=[set(u for u,l in zip(cells,lab) if l==j) for j in range(k)]
        Q=[b for b in Q if b]
        c1=criterion(p,Q); c2=direct(p,Q,2); c3=direct(p,Q,3)
        tested+=1; agree+= (c1==c2==c3); inv+=c1
    print(f"p={p}: criterion agrees with direct check on {agree}/{tested} random partitions; {inv} were invariant")
    # canonical families
    cols=[{(m,c) for m in range(p)} for c in range(p)]
    rows=[{(m,c) for c in range(p)} for m in range(p)]
    print("   column partition invariant:", criterion(p,cols), " row partition:", criterion(p,rows))
    # coarsenings
    colcoarse=[cols[0]|cols[1]]+cols[2:]
    rowcoarse_nozero=[rows[1]|rows[2]]+[rows[0]]+rows[3:]
    rowcoarse_zero=[rows[0]|rows[1]]+rows[2:]
    print("   merge two columns:", criterion(p,colcoarse),
          "| merge two non-zero rows:", criterion(p,rowcoarse_nozero),
          "| merge row 0 with another:", criterion(p,rowcoarse_zero))
