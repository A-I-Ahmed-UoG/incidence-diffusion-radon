import numpy as np
def radon(p):
    M=np.zeros((p*p,p*p),dtype=np.int64)
    for m in range(p):
        for c in range(p):
            for x in range(p):
                M[m*p+c,((m*x+c)%p)*p+x]=1
    return M
def rref(A,p):
    A=A.copy()%p; n,m=A.shape; r=0; piv=[]
    for c in range(m):
        nz=np.nonzero(A[r:,c])[0]
        if len(nz)==0: continue
        s=r+nz[0]; A[[r,s]]=A[[s,r]]; A[r]=A[r]*pow(int(A[r,c]),-1,p)%p
        col=A[:,c].copy(); col[r]=0
        A=(A-np.outer(col,A[r]))%p; piv.append(c); r+=1
        if r==n: break
    return A,piv
def rank(A,p): return len(rref(A,p)[1])
def minpoly(M,p):
    n=M.shape[0]; pows=[np.eye(n,dtype=np.int64)]
    while True:
        B=np.array([P.ravel() for P in pows]).T  # columns are powers
        A,piv=rref(B,p)
        if len(piv)<len(pows):
            k=len(pows)-1  # last is dependent
            coeffs=A[:k,k]  # M^k = sum coeffs_i M^i
            return [(-int(c))%p for c in coeffs]+[1]  # poly coeffs low->high
        pows.append(pows[-1]@M%p)
