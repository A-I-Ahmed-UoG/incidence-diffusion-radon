"""All partitions Q of the p x p cells with W_Q invariant under M (equivalently
under M + aI), for p = 3, by exhaustive enumeration of set partitions."""
import itertools, numpy as np
from lib import radon

def partitions(collection):
    collection=list(collection)
    if len(collection)==1:
        yield [collection]; return
    first=collection[0]
    for smaller in partitions(collection[1:]):
        for n,subset in enumerate(smaller):
            yield smaller[:n]+[[first]+subset]+smaller[n+1:]
        yield [[first]]+smaller

def line_points(p,m,c):
    return {( (m*x+c)%p, x) for x in range(p)}

def invariant(p, Q):
    """Q: list of blocks, each a set of cells (m,c). Criterion:
    cells in the same block must meet every block equally often mod p."""
    idx={}
    for b,blk in enumerate(Q):
        for u in blk: idx[u]=b
    prof={}
    for u in idx:
        L=line_points(p,u[0],u[1])
        prof[u]=tuple(len(L & set(blk))%p for blk in Q)
    for blk in Q:
        blk=list(blk)
        if any(prof[u]!=prof[blk[0]] for u in blk): return False
    return True

p=3
cells=[(m,c) for m in range(p) for c in range(p)]
inv=[]
for part in partitions(cells):
    Q=[set(b) for b in part]
    if invariant(p,Q): inv.append(Q)
print(f"p={p}: {len(inv)} invariant partitions out of Bell(9)=21147")

def describe(Q,p):
    def is_union_of_columns(blk): 
        cols={c for (m,c) in blk}
        return all((m,c) in blk for c in cols for m in range(p))
    def is_union_of_rows(blk):
        rows={m for (m,c) in blk}
        return all((m,c) in blk for m in rows for c in range(p))
    if all(len(b)==1 for b in Q): return "discrete"
    if len(Q)==1: return "one block"
    if all(is_union_of_columns(b) for b in Q): return "union of columns"
    if all(is_union_of_rows(b) for b in Q):
        merged0=[b for b in Q if any(m==0 for (m,c) in b)]
        return "union of rows" + (" (row 0 merged)" if len(merged0[0])>p else "")
    return "OTHER"
from collections import Counter
cnt=Counter(describe(Q,p) for Q in inv)
print(cnt)
for Q in inv:
    if describe(Q,p)=="OTHER":
        print("  other:", sorted(tuple(sorted(b)) for b in Q))
