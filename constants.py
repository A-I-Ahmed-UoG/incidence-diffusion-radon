"""The round constants of Construction 8.1 separate every pair of cells."""
import numpy as np
for p,R in [(23,8),(29,6),(17,14),(11,24),(47,4)]:
    if p%3!=2: continue
    prof={}
    clash=0
    for y in range(p):
        for x in range(p):
            key=tuple(pow((y+(r+1)*x+r+1)%p,3,p) for r in range(R))
            if key in prof: clash+=1
            prof[key]=(y,x)
    print(f"p={p}, R={R}: distinct constant profiles {len(prof)} of {p*p}, collisions {clash}")
    # also: cubing is a bijection exactly when p is not 1 mod 3
    cubes={pow(z,3,p) for z in range(p)}
    print(f"   cubing bijective on F_{p}: {len(cubes)==p}")
