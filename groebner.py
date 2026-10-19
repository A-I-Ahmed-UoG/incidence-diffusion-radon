#!/usr/bin/env python3
"""Groebner-basis experiments on reduced CICO instances of Construction 8.1.

A CICO instance: c input cells U are unknown and the other input cells are
fixed; c output cells O must take prescribed values (taken from a random
point, so that a solution exists).  Modelling: the first S-box layer acts on
U bijectively, so its outputs u are the unknowns.  Every later S-box whose
input depends on u gets a variable y and, in the quadratic model, the
equation y*v - 1 = 0, v being its (linear) input; in the exact model the two
cubics v^2 y - v and v y^2 - y of Proposition 7.3.  The output constraints
are linear in the last layer of variables.

For each instance Singular computes a Groebner basis (degree reverse
lexicographic order) and reports the number of variables, the largest
degree in the basis, the degree D of the ideal (vdim, the number of
solutions over the algebraic closure counted with multiplicity) and the
time.  D is a lower bound on the cost of any method that outputs the
solutions from a Groebner basis, since the eliminating polynomial has
degree D.

Runtime a few minutes (instances that exceed the time limit are reported).
"""
import itertools
import math
import os
import re
import subprocess
import tempfile
import time

import numpy as np

from zerosum import radon_layer, constants, sbox_table

rng = np.random.default_rng(4242)
LIMIT = 400


def build(p, R, c, model="quad"):
    n = p * p
    A = radon_layer(p, 2)
    K = constants(p, R)
    s = sbox_table(p)
    while True:
        U = sorted(rng.choice(n, size=c, replace=False).tolist())
        # output cells reached from every unknown cell by walks of length R,
        # so that every equation involves every unknown
        Nm = (A != 0).astype(np.int64)
        reach = np.ones(n, dtype=bool)
        for uu in U:
            w = np.zeros(n, dtype=np.int64)
            w[uu] = 1
            for _ in range(R):
                w = np.minimum(Nm @ w, 1)
            reach &= w > 0
        O = sorted(rng.choice(np.nonzero(reach)[0], size=c,
                              replace=False).tolist())
        x = rng.integers(0, p, size=n)
        # numeric run, recording S-box inputs
        y = (s[x] + K[0]) % p
        vs = []
        state = y
        for r in range(1, R):
            v = (A @ state) % p
            vs.append(v)
            state = (s[v] + K[r]) % p
        out = (A @ state) % p
        # symbolic dependence
        dep = np.zeros(n, dtype=bool)
        dep[U] = True
        deps = []
        for r in range(1, R):
            dep = (A[:, dep] != 0).any(axis=1)
            deps.append(dep.copy())
        ok = all((vs[r][deps[r]] != 0).all() for r in range(R - 1))
        if ok or model == "exact":
            break

    names = [f"u{i}" for i in range(c)]
    # linear forms as dict var -> coeff, plus constant
    cur = []
    for k in range(n):
        if k in U:
            cur.append(({f"u{U.index(k)}": 1}, 0))
        else:
            cur.append(({}, int(y[k])))
    eqs = []
    for r in range(1, R):
        nxt = []
        for k in range(n):
            lin, const = {}, 0
            for jj in np.nonzero(A[k])[0]:
                a = int(A[k, jj])
                for vname, co in cur[jj][0].items():
                    lin[vname] = (lin.get(vname, 0) + a * co) % p
                const = (const + a * cur[jj][1]) % p
            lin = {kk: vv for kk, vv in lin.items() if vv}
            if lin:
                yname = f"y{r}_{k}"
                names.append(yname)
                vstr = "+".join([f"{co}*{vn}" for vn, co in lin.items()]
                                + [str(const)])
                if model == "quad":
                    eqs.append(f"{yname}*({vstr})-1")
                else:
                    eqs.append(f"({vstr})^2*{yname}-({vstr})")
                    eqs.append(f"({vstr})*{yname}^2-{yname}")
                nxt.append(({yname: 1}, int(K[r][k])))
            else:
                val = (int(s[const]) + int(K[r][k])) % p
                nxt.append(({}, val))
        cur = nxt
    for o in O:
        lin, const = {}, 0
        for jj in np.nonzero(A[o])[0]:
            a = int(A[o, jj])
            for vname, co in cur[jj][0].items():
                lin[vname] = (lin.get(vname, 0) + a * co) % p
            const = (const + a * cur[jj][1]) % p
        vstr = "+".join([f"{co}*{vn}" for vn, co in lin.items() if co]
                        + [str((const - int(out[o])) % p)])
        eqs.append(vstr)
    return names, eqs


def singular(p, names, eqs):
    script = f"""
ring r = {p}, ({",".join(names)}), dp;
option(redSB);
ideal I = {",".join(eqs)};
int t = timer;
ideal G = std(I);
int md = 0; int i;
for (i = 1; i <= size(G); i++) {{ if (deg(G[i]) > md) {{ md = deg(G[i]); }} }}
print("RESULT " + string(vdim(G)) + " " + string(md) + " " + string(timer - t));
quit;
"""
    with tempfile.NamedTemporaryFile("w", suffix=".sing", delete=False) as f:
        f.write(script)
        path = f.name
    t0 = time.time()
    try:
        res = subprocess.run(["Singular", "-q", path], capture_output=True,
                             text=True, timeout=LIMIT)
        m = re.search(r"RESULT (-?\d+) (\d+)", res.stdout)
        if not m:
            return None, None, time.time() - t0
        return int(m.group(1)), int(m.group(2)), time.time() - t0
    except subprocess.TimeoutExpired:
        return None, None, None
    finally:
        os.unlink(path)


def main():
    print("=" * 78)
    print("CICO with c unknown input cells and c prescribed output cells,")
    print("quadratic model, degree reverse lexicographic basis in Singular")
    print("=" * 78)
    for p, R, c, trials in [(5, 2, 1, 3), (5, 2, 2, 3), (5, 3, 1, 5),
                            (5, 3, 2, 5), (7, 3, 1, 4), (5, 4, 1, 1)]:
        res = []
        nv = None
        for _ in range(trials):
            names, eqs = build(p, R, c, "quad")
            nv = len(names)
            D, md, t = singular(p, names, eqs)
            res.append("timeout" if t is None else f"D={D} ({t:.0f}s)")
            if t is None:
                break
        print(f"  p = {p}, R = {R}, c = {c}, {nv} variables: " + ", ".join(res),
              flush=True)
    print()
    print("  Exhaustive search over the unknown cells costs p^c evaluations.")


if __name__ == "__main__":
    main()
