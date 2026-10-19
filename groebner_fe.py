#!/usr/bin/env python3
"""Groebner bases for reduced CICO instances with the field equations.

The instances are those of groebner.py (quadratic model, or the exact model
with the two cubics of Proposition 7.3, c unknown input cells, c prescribed
output cells reached by every unknown), to which the
field equations u^p - u of the unknown input cells are added, so that the
solutions are exactly those in F_p.  The basis is computed with msolve (F4,
degree reverse lexicographic order), and we record

  * the solving degree: the largest degree of the pairs F4 selects;
  * the largest Macaulay matrix;
  * the time;

and compare with exhaustive search over the p^c values of the unknowns.
The quadratic instances are also run without the field equations.

Needs msolve (apt install msolve).  Runtime about 30 minutes.
"""
import os
import re
import subprocess
import tempfile
import time

import sympy as sp

import groebner as g

LIMIT = 1500


def to_msolve(p, names, eqs, field):
    ren = {nm: f"v{i}" for i, nm in enumerate(names)}
    syms = sp.symbols(" ".join(ren.values()))
    loc = {ren[nm]: syms[i] for i, nm in enumerate(names)}
    polys = []
    for e in eqs:
        for nm in sorted(names, key=len, reverse=True):
            e = e.replace(nm, ren[nm])
        expr = sp.expand(sp.sympify(e, locals=loc))
        P = sp.Poly(expr, *syms, modulus=p)
        polys.append(P)
    if field:
        for i, nm in enumerate(names):
            if nm.startswith("u"):
                polys.append(sp.Poly(syms[i] ** p - syms[i], *syms, modulus=p))
    lines = []
    for P in polys:
        terms = []
        for mon, co in P.terms():
            co = int(co) % p
            if co == 0:
                continue
            m = "*".join(f"{syms[k]}^{e}" if e > 1 else f"{syms[k]}"
                         for k, e in enumerate(mon) if e)
            terms.append(f"{co}*{m}" if m else f"{co}")
        lines.append("+".join(terms) if terms else "0")
    return ",".join(str(s_) for s_ in syms), lines


def msolve_run(p, var_line, polys):
    with tempfile.NamedTemporaryFile("w", suffix=".ms", delete=False) as f:
        f.write(var_line + "\n" + str(p) + "\n" + ",\n".join(polys) + "\n")
        path = f.name
    out = path + ".out"
    t0 = time.time()
    try:
        res = subprocess.run(["msolve", "-f", path, "-o", out, "-g", "2",
                              "-v", "2"], capture_output=True, text=True,
                             timeout=LIMIT)
    except subprocess.TimeoutExpired:
        return None, None, None
    finally:
        os.unlink(path)
        if os.path.exists(out):
            os.unlink(out)
    t = time.time() - t0
    degs, mats = [], []
    for line in res.stdout.splitlines():
        m = re.match(r"\s*(\d+)\s+\d+\s+\d+\s+(\d+) x (\d+)", line)
        if m:
            degs.append(int(m.group(1)))
            mats.append((int(m.group(2)), int(m.group(3))))
    sd = max(degs) if degs else None
    mat = max(mats, key=lambda rc: rc[0] * rc[1]) if mats else None
    return sd, mat, t


def main():
    print("=" * 86)
    print("CICO with the field equations of the unknowns, msolve (F4, DRL)")
    print("=" * 86)
    print(f"  {'p':>2} {'R':>2} {'c':>2} {'model':>6} {'vars':>5} {'field':>6} {'solv.deg':>9} "
          f"{'largest matrix':>18} {'time s':>8} {'p^c':>7}")
    runs = [(5, 2, 1, "quad"), (5, 3, 1, "quad"), (5, 4, 1, "quad"),
            (5, 2, 2, "quad"), (5, 3, 2, "quad"),
            (5, 2, 3, "quad"), (5, 3, 3, "quad"),
            (7, 2, 1, "quad"), (7, 3, 1, "quad"),
            (7, 2, 2, "quad"), (7, 3, 2, "quad"),
            (5, 3, 1, "exact"), (5, 4, 1, "exact"), (5, 5, 1, "exact"),
            (5, 6, 1, "exact"), (5, 3, 2, "exact"), (7, 3, 1, "exact")]
    for p, R, c, model in runs:
        names, eqs = g.build(p, R, c, model)
        for field in ((True, False) if model == "quad" else (True,)):
            vl, polys = to_msolve(p, names, eqs, field)
            sd, mat, t = msolve_run(p, vl, polys)
            ms = f"{mat[0]} x {mat[1]}" if mat else "--"
            ts = f"{t:.1f}" if t is not None else f">{LIMIT}"
            print(f"  {p:>2} {R:>2} {c:>2} {model:>6} {len(names):>5} "
                  f"{'yes' if field else 'no':>6} {str(sd):>9} {ms:>18} "
                  f"{ts:>8} {p**c:>7}", flush=True)


if __name__ == "__main__":
    main()
