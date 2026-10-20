# Diffusion layers from finite incidence structures: code

Code and data for the paper

> A. I. Ahmed, G. Loukas, A. S. Sani, E. Panaousis.
> *Diffusion Layers from Finite Incidence Structures and the Discrete Radon
> Transform over F_p.* Submitted to *Cryptography and Communications*.

The scripts check the computational statements in the paper and generate the
data behind its figures. Each script is self-contained and prints what it
checks, so the output can be read next to the statement it refers to.

## Requirements

- Python 3.8 or later
- NumPy
- SymPy, only for `code/groebner_fe.py`
- [Singular](https://www.singular.uni-kl.de/) for `code/groebner.py` and
  [msolve](https://msolve.lip6.fr/) for `code/groebner_fe.py` (on Debian or
  Ubuntu: `apt install singular msolve`)

```
pip install -r requirements.txt
```

## Layout

```
make_figdata.py   data for Figures 3, 4, 6, 7, 10 and 11
lib.py            Radon matrix, rank and minimal polynomial over F_p
                  (code/lib.py is the same file; code/modp.py has linear
                  algebra mod p and the permutation, for Sections 8 and 9)
figdata/          the data files read by the figures
code/             one script per result or group of results
```

## Quick start

```
python3 make_figdata.py        # regenerates figdata/, about 1.5 min
cd code
python3 verify_all.py          # about 5 min, ends with "ALL MANUSCRIPT CHECKS PASSED"
```

`verify_all.py` covers most of the computational statements in Sections 3 to 8.
The other scripts go into individual results in more depth. They can be run
from any directory.

## Scripts

Runtimes are rough figures on a laptop.

| Script | What it checks | Time |
|---|---|---|
| `verify_all.py` | Most computational statements of Sections 3 to 8, including Propositions 4.3, 4.7, 4.10, 8.3(iii), 8.4 and the reach of Corollary 8.8 | 5 min |
| **Branch numbers (Section 3)** | | |
| `arcs.py` | Remark 3.7: the p²(p−1) arcs at p = 3, 5, 7, 11, and the counts in the caption of Figure 2 | 15 s |
| `char2.py` | Theorem 3.6 in characteristic two: PG(2,3) and AG(2,4) without a parallel class, over F_2 and F_4 | 10 s |
| `casec_general.py` | Lemma 3.5 and Theorem 3.6 by the original counting step (k ≥ 7): exceptional supports, the parabola, Mantel's threshold | 4 min |
| `casec_k4.py` | Theorem 3.6 with the sharper count (k ≥ 4) | 3 min |
| `sharpness.py` | Proposition 3.8: on the Fano plane the branch number can be k + 1, so k ≥ 4 is sharp | 5 min |
| `casec.py` | Corollary 3.9: a bijection with no diagonal one at p = 5, and the branch numbers of its shifts | 15 min |
| **The Radon matrix (Section 4)** | | |
| `cost.py` | Algorithms 1 and 2 against dense linear algebra, and the operation counts of Proposition 4.7 for an implementation without the shift by zero | 1 min |
| `cost_literal.py` | Proposition 4.7: the exact counts of Algorithms 1 and 2 as printed (inverse p³ + 3p² + 3p − 3 additions, 3p² + 2p + 2 multiplications, p + 1 inversions) | seconds |
| `fourier_slice.py` | The Fourier-slice identity for M v (Section 4.3) | seconds |
| `kernel_jordan.py` | Proposition 4.3 and Remark 4.4: ker M = GRM_p(p−2, 2), and the Jordan forms of M and of the diagonal-free N | seconds |
| `diagfree.py` | Proposition 4.10 and Remark 4.11: the diagonal-free identification, its inverse, and branch number p + 2 at p = 5, 7 (`--quick` skips p = 7) | 18 min |
| **Projective plane (Section 5)** | | |
| `pg.py`, `pg2.py` | Theorem 5.1 for PG(2,q): identities, ranks, minimal polynomial, invertible shifts, explicit inverse, branch numbers | seconds |
| **Invariant subspaces (Section 6)** | | |
| `invariant.py` | All invariant partitions at p = 3 | seconds |
| `invariant2.py` | Theorem 6.3: the criterion against a direct check on random partitions | seconds |
| `signed.py` | Theorem 6.6: the signed criterion, and signed subspaces the unsigned one misses | 2 min |
| **The permutation and the sponge (Section 8)** | | |
| `constants.py` | Proposition 8.2: the round constants separate every pair of cells | seconds |
| `constants2.py` | Proposition 8.3: the constants and the signed-partition subspaces | 3 min |
| `round_invariant.py` | The witness of Proposition 8.3(iii), Proposition 8.4 (per-round subspaces W_r), and dim W_A(D) for the criterion of Beierle et al. | 1 min |
| `params.py` | Table 3: instance parameters and round counts | seconds |
| `compare.py` | Table 4: comparison with AES-like layouts over F_p and F_{p²} and a single F_{p²}-MDS matrix, one row per p and layout; the F_{p²} rows of Table 1 | seconds |
| `instance23.py` | Reference implementation of the p = 23 instance: inverse, encoding, avalanche, output statistics | seconds |
| `singlepath.py` | Propositions 8.6 and 8.9, Lemma 8.7 (first part), Corollary 8.8 without round-skipping: three-round integrals and the reach of zero-sums | 3 min |
| `zerosum_skip.py` | Lemma 8.7 (second part) and Corollary 8.8: round-skipping, the rows of Table 6 marked "chosen", six rounds within 2^256 | 6 min |
| `integral.py` | Integral and zero-sum distinguishers at p = 5, 7 (Table 6) | 30 min |
| `zerosum.py` | The same pushed to larger k, at p = 5, 7, 11 | over 90 min |
| `rebound.py` | The counting bound for rebound attacks and the size of the algebraic system | seconds |
| `rebound2.py` | Proposition 8.10 and Figure 12: the five-round rebound distinguisher, built and checked at p = 11, 13, 23 | 1 min |
| `subspacetrail.py` | Subspace trails after Proposition 8.11 | 4 min |
| `eigvec_search.py` | The signed eigenvectors and the invariant-line search after Proposition 8.11 | seconds |
| `unidegree.py` | Lemma 8.12 and Figure 13: degree of the one-cell CICO equation | 20 min |
| `groebner.py` | Gröbner bases of reduced CICO instances (needs Singular) | minutes |
| `groebner_fe.py` | Table 7: the same with field equations (needs msolve and SymPy) | 30 min |
| **Two-round bounds (Section 9)** | | |
| `hull_check.py` | Theorem 9.3 on a toy two-cell SPN | 4 min |
| `tworound.py` | Theorem 9.3: exact two-round EDP for small MDS and non-MDS layers | seconds |
| `tworound2.py` | Theorem 9.3 and Lemma 9.2 for the Fano incidence layer, and the Singleton bound on dim U | 5 min |
| `tightness.py` | Theorem 9.3: exact two-round EDP against the trail, global and per-pair bounds, up to F_17 | 1 min |
| `edp_radon.py` | Exact two-round EDP of M + αI from weight-one inputs at p = 5, 7 (Section 9) | seconds |
| `hull_linear.py` | The linear half of Theorem 9.3 and formula (15): exact ELP for three-cell layers | 30 s |

`rebound2.py` and `unidegree.py` also write the data for Figures 12 and 13 to
`figdata/`.

## Conventions

A state of p² cells is indexed by (y, x) in row-major order, cell
`y * p + x`. The Radon matrix M has rows indexed by lines (m, c) and columns by
points, with M[(m, c), (mx + c, x)] = 1. The layers used throughout are
A = M + αI, mostly with α = 2, and the S-box is x ↦ x^(p−2), the inverse map
with 0 ↦ 0.

`diagfree.py`, `kernel_jordan.py` and `arcs.py` also use the diagonal-free
identification of Proposition 4.10: with e a non-square, cell (y, x) is
identified with the line ℓ_{x, y−e}, so that
N[(y, x), (y′, x′)] = 1[y′ = x x′ + y − e]. Its rows are those of M, permuted.
The other scripts use M.

## Citation

If you use this code, please cite the paper above.




