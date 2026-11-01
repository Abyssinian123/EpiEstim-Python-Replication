# -*- coding: utf-8 -*-
"""Faithful Python port of R's loess() for the 1-dimensional case
(span default 0.75, tricube, degrees 0/1/2, family "gaussian" without
robustness iterations, surface "interpolate"), following R's
src/library/stats/src/loessf.f (Cleveland, Grosse & Shyu; R Core Team).

The ported path is exactly the one R executes for predict() on a 1-D
loess fit built with the defaults:

  loessc.c: nf = min(n, floor(n*span + 1e-5))          (q = 3 for n = 5)
  lowesb -> ehg131 -> ehg126 (bbox vertices) -> ehg124/ehg125
            (median-split k-d tree, "ties go with the hi son",
             degenerate splits at a boundary vertex become leaves)
  ehg139 -> ehg127: at each vertex, fit the local polynomial of the
            given degree on the nf nearest points with tricube weights
            w = (1 - (d/dmax)^3)^3; the linear system is solved on the
            column-equilibrated design with the singular values below
            sigma_1 * 100 * 2^-52 dropped, then compensated
            (so in the singular degree-2 case the result is the
            D-weighted minimum-norm solution, as in R).
  lowese -> ehg133 -> ehg128 (d = 1): Hermite interpolation between
            the two vertices bracketing the evaluation point, using
            the local fit value and slope at each vertex. No
            extrapolation (R errors on it).

Only predictions inside the bounding box are valid, which is always
the case in the loess study (predictions at the fit x values).

Validated against the paper's stored loess_rt.pdf curves
(validate_loess_vs_pdf.py): all degrees and both options agree to
< 0.005 PDF points (~3e-5 in R units).
"""

import numpy as np

_MACHEP = 2.220446049250313e-16   # R: d1mach(4)^-1 = 2^-52


def _fit_at_vertex(x, y, q, nf, span, degree):
    """ehg127: local polynomial fit centred at vertex q."""
    d2 = (x - q) ** 2
    idx = np.argsort(d2, kind="stable")[:nf]
    rho = d2[idx[nf - 1]] * max(1.0, span)
    u = np.sqrt(d2[idx] / rho)
    w = np.sqrt((1.0 - u ** 3) ** 3)          # rw == 1 (no robustness iter.)

    cols = [w]
    if degree >= 1:
        cols.append(w * (x[idx] - q))
    if degree >= 2:
        cols.append(w * (x[idx] - q) ** 2)
    b = np.column_stack(cols)
    colnor = np.linalg.norm(b, axis=0)
    colnor[colnor <= 0] = 1.0

    # R solves the equilibrated system with a pseudoinverse (sigma_j
    # dropped when <= sigma_1 * 100 * 2^-52) and compensates afterwards;
    # mathematically that is the colnor-weighted minimum-norm solution.
    bsc = b / colnor
    c_tilde = np.linalg.pinv(bsc, rcond=100.0 * _MACHEP) @ (w * y[idx])
    s = c_tilde / colnor
    return s                     # coefficients of 1, (x-q), (x-q)^2, ...


def r_loess(x, y, degree, x_eval, span=0.75):
    """R loess() + predict() equivalent (1-D). Returns predictions at
    x_eval. Mirrors lowesd/ehg126/ehg124/ehg125/ehg139/ehg127/ehg128."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)
    n = len(x)
    nf = min(n, int(np.floor(n * span + 1e-5)))

    # ---- bbox (ehg126): vertices 1 and 2 just outside the data range
    rng = x.max() - x.min()
    mu = 0.005 * max(rng, 1e-10 * max(abs(x.min()), abs(x.max())) + 1e-30)
    vert = [x.min() - mu, x.max() + mu]

    # ---- k-d tree (ehg124/ehg125); node 1 = root
    a, xi, lo, hi, cell = [0], [0.0], [0], [0], [[0, 1]]
    l, u = [0], [0]                    # data index range (1-based here)
    l[0], u[0] = 0, n - 1              # 0-based internally
    p = 0
    while p < len(a):
        is_leaf = False
        diameter = vert[cell[p][1]] - vert[cell[p][0]]
        if (u[p] - l[p]) + 1 <= 0 or diameter <= 0.0:
            is_leaf = True
        else:
            m = (l[p] + u[p]) // 2
            # median x value in the cell (ehg106; distinct values here)
            xs = np.sort(x[l[p]:u[p] + 1])
            xm = xs[m - l[p]]
            if vert[cell[p][0]] == xm or vert[cell[p][1]] == xm:
                is_leaf = True          # split at a boundary -> degenerate
            else:
                new_v = len(vert)
                vert.append(xm)
                left, right = len(a), len(a) + 1
                a.append(0); xi.append(0.0)
                lo.append(0); hi.append(0)
                l.append(l[p]); u.append(m)
                cell.append([cell[p][0], new_v])
                a.append(0); xi.append(0.0)
                lo.append(0); hi.append(0)
                l.append(m + 1); u.append(u[p])
                cell.append([new_v, cell[p][1]])
                a[p] = 1
                xi[p] = xm
                lo[p], hi[p] = left, right
        if is_leaf:
            a[p] = 0
        p += 1

    # ---- vertex fits (ehg139/ehg127)
    vval = []
    for q in vert:
        s = _fit_at_vertex(x, y, q, nf, span, degree)
        vval.append(s)

    # ---- evaluation (ehg128, d = 1): walk to the leaf, Hermite blend
    out = np.empty(len(x_eval))
    for i, z in enumerate(x_eval):
        t = 0
        while a[t] != 0:
            t = lo[t] if z <= xi[t] else hi[t]
        vl, vr = vert[cell[t][0]], vert[cell[t][1]]
        gl, gr = vval[cell[t][0]], vval[cell[t][1]]
        # ehg127 sets s(j) = 0 for j >= k: a degree-0 fit has slope 0, a
        # degree-1 fit no quadratic term, so only the value and (if the
        # degree is >= 1) the slope enter the Hermite blend.
        slope_l = gl[1] if len(gl) > 1 else 0.0
        slope_r = gr[1] if len(gr) > 1 else 0.0
        h = (z - vl) / (vr - vl)
        if h < -0.001 or h > 1.001:
            raise ValueError(
                "extrapolation not allowed with blending (R loess error)")
        phi0 = (1 - h) ** 2 * (1 + 2 * h)
        phi1 = h ** 2 * (3 - 2 * h)
        psi0 = h * (1 - h) ** 2
        psi1 = h ** 2 * (h - 1)
        out[i] = (phi0 * gl[0] + phi1 * gr[0]
                  + (psi0 * slope_l + psi1 * slope_r) * (vr - vl))
    return out


def loess_predict(x, y, degree, x_eval, span=0.75):
    """Convenience alias of r_loess."""
    return r_loess(x, y, degree, x_eval, span)


if __name__ == "__main__":
    # The authoritative check is validate_loess_vs_pdf.py, which compares
    # the full pipeline (loess degrees 0/1/2 -> estimate_R) with the curves
    # stored in the paper's loess_rt.pdf: agreement to <0.005 PDF points.
    x = np.array([7.0, 14.0, 21.0, 28.0, 35.0])
    y = np.array([12.87752, 13.46683, 14.02367, 12.70065, 10.87572])
    for d in (0, 1, 2):
        pred = r_loess(x, y, d, np.arange(7.0, 36.0))
        print(f"degree {d}: pred[7] = {pred[0]:.10f}  pred[35] = {pred[-1]:.10f}")
