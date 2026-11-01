# -*- coding: utf-8 -*-
"""Validate loess.py against the paper's stored loess_rt.pdf vector curves
(no R needed).

Strategy: the grey (reported daily) and black (EM reconstruction) curves are
known exactly from the stored RDS files; they calibrate the affine
PDF->data map for each panel.  The red/dodgerblue/magenta curves are the
LOESS pipeline (degree 0/1/2, options 1/2) which is what we verify, plus
the ribbons (95% CrI) which we check against the extracted polygon fills.

PDF geometry (cairo_pdf, top-of-page coordinates):
  panel A (opt1, top): y in [5.5, 203.1], curves grey t_end 8..35,
                       black/red/blue/magenta t_end 14..35 (t_end + 6)
  panel B (opt2, bottom): y in [257.5, 455.1], loess curves 10..31
                          (t_end + 2)
  shared x-scale: 10/20/30 ticks at pdf x ~127.5/350.3/573.1
"""
import os
import sys

import numpy as np
import pdfplumber
import pyreadr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from epiestim.rds_reader import read_rds, list_to_dict, to_df  # noqa: E402
from epiestim import estimate_R, aggregate_inc  # noqa: E402
from epiestim.loess import r_loess  # noqa: E402

PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "EM_EpiEstim_Nash2023-main", "supplementary_figures", "loess_rt.pdf")
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "EM_EpiEstim_Nash2023-main")

GREY = (0.745, 0.745, 0.745)
BLACK = (0.0, 0.0, 0.0)
RED = (1.0, 0.0, 0.0)
BLUE = (0.118, 0.565, 1.0)
MAGENTA = (1.0, 0.0, 1.0)
LINE_COLORS = {"grey": GREY, "black": BLACK, "red": RED,
               "blue": BLUE, "magenta": MAGENTA}


def _col(c):
    if c is None:
        return None
    if isinstance(c, (int, float)):      # gray level
        return (round(float(c), 3),) * 3
    return tuple(round(float(x), 3) for x in c)


def pdf_curves():
    """Return {panel: {color: (line_pts, ribbon_polys)}}.

    Stroked-only paths carry fill = 0 (int, the PDF default); the
    geom_ribbon polygons carry a tuple fill."""
    with pdfplumber.open(PDF) as pdf:
        page = pdf.pages[0]
        objs = page.objects["curve"]
    out = {"A": {}, "B": {}}
    for obj in objs:
        fill = obj.get("non_stroking_color")
        sc = _col(obj.get("stroking_color"))
        pts = np.array(obj["pts"])
        if len(pts) <= 2 or pts[:, 0].min() > 600:   # legend swatches
            continue
        panel = "A" if pts[:, 1].mean() < 230 else "B"
        if isinstance(fill, (tuple, list)):
            fc = _col(fill)
            entry = out[panel].setdefault(fc, (None, []))
            entry[1].append(pts)
        else:
            entry = out[panel].setdefault(sc, (None, []))
            out[panel][sc] = (pts, entry[1])
    return out


def known_data():
    """(t_end, Mean(R), Q2.5, Q97.5) for grey (r_rep) and black (r_em)
    from the stored RDS files."""
    r = list_to_dict(read_rds(
        f"{REPO}/supplementary_analysis/flu/r_rep_sliding_result.rds"))
    rr = to_df(r["R"])
    grey = (np.array(rr["t_end"], float), np.array(rr["Mean(R)"], float),
            np.array(rr["Quantile.0.025(R)"], float),
            np.array(rr["Quantile.0.975(R)"], float))
    em = list_to_dict(read_rds(
        f"{REPO}/supplementary_analysis/flu/R_agg_sliding_result.rds"))
    ee = to_df(em["R"])
    black = (np.array(ee["t_end"], float), np.array(ee["Mean(R)"], float),
             np.array(ee["Quantile.0.025(R)"], float),
             np.array(ee["Quantile.0.975(R)"], float))
    return {"grey": grey, "black": black}


def my_loess_curves():
    """My pipeline: naive weekly/7 -> loess d=0/1/2 at opt1/opt2 ->
    estimate_R -> (t_end+6 | +2, Mean(R), Q2.5, Q97.5)."""
    flu = pyreadr.read_r(f"{REPO}/real_data/daily_flu.rds")[None]
    incid = flu["Incidence"].to_numpy(float)
    weekly = aggregate_inc(incid, dt=7)
    naive = weekly / 7
    out = {}
    for opt_name, opt, offset in (("opt1", np.arange(7.0, 36.0, 7.0), 6),
                                  ("opt2", np.arange(3.0, 32.0, 7.0), 2)):
        days = np.arange(opt[0], opt[-1] + 1.0)
        for d in (0, 1, 2):
            y = r_loess(opt, naive, d, days)
            res = estimate_R(y, mean_si=3.6, std_si=1.6)["R"]
            out[(opt_name, d)] = (
                res["t_end"].to_numpy(float) + offset,
                res["Mean(R)"].to_numpy(float),
                res["Quantile.0.025(R)"].to_numpy(float),
                res["Quantile.0.975(R)"].to_numpy(float))
    return out


def fit_map(te, r, pts):
    """Affine least-squares map (t_end, R) -> (pdf_x, pdf_y), using the
    polyline points in data order."""
    m = min(len(te), len(pts))
    A = np.column_stack([te[:m], r[:m], np.ones(m)])
    sol, res, rank, sv = np.linalg.lstsq(A, pts[:m], rcond=None)
    resid = np.sqrt(((A @ sol - pts[:m]) ** 2).sum(1))
    return sol, resid.max()


def curve_dist(pred_pts, pdf_pts):
    """Mean and max distance from predicted pdf points to the extracted
    polyline (point-to-segment)."""
    v = pdf_pts[1:] - pdf_pts[:-1]
    vv = (v * v).sum(1)
    dists = []
    for p in pred_pts:
        w = p - pdf_pts[:-1]
        t = np.clip((w * v).sum(1) / np.maximum(vv, 1e-12), 0.0, 1.0)
        d = np.sqrt(((w - t[:, None] * v) ** 2).sum(1))
        dists.append(d.min())
    dists = np.array(dists)
    return dists.mean(), dists.max()


def main():
    curves = pdf_curves()
    known = known_data()
    mine = my_loess_curves()

    # sanity: report map fit quality from the known curves
    print("calibration (known curves -> extracted pdf), max resid in pt:")
    for panel in ("A", "B"):
        for cname in ("grey", "black"):
            te, r, _, _ = known[cname]
            pts = curves[panel][LINE_COLORS[cname]][0]
            _, mx = fit_map(te, r, pts)
            print(f"  panel {panel} {cname}: {mx:.3f}")

    print("\nLOESS pipeline vs paper PDF (distances in pt, y ~ 120 pt per "
          "R unit):")
    ok = True
    for panel, opt_name in (("A", "opt1"), ("B", "opt2")):
        # per-panel map from grey+black (shared affine)
        sols = []
        for cname in ("grey", "black"):
            te, r, _, _ = known[cname]
            pts = curves[panel][LINE_COLORS[cname]][0]
            sol, _ = fit_map(te, r, pts)
            sols.append(sol)
        sol = np.mean(sols, axis=0)
        for d, cname in ((0, "red"), (1, "blue"), (2, "magenta")):
            te, r, q2, q97 = mine[(opt_name, d)]
            pred = np.column_stack([te, r, np.ones(len(te))]) @ sol
            pdf_pts = curves[panel][LINE_COLORS[cname]][0]
            mean_d, max_d = curve_dist(pred, pdf_pts)
            # ribbon: upper = polygon first half, lower = second half
            rib = curves[panel][LINE_COLORS[cname]][1]
            mean_u = mean_l = max_u = max_l = np.nan
            if rib:
                pu = rib[0]
                n = len(pu) // 2
                du = curve_dist(np.column_stack([te, q97, np.ones(len(te))])
                                @ sol, pu[:n])
                dl = curve_dist(np.column_stack([te, q2, np.ones(len(te))])
                                @ sol, pu[n:2 * n])
                mean_u, max_u, mean_l, max_l = du[0], du[1], dl[0], dl[1]
            status = "OK" if max_d < 1.5 else "MISMATCH"
            ok &= max_d < 1.5
            print(f"  panel {panel} d={d}: R line mean {mean_d:.3f} max "
                  f"{max_d:.3f} | CrI upper mean {mean_u:.3f} max {max_u:.3f}"
                  f" | CrI lower mean {mean_l:.3f} max {max_l:.3f}  [{status}]")
    print("\nLOESS PORT VALIDATED" if ok else "\nLOESS PORT MISMATCH")


if __name__ == "__main__":
    main()
