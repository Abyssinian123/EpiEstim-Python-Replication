# -*- coding: utf-8 -*-
"""Decode weekend_effects.pdf.

Figure layout: columns const/sud/grad; rows 2-4 nonsliding/sliding/twsliding.

Round 3: fit the affine map from the GREEN line vs the TRUE vectors
(exact values, no clipping), then identify the black dashed line and the
ribbons in data space."""
import os
import sys

import numpy as np
import pdfplumber

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from epiestim.rds_reader import read_rds, list_to_dict, to_df  # noqa: E402

PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "EM_EpiEstim_Nash2023-main", "supplementary_figures", "weekend_effects.pdf")
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "EM_EpiEstim_Nash2023-main", "supplementary_analysis", "weekend_effects")

DGREY = (0.66, 0.66, 0.66)
GREEN = (0.0, 0.39, 0.0)
GFILL = (0.24, 0.7, 0.44)
GRFILL = (0.75, 0.75, 0.75)
BLACK = (0.0, 0.0, 0.0)

COL_SC = {1: "const", 2: "sud", 3: "grad"}
ROW_KIND = {2: ("nonsliding", 9), 3: ("sliding", 14), 4: ("twsliding", 21)}

TRUE = {
    "const": np.repeat(1.5, 70),
    "sud": np.r_[np.repeat(1.5, 35), np.repeat(1.25, 35)],
    "grad": np.r_[np.repeat(1.25, 18),
                  np.linspace(1.25, 1.5, 32), np.repeat(1.5, 20)],
}


def stored(scenario, kind):
    r = list_to_dict(read_rds(os.path.join(
        DIR, f"result_rt_{scenario}_{kind}.rds")))
    te, mr = [], []
    for sim in r:
        R = to_df(list_to_dict(sim)["R"])
        te.append([float(x) for x in R["t_end"]])
        mr.append([float(x) for x in R["Mean(R)"]])
    return np.array(te), np.array(mr)


def get_curves():
    with pdfplumber.open(PDF) as pdf:
        page = pdf.pages[0]
        objs = []
        for otype in ("curve", "line", "rectline", "polyline"):
            objs += [(otype, o) for o in page.objects.get(otype, [])]
    out = []
    for otype, obj in objs:
        sc = obj.get("stroking_color")
        if sc is None:
            continue
        sc = tuple(round(float(c), 2) for c in sc)
        fc = obj.get("non_stroking_color")
        fc = (tuple(round(float(c), 2) for c in fc)
              if isinstance(fc, (tuple, list)) else None)
        pts = np.array(obj["pts"])
        if len(pts) < 2:
            continue
        out.append((sc, fc, pts, otype))
    return out


def panel_of(pts):
    x, y = pts[:, 0].mean(), pts[:, 1].mean()
    row = 1 if y < 200 else 2 if y < 430 else 3 if y < 630 else 4
    col = 1 if x < 330 else 2 if x < 620 else 3
    return row, col


def fit(te, r, pts):
    m = min(len(te), len(pts))
    A = np.column_stack([te[:m], r[:m], np.ones(m)])
    best = (None, 1e9, None)
    for B, tag in ((pts[:m], "fwd"), (pts[:m][::-1], "rev")):
        sol, *_ = np.linalg.lstsq(A, B, rcond=None)
        res = np.sqrt(((A @ sol - B) ** 2).sum(1)).max()
        if res < best[1]:
            best = (sol, res, tag)
    return best


def main():
    curves = get_curves()
    panels = {}
    for sc, fc, pts, otype in curves:
        row, col = panel_of(pts)
        panels.setdefault((row, col), []).append((sc, fc, pts, otype))

    for row in (2, 3, 4):
        kind, minte = ROW_KIND[row]
        for col in (1, 2, 3):
            sc = COL_SC[col]
            objs = panels.get((row, col), [])
            print(f"\n== panel ({row},{col}) {sc} {kind}")
            # 1) map from green vs true (both t_end 9..70 / 14..70 / 21..70)
            for o in objs:
                if o[0] != GREEN or len(o[2]) < 20:
                    continue
                te_t = np.arange(1, 71.0)[minte - 1:]
                sol, res, tag = fit(te_t, TRUE[sc][minte - 1:], o[2])
                print(f"  map from GREEN vs true_{sc}: maxres {res:.3f} pt "
                      f"({tag})  ylim -> R=0 at y={sol[2][1]:.1f}, "
                      f"R=1.5 at y={sol[2][1]+1.5*sol[1][1]:.1f}")
                # 2) black dashes in data coords
                for bo in objs:
                    if bo[0] == BLACK and bo[1] in (None, (0.0, 0.0, 0.0)):
                        p = bo[2]
                        xd = (p[:, 0] - sol[2][0]) / sol[0][0]
                        yd = (p[:, 1] - sol[2][1]) / sol[1][1]
                        if xd.min() > 8 and xd.max() < 65:
                            print(f"   black dash n={len(p)}: "
                                  f"t_end[{xd.min():.1f},{xd.max():.1f}] "
                                  f"R[{yd.min():.2f},{yd.max():.2f}]")
                # 3) ribbons in data coords
                for bo in objs:
                    if bo[0] == BLACK and bo[1] in (GFILL, GRFILL):
                        p = bo[2]
                        xd = (p[:, 0] - sol[2][0]) / sol[0][0]
                        yd = (p[:, 1] - sol[2][1]) / sol[1][1]
                        if len(p) < 20:
                            continue
                        tagr = "weekly(agg?)" if bo[1] == GFILL else "daily(ref)"
                        print(f"   ribbon {tagr} n={len(p)}: "
                              f"t_end[{xd.min():.1f},{xd.max():.1f}] "
                              f"R[{yd.min():.2f},{yd.max():.2f}]")
                break  # only the main green object


if __name__ == "__main__":
    main()
